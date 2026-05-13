# Copyright 2026 Jeremy Lilley (jeremy@jlilley.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core classifier: detects whether CJ text is Japanese, Chinese Simplified,
or Chinese Traditional.

Classification of ideographs is based on a unigram + bigram statistical model
built from Japanese and Chinese language Wikipedia corpora. For every character
and character-pair in the CJ range, the model stores per-language
log-probabilities. At classification time the library sums these
log-probabilities across the input and picks the language with the highest
score."""

from __future__ import annotations

import array
import gzip
import importlib.resources
import io
import logging
from typing import Dict, List, Optional, Tuple

from cjclassifier.language import CJLanguage

logger = logging.getLogger(__name__)

_CJ_RANGE_START = 0x3400
_CJ_RANGE_END = 0x9FFF
_CJ_RANGE_SIZE = _CJ_RANGE_END - _CJ_RANGE_START + 1
_LANG_COUNT = 3  # zh-hans=0, zh-hant=1, ja=2

CJ_LANGUAGES: List[CJLanguage] = [
    CJLanguage.CHINESE_SIMPLIFIED,
    CJLanguage.CHINESE_TRADITIONAL,
    CJLanguage.JAPANESE,
]

# Singleton cache
_cache: Dict[str, "CJClassifier"] = {}


def _is_kana(c: int) -> bool:
    """Check if a Unicode code point is Hiragana, Katakana, or half-width Katakana."""
    if 0x3040 <= c <= 0x30FF:
        return True
    if 0x31F0 <= c <= 0x31FF:
        return True
    if 0xFF65 <= c <= 0xFF9F:
        return True
    return False


def _in_main_cj_range(c: int) -> bool:
    """Check if a Unicode code point is in the main CJ Unified Ideograph range."""
    return _CJ_RANGE_START <= c <= _CJ_RANGE_END


# ========================================================================
# BigramMap: open-addressing hash map using flat array.array storage
# ========================================================================
# We could just use a Dict[int, int] here, but this cuts the model's
# memory overhead from 109MB to 29MB, since all the int boxing and
# hashtable overhead adds up.
#
# This is just a port of the Java BigramToFloatArrayMap. It avoids Python
# object boxing by storing keys and value-offsets in array.array('i')
# (4 bytes per slot) instead of a dict of Python int objects
# (~28-32 bytes each).

_EMPTY = 0
_MAX_LOAD_FACTOR = 0.75


def _mix(k: int) -> int:
    """Hash mixing function to spread bits of the key."""
    k = (k ^ (k >> 16)) & 0xFFFFFFFF
    k = (k * 0x85EBCA6B) & 0xFFFFFFFF
    k = (k ^ (k >> 13)) & 0xFFFFFFFF
    return k


def _table_size_for(expected: int) -> int:
    """Return the smallest power of 2 that can hold expected entries at <=75% load."""
    min_cap = int(expected / _MAX_LOAD_FACTOR) + 1
    cap = 1
    while cap < min_cap:
        cap <<= 1
    return max(cap, 16)


class _BigramMap:
    """Open-addressing hash map: bigram key (int) -> offset into prob_data.

    Keys and value-offsets are stored in array.array('i') to avoid the overhead
    of Python int objects. Probabilities are in a separate array.array('f').
    """

    __slots__ = ("_keys", "_value_indices", "prob_data", "_mask")

    def __init__(self, keys, value_indices, prob_data, mask):
        self._keys = keys
        self._value_indices = value_indices
        self.prob_data = prob_data
        self._mask = mask

    def get_offset(self, c1: int, c2: int) -> int:
        """Look up the offset for a bigram. Returns 0 if not present."""
        key = (c1 << 16) | c2
        keys = self._keys
        value_indices = self._value_indices
        mask = self._mask
        idx = _mix(key) & mask
        while True:
            k = keys[idx]
            if k == key:
                return value_indices[idx]
            if k == _EMPTY:
                return 0
            idx = (idx + 1) & mask


class _BigramMapBuilder:
    """Mutable builder for constructing a _BigramMap."""

    __slots__ = ("_keys", "_value_indices", "_prob_data", "_prob_data_size",
                 "_size", "_mask", "_threshold")

    def __init__(self, expected_size: int):
        capacity = _table_size_for(expected_size)
        self._keys = array.array('I', [0]) * capacity
        self._value_indices = array.array('I', [0]) * capacity
        self._mask = capacity - 1
        self._threshold = int(capacity * _MAX_LOAD_FACTOR)
        self._prob_data = array.array('f', [0.0])  # offset 0 reserved
        self._prob_data_size = 1
        self._size = 0

    def put(self, c1: int, c2: int, probs: List[float]):
        """Store probabilities for a bigram key. Copies from probs list."""
        key = (c1 << 16) | c2
        if self._size >= self._threshold:
            self._resize()
        idx = _mix(key) & self._mask
        while True:
            k = self._keys[idx]
            if k == _EMPTY:
                self._keys[idx] = key
                self._value_indices[idx] = len(self._prob_data)
                self._prob_data.extend(probs)
                self._size += 1
                return
            if k == key:
                # Update in place
                off = self._value_indices[idx]
                for i in range(len(probs)):
                    self._prob_data[off + i] = probs[i]
                return
            idx = (idx + 1) & self._mask

    def build(self) -> _BigramMap:
        """Build an immutable _BigramMap."""
        return _BigramMap(self._keys, self._value_indices,
                          self._prob_data, self._mask)

    def _resize(self):
        new_capacity = (self._mask + 1) << 1
        old_keys = self._keys
        old_value_indices = self._value_indices
        self._keys = array.array('I', [0]) * new_capacity
        self._value_indices = array.array('I', [0]) * new_capacity
        self._mask = new_capacity - 1
        self._threshold = int(new_capacity * _MAX_LOAD_FACTOR)
        self._size = 0
        for i in range(len(old_keys)):
            if old_keys[i] != _EMPTY:
                self._rehash_put(old_keys[i], old_value_indices[i])

    def _rehash_put(self, key: int, value_index: int):
        idx = _mix(key) & self._mask
        while True:
            if self._keys[idx] == _EMPTY:
                self._keys[idx] = key
                self._value_indices[idx] = value_index
                self._size += 1
                return
            idx = (idx + 1) & self._mask


class Scores:
    """Accumulates per-language scoring data during CJ text processing."""

    __slots__ = (
        "unigram_scores", "bigram_scores",
        "unigram_hits_per_lang", "bigram_hits_per_lang",
        "kana_count", "cj_char_count",
    )

    def __init__(self):
        self.unigram_scores = [0.0] * _LANG_COUNT
        self.bigram_scores = [0.0] * _LANG_COUNT
        self.unigram_hits_per_lang = [0] * _LANG_COUNT
        self.bigram_hits_per_lang = [0] * _LANG_COUNT
        self.kana_count = 0
        self.cj_char_count = 0

    def clear(self):
        """Reset all accumulated scores and counts to zero."""
        for i in range(_LANG_COUNT):
            self.unigram_scores[i] = 0.0
            self.bigram_scores[i] = 0.0
            self.unigram_hits_per_lang[i] = 0
            self.bigram_hits_per_lang[i] = 0
        self.kana_count = 0
        self.cj_char_count = 0

    def any_hits(self) -> bool:
        """True if at least one unigram was matched in any language."""
        return max(self.unigram_hits_per_lang) > 0


class Results:
    """Detection results, including accumulated Scores and the computed result."""

    __slots__ = ("scores", "total_scores", "boosts", "result", "gap")

    def __init__(self, scores: Optional[Scores] = None):
        self.scores = scores if scores is not None else Scores()
        self.total_scores = [0.0] * _LANG_COUNT
        self.boosts = [0.0] * _LANG_COUNT
        self.result: Optional[CJLanguage] = None
        self.gap: float = 0.0

    def clear(self):
        """Reset all scores, boosts, and the computed result."""
        self.scores.clear()
        for i in range(_LANG_COUNT):
            self.total_scores[i] = 0.0
            self.boosts[i] = 0.0
        self.result = None
        self.gap = 0.0

    def _compute_totals(self, placeholder_score: float):
        max_unigram = max(self.scores.unigram_hits_per_lang)
        max_bigram = max(self.scores.bigram_hits_per_lang)
        for i in range(_LANG_COUNT):
            self.total_scores[i] = (
                self.scores.unigram_scores[i]
                + (max_unigram - self.scores.unigram_hits_per_lang[i]) * placeholder_score
                + self.scores.bigram_scores[i]
                + (max_bigram - self.scores.bigram_hits_per_lang[i]) * placeholder_score
            )
            # Implement boosts: since logprob values are negative, a favorable boost
            # is negative.
            self.total_scores[i] -= self.boosts[i] * self.total_scores[i]

    def to_short_string(self) -> str:
        """Compact comma-separated representation of per-language relative scores.

        Example: 'zh-hans:1.00,zh-hant:0.97,ja:0.85'
        Returns '' if no result has been computed or the result is UNKNOWN.
        """
        if self.result is None or self.result is CJLanguage.UNKNOWN:
            return ""
        if self.scores.kana_count > 0 and self.result is CJLanguage.JAPANESE:
            return "ja:1.0,zh-hans:0,zh-hant:0"

        # Sort indices by total_scores descending
        order = sorted(range(_LANG_COUNT), key=lambda i: self.total_scores[i], reverse=True)
        best = self.total_scores[order[0]]
        if best == 0:
            return ""

        parts = []
        for li in order:
            ratio = best / self.total_scores[li] if self.total_scores[li] != 0 else 0
            parts.append(f"{CJ_LANGUAGES[li].iso_code}:{ratio:.2f}")
        return ",".join(parts)

    def __repr__(self) -> str:
        code = self.result.iso_code if self.result is not None else "None"
        parts = [f"Results(result={code}"]
        if self.scores.kana_count > 0:
            total = self.scores.kana_count + self.scores.cj_char_count
            parts.append(f" kana={self.scores.kana_count}/{total}")
        for li in range(_LANG_COUNT):
            parts.append(
                f" | {CJ_LANGUAGES[li].iso_code}:"
                f" uni={self.scores.unigram_scores[li]:.2f}"
                f" bi={self.scores.bigram_scores[li]:.2f}"
                f" total={self.total_scores[li]:.2f}"
                f" biHits={self.scores.bigram_hits_per_lang[li]}"
            )
        parts.append(")")
        return "".join(parts)


class CJClassifier:
    """Detects whether CJ text is Japanese, Chinese Simplified, or Chinese Traditional.

    Load with ``CJClassifier.load()``. The model is cached as a singleton.
    Call ``detect(text)`` to classify text.

    Example::

        cjc = CJClassifier.load()
        cjc.detect("今天天气很好")  # => CJLanguage.CHINESE_SIMPLIFIED
        cjc.detect("事務所")        # => CJLanguage.JAPANESE
    """

    # Class-level constants exposed for user code
    CJ_LANGUAGES = CJ_LANGUAGES

    def __init__(
        self,
        unigram_log_probs: List[float],
        bigram_map: _BigramMap,
        default_log_prob: float,
    ):
        self._unigram_log_probs = unigram_log_probs
        self._bigram_map = bigram_map
        self._default_log_prob = default_log_prob
        self._tolerated_kana_threshold = 0.01

    @property
    def tolerated_kana_threshold(self) -> float:
        """Kana fraction threshold above which text is classified as Japanese."""
        return self._tolerated_kana_threshold

    @tolerated_kana_threshold.setter
    def tolerated_kana_threshold(self, value: float):
        self._tolerated_kana_threshold = value

    # ========================================================================
    # Loading
    # ========================================================================

    @classmethod
    def load(cls, path: Optional[str] = None, log_prob_floor: float = 0.0) -> "CJClassifier":
        """Load a CJClassifier, using a cached singleton.

        Args:
            path: Filesystem path to a model file (may be gzipped).
                  If None, loads the bundled model from the package.
            log_prob_floor: Custom log-probability floor.
                  If 0, uses the file's MinLogProb header value.

        Returns:
            A cached (or newly-loaded) CJClassifier instance.
        """
        cache_key = f"{path or 'bundled'}:{log_prob_floor}"
        if cache_key in _cache:
            return _cache[cache_key]

        if path is not None:
            instance = cls._load_from_file(path, log_prob_floor)
        else:
            instance = cls._load_bundled(log_prob_floor)

        _cache[cache_key] = instance
        return instance

    @classmethod
    def clear_cached_models(cls):
        """Clear loaded model(s), primarily for testing."""
        _cache.clear()

    @classmethod
    def _load_bundled(cls, log_prob_floor: float) -> "CJClassifier":
        """Load the bundled model from package resources."""
        # Compatible with Python 3.8+
        try:
            # Python 3.9+
            ref = importlib.resources.files("cjclassifier").joinpath("cjlogprobs.gz")
            data = ref.read_bytes()
        except AttributeError:
            # Python 3.8 fallback
            with importlib.resources.open_binary("cjclassifier", "cjlogprobs.gz") as f:
                data = f.read()

        with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
            return cls._parse_model(f, "bundled:cjlogprobs.gz", log_prob_floor)

    @classmethod
    def _load_from_file(cls, path: str, log_prob_floor: float) -> "CJClassifier":
        """Load a model from a filesystem path."""
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return cls._parse_model(f, path, log_prob_floor)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return cls._parse_model(f, path, log_prob_floor)

    @classmethod
    def _parse_model(cls, f, label: str, log_prob_floor: float) -> "CJClassifier":
        """Parse the model file format line-by-line from a file-like object."""
        # Parse header
        header = f.readline().rstrip("\n")
        if not header.startswith("Languages: "):
            raise ValueError(f"Invalid model file (bad header): {label}")

        header_parts = header.split(" ")
        lang_codes = header_parts[1].split(",")
        lang_map = []
        for code in lang_codes:
            lang = CJLanguage.from_string(code)
            if lang is CJLanguage.UNKNOWN:
                raise ValueError(f"Unknown CJ language in header: {code} in {label}")
            lang_map.append(int(lang))

        # Parse MinLogProb from header
        parsed_min_prob = None
        for i in range(len(header_parts) - 1):
            if header_parts[i] == "MinLogProb:":
                parsed_min_prob = float(header_parts[i + 1])
                break

        if log_prob_floor == 0.0:
            if parsed_min_prob is None:
                raise ValueError(
                    f"No MinLogProb in header and no explicit log_prob_floor: {label}"
                )
            default_log_prob = parsed_min_prob
        else:
            if parsed_min_prob is not None:
                default_log_prob = max(log_prob_floor, parsed_min_prob)
            else:
                default_log_prob = log_prob_floor

        # Parse unigram and bigram lines.
        # Bigram data uses an open-addressing hash map backed by array.array
        # to avoid the overhead of Python object boxing (~28 bytes per int/float
        # vs 4 bytes in a C-level array).
        unigram_log_probs = [0.0] * (_CJ_RANGE_SIZE * _LANG_COUNT)
        bigram_builder = _BigramMapBuilder(16 * 1024)

        unigram_count = 0
        bigram_count = 0
        probs = [0.0] * _LANG_COUNT  # reused across bigram lines

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            key = parts[0]
            if len(parts) != len(lang_map) + 1:
                raise ValueError(f"Column count mismatch on line: {line} in {label}")

            if len(key) == 1:
                # Unigram
                c = ord(key)
                idx = c - _CJ_RANGE_START
                if 0 <= idx < _CJ_RANGE_SIZE:
                    for col in range(len(lang_map)):
                        val = max(float(parts[col + 1]), default_log_prob)
                        unigram_log_probs[idx * _LANG_COUNT + lang_map[col]] = val
                    unigram_count += 1

            elif len(key) == 2:
                # Bigram
                any_higher = False
                for col in range(len(lang_map)):
                    val = float(parts[col + 1])
                    if val < default_log_prob or val == 0.0:
                        val = default_log_prob
                    else:
                        any_higher = True
                    probs[lang_map[col]] = val
                if any_higher:
                    bigram_builder.put(ord(key[0]), ord(key[1]), probs)
                    bigram_count += 1

        logger.info(
            "Loaded %d unigrams and %d bigrams, minLogProb: %.2f, from %s",
            unigram_count, bigram_count, default_log_prob, label,
        )

        return cls(unigram_log_probs, bigram_builder.build(), default_log_prob)

    # ========================================================================
    # Detection
    # ========================================================================

    def detect(self, text: str, results: Optional[Results] = None) -> CJLanguage:
        """Detect the language of the given CJ text.

        Args:
            text: The text to classify.
            results: Optional Results instance to populate with scoring details.

        Returns:
            The detected CJLanguage, or CJLanguage.UNKNOWN if undetermined.
        """
        if results is None:
            results = Results()
        else:
            results.clear()
        self.add_text(text, results.scores)
        return self.compute_result(results)

    def add_text(self, text: str, scores: Scores):
        """Add text to the scoring accumulator.

        This is the lower-level API for incremental classification.
        Call compute_result() when all text has been added.

        Args:
            text: A chunk of CJ text to process.
            scores: The Scores instance to accumulate into.
        """
        prev = 0
        prev_in_range = False

        unigram_log_probs = self._unigram_log_probs
        bigram_map = self._bigram_map
        bigram_prob_data = bigram_map.prob_data
        default_log_prob = self._default_log_prob

        for ch in text:
            c = ord(ch)
            if _is_kana(c):
                scores.kana_count += 1
                continue
            in_range = _in_main_cj_range(c)
            if in_range:
                scores.cj_char_count += 1
                # Unigram scoring
                idx = c - _CJ_RANGE_START
                base = idx * _LANG_COUNT
                for li in range(_LANG_COUNT):
                    u_prob = unigram_log_probs[base + li]
                    scores.unigram_scores[li] += max(u_prob, default_log_prob)
                    if u_prob != 0:
                        scores.unigram_hits_per_lang[li] += 1

                # Bigram scoring
                if prev_in_range:
                    b_off = bigram_map.get_offset(prev, c)
                    if b_off != 0:
                        for li in range(_LANG_COUNT):
                            bp = bigram_prob_data[b_off + li]
                            scores.bigram_scores[li] += max(bp, default_log_prob)
                            if bp != 0:
                                scores.bigram_hits_per_lang[li] += 1

            prev = c
            prev_in_range = in_range

    def compute_result(self, results: Results) -> CJLanguage:
        """Compute the final language result from accumulated scores.

        Args:
            results: The Results instance containing accumulated scores.

        Returns:
            The detected CJLanguage, or CJLanguage.UNKNOWN if undetermined.
        """
        scores = results.scores

        if scores.kana_count > 0:
            kana_ratio = scores.kana_count / (scores.kana_count + scores.cj_char_count)
            if kana_ratio > self._tolerated_kana_threshold:
                results.result = CJLanguage.JAPANESE
                results.gap = 1.0
                results.total_scores[CJLanguage.JAPANESE] = 1.0
                return results.result

        if not scores.any_hits():
            results.result = CJLanguage.UNKNOWN
            results.gap = 0.0
            return results.result

        results._compute_totals(self._default_log_prob)

        # Find best and second-best language
        best_idx = 0
        second_idx = -1
        for li in range(1, _LANG_COUNT):
            if results.total_scores[li] > results.total_scores[best_idx]:
                second_idx = best_idx
                best_idx = li
            elif second_idx < 0 or results.total_scores[li] > results.total_scores[second_idx]:
                second_idx = li

        results.result = CJ_LANGUAGES[best_idx]

        # Compute gap: 1 - (best / second). Scores are negative logprobs,
        # so best is least negative.
        best = results.total_scores[best_idx]
        second = results.total_scores[second_idx] if second_idx >= 0 else best
        results.gap = (1.0 - (best / second)) if second != 0.0 else 0.0

        return results.result
