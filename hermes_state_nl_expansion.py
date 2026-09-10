# Natural-language query expansion for FTS5 session search.
#
# This module implements language-agnostic NL expansion: stopword removal,
# light suffix stripping, and prefix wildcards for inflectional languages.
# Language data lives in ``_NL_LANG_PACKS`` as pure dictionaries — adding a
# language is just adding a new entry, no mechanism changes required.
#
# Usage in hermes_state_search.py:
#   try:
#       from . import hermes_state_nl_expansion as _nle
#   except ImportError:
#       _nle = None  # optional feature, don't break core search
#   ...
#   nl_support = _nle.NLSupport() if _nle else None
#   ...
#   expanded = nl_support.expand_nl_query(query) if nl_support else None

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Collection, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pack schema (all fields documented inline above each pack below)
# ---------------------------------------------------------------------------
#   stopwords           frozenset[str]   words dropped from the query
#   affinity_stopwords  frozenset[str]   small function-word set used ONLY
#                                        for language detection scoring
#   suffixes            tuple[str, ...]  light suffixes stripped first
#   endings             frozenset[str]   2-char flexion endings → drop 2
#   vowels              str              vowel set for trailing-vowel drop
#   trailing_vowel_drop bool             tail vowel counts as flexion
#   min_stem            int              shortest prefix kept (precision)
#   fallback            str              "keep" (stem already) | "drop1"


# ===========================================================================
# LANGUAGE PACKS — pure data, no logic
# ===========================================================================
_NL_LANG_PACKS: Dict[str, Dict[str, Any]] = {
    "default": {
        # English + conservative universal layer. Latin terms are usually
        # already stems after suffix strip; keep them whole with *.
        "stopwords": frozenset(
            """
            a an and are as at be but by for if in into is it no not of on or
            s such t that the their then there these they this to was we will
            with you your do does did how what why when where which who whom
            whose can could should would may might must shall please help show
            find tell explain check make give get
            """.split()
        ),
        "affinity_stopwords": frozenset(),  # default is fallback, no score
        "suffixes": ("ing", "ed", "es", "'s", "s"),
        "endings": frozenset(),
        "vowels": "",
        "trailing_vowel_drop": False,
        "min_stem": 4,
        "fallback": "keep",
    },
}


# ===========================================================================
# DETECTION
# ===========================================================================
def detect_lang(query: str) -> str:
    """Pick a language pack for the raw query (two-stage detection).

    Stage 1 — script: non-Latin scripts are unambiguous. Cyrillic anywhere
    in the query narrows candidates to ru/be/uk. Everything else falls
    through to stage 2.

    Stage 2 — affinity: each pack's ``affinity_stopwords`` (small function-
    word set) is scored against query tokens; the best-scoring pack wins.
    Ties and zero scores degrade to ``default``.
    """
    if re.search(r"[а-яё]", query, re.IGNORECASE):
        # Among Cyrillic packs, choose the best-affinity one
        tokens = set(re.findall(r"[^\W_]+", query.lower()))
        best_score, best_lang = 0, "ru"  # ru is the largest Cyrillic pack
        for lang, pack in _NL_LANG_PACKS.items():
            if lang == "default":
                continue
            aff = pack.get("affinity_stopwords")
            if not aff:
                continue
            score = len(tokens & aff)
            if score > best_score:
                best_score = score
                best_lang = lang
        return best_lang

    # Latin-script or digits-only: score all packs
    tokens = set(re.findall(r"[^\W_]+", query.lower()))
    best_lang, best_score = "default", 0
    for lang, pack in _NL_LANG_PACKS.items():
        aff = pack.get("affinity_stopwords")
        if not aff:
            continue  # default has no affinity set
        score = len(tokens & aff)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


# ===========================================================================
# MORPHOLOGY + EXPANSION
# ===========================================================================
def morph_prefix(
    tok: str,
    *,
    suffixes: Tuple[str, ...] = (),
    endings: Collection[str] = frozenset(),
    vowels: str = "",
    min_stem: int = 4,
    trailing_vowel_drop: bool = True,
    fallback: str = "drop1",
) -> str:
    """Prefix wildcard for one term; heuristics guided by pack data.

    Inflection lives in the suffix for most natural languages. The heuristic
    is recall-oriented and needs no morphology library:

      - explicit light ``suffixes`` (s/es/ed/ing/'s …) stripped first
        when enough stem remains;
      - trailing vowel → drop that one char, when the pack says the tail
        is flexion ("servers"→"server*");
      - 2-char flexion ``endings`` → drop 2;
      - otherwise the pack ``fallback`` decides: ``keep`` (Latin stems
        are usually already the stem: "config"→"config*") or
        ``drop1`` (agglutinative tails carry flexion).

    Tokens shorter than ``min_stem`` are returned unchanged.
    """
    if len(tok) < min_stem:
        return tok
    low = tok.lower()
    # 1. Explicit suffix strip (highest priority)
    for suf in suffixes:
        if suf and low.endswith(suf) and len(tok) - len(suf) >= min_stem:
            return f"{tok[: len(tok) - len(suf)]}*"
    # 2. Trailing vowel drop
    if trailing_vowel_drop and vowels and low[-1] in vowels:
        return f"{tok[:-1]}*"
    # 3. 2-char endings table
    if len(tok) >= min_stem + 2 and tok[-2:].lower() in endings:
        return f"{tok[:-2]}*"
    # 4. Default fallback
    if fallback == "keep":
        return f"{tok}*"
    if len(tok) == min_stem:
        return f"{tok}*"
    return f"{tok[:-1]}*"


class NLSupport:
    """Build bounded FTS5 expansions for plain conversational text."""

    _CACHE_MAXSIZE = 256
    # Expansion adds up to two FTS5 retries. Bound it independently of the
    # caller so a long conversational prompt cannot amplify search work.
    _MAX_QUERY_CHARS = 512
    _MAX_MEANINGFUL_TERMS = 8

    def __init__(self) -> None:
        # Queries originate with users; do not retain an unbounded input log.
        self._cache: OrderedDict[str, Optional[Dict[str, str]]] = OrderedDict()

    def expand_nl_query(self, query: str) -> Optional[Dict[str, str]]:
        """Expand a natural-language query into FTS5-friendly variants.

        Returns ``None`` when the query has fewer than two meaningful terms
        (nothing to gain from expansion) or is entirely stopwords.
        """
        # Do not cache or expand unusually long user input. Search still
        # continues through the existing bounded fallback chain unchanged.
        if len(query) > self._MAX_QUERY_CHARS:
            return None

        # Check cache first
        cache_key = query
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        lang = detect_lang(query)
        pack = _NL_LANG_PACKS.get(lang, _NL_LANG_PACKS["default"])
        stopwords = pack["stopwords"]
        suffixes = tuple(pack.get("suffixes", ()))
        endings = pack.get("endings", frozenset())
        vowels = pack.get("vowels", "")
        min_stem = pack.get("min_stem", 4)
        vowel_drop = bool(pack.get("trailing_vowel_drop", True))
        fallback = pack.get("fallback", "drop1")

        meaningful: List[str] = []
        and_parts: List[str] = []
        or_parts: List[str] = []

        def _add_subtoken(sub: str) -> None:
            if not sub or not re.search(r"[^\W\d_]", sub):
                return
            if sub.lower() in stopwords:
                return
            meaningful.append(sub)
            prefixed = morph_prefix(
                sub, suffixes=suffixes, endings=endings,
                vowels=vowels, min_stem=min_stem,
                trailing_vowel_drop=vowel_drop, fallback=fallback,
            )
            and_parts.append(prefixed)
            or_parts.append(prefixed)

        for raw_tok in re.findall(r'"[^"]+"|\S+', query):
            if raw_tok.startswith('"') and raw_tok.endswith('"'):
                phrase = raw_tok[1:-1].strip()
                if not phrase:
                    continue
                if re.search(r"[^\w\s]", phrase):
                    for sub in re.split(r"[^\w]+", phrase):
                        _add_subtoken(sub)
                else:
                    meaningful.append(phrase)
                    and_parts.append(raw_tok)
                    or_parts.append(raw_tok)
                continue
            tok = raw_tok.strip('"').strip("*").strip()
            if not tok or tok.upper() in {"AND", "OR", "NOT", "NEAR"}:
                continue
            for sub in re.split(r"[^\w]+", tok):
                _add_subtoken(sub)

        if len(meaningful) < 2:
            result = None
        else:
            # Keep the first terms in user order. This is deterministic and
            # caps both MATCH expression size and the broad OR retry cost.
            meaningful = meaningful[:self._MAX_MEANINGFUL_TERMS]
            and_parts = and_parts[:self._MAX_MEANINGFUL_TERMS]
            or_parts = or_parts[:self._MAX_MEANINGFUL_TERMS]
            result = {
                "and": " AND ".join(and_parts) if len(and_parts) > 1 else and_parts[0],
                "or": " OR ".join(or_parts),
                "bare": " ".join(meaningful),
                # Metadata only; callers never send it to FTS5.
                "language": lang,
            }

        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self._CACHE_MAXSIZE:
            self._cache.popitem(last=False)
        return result
