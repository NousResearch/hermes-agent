"""
YAKE (Yet Another Keyword Extractor) — unsupervised statistical keyphrase extraction.

Ported from HiveMind's Rust implementation (memory.rs extract_keywords).
Extracts multi-word keyphrases (1-3 grams) using casing, position, frequency,
context diversity, and sentence spread. Lower internal scores = better keywords.
No external dependencies — stdlib only.
"""

import math
import re
from collections import defaultdict


STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "mine",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then", "than",
    "too", "very", "just", "about", "above", "after", "again", "all",
    "also", "any", "because", "before", "between", "both", "by", "come",
    "each", "few", "for", "from", "get", "got", "here", "how", "in",
    "into", "like", "make", "many", "more", "most", "much", "of", "on",
    "one", "only", "other", "out", "over", "said", "same", "see", "some",
    "still", "such", "take", "tell", "there", "to", "up", "use", "want",
    "way", "when", "where", "with", "don't", "i'm", "it's", "that's",
    "let", "let's", "sure", "going", "think", "know", "thing", "things",
    "really", "actually", "basically", "yes", "yeah", "okay", "well",
    "right", "good", "new", "now", "even", "back", "first", "last",
    "long", "great", "little", "own", "old", "big", "high", "different",
    "small", "large", "next", "early", "young", "important", "public",
    "bad", "same", "able", "try", "ask", "keep", "around", "however",
    "work", "using", "used", "also", "while", "something", "without",
])

# Tokenization regex: split on anything that's not alphanumeric, underscore, or hyphen
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    """Split text into tokens, keeping original casing. Filters by length [2,30]."""
    return [t for t in _TOKEN_SPLIT.split(text) if 2 <= len(t) <= 30]


def _segment_sentences(text: str) -> list[str]:
    """Split on .!?\\n, keep sentences with >= 2 whitespace-separated words."""
    raw = re.split(r"[.!?\n]", text)
    sentences = [s for s in raw if len(s.split()) >= 2]
    if not sentences:
        sentences = [text]
    return sentences


def extract_keywords(content: str) -> list[str]:
    """
    YAKE keyword extraction — returns up to 8 keywords/keyphrases.

    Implements 5-feature scoring: TCase, TPosition, TFrequency, TRelatedness, TDifferent.
    Generates 1-3 gram candidates, scores them, deduplicates, returns top 8.
    """
    if not content or not content.strip():
        return []

    # 1. Sentence segmentation
    sentences = _segment_sentences(content)
    n_sentences = max(len(sentences), 1)

    # 2. Tokenize each sentence (preserving original casing)
    sentence_tokens: list[list[str]] = [_tokenize(s) for s in sentences]

    # 3. Compute per-word statistics
    # Stats: tf, tf_upper, tf_acronym, sent_positions, left_ctx, right_ctx
    stats: dict[str, dict] = {}

    for sent_idx, tokens in enumerate(sentence_tokens):
        for tok_idx, token in enumerate(tokens):
            key = token.lower()
            if len(key) < 2 or key in STOPWORDS:
                continue
            if key.isdigit():
                continue

            if key not in stats:
                stats[key] = {
                    "tf": 0.0,
                    "tf_upper": 0.0,
                    "tf_acronym": 0.0,
                    "sent_positions": [],
                    "left_ctx": set(),
                    "right_ctx": set(),
                }

            entry = stats[key]
            entry["tf"] += 1.0
            entry["sent_positions"].append(sent_idx)

            # TCase features — check original casing
            if tok_idx > 0 and token[0].isupper():
                entry["tf_upper"] += 1.0
            alpha_chars = [c for c in token if c.isalpha()]
            if len(token) >= 2 and alpha_chars and all(c.isupper() for c in alpha_chars):
                entry["tf_acronym"] += 1.0

            # Context diversity — unique neighbors (ignoring stopwords)
            if tok_idx > 0:
                prev = tokens[tok_idx - 1].lower()
                if len(prev) >= 2 and prev not in STOPWORDS:
                    entry["left_ctx"].add(prev)
            if tok_idx + 1 < len(tokens):
                nxt = tokens[tok_idx + 1].lower()
                if len(nxt) >= 2 and nxt not in STOPWORDS:
                    entry["right_ctx"].add(nxt)

    if not stats:
        return []

    # 4. Global TF statistics for normalization
    tf_values = [ws["tf"] for ws in stats.values()]
    mean_tf = sum(tf_values) / len(tf_values)
    var_tf = sum((tf - mean_tf) ** 2 for tf in tf_values) / len(tf_values)
    std_tf = math.sqrt(var_tf)

    # 5. YAKE score per word (lower = better keyword)
    word_scores: dict[str, float] = {}

    for word, ws in stats.items():
        # TCase: casing relevance
        t_case = max(max(ws["tf_upper"], ws["tf_acronym"]) / (1.0 + math.log(ws["tf"])), 0.01)

        # TPos: positional relevance — earlier = more important
        pos = sorted(ws["sent_positions"])
        median_pos = pos[len(pos) // 2]
        t_pos = max(math.log(math.log(3.0 + median_pos)), 0.01)

        # TFreq: normalized frequency
        t_freq = ws["tf"] / (mean_tf + std_tf + 1.0)

        # TRel: context diversity
        t_rel = 1.0 + (len(ws["left_ctx"]) + len(ws["right_ctx"])) / (2.0 * ws["tf"] + 1.0)

        # TDif: sentence spread
        unique_sents = len(set(ws["sent_positions"]))
        t_dif = unique_sents / n_sentences

        # YAKE composite score
        score = (t_rel * t_pos) / (t_case + t_freq / t_rel + t_dif / t_rel + 0.001)
        word_scores[word] = score

    # 6. Generate n-gram candidates (1-3 word phrases)
    candidates: list[tuple[str, float]] = []

    # Single words
    for word, score in word_scores.items():
        candidates.append((word, score))

    # Multi-word phrases (bigrams and trigrams)
    for tokens in sentence_tokens:
        lower_tokens = [t.lower() for t in tokens]

        for n in range(2, 4):  # 2 and 3
            if len(lower_tokens) < n:
                continue
            for i in range(len(lower_tokens) - n + 1):
                gram = lower_tokens[i : i + n]

                # Skip if any component is a stopword, too short, or all numeric
                if any(w in STOPWORDS or len(w) < 2 or w.isdigit() for w in gram):
                    continue

                # All words must have computed scores
                scores = [word_scores[w] for w in gram if w in word_scores]
                if len(scores) != n:
                    continue

                # N-gram score: product of member scores / (1 + sum)
                product = 1.0
                for s in scores:
                    product *= s
                total = sum(scores)
                ng_score = product / (1.0 + total)

                candidates.append((" ".join(gram), ng_score))

    # 7. Sort by YAKE score (lower = better)
    candidates.sort(key=lambda x: x[1])

    # 8. Deduplicate — prefer longer phrases, skip substrings of already-selected
    result: list[str] = []
    for candidate, _ in candidates:
        if len(result) >= 8:
            break
        is_redundant = any(
            candidate in r or r in candidate for r in result
        )
        if not is_redundant:
            result.append(candidate)

    return result


if __name__ == "__main__":
    sample = """
    Machine learning is a subset of artificial intelligence that focuses on building
    systems that learn from data. Deep learning, a specialized form of machine learning,
    uses neural networks with many layers. Natural language processing enables computers
    to understand human language. Computer vision allows machines to interpret visual data.
    Reinforcement learning trains agents through reward signals in an environment.
    """
    keywords = extract_keywords(sample)
    print(f"Extracted {len(keywords)} keywords:")
    for kw in keywords:
        print(f"  - {kw}")
