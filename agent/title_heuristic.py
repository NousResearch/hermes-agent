"""Heuristic session title extraction from the first user message.

Runs synchronously before the LLM title generation call to provide an
instant placeholder title.  The LLM result (which runs in a background
thread) overwrites this when it returns.

Uses pattern matching inspired by RAKE (Rapid Automatic Keyword Extraction):
  1. Strip conversational prefixes ("Can you", "Hey, I need help", …)
  2. Match action-verb patterns ("fix X", "review X", "refactor X")
  3. Match question patterns ("how does X work?" → "How X works")
  4. RAKE-inspired keyphrase fallback for unstructured messages

No external dependencies — pure Python regex + scoring.
"""

import re

__all__ = ["extract_title"]


# ── Stopwords for keyphrase scoring ──────────────────────────────
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "my", "your", "our",
    "i", "you", "we", "they", "he", "she", "me", "us", "them",
    "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "do", "does", "did", "will", "would", "should", "can", "could",
    "have", "has", "had", "not", "no", "just", "very", "really",
    "still", "even", "also", "only", "up", "down", "out", "about",
})

# Constituent boundary: conjunctions, relative pronouns, punctuation
_BOUNDARY = re.compile(
    r"(?:\s+(?:and|but|or|so|because|while|when|that|which|who|where|—|–)\s+)"
    r"|(?:[.,;:!?]\s+)",
    re.I,
)

# Conversational prefixes, stripped iteratively (up to 3 passes)
_PREFIX = re.compile(
    r"^(?:hi|hello|hey|yo|sup|so|yeah|ok|okay|um|uh|right|alright|"
    r"can you|could you|would you|please|i need to|i want to|i'd like to|"
    r"help me|help|i'm trying to|i'm looking to|i'm working on|"
    r"i need help|i need help with|i need help understanding|"
    r"let's|lets|we need to|we should|can we|could we|"
    r"i have a|here's a|here is a|there's a|"
    r"here's what's happening|here's what is happening|"
    r"here is what's happening|here is what is happening|"
    r"so here's|"
    r"what's happening is|the thing is|"
    r"any ideas on|any thoughts on|any suggestions for)"
    r",?\s+",
    re.I,
)

_LEADING_DASH = re.compile(r"^[—–\-]+\s*")
_LEADING_FILLER = re.compile(
    r"^(?:hey|hello|hi|yo|sup|so|well|look|right|okay|yeah|um|uh|—|–|-)\s+",
    re.I,
)

# Action-verb patterns: (verb_group, optional_article_group, object_group)
# The verb is preserved as-is (no normalization).
_ACTION_PATTERNS = [
    # fix / repair / patch / debug / resolve
    (r"(fix|repair|patch|debug|resolve|troubleshoot)\s+(?:the\s+|a\s+|an\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # add / create / implement / build
    (r"(add|create|implement|build|introduce|set up|setup)\s+(?:a\s+|an\s+|support for\s+|support\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # remove / delete / drop / disable
    (r"(remove|delete|drop|disable|turn off|strip out|get rid of)\s+(?:the\s+|a\s+|an\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # update / change / modify / adjust
    (r"(update|change|modify|adjust|tweak|bump)\s+(?:the\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # review / check / audit / inspect
    (r"(review|check|audit|inspect|examine)\s+(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # write / draft / compose
    (r"(write|draft|compose)\s+(?:a\s+|an\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # run / test / execute
    (r"(run|test|execute)\s+(?:the\s+|a\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # explain / describe
    (r"(explain|describe)\s+(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # refactor / restructure / reorganize
    (r"(refactor|restructure|reorganize|clean up|rework)\s+(?:the\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # investigate / look into / figure out
    (r"(investigate|look into|find out|figure out|dig into)\s+(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # enable / turn on
    (r"(enable|turn on)\s+(?:the\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # configure / wire up / connect
    (r"(configure|wire up|connect)\s+(?:the\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # monitor / track / watch
    (r"(monitor|track|watch)\s+(?:the\s+)?(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # compare / benchmark
    (r"(compare|benchmark)\s+(.+)",
     lambda m: f"{_cap(m.group(1))} {_clean_obj(m.group(2))}"),
    # talk about / discuss
    (r"(talk about|discuss)\s+(.+)",
     lambda m: f"{_cap(m.group(1).split()[-1])} {_clean_obj(m.group(2))}"),
]

# Question patterns → clean topic phrases
_QUESTION_PATTERNS = [
    (r"how\s+(?:does|do|can|should|would)\s+(?:I|we|you)?\s*(.+?)(?:\s+work(?:s)?)?(?:\?|$)",
     lambda m: f"How {_clean_q(m.group(1))} works"),
    (r"what(?:'s|\s+is|\s+are)\s+(?:the\s+)?difference\s+between\s+(.+?)(?:\?|$)",
     lambda m: f"Difference between {_clean_obj(m.group(1))}"),
    (r"what(?:'s|\s+is|\s+are)\s+(.+?)(?:\?|$)",
     lambda m: f"What {_clean_q(m.group(1))} is"),
    (r"why\s+(?:does|is|do|are|did)\s+(.+?)(?:\?|$)",
     lambda m: f"Why {_clean_q(m.group(1))}"),
    (r"when\s+(?:does|is|do|should|did)\s+(.+?)(?:\?|$)",
     lambda m: f"When {_clean_q(m.group(1))}"),
    (r"where\s+is\s+(.+?)(?:\?|$)",
     lambda m: f"Where {_clean_q(m.group(1))}"),
    (r"where\s+(?:does|are|do)\s+(.+?)(?:\?|$)",
     lambda m: f"Where {_clean_q(m.group(1))}"),
    (r"can\s+(?:we|I)\s+(.+?)(?:\?|$)",
     lambda m: f"{_cap(m.group(1).split()[0])} {_clean_obj(' '.join(m.group(1).split()[1:]))}"),
    (r"(?:is there|does)\s+(.+?)(?:\?|$)",
     lambda m: _clean_obj(m.group(1))),
]

# Problem/topic statement patterns
_TOPIC_PATTERNS = [
    (r"(.+?)\s+(?:is broken|doesn't work|isn't working|failed|fails|crashes|is buggy)",
     lambda m: f"Fix {_clean_obj(m.group(1))}"),
    (r"(.+?)\s+(?:is slow|is laggy|is sluggish|performs badly)",
     lambda m: f"Improve {_clean_obj(m.group(1))}"),
    (r"(.+?)\s+(?:are getting|is getting|keeps getting|keeps being)\s+(.+)",
     lambda m: f"Fix {_clean_obj(m.group(1))} {m.group(2).strip().rstrip('.,;:!?')}"),
]


def extract_title(message: str) -> str:
    """Extract a short title from the first user message.

    Returns a title string (3-45 chars).  Never raises — always returns
    *something*, even if it's just the first few words of the message.
    """
    if not message or not message.strip():
        return ""

    msg = message.strip()
    msg = _strip_prefixes(msg)
    msg = _LEADING_DASH.sub("", msg)
    msg = msg.rstrip(".,;:!?")

    if not msg:
        return ""

    # Action verb patterns
    for pattern, formatter in _ACTION_PATTERNS:
        m = re.search(pattern, msg, re.I)
        if m:
            title = _title_case(formatter(m))
            raw_obj = m.group(2) if (m.lastindex or 0) >= 2 else m.group(1)
            title = _length_guard(title, raw_obj)
            return title

    # Question patterns
    for pattern, formatter in _QUESTION_PATTERNS:
        m = re.search(pattern, msg, re.I)
        if m:
            result = _title_case(formatter(m))
            if len(result) > 45:
                result = result[:42].rsplit(" ", 1)[0] + "…"
            return result

    # Problem/topic statements
    for pattern, formatter in _TOPIC_PATTERNS:
        m = re.search(pattern, msg, re.I)
        if m:
            return _title_case(formatter(m))

    # RAKE-inspired keyphrase fallback
    return _keyphrase_fallback(msg)


# ── Internal helpers ─────────────────────────────────────────────

def _strip_prefixes(msg: str) -> str:
    """Iteratively strip conversational prefixes (up to 3 passes)."""
    for _ in range(3):
        new_msg = _PREFIX.sub("", msg)
        if new_msg == msg:
            break
        msg = new_msg
    return msg.strip()


def _clean_obj(text: str) -> str:
    """Clean an object phrase: strip trailing clauses, truncate at word boundary."""
    t = text.strip().rstrip(".,;:!?")
    # Cut at relative clauses
    t = re.sub(r"\s+(?:when|that|which|while|where|because|since|if)\s+.+$", "", t, flags=re.I)
    # Cut at coordinating conjunctions (keep "and" for compound objects like "X and Y")
    t = re.split(r"\s+(?:but|or|so)\s+", t, maxsplit=1, flags=re.I)[0]
    # Strip "keep getting/keeps being" progressive patterns
    t = re.sub(r"\s+(?:keep|keeps)\s+(?:getting|being)\s+.+$", "", t, flags=re.I)
    # Strip trailing prepositional phrases (broad char class for paths/slashes)
    t = re.sub(
        r"\s+(?:in|at|from|for|on|to|of|with|under|over|into|between)"
        r"\s+(?:the\s+|a\s+|an\s+)?[\w][\w\s\-/\.]{0,30}$",
        "", t, flags=re.I,
    )
    # Strip orphaned trailing prepositions
    t = re.sub(
        r"\s+(?:in|at|from|for|on|to|of|with|under|over|into|between)\b\s*$",
        "", t, flags=re.I,
    )
    if len(t) > 30:
        t = t[:30].rsplit(" ", 1)[0]
    return t.strip()


def _clean_q(text: str) -> str:
    """Extract a noun phrase from a question target."""
    t = text.strip().rstrip(".,;:!?")
    t = re.sub(r"\s+works?$", "", t, flags=re.I)
    t = re.sub(r"\s+(?:right|correct|isn't it|aren't they)\??$", "", t, flags=re.I)
    t = re.split(r"\s+(?:and|but|or)\s+", t, maxsplit=1, flags=re.I)[0]
    t = re.sub(
        r"\s+(?:in|at|from|for|on|to|of|with|under|over|into)"
        r"\s+(?:the\s+)?[\w][\w\s\-/\.]{0,30}$",
        "", t, flags=re.I,
    )
    t = re.sub(r"\s+(?:in|at|from|for|on|to|of|with)\b\s*$", "", t, flags=re.I)
    if len(t) > 35:
        t = t[:35].rsplit(" ", 1)[0]
    return t.strip()


def _length_guard(title: str, raw_object: str) -> str:
    """Progressive relaxation: if the title is < 3 words, retry with less aggressive cleanup."""
    if len(title.split()) >= 3:
        return title
    minimal = raw_object.strip().rstrip(".,;:!?")
    # First try: only strip relative clauses, keep prepositional phrases
    minimal = re.sub(r"\s+(?:when|that|which|while|because|since|if)\s+.+$", "", minimal, flags=re.I)
    minimal = re.split(r"\s+(?:but|or|so)\s+", minimal, maxsplit=1, flags=re.I)[0]
    if len(minimal.split()) >= 2:
        if len(minimal) > 40:
            minimal = minimal[:40].rsplit(" ", 1)[0]
        verb = title.split()[0] if title else ""
        return _title_case(f"{verb} {minimal.strip()}")
    # Second try: keep everything, just truncate
    minimal = raw_object.strip().rstrip(".,;:!?")
    if len(minimal) > 40:
        minimal = minimal[:40].rsplit(" ", 1)[0]
    verb = title.split()[0] if title else ""
    return _title_case(f"{verb} {minimal.strip()}")


def _keyphrase_fallback(msg: str) -> str:
    """RAKE-inspired keyphrase extraction for fallback cases."""
    phrases = [p.strip() for p in _BOUNDARY.split(msg) if p.strip()]
    if not phrases:
        words = msg.split()[:7]
        return _title_case(" ".join(words))[:45].rstrip(".,;:!?")

    # TF scoring across all phrases
    all_words = re.findall(r"[A-Za-z][\w\-/\.]{0,30}", msg.lower())
    word_freq: dict[str, int] = {}
    for w in all_words:
        if w in _STOPWORDS:
            continue
        word_freq[w] = word_freq.get(w, 0) + 1

    best_score = -1.0
    best_phrase = phrases[0]

    for i, phrase in enumerate(phrases):
        words = phrase.split()
        if not words:
            continue
        content_words = [w for w in words if w.lower().rstrip(".,;:!?") not in _STOPWORDS]
        if not content_words:
            continue
        tf_score = sum(word_freq.get(w.lower().rstrip(".,;:!?"), 1) for w in words)
        length_bonus = 1.0 if 3 <= len(words) <= 7 else 0.6
        position_bonus = 1.5 if i == 0 else (1.2 if i == 1 else 1.0)
        score = tf_score * length_bonus * position_bonus * len(content_words)
        if score > best_score:
            best_score = score
            best_phrase = phrase

    title = best_phrase.strip().rstrip(".,;:!?")
    title = _LEADING_FILLER.sub("", title).strip()
    if len(title) > 45:
        title = title[:45].rsplit(" ", 1)[0]
    return _title_case(title.rstrip(".,;:!?"))


def _cap(word: str) -> str:
    """Capitalize first letter, preserve ALL-CAPS tokens."""
    if word.isupper() and len(word) > 1:
        return word
    return word[0].upper() + word[1:] if word else word


def _title_case(text: str) -> str:
    """Title-case preserving ALL-CAPS tokens (PR, API, TUI) and snake_case identifiers."""
    words = text.split()
    result = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) > 1:
            result.append(w)
        elif "_" in w and w.replace("_", "").isalnum():
            result.append(w)
        elif i == 0:
            result.append(w[0].upper() + w[1:] if w else w)
        else:
            result.append(w)
    return " ".join(result)
