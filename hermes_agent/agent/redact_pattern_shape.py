"""Structural checks on credential regexes before they join the redaction prefix matcher."""


def _skip_char_class(pattern: str, i: int) -> int:
    """Given ``pattern[i] == "["``, return the index just past the closing ``]``."""
    i += 2 if pattern[i + 1:i + 2] == "]" else 1  # a leading "]" is literal
    while i < len(pattern) and pattern[i] != "]":
        i += 2 if pattern[i] == "\\" else 1
    return i


def _unbounded_quantifier_follows(pattern: str, j: int) -> bool:
    """True if an open-ended quantifier (``*``, ``+``, ``{m,}``) starts at ``pattern[j]``."""
    if j >= len(pattern):
        return False
    if pattern[j] in "*+":
        return True
    if pattern[j] == "{":
        k = pattern.find("}", j)
        body = pattern[j + 1:k] if k != -1 else ""
        return body[:-1].isdigit() and body.endswith(",")  # {m,} is open-ended; {m} / {m,n} bounded
    return False


def _pattern_structure(pattern: str) -> tuple[bool, bool]:
    """One scan → ``(has_top_level_alternation, has_nested_unbounded_repeat)``.

    Top-level ``|`` defeats the literal-prefix guarantee (in ``ab|.*`` the prefix
    binds only the first branch; ``ab(?:x|y)`` is fine). An unbounded quantifier
    on a group containing one (``(a+)+``, ``(a{2,})+``) is the canonical ReDoS
    shape. Structural only; overlapping branches (``(a|aa)+``) are not detected.
    """
    top_level_alt = nested = False
    contains_unbounded = [False]  # per open group: does it contain an unbounded repeat?
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            i = _skip_char_class(pattern, i)
        elif ch == "(":
            contains_unbounded.append(False)
        elif ch == ")":
            inner = contains_unbounded.pop() if len(contains_unbounded) > 1 else False
            if inner and _unbounded_quantifier_follows(pattern, i + 1):
                nested = True
            contains_unbounded[-1] = contains_unbounded[-1] or inner
        elif ch == "|" and len(contains_unbounded) == 1:
            top_level_alt = True
        elif _unbounded_quantifier_follows(pattern, i):
            contains_unbounded[-1] = True
            if ch == "{":
                i = pattern.find("}", i)  # skip the {m,} body
        i += 1
    return top_level_alt, nested
