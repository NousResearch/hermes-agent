"""Partial private-key prints for agent.redact: a ``head``/``tail`` of a key file keeps one armor line."""

import re

# A partial print keeps only one armor line: ``head`` of a key file has BEGIN + body, ``tail``
# has body + END. The full-block pattern needs both, so those body lines leaked verbatim.
_PRIVATE_KEY_ARMOR_RE = re.compile(r"-----(BEGIN|END)[A-Z ]*PRIVATE KEY-----")
# A pure-hex line is a digest in docs, not key material: a base64 body line is never all hex.
_PEM_BODY_LINE_RE = re.compile(r"[ \t]*(?![0-9A-Fa-f]+[ \t]*$)[A-Za-z0-9+/=]{16,}[ \t]*")
_PEM_TRAILER_RE = re.compile(r"[ \t]*(?:[A-Za-z0-9+/]{1,15}={0,2}|==?)[ \t]*")  # ecdsa can end on a bare ``==``


def _redact_partial_private_keys(text: str) -> str:
    """Mask the base64 body run next to a lone BEGIN (forward) or END (backward) armor line.

    Walks lines instead of regex-backtracking over the run, so a megabyte base64 blob with a
    stray armor line elsewhere stays linear. A body slice with no armor at all is invisible to
    any shape rule; secret-source values cover that case through exact-line registration.
    """
    lines = text.split("\n")
    drop: set[int] = set()
    for i, line in enumerate(lines):
        m = _PRIVATE_KEY_ARMOR_RE.search(line)
        if not m:
            continue
        step = 1 if m.group(1) == "BEGIN" else -1
        j = i + step
        if step < 0 and j > 0 and _PEM_TRAILER_RE.fullmatch(lines[j].rstrip("\r")) and _PEM_BODY_LINE_RE.fullmatch(lines[j - 1].rstrip("\r")):
            drop.add(j)
            j -= 1  # a key's last body line can be short: ed25519 with a 30-35 char comment ends on ``QFBgc=``
        while 0 <= j < len(lines) and j not in drop and _PEM_BODY_LINE_RE.fullmatch(lines[j].rstrip("\r")):
            drop.add(j)
            j += step
        # Forward, only a padded line: an unpadded short word after the run is output (``echo $?``, ``done``).
        if step > 0 and i + 1 < j < len(lines) and _PEM_TRAILER_RE.fullmatch(lines[j].rstrip("\r")) \
                and lines[j].rstrip().endswith("="):
            drop.add(j)
        if j != i + step:
            lines[i] = "[REDACTED PRIVATE KEY]"
    if not drop:
        return text
    return "\n".join(line for k, line in enumerate(lines) if k not in drop)
