"""Fenced draft-body warning and its strip, shared by the entrypoint and publish.

Immutable releases take no edits after publication, so the draft carries a
caution against publishing it by hand at both ends of the body and the
publication pass strips both fenced blocks while the release is still a draft.
The helpers live here so the entrypoint and the publish step can import them
without a cycle.
"""
from __future__ import annotations

WARNING_OPEN = "<!-- hermes-release:draft-warning -->"
WARNING_CLOSE = "<!-- /hermes-release:draft-warning -->"

_WARNING = """> [!CAUTION]
> ## **DO NOT PUBLISH THIS RELEASE FROM GITHUB.**
> **This is attempt `{attempt_ref}`. Publishing it here skips the `v{version}` receipt tag, the update feeds, the Docker aliases, and the Store check. Releases are immutable, so a release published here cannot be fixed.**
>
> **Run this instead:**
> ```
> python scripts/release.py publish --version {version}
> ```
> **To drop this attempt:** `python scripts/release.py abandon --version {version}`"""


def draft_body(*, version: str, attempt_ref: str, notes: str) -> str:
    """The draft body: fenced warning block, generated notes, fenced warning block."""
    block = "\n".join([WARNING_OPEN, _WARNING.format(version=version, attempt_ref=attempt_ref),
                       WARNING_CLOSE])
    return "\n".join([block, notes, block])


def strip_draft_warning(body: str) -> str:
    """Remove each fenced warning block, fence lines included.

    The fences must pair up in order; anything else means the draft body was
    edited underneath the tool, and publish refuses rather than publishing a
    mangled body.
    """
    kept: list[str] = []
    inside = False
    for line in body.splitlines():
        stripped = line.strip()
        if inside:
            if stripped == WARNING_OPEN:
                raise ValueError("draft warning fences are unbalanced")
            if stripped == WARNING_CLOSE:
                inside = False
            continue
        else:
            if stripped == WARNING_OPEN:
                inside = True
                continue
            if stripped == WARNING_CLOSE:
                raise ValueError("draft warning fences are unbalanced")
            kept.append(line)
    text = "\n".join(kept)
    if inside or WARNING_OPEN in text or WARNING_CLOSE in text:
        raise ValueError("draft warning fences are unbalanced")
    return text
