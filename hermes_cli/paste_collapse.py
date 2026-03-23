from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

PASTE_REF_RE = re.compile(r"\[Pasted text #(\d+): \d+ lines → (.+?)\]")


def should_collapse_pasted_text(pasted_text: str, *, min_lines: int = 5) -> bool:
    if not pasted_text:
        return False
    return pasted_text.count("\n") >= min_lines


def write_pasted_text_reference(
    pasted_text: str,
    *,
    paste_dir: Path,
    counter: int,
    now: datetime,
) -> str:
    paste_dir.mkdir(parents=True, exist_ok=True)
    paste_file = paste_dir / f"paste_{counter}_{now.strftime('%H%M%S')}.txt"
    paste_file.write_text(pasted_text, encoding="utf-8")
    line_count = pasted_text.count("\n") + 1
    return f"[Pasted text #{counter}: {line_count} lines → {paste_file}]"


def materialize_paste_for_insertion(
    pasted_text: str,
    *,
    current_buffer_text: str,
    paste_dir: Path,
    counter: int,
    now: datetime,
    min_lines: int = 5,
) -> tuple[str, bool]:
    if not pasted_text:
        return "", False
    if (current_buffer_text or "").startswith("/"):
        return pasted_text, False
    if not should_collapse_pasted_text(pasted_text, min_lines=min_lines):
        return pasted_text, False
    return (
        write_pasted_text_reference(
            pasted_text,
            paste_dir=paste_dir,
            counter=counter,
            now=now,
        ),
        True,
    )


def expand_paste_references(text: str) -> str:
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        paste_path = Path(match.group(2))
        if not paste_path.exists():
            return match.group(0)
        return paste_path.read_text(encoding="utf-8")

    return PASTE_REF_RE.sub(repl, text)
