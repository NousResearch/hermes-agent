"""Input parsing, history helpers, autocomplete support, and file-drop handling."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from hermes_constants import is_termux as _is_termux_environment

from .constants import _IMAGE_EXTENSIONS, _TERMINAL_INPUT_MODE_RESET_SEQ


from hermes_constants import is_termux as _is_termux_environment


def _termux_example_image_path(filename: str = "cat.png") -> str:
    """Return a realistic example media path for the current Termux setup."""
    candidates = [
        os.path.expanduser("~/storage/shared"),
        "/sdcard",
        "/storage/emulated/0",
        "/storage/self/primary",
    ]
    for root in candidates:
        if os.path.isdir(root):
            return os.path.join(root, "Pictures", filename)
    return os.path.join("~/storage/shared", "Pictures", filename)


def _split_path_input(raw: str) -> tuple[str, str]:
    r"""Split a leading file path token from trailing free-form text.

    Supports quoted paths and backslash-escaped spaces so callers can accept
    inputs like:
      /tmp/pic.png describe this
      ~/storage/shared/My\ Photos/cat.png what is this?
      "/storage/emulated/0/DCIM/Camera/cat 1.png" summarize
    """
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == '\\' and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                token = raw[1:pos]
                remainder = raw[pos + 1 :].strip()
                return token, remainder
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == '\\' and pos + 1 < len(raw) and raw[pos + 1] == ' ':
            pos += 2
        elif ch == ' ':
            break
        else:
            pos += 1

    token = raw[:pos].replace('\\ ', ' ')
    remainder = raw[pos:].strip()
    return token, remainder


def _resolve_attachment_path(raw_path: str) -> Path | None:
    """Resolve a user-supplied local attachment path.

    Accepts quoted or unquoted paths, expands ``~`` and env vars, and resolves
    relative paths from ``TERMINAL_CWD`` when set (matching terminal tool cwd).
    Returns ``None`` when the path does not resolve to an existing file.
    """
    token = str(raw_path or "").strip()
    if not token:
        return None

    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()
    token = token.replace('\\ ', ' ')
    if not token:
        return None

    expanded = token
    if token.startswith("file://"):
        try:
            parsed = urlparse(token)
            if parsed.scheme == "file":
                expanded = unquote(parsed.path or "")
                if parsed.netloc and os.name == "nt":
                    expanded = f"//{parsed.netloc}{expanded}"
        except Exception:
            expanded = token
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if os.name != "nt":
        normalized = expanded.replace("\\", "/")
        if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/" and normalized[0].isalpha():
            expanded = f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    path = Path(expanded)
    if not path.is_absolute():
        base_dir = Path(os.getenv("TERMINAL_CWD", os.getcwd()))
        path = base_dir / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    # Path.exists() / is_file() invoke os.stat(), which raises OSError when
    # the candidate string is structurally invalid as a path — most commonly
    # ENAMETOOLONG (errno 63 on macOS, errno 36 on Linux) when the input
    # exceeds NAME_MAX (typically 255 bytes). This bites pasted slash
    # commands like `/goal <long prose>` because `_detect_file_drop()`'s
    # `starts_like_path` prefilter accepts any input starting with `/`,
    # then this resolver tries to stat it before short-circuiting on the
    # slash-command path. Without this guard the OSError propagates up to
    # the process_loop catch-all in _interactive_loop and the user input
    # is silently lost (the warning ends up in agent.log but the user sees
    # nothing — the prompt just hangs).
    try:
        if not resolved.exists() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved





def _detect_file_drop(user_input: str) -> "dict | None":
    """Detect if *user_input* starts with a real local file path.

    This catches dragged/pasted paths before they are mistaken for slash
    commands, and also supports Termux-friendly paths like ``~/storage/...``.

    Returns a dict on match::

        {
            "path": Path,          # resolved file path
            "is_image": bool,      # True when suffix is a known image type
            "remainder": str,      # any text after the path
        }

    Returns ``None`` when the input is not a real file path.
    """
    if not isinstance(user_input, str):
        return None

    stripped = user_input.strip()
    if not stripped:
        return None

    starts_like_path = (
        stripped.startswith("/")
        or stripped.startswith("~")
        or stripped.startswith("./")
        or stripped.startswith("../")
        or stripped.startswith("file://")
        or (len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in {"\\", "/"} and stripped[0].isalpha())
        or stripped.startswith('"/')
        or stripped.startswith('"~')
        or stripped.startswith("'/")
        or stripped.startswith("'~")
        or stripped.startswith('"./')
        or stripped.startswith('"../')
        or stripped.startswith("'./")
        or stripped.startswith("'../")
        or (len(stripped) >= 4 and stripped[0] in {"'", '"'} and stripped[2] == ":" and stripped[3] in {"\\", "/"} and stripped[1].isalpha())
    )
    if not starts_like_path:
        return None

    direct_path = _resolve_attachment_path(stripped)
    if direct_path is not None:
        return {
            "path": direct_path,
            "is_image": direct_path.suffix.lower() in _IMAGE_EXTENSIONS,
            "remainder": "",
        }

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None and " " in stripped and stripped[0] not in {"'", '"'}:
        space_positions = [idx for idx, ch in enumerate(stripped) if ch == " "]
        for pos in reversed(space_positions):
            candidate = stripped[:pos].rstrip()
            resolved = _resolve_attachment_path(candidate)
            if resolved is not None:
                drop_path = resolved
                remainder = stripped[pos + 1 :].strip()
                break
    if drop_path is None:
        return None

    return {
        "path": drop_path,
        "is_image": drop_path.suffix.lower() in _IMAGE_EXTENSIONS,
        "remainder": remainder,
    }


def _format_image_attachment_badges(attached_images: list[Path], image_counter: int, width: int | None = None) -> str:
    """Format the attached-image badge row for the interactive CLI.

    Narrow terminals such as Termux should get a compact summary that fits on a
    single row, while wider terminals can show the classic per-image badges.
    """
    if not attached_images:
        return ""

    width = width or shutil.get_terminal_size((80, 24)).columns

    def _trunc(name: str, limit: int) -> str:
        return name if len(name) <= limit else name[: max(1, limit - 3)] + "..."

    if width < 52:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 20)}]"
        return f"[📎 {len(attached_images)} images attached]"

    if width < 80:
        if len(attached_images) == 1:
            return f"[📎 {_trunc(attached_images[0].name, 32)}]"
        first = _trunc(attached_images[0].name, 20)
        extra = len(attached_images) - 1
        return f"[📎 {first}] [+{extra}]"

    base = image_counter - len(attached_images) + 1
    return " ".join(
        f"[📎 Image #{base + i}]"
        for i in range(len(attached_images))
    )


def _should_auto_attach_clipboard_image_on_paste(pasted_text: str) -> bool:
    """Auto-attach clipboard images only for image-only paste gestures."""
    return not pasted_text.strip()


def _strip_leaked_bracketed_paste_wrappers(text: str) -> str:
    """Strip leaked bracketed-paste wrapper markers from user-visible text.

    Defensive normalization for cases where terminal/prompt_toolkit parsing
    fails and bracketed-paste markers end up in the buffer as literal text.

    We strip canonical wrappers unconditionally and also handle degraded
    visible forms like ``[200~`` / ``[201~`` and ``00~`` / ``01~`` when they
    look like wrapper boundaries, not arbitrary user content.
    """
    if not text:
        return text

    text = (
        text.replace("\x1b[200~", "")
        .replace("\x1b[201~", "")
        .replace("^[[200~", "")
        .replace("^[[201~", "")
    )
    text = re.sub(r"(^|[\s\n>:\]\)])\[200~", r"\1", text)
    text = re.sub(r"\[201~(?=$|[\s\n<\[\(\):;.,!?])", "", text)
    text = re.sub(r"(^|[\s\n>:\]\)])00~", r"\1", text)
    text = re.sub(r"01~(?=$|[\s\n<\[\(\):;.,!?])", "", text)
    return text


# Cursor Position Report (CPR / DSR) response, format ``ESC[<row>;<col>R``.
# prompt_toolkit's _on_resize() + renderer send ``ESC[6n`` queries to the
# terminal; under resize storms or tab switches the terminal's reply can
# race past the input parser and end up in the input buffer as literal
# text (see issue #14692). Also matches the visible-form ``^[[<row>;<col>R``
# that appears when the ESC byte was stripped by a prior filter.
_DSR_CPR_ESC_RE = re.compile(r"\x1b\[\d+;\d+R")
_DSR_CPR_VISIBLE_RE = re.compile(r"\^\[\[\d+;\d+R")
_SGR_MOUSE_ESC_RE = re.compile(r"\x1b\[<\d+;\d+;\d+[Mm]")
_SGR_MOUSE_VISIBLE_RE = re.compile(r"\^\[\[<\d+;\d+;\d+[Mm]")
# Some terminals/filters can drop ESC and literal "^[[", leaving only
# "<btn;col;rowM" fragments in the buffer. Keep this broad on purpose:
# these fragments are extremely unlikely to be intentional user input, and
# stripping them is better than sending corrupted prompts.
_SGR_MOUSE_BARE_RE = re.compile(r"<\d+;\d+;\d+[Mm]")


def _strip_leaked_terminal_responses_with_meta(text: str) -> tuple[str, bool]:
    """Strip leaked terminal control-response sequences from user input.

    Covers Cursor Position Report (CPR / DSR) responses — ``ESC[<row>;<col>R``
    and the visible ``^[[<row>;<col>R`` form. These are replies the terminal
    sends back to queries prompt_toolkit makes during ``_on_resize`` /
    ``_request_absolute_cursor_position``. When the input parser drops one
    (resize storms, multiplexer focus changes, slow PTYs) the response
    lands in the input buffer as literal text and corrupts what the user
    typed.

    Also strips leaked SGR mouse-report fragments (``ESC[<...M/m`` and
    degraded visible forms). Returns ``(cleaned_text, had_mouse_reports)``
    so callers can trigger an in-place terminal mode recovery when needed.
    """
    if not text:
        return text, False

    has_esc = "\x1b[" in text
    has_visible = "^[" in text
    has_bare_mouse = "<" in text and ";" in text and ("M" in text or "m" in text)
    if not (has_esc or has_visible or has_bare_mouse):
        return text, False

    had_mouse_reports = False

    if has_esc:
        text = _DSR_CPR_ESC_RE.sub("", text)
        text, count = _SGR_MOUSE_ESC_RE.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0

    if has_visible:
        text = _DSR_CPR_VISIBLE_RE.sub("", text)
        text, count = _SGR_MOUSE_VISIBLE_RE.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0

    if has_bare_mouse:
        text, count = _SGR_MOUSE_BARE_RE.subn("", text)
        had_mouse_reports = had_mouse_reports or count > 0

    return text, had_mouse_reports


def _strip_leaked_terminal_responses(text: str) -> str:
    """Compatibility wrapper returning only cleaned text."""
    cleaned, _ = _strip_leaked_terminal_responses_with_meta(text)
    return cleaned


def _collect_query_images(query: str | None, image_arg: str | None = None) -> tuple[str, list[Path]]:
    """Collect local image attachments for single-query CLI flows."""
    message = query or ""
    images: list[Path] = []

    if isinstance(message, str):
        dropped = _detect_file_drop(message)
        if dropped and dropped.get("is_image"):
            images.append(dropped["path"])
            message = dropped["remainder"] or f"[User attached image: {dropped['path'].name}]"

    if image_arg:
        explicit_path = _resolve_attachment_path(image_arg)
        if explicit_path is None:
            raise ValueError(f"Image file not found: {image_arg}")
        if explicit_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Not a supported image file: {explicit_path}")
        images.append(explicit_path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for img in images:
        key = str(img)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(img)
    return message, deduped


