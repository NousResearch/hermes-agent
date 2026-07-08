"""Decode-safe subprocess text helpers."""

from __future__ import annotations

from dataclasses import dataclass
import locale
import subprocess
from typing import Any


@dataclass(frozen=True)
class DecodedSubprocessText:
    text: str
    encoding: str
    used_fallback: bool
    had_replacement: bool


_CP1250_POLISH_HINTS = set(
    "\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c"
    "\u0104\u0106\u0118\u0141\u0143\u00d3\u015a\u0179\u017b"
)


def _fallback_encodings() -> list[str]:
    candidates = ["cp1250"]
    try:
        preferred = locale.getpreferredencoding(False)
    except Exception:
        preferred = None
    if preferred:
        candidates.append(preferred)
    seen: set[str] = set()
    result: list[str] = []
    for encoding in candidates:
        key = encoding.lower().replace("_", "-")
        if key in {"utf-8", "utf8"} or key in seen:
            continue
        seen.add(key)
        result.append(encoding)
    return result


def _fallback_is_useful(candidate: str, utf8_replaced: str) -> bool:
    if "\ufffd" not in utf8_replaced:
        return False
    return any(ch in _CP1250_POLISH_HINTS for ch in candidate)


def decode_subprocess_bytes(
    data: bytes | bytearray | memoryview | str | None,
    *,
    allow_fallback: bool = True,
) -> DecodedSubprocessText:
    """Decode subprocess bytes without crashing reader threads."""

    if data is None:
        return DecodedSubprocessText("", "utf-8", False, False)
    if isinstance(data, str):
        return DecodedSubprocessText(data, "utf-8", False, "\ufffd" in data)
    raw = bytes(data)
    try:
        return DecodedSubprocessText(raw.decode("utf-8"), "utf-8", False, False)
    except UnicodeDecodeError:
        utf8_replaced = raw.decode("utf-8", errors="replace")
    if allow_fallback:
        for encoding in _fallback_encodings():
            try:
                candidate = raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
            if _fallback_is_useful(candidate, utf8_replaced):
                return DecodedSubprocessText(candidate, encoding, True, False)
    return DecodedSubprocessText(
        utf8_replaced,
        "utf-8",
        False,
        "\ufffd" in utf8_replaced,
    )


def text_subprocess_kwargs(**kwargs: Any) -> dict[str, Any]:
    merged = dict(kwargs)
    merged.setdefault("text", True)
    merged.setdefault("encoding", "utf-8")
    merged.setdefault("errors", "replace")
    return merged


def popen_text_kwargs(**kwargs: Any) -> dict[str, Any]:
    return text_subprocess_kwargs(**kwargs)


def _apply_lossy_text_defaults(kwargs: dict[str, Any]) -> None:
    if not (kwargs.get("text") or kwargs.get("universal_newlines")):
        return
    if kwargs.get("encoding") is None:
        kwargs["encoding"] = "utf-8"
    if kwargs.get("errors") is None:
        kwargs["errors"] = "replace"


def install_lossy_text_subprocess_defaults() -> None:
    """Make ad-hoc text-mode subprocess calls decode lossy instead of crashing.

    Shared code should prefer ``run_text_subprocess``/``popen_text_kwargs``.
    This process-local guard covers older direct ``subprocess.run(..., text=True)``
    calls whose internal reader threads would otherwise raise UnicodeDecodeError.
    """

    current_popen = subprocess.Popen
    if getattr(current_popen, "_hermes_lossy_text_defaults", False):
        return

    if isinstance(current_popen, type):
        base_popen = current_popen

        class HermesLossyTextPopen(base_popen):  # type: ignore[misc, valid-type]
            _hermes_lossy_text_defaults = True

            def __init__(self, *popenargs: Any, **kwargs: Any) -> None:
                _apply_lossy_text_defaults(kwargs)
                super().__init__(*popenargs, **kwargs)

        HermesLossyTextPopen.__name__ = getattr(base_popen, "__name__", "Popen")
        HermesLossyTextPopen.__qualname__ = getattr(base_popen, "__qualname__", "Popen")
        subprocess.Popen = HermesLossyTextPopen  # type: ignore[assignment]
        return

    def hermes_lossy_text_popen(*popenargs: Any, **kwargs: Any) -> Any:
        _apply_lossy_text_defaults(kwargs)
        return current_popen(*popenargs, **kwargs)

    hermes_lossy_text_popen._hermes_lossy_text_defaults = True  # type: ignore[attr-defined]
    subprocess.Popen = hermes_lossy_text_popen  # type: ignore[assignment]


def run_text_subprocess(*popenargs: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and decode captured output through shared safe logic."""

    run_kwargs = dict(kwargs)
    check = bool(run_kwargs.pop("check", False))
    input_value = run_kwargs.get("input")
    input_encoding = run_kwargs.pop("encoding", "utf-8")
    input_errors = run_kwargs.pop("errors", "replace")
    run_kwargs.pop("text", None)
    run_kwargs.pop("universal_newlines", None)
    if isinstance(input_value, str):
        run_kwargs["input"] = input_value.encode(input_encoding, errors=input_errors)

    completed = subprocess.run(*popenargs, check=False, **run_kwargs)
    stdout = (
        decode_subprocess_bytes(completed.stdout).text
        if isinstance(completed.stdout, (bytes, bytearray, memoryview))
        else completed.stdout
    )
    stderr = (
        decode_subprocess_bytes(completed.stderr).text
        if isinstance(completed.stderr, (bytes, bytearray, memoryview))
        else completed.stderr
    )
    text_completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        stdout,
        stderr,
    )
    if check and completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=stdout,
            stderr=stderr,
        )
    return text_completed
