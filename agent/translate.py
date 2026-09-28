"""On-demand translation layer for tool output, errors and fetched content (#123591).

The actual translation is done by the agent model itself (that surface only the
agent can reach); this module is the deterministic layer around it that enforces
the issue's hard requirements without any new dependency:

- machine-exact text (fenced code, inline code, URLs, file paths, CLI flags)
  is placeholder-substituted before the model ever sees it, so it comes back
  verbatim and is never "translated";
- the rendered block always keeps the original alongside the translation;
- uncertainty is flagged: an explicit ROUGH marker from the model, or a lost
  placeholder (evidence the model mangled the text), marks the result rough;
- the target language follows the UI locale (:func:`agent.i18n.get_language`)
  with an independent per-session override;
- automatic translation exists as a per-session toggle, OFF by default.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable

from agent import i18n

__all__ = [
    "TranslationResult",
    "resolve_target",
    "effective_target",
    "protect",
    "restore",
    "build_translation_prompt",
    "translate_text",
    "render_translation",
    "set_target",
    "set_auto",
    "auto_enabled",
    "should_auto_translate",
]

#: Backend: ``backend(protected_prompt, target_lang) -> raw model reply``.
#: ``None`` selects the built-in one-shot model call (needs provider credentials).
Backend = Callable[[str, str], str]

_PLACEHOLDER_RE = re.compile(r"⟦KEEP-(\d+)⟧")

_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://[^\s\"'`<>\]\)]+|www\.[^\s\"'`<>\]\)]+")
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|~/|\.{1,2}/|/)[^\s\"'`<>\]\)]*[^\s\"'`<>\]\)\.,;:!?]"
)
_FLAG_RE = re.compile(r"(?<!\S)--[A-Za-z][A-Za-z0-9_-]*|(?<!\S)-[A-Za-z](?!\S)")

# Order matters: fences first so inner backticks/paths never match on their own.
_PROTECT_PATTERNS = (_FENCE_RE, _INLINE_CODE_RE, _URL_RE, _PATH_RE, _FLAG_RE)

_ROUGH_RE = re.compile(r"^\s*(?:ROUGH|UNCERTAIN|LOW CONFIDENCE)\s*:.*$", re.IGNORECASE | re.MULTILINE)

_LANGUAGE_NAMES = {
    "en": "English", "zh": "Simplified Chinese", "zh-hant": "Traditional Chinese",
    "ja": "Japanese", "de": "German", "es": "Spanish", "fr": "French",
    "tr": "Turkish", "uk": "Ukrainian", "af": "Afrikaans", "ko": "Korean",
    "it": "Italian", "ga": "Irish", "pt": "Portuguese", "ru": "Russian",
    "hu": "Hungarian", "ar": "Arabic",
}

_SYSTEM_PROMPT = (
    "You translate technical text (tool output, error messages, fetched documentation) "
    "for a developer reading in another language. Rules:\n"
    "- Translate the prose. Leave every ⟦KEEP-n⟧ placeholder EXACTLY as-is, in place.\n"
    "- Never translate, reformat or drop code, paths, identifiers, flags or URLs — "
    "they are already hidden inside placeholders; do not guess what they hide.\n"
    "- Reply with ONLY the translation. If any idiom or technical term is a rough "
    "guess, append one final line starting with 'ROUGH:' naming it."
)


@dataclass
class TranslationResult:
    """One translated block: the machine text, the untouched source, and flags."""

    source: str
    translation: str
    target: str
    uncertain: bool


def resolve_target(explicit: str | None) -> str:
    """Target language: explicit override, else the UI locale, normalized to a
    supported code (unknown values fall back to English, same as i18n)."""
    if explicit and explicit.strip():
        # ponytail: reuse the alias table in agent.i18n instead of a second copy;
        # upgrade path is a public normalize() if another caller needs it.
        return i18n._normalize_lang(explicit)
    return i18n.get_language()


def _placeholder(index: int) -> str:
    return f"⟦KEEP-{index}⟧"


def protect(text: str) -> tuple[str, list[str]]:
    """Hide machine-exact spans behind ``⟦KEEP-n⟧`` placeholders.

    Returns ``(protected_text, table)``; :func:`restore` maps them back.
    """
    table: list[str] = []

    def _stash(match: re.Match) -> str:
        table.append(match.group(0))
        return _placeholder(len(table) - 1)

    protected = text
    for pattern in _PROTECT_PATTERNS:
        protected = pattern.sub(_stash, protected)
    return protected, table


def restore(text: str, table: list[str]) -> str:
    """Swap placeholders back to the stashed spans (unknown indices left as-is)."""

    def _unstash(match: re.Match) -> str:
        index = int(match.group(1))
        return table[index] if 0 <= index < len(table) else match.group(0)

    return _PLACEHOLDER_RE.sub(_unstash, text)


def _expected_placeholders(protected: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(protected))


def build_translation_prompt(protected_source: str, target: str) -> str:
    """User-side prompt for the model call (placeholders already substituted)."""
    name = _LANGUAGE_NAMES.get(target, target)
    return (
        f"Translate the following text into {name} (language code: {target}).\n\n"
        f"{protected_source}"
    )


def _parse_reply(reply: str, protected: str, table: list[str]) -> tuple[str, bool]:
    """Restore placeholders; flag rough when the model says so or drops any."""
    uncertain = False
    if _ROUGH_RE.search(reply):
        uncertain = True
        reply = _ROUGH_RE.sub("", reply).strip()
    if _expected_placeholders(protected) - set(_PLACEHOLDER_RE.findall(reply)):
        # The model dropped or rewrote a placeholder: machine-exact text may be
        # corrupted, so present the result as rough rather than confident.
        uncertain = True
    return restore(reply.strip(), table), uncertain


def _oneshot_backend(prompt: str, target: str) -> str:
    from agent.oneshot import run_oneshot

    return run_oneshot(
        instructions=_SYSTEM_PROMPT,
        user_input=prompt,
        task="translation",
        max_tokens=2048,
        temperature=0.2,
        timeout=60.0,
    )


def translate_text(
    source: str,
    *,
    target: str | None = None,
    backend: Backend | None = None,
    session_id: str | None = None,
) -> TranslationResult:
    """Translate ``source`` prose, keeping code/paths/flags verbatim.

    ``target`` wins over the per-session override, which wins over the UI
    locale. ``backend`` defaults to a one-shot model call (raises RuntimeError
    with no provider configured); tests inject a stub.
    """
    resolved = effective_target(session_id, target)
    if not (source or "").strip():
        return TranslationResult(source=source, translation="", target=resolved, uncertain=False)
    protected, table = protect(source)
    reply = (backend or _oneshot_backend)(build_translation_prompt(protected, resolved), resolved)
    translation, uncertain = _parse_reply(reply or "", protected, table)
    return TranslationResult(source=source, translation=translation, target=resolved, uncertain=uncertain)


def render_translation(result: TranslationResult) -> str:
    """Bilingual block: translation first, original always kept, roughness flagged."""
    lines = [
        i18n.t("translate.header", target=result.target),
        result.translation,
        "",
        f"{i18n.t('translate.original_label')} {result.source}",
    ]
    if result.uncertain:
        lines.append(i18n.t("translate.uncertain_note"))
    return "\n".join(lines).strip()


# ---- per-session preferences (in-memory; off by default) --------------------

_state_lock = threading.Lock()
_session_targets: dict[str, str] = {}
_session_auto: dict[str, bool] = {}


def _session_key(session_id: str | None) -> str:
    return str(session_id or "default")


def set_target(session_id: str | None, lang: str | None) -> None:
    """Independent target override (None clears it back to the UI locale)."""
    key = _session_key(session_id)
    with _state_lock:
        if lang and lang.strip():
            _session_targets[key] = i18n._normalize_lang(lang)
        else:
            _session_targets.pop(key, None)


def effective_target(session_id: str | None, explicit: str | None) -> str:
    """Explicit argument > per-session override > UI locale."""
    if explicit and explicit.strip():
        return i18n._normalize_lang(explicit)
    with _state_lock:
        override = _session_targets.get(_session_key(session_id))
    if override:
        return override
    return i18n.get_language()


def set_auto(session_id: str | None, enabled: bool) -> None:
    """Per-session automatic-translation toggle (default off)."""
    with _state_lock:
        _session_auto[_session_key(session_id)] = bool(enabled)


def auto_enabled(session_id: str | None) -> bool:
    """Automatic translation is opt-in: False unless explicitly enabled."""
    with _state_lock:
        return _session_auto.get(_session_key(session_id), False)


def _looks_foreign(text: str, target: str) -> bool:
    # ponytail: naive script heuristic (non-Latin script vs English; ASCII-only
    # vs a non-English target, the common fetched-docs case). Upgrade path is a
    # real language-ID pass when auto mode grows beyond opt-in.
    if target == "en":
        return any(ch.isalpha() and ord(ch) > 127 for ch in text)
    return bool(text.strip()) and all(ord(ch) < 128 for ch in text)


def should_auto_translate(
    session_id: str | None, text: str, target: str | None = None
) -> bool:
    """True only when auto mode is on AND the text looks foreign to the target."""
    if not auto_enabled(session_id) or not (text or "").strip():
        return False
    return _looks_foreign(text, effective_target(session_id, target))
