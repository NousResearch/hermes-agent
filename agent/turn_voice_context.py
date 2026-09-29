"""Per-turn voice/input-modality context, carried from a client entry point (CLI, TUI,
Desktop-via-gateway) into the turn loop and on into ``pre_llm_call`` plugin/shell hooks.

Ephemeral by construction: a cached gateway agent sees many turns over its lifetime, so this
is read fresh per turn (mirrors ``agent/turn_author.py``) and never folded into the persisted
user message, conversation history, or the cached system prompt — only the API-local copy of
the current turn ever sees it. A turn that does not pass one clears whatever the previous turn
set (#109455).

Trust varies by entry point, and callers must not conflate them: the CLI's ``voice_input``
flag reflects this process's own mic capture, so it is as trustworthy as any other local CLI
state. The TUI/Desktop-gateway path is different — ``prompt.submit``'s ``surface`` param
(``tui_gateway/methods_prompt.py``) is a plain client-supplied string with no server-side proof
of an actual live voice session behind it, unlike ``turn_author`` (gated by a
``DeliveryAuthor`` the client cannot construct, see 4124). So the resulting
``client_surface``/``input_modality``/``voice_session_active``/``voice_engine`` a hook sees for a
gateway turn is *client-declared UI state*, at the same trust level as the pre-existing ``display_kind``/
``surface`` params it rides alongside — useful for a plugin adapting tone or formatting, but
hooks and plugins MUST NOT use it as an authorization or identity signal.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

_VALID_MODALITIES = frozenset({"voice", "text"})
_MAX_LABEL_LEN = 40


def _clean_label(value: Any) -> str:
    """A short, trimmed label; ``""`` for anything that isn't a non-empty string. No fixed enum —
    a new client or a new voice engine needs no code change here."""
    if not isinstance(value, str):
        return ""
    return value.strip()[:_MAX_LABEL_LEN]


def _truthy(value: Any) -> bool:
    """Same coercion as ``turn_author._bot_flag``: a recognized truthy string, or a real
    bool/int — never Python's own truthiness (``bool("false")`` is ``True``, which would be a
    silent inversion of an explicit ``voice_session_active: "false"``)."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return isinstance(value, (bool, int)) and bool(value)


def parse_voice_context(raw: Any) -> Dict[str, Any]:
    """Normalize *raw* into ``{"input_modality", "voice_session_active", "client_surface"}``,
    plus ``"voice_engine"`` when one was named.

    These are separate axes, and a caller must not collapse them:
    ``client_surface`` is *where* the turn came from (which client: ``"cli"``, ``"tui"``,
    ``"desktop"``, ...), ``input_modality`` is *how* it was entered, and ``voice_engine`` — present
    only when a live-voice engine handled the turn — is *what* ran it (``"voice-live"`` for
    Desktop's GPT-Live). Putting an engine id in ``client_surface`` would break the plain
    ``client_surface == "desktop"`` test a plugin writes to mean "every desktop turn", and would
    need re-deciding for every new engine — so engines get their own field instead.

    ``voice_session_active`` is whether a voice interaction is ongoing. It is a separate axis
    from ``input_modality`` by design (a user can type one message mid voice-session), though not
    every entry point can express that today: the TUI/Desktop gateway derives all three fields
    from one ``surface == "voice-live"`` boolean, so a typed message mid voice-session arrives
    there as no signal at all. The CLI sets it from its own voice-mode state, independently.

    Anything that isn't a mapping, or a mapping asserting nothing (default modality, inactive, no
    surface), normalizes to ``{}`` — falsy, so callers can treat "no signal" and "garbage input"
    identically instead of special-casing either. A bare ``voice_engine`` asserts nothing on its
    own and does not by itself make the result non-empty. Every field is normalized defensively
    (an unhashable ``input_modality`` must not raise) since callers span the CLI's own trusted
    ``voice_input`` flag down to a gateway's client-declared, unauthenticated ``surface`` param —
    see the module docstring's trust note."""
    if not isinstance(raw, Mapping):
        return {}
    modality = raw.get("input_modality")
    modality = modality if isinstance(modality, str) and modality in _VALID_MODALITIES else "text"
    active = _truthy(raw.get("voice_session_active"))
    surface = _clean_label(raw.get("client_surface"))
    if modality == "text" and not active and not surface:
        return {}
    parsed = {"input_modality": modality, "voice_session_active": active, "client_surface": surface}
    # Omitted rather than carried as "" when absent: most turns name no engine, and the common
    # shape stays the three fields #109455 specified.
    if (engine := _clean_label(raw.get("voice_engine"))):
        parsed["voice_engine"] = engine
    return parsed
