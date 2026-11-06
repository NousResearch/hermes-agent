"""Example inbox plugin: the policy half of the session-input primitives.

The install tree owns everything durable: the admission transaction, the
versioned batch record and its retry alias, the deadline the claim path honors,
the cancellation-safe transfer and ticket settlement, and the two checkpoint
seams. This plugin only answers four questions for the profile it is enabled in:

* may these payloads share one admission, and with which windows (``policy``),
* how does a batch render as one user message (``render``),
* how does one ticket look when it is appended mid-turn (``format_part``),
* should an admitted batch go to the running turn, and as a steer or a redirect
  (``admitted`` / ``transfer_mode``).

Nothing here sorts "correction / question / new task"; the model decides that. Disabling the plugin (or removing it) stops new batching; text that
was already accepted still finishes through the install tree's primitives.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

log = logging.getLogger("plugins.session-input-inbox")

BLOCK = "session_input_inbox"
MARKER = "[追加消息 "
DEFAULTS = {"first_window_seconds": 2.0, "quiet_seconds": 2.0, "hard_cap_seconds": 8.0}


def _iso(epoch=None) -> str:
    moment = (datetime.fromtimestamp(epoch, timezone.utc) if epoch is not None else datetime.now(timezone.utc))
    return moment.astimezone().isoformat(timespec="milliseconds")


def _signature(payload):
    """Only text with identical execution policy/provenance may share an admission.

    Returns ``None`` for anything this plugin must not fold: commands, non-text
    surfaces, edits, automation, media, whole-turn options.
    """
    from session_input_primitives import input_payload_fields
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or text.lstrip().startswith("/"):
        return None
    allowed = {"text", "surface_v1", "display_v1", "local_operator_v1", "cli_turn_v1",
               "native_text_v1", "api_turn_v1"} | set(input_payload_fields())
    if set(payload) - allowed:
        return None
    signature = deepcopy(payload)
    signature.pop("text")
    for key in input_payload_fields():
        signature.pop(key, None)
    cli = signature.get("cli_turn_v1", {})
    if set(cli) - {"turn_author", "clarify_batch"}:
        return None  # api_text, model_override are whole-turn semantics
    native = signature.get("native_text_v1")
    if native is not None:
        if any(key in native for key in ("automation", "media")) or native.get("message_type", "text") != "text":
            return None
        native.pop("timestamp", None)
        native.pop("ingress_sha256", None)
        for key in ("message_id", "platform_update_id"):
            native.get("event", {}).pop(key, None)
        if native.get("event", {}).get("edit_revision"):
            return None
    display = signature.get("display_v1", {})
    if display.get("kind") in {"hidden", "internal_notification"}:
        return None
    return signature


class InboxExampleProvider:
    """The profile-scoped policy snapshot used by the install tree's primitives."""

    def __init__(self, profile_home):
        self.profile_home = str(Path(profile_home).resolve())
        self._cache: dict[str, tuple[int, dict | None]] = {}

    # -- configuration ---------------------------------------------------
    def _settings(self, profile_id: str) -> dict | None:
        if str(profile_id) != self.profile_home:
            return None  # a shared process never applies this profile's windows to another one
        path = Path(profile_id) / "config.yaml"
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None
        cached = self._cache.get(str(path))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        from hermes_cli.config import read_user_config_raw
        block = read_user_config_raw(path).get(BLOCK, {})
        settings = None
        if isinstance(block, dict) and block.get("enabled") is True:
            settings = {key: block.get(key, default) for key, default in DEFAULTS.items()}
        self._cache[str(path)] = (stamp, settings)
        return settings

    # -- provider protocol ----------------------------------------------
    def policy(self, profile_id: str, payload):
        settings = self._settings(profile_id)
        if settings is None:
            return None
        if payload is None:
            # A turn-bound binding asks only for the windows (a checkpoint wait), not eligibility.
            return {"eligible": True, **settings, "signature": "profile"}
        signature = _signature(payload)
        if signature is None:
            return None
        return {"eligible": True, **settings,
                "signature": json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str)}

    def render(self, parts) -> str:
        total = len(parts)
        return "\n\n".join(
            f"{MARKER}{index}/{total} · {_iso(part['at'])}]\n{part['text']}"
            for index, part in enumerate(parts, 1)
        )

    def format_part(self, text: str, *, count: int, arrived=None) -> str:
        if text.startswith(MARKER):
            return text  # an already merged batch keeps its own numbering
        return f"{MARKER}{count} · {_iso(arrived)}]\n{text}"

    def admitted(self, snapshot) -> bool:
        # Accepted text still finishes when the block is switched off after admission.
        return snapshot.profile_id == self.profile_home and bool(snapshot.parts)

    def transfer_mode(self, phase) -> str:
        if phase.executing_tools:
            return "steer"  # delivered once, at the batch-end checkpoint
        if phase.model_request_active or phase.redirect_pending:
            return "redirect"  # cancel just the request, append, rebuild
        return "steer"


def register(ctx) -> None:
    from hermes_constants import get_hermes_home
    from session_input_primitives import register_provider, unregister_provider
    provider = InboxExampleProvider(get_hermes_home())
    register_provider(provider)
    ctx.on_unload(lambda: unregister_provider(provider))
    log.info("session-input-inbox active for %s", provider.profile_home)
