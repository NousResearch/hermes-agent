"""Active ContextOps cognitive hydration for the Hermes answer path.

This is the *active* (pre-answer) hydration adapter — distinct from the
read-only :mod:`contextops.hydrate` preview. It is invoked once per turn
inside the conversation loop to construct a compact restore/avoid/epistemic-
mode block that is appended to the API-call user context **before generation**.

Hard constraints (enforced by tests in
``tests/contextops/test_active_hydration.py``):

* **Metadata/context only.** This module never sends a message, mutates a
  Kanban board, writes memory, restarts the gateway, dispatches tools, or
  shells out. The injected text is text-only and ephemeral.
* **Disabled by default.** With no config — or an explicit ``enabled: false``
  — :func:`build_active_context` returns ``(None, {...})`` and the API-call
  context is unchanged.
* **Allowlist-gated.** Even when enabled, live injection only happens when the
  current channel identifier matches an explicit ``channel_allowlist`` entry.
* **Fail closed.** Missing seed, unreadable seed, missing channel identity, or
  any internal exception yields ``(None, {...})`` with a ``skipped_reason``;
  the caller's API-call context is left untouched.
* **No raw IDs, paths, transcripts, or tokens** appear in the injected text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from contextops.hydrate import build_hydration_preview

__all__ = ["ACTIVE_HYDRATION_CONFIG_PATH", "build_active_context"]

logger = logging.getLogger(__name__)

# Naming reflects *active cognitive context*, not watchdog/suggestions. This
# is the config namespace callers (and tests) anchor on.
ACTIVE_HYDRATION_CONFIG_PATH: tuple[str, str] = (
    "contextops",
    "active_cognitive_hydration",
)

_MAX_RESTORE_ITEMS = 6
_MAX_AVOID_ITEMS = 5
_MAX_ITEM_CHARS = 280
_DEFAULT_PACK_ID = "pack-contextops-active"


# --- ContextPack render-safety sanitization --------------------------------
#
# Active hydration is a *cognitive phase restoration* injection, not a free
# pipe from upstream ContextPack content into the model. An adversarial seed
# (or any future ContextPack mutation) must never be able to smuggle private
# paths, secrets, raw IDs, or transcript-shaped payloads into the model's
# pre-answer context. Two layers of defence:
#
#   * Payload-shaped items (JSON/dict/list/transcript role-content/provider)
#     are treated as "unsafe beyond safe redaction": the whole active block
#     is dropped with skipped_reason="unsafe_context_pack".
#   * Items that carry a normal cognitive phrase plus an embedded path/token/
#     raw ID are redacted in place; the compact phrase survives.
#
# Sanitizers are intentionally local to this module so the read-only hydrate
# preview builder remains unmodified.

class _UnsafeContextPack(Exception):
    """Raised when a ContextPack item cannot be safely redacted for injection."""


# Absolute POSIX-style paths: any /a/b style with >=2 segments. Broader than
# the original allowlist so adversarial roots like /secret/path or /opt/leak
# are caught even when they fall outside the well-known prefix set.
_ABS_PATH_RE = re.compile(r"/(?:[\w.\-+]+/)+[\w.\-+]+")
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[\w\\.\-+]+")
# ``~/foo`` or ``~user/foo`` — private home shorthand.
_TILDE_PATH_RE = re.compile(r"~(?:[A-Za-z][\w]*)?/[\w./\-+]+")
# ``.env``, ``.env.local``, ``.env.production`` — env/dotfile config paths.
_ENV_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9])\.env(?:\.[A-Za-z0-9_\-]+)?(?![A-Za-z0-9])"
)
# Secret/credential filenames with explicit extension (``secrets.json``,
# ``credentials.yaml``, ``keyfile.pem``) plus standalone SSH key filenames.
_SECRET_FILE_RE = re.compile(
    r"\b(?:secrets?|credentials?|keyfile|api[_-]?keys?|service[_-]?account|"
    r"serviceaccount|client[_-]?secret|gha[_-]?secrets?|env[_-]?secrets?)"
    r"\.(?:json|ya?ml|toml|ini|conf|cfg|pem|key|p12|pfx|txt|enc)\b",
    re.IGNORECASE,
)
_SSH_KEY_RE = re.compile(
    r"\b(?:id_rsa|id_ed25519|id_dsa|id_ecdsa)(?:\.[a-z]+)?\b",
    re.IGNORECASE,
)
_TOKEN_PREFIX_RE = re.compile(
    r"\b(?:sk|tok|ghp|gho|ghu|ghs|xox[abprs]|pat|Bearer|api[_-]?key)[-_:= ]"
    r"[A-Za-z0-9_\-./+=]{8,}",
    re.IGNORECASE,
)
# Additional vendor token prefixes not covered by ``_TOKEN_PREFIX_RE``: AWS
# access keys (``AKIA``/``ASIA``), Google API keys (``AIza``), GitLab PATs
# (``glpat-``), HuggingFace (``hf_``), Anthropic (``sk-ant-``), npm (``npm_``),
# Google OAuth (``ya29.``), Stripe (``rk_``/``pk_live_``/``sk_live_``).
_EXTRA_TOKEN_RE = re.compile(
    r"\b(?:AKIA|ASIA|AIza|glpat-|hf_|npm_|sk-ant-[A-Za-z0-9_\-]*|ya29\.|"
    r"rk_live_|sk_live_|pk_live_|sk_test_|pk_test_|shpat_|shppa_|shpss_|"
    r"shpca_|ey[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]+)[A-Za-z0-9_\-./+=]{8,}",
)
_JWT_RE = re.compile(
    r"\b[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"
)
# Base64-ish blobs of >=32 chars that mix case and digits — catches generic
# opaque secrets that don't carry a recognizable vendor prefix.
_BASE64_LIKE_RE = re.compile(
    r"(?<![A-Za-z0-9+/])"
    r"(?=[A-Za-z0-9+/]{32,}={0,3}(?:[^A-Za-z0-9+/=]|$))"
    r"(?=[A-Za-z0-9+/]*[a-z])"
    r"(?=[A-Za-z0-9+/]*[A-Z])"
    r"(?=[A-Za-z0-9+/]*\d)"
    r"[A-Za-z0-9+/]{32,}={0,3}"
)
_LONG_HEX_RE = re.compile(r"\b[a-fA-F0-9]{32,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# Kanban task ids look like ``t_<hex>`` (e.g. ``t_d52bb1c6``).
_KANBAN_TASK_RE = re.compile(r"\bt_[a-fA-F0-9]{6,}\b")
# Raw IDs we deliberately keep out of the active block. ``tension`` and
# ``stance`` are NOT listed because they are legitimate cognitive nouns that
# appear in human-readable restore phrases like "Restore unresolved tension: ...".
_RAW_ID_RE = re.compile(
    r"\b(?:evt|msg|thr|thread|chan|channel|usr|event|message|gateway|session|"
    r"pack|board|kanban|workflow|wf|task|sess|sid|conv|chat|sock|conn|gw)"
    r"[-:_][A-Za-z0-9_\-:.]+",
    re.IGNORECASE,
)
# Raw ``<id-noun>=<value>`` / ``<id-noun>: <value>`` shapes the upstream
# review flagged as leaking the rhs value: the lhs key matched
# ``_RAW_ID_RE`` and became ``[redacted-id]``, but the actual identifier
# after ``=`` / ``:`` survived (e.g. ``message_id=abc123XYZ`` →
# ``[redacted-id]=abc123XYZ``). Covers underscore, hyphen, and camelCase
# variants of the documented id-noun set plus a few practical neighbors.
_RAW_ID_KV_RE = re.compile(
    r"\b(?:message|channel|gateway|user|session|thread|chat|task|event|run|"
    r"conversation|conv|msg|chan|usr|evt|sess|sock|conn|gw|board|kanban|"
    r"workflow|wf|pack|sid|sender|author|guild|server|room)"
    r"[_\-]?id"
    r"\s*[:=]\s*"
    r"[^\s,;]+",
    re.IGNORECASE,
)
# Discord/Twitter/Slack-style platform snowflakes: standalone numeric IDs
# with 15-22 digits. Bounded by ``(?<!\d)`` / ``(?!\d)`` so adjacent digit
# runs aren't partially consumed and ordinary small numbers in cognitive
# prose (counts, years, ordinals) are left untouched.
_SNOWFLAKE_ID_RE = re.compile(r"(?<!\d)\d{15,22}(?!\d)")
_RESTORED_ID_RE = re.compile(
    # ContextPack avoid guards can contain raw thread ids without a prefix,
    # e.g. "Do not restore recent_topic_only: ...". Keep the cognitive guard
    # but redact the raw identifier after the fixed phrase.
    r"(?i)(\bDo not restore\s+)([A-Za-z][A-Za-z0-9_.\-:]{2,})(\s*:)",
)
# Key/value secret shapes: ``password=hunter2``, ``api_key: xyz123``,
# ``auth_token = abc...``. Restricted to recognized secret-noun lhs so
# legitimate cognitive phrases like "Restore stance: keep ..." don't trigger.
_KV_SECRET_RE = re.compile(
    r"\b(?:password|passwd|secret|api[_-]?key|auth[_-]?token|auth[_-]?key|"
    r"access[_-]?token|access[_-]?key|refresh[_-]?token|private[_-]?key|"
    r"client[_-]?secret|api[_-]?secret|bearer[_-]?token|session[_-]?token)"
    r"\s*[:=]\s*\S{3,}",
    re.IGNORECASE,
)
# Env-var-shaped secret assignments: ``AWS_SECRET_ACCESS_KEY=...``,
# ``GITHUB_TOKEN=...``, ``DATABASE_PASSWORD=...``.
_ENV_VAR_SECRET_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]*"
    r"(?:SECRET|TOKEN|PASSWORD|PASSWD|APIKEY|API_KEY|PRIVATEKEY|PRIVATE_KEY|"
    r"ACCESSKEY|ACCESS_KEY|AUTH|CREDENTIAL|CREDENTIALS|KEY_ID)"
    r"[A-Z0-9_]*\s*=\s*\S+"
)

# Transcript / provider payload markers — if any of these appear, the item is
# treated as carrying raw conversational/provider state that has no business
# being projected back into the model context.
_PAYLOAD_KEY_MARKERS: tuple[str, ...] = (
    '"role"', "'role'",
    '"content"', "'content'",
    '"provider"', "'provider'",
    '"messages"', "'messages'",
    '"author"', "'author'",
    '"sender"', "'sender'",
    '"tool_use"', "'tool_use'",
    '"tool_result"', "'tool_result'",
    '"system"', "'system'",
    '"assistant"', "'assistant'",
    '"user"', "'user'",
)
# Transcript-shaped sender prefixes: "assistant: ...", "User: ...", "system >".
_TRANSCRIPT_SENDER_RE = re.compile(
    r"(?:^|\n|\|)\s*(?:assistant|user|system|provider|human|ai|bot)\s*[:>]\s*",
    re.IGNORECASE,
)


def _looks_like_payload(item: str) -> bool:
    """True when an item carries raw JSON / dict / list / transcript payload."""

    stripped = item.strip()
    if not stripped:
        return False
    if stripped[0] in "{[" or stripped[-1] in "}]":
        return True
    if any(marker in item for marker in _PAYLOAD_KEY_MARKERS):
        return True
    if _TRANSCRIPT_SENDER_RE.search(item):
        return True
    return False


def _redact(item: str) -> str:
    """Strip paths, tokens, and raw IDs out of an otherwise-cognitive phrase."""

    out = item
    # Tilde/dotfile shorthand first so the full ``~/.aws/credentials`` form is
    # consumed before the generic absolute-path matcher sees ``/.aws/...``.
    out = _TILDE_PATH_RE.sub("[redacted-path]", out)
    out = _ENV_FILE_RE.sub("[redacted-path]", out)
    out = _SECRET_FILE_RE.sub("[redacted-path]", out)
    out = _SSH_KEY_RE.sub("[redacted-path]", out)
    out = _ABS_PATH_RE.sub("[redacted-path]", out)
    out = _WIN_PATH_RE.sub("[redacted-path]", out)
    # Secret key=value / env-var shapes before token-prefix so generic
    # ``password=...`` / ``AWS_SECRET_ACCESS_KEY=...`` blobs are caught even
    # when they lack a recognized vendor prefix.
    out = _ENV_VAR_SECRET_RE.sub("[redacted-secret]", out)
    out = _KV_SECRET_RE.sub("[redacted-secret]", out)
    out = _TOKEN_PREFIX_RE.sub("[redacted-token]", out)
    out = _EXTRA_TOKEN_RE.sub("[redacted-token]", out)
    out = _JWT_RE.sub("[redacted-token]", out)
    out = _UUID_RE.sub("[redacted-id]", out)
    out = _LONG_HEX_RE.sub("[redacted-token]", out)
    out = _BASE64_LIKE_RE.sub("[redacted-token]", out)
    out = _KANBAN_TASK_RE.sub("[redacted-id]", out)
    # Run the explicit key/value form first so it consumes both the lhs key
    # AND the rhs value before the generic ``_RAW_ID_RE`` can swallow only
    # the lhs and leave the rhs identifier exposed.
    out = _RAW_ID_KV_RE.sub("[redacted-id]", out)
    out = _SNOWFLAKE_ID_RE.sub("[redacted-id]", out)
    out = _RAW_ID_RE.sub("[redacted-id]", out)
    out = _RESTORED_ID_RE.sub(r"\1[redacted-id]\3", out)
    # Collapse whitespace introduced by redaction.
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


# Patterns that must NOT survive into the rendered active block. Applied as a
# residual guard after per-item redaction so any unsafe class the redactor
# missed (or that arrived through a novel item shape) drops the whole block
# via ``_UnsafeContextPack`` instead of leaking into the model context.
_RESIDUAL_CHECKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abs_path", _ABS_PATH_RE),
    ("win_path", _WIN_PATH_RE),
    ("tilde_path", _TILDE_PATH_RE),
    ("env_file", _ENV_FILE_RE),
    ("secret_file", _SECRET_FILE_RE),
    ("ssh_key", _SSH_KEY_RE),
    ("token_prefix", _TOKEN_PREFIX_RE),
    ("extra_token", _EXTRA_TOKEN_RE),
    ("jwt", _JWT_RE),
    ("uuid", _UUID_RE),
    ("long_hex", _LONG_HEX_RE),
    ("base64_like", _BASE64_LIKE_RE),
    ("kanban_task", _KANBAN_TASK_RE),
    ("raw_id_kv", _RAW_ID_KV_RE),
    ("snowflake_id", _SNOWFLAKE_ID_RE),
    ("raw_id", _RAW_ID_RE),
    ("kv_secret", _KV_SECRET_RE),
    ("env_var_secret", _ENV_VAR_SECRET_RE),
)


_PLACEHOLDER_RE = re.compile(r"\[redacted-[a-z\-]+\]")


def _residual_unsafe(text: str) -> str | None:
    """Return the name of a residual unsafe pattern in ``text``, or ``None``.

    Runs after per-item redaction has already inserted ``[redacted-*]``
    placeholders; those placeholders are stripped out before scanning so they
    don't false-positive on the structural patterns above.
    """

    scrubbed = _PLACEHOLDER_RE.sub("", text)
    for marker in _PAYLOAD_KEY_MARKERS:
        if marker in scrubbed:
            return f"payload_marker:{marker}"
    if _TRANSCRIPT_SENDER_RE.search(scrubbed):
        return "transcript_sender"
    for name, rx in _RESIDUAL_CHECKS:
        if rx.search(scrubbed):
            return name
    return None


def _safe_item(raw: Any) -> str | None:
    """Return a safe, compact cognitive phrase or ``None`` to drop the item.

    Raises :class:`_UnsafeContextPack` when an item is non-string, payload-
    shaped, or otherwise carries content that cannot be safely projected even
    after redaction.
    """

    if not isinstance(raw, str):
        raise _UnsafeContextPack("non-string context pack item")
    if _looks_like_payload(raw):
        raise _UnsafeContextPack("payload-shaped context pack item")
    if len(raw) > _MAX_ITEM_CHARS:
        # Oversize items are very likely arbitrary user-controlled snippets;
        # fail closed instead of trying to summarize them.
        raise _UnsafeContextPack("oversize context pack item")
    redacted = _redact(raw)
    if not redacted:
        return None
    # If redaction consumed everything meaningful (only placeholders remain),
    # drop the item rather than render a stub.
    placeholder_free = re.sub(r"\[redacted-[a-z\-]+\]", "", redacted).strip(" :;,.-")
    if not placeholder_free:
        return None
    return redacted


def _cfg_section(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    cur: Any = config
    for key in ACTIVE_HYDRATION_CONFIG_PATH:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if not isinstance(cur, dict):
        return None
    return cur


def _channel_identity(agent: Any) -> str | None:
    """Derive a stable channel identifier from agent attributes.

    Prefers the gateway session key (already namespaced by platform/chat/thread)
    and falls back to a synthesized ``platform:chat:thread`` triple. Returns
    ``None`` if nothing usable is available, which forces fail-closed.
    """

    key = getattr(agent, "_gateway_session_key", None)
    if isinstance(key, str) and key.strip():
        return key.strip()
    platform = getattr(agent, "platform", None)
    chat_id = getattr(agent, "_chat_id", None)
    thread_id = getattr(agent, "_thread_id", None)
    if not platform or not chat_id:
        return None
    parts = [str(platform), str(chat_id)]
    if thread_id:
        parts.append(str(thread_id))
    return ":".join(parts)


def _health(
    *,
    enabled: bool,
    channel: str | None,
    allowlisted: bool,
    skipped_reason: str | None,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "channel": channel,
        "allowlisted": allowlisted,
        "skipped_reason": skipped_reason,
    }


def _sanitize_items(raw_items: Any, limit: int) -> list[str]:
    """Sanitize a slice of ContextPack restore/avoid items.

    Raises :class:`_UnsafeContextPack` if any candidate carries payload-shaped
    or otherwise irredeemable content (forces the caller to fail closed on the
    whole active block rather than render a partially redacted view).
    """

    if not isinstance(raw_items, (list, tuple)):
        raise _UnsafeContextPack("non-list context pack section")
    out: list[str] = []
    for raw in list(raw_items)[:limit]:
        safe = _safe_item(raw)
        if safe is None:
            continue
        out.append(safe)
    return out


def _render_active_block(state: Any, channel: str) -> str:
    """Render the compact active-hydration block.

    Only emits Thread/Tension-derived restore directives, the contamination-
    avoidance guards, and the epistemic-mode marker. Each item passes through
    :func:`_safe_item`; if any item is payload-shaped or otherwise unsafe
    beyond safe redaction, :class:`_UnsafeContextPack` is raised so the caller
    can fail closed with ``skipped_reason="unsafe_context_pack"`` rather than
    inject a half-redacted block.
    """

    pack = getattr(state, "context_pack", None)
    if pack is None:
        raise _UnsafeContextPack("missing context pack")

    restore_items = _sanitize_items(getattr(pack, "restore", None), _MAX_RESTORE_ITEMS)
    avoid_items = _sanitize_items(getattr(pack, "avoid", None), _MAX_AVOID_ITEMS)

    lines = [
        "[ContextOps active cognitive context — metadata only, not user-visible]",
        "Epistemic mode: restore unresolved cognitive pressure for this channel; "
        "do not flatten open tensions into closed answers.",
        "",
        "Restore (Thread/Tension cognitive directives):",
    ]
    if restore_items:
        for item in restore_items:
            lines.append(f"  - {item}")
    else:
        lines.append("  - (no active restore directives)")
    lines.append("")
    lines.append("Avoid (contamination guards):")
    if avoid_items:
        for item in avoid_items:
            lines.append(f"  - {item}")
    else:
        lines.append("  - (no active avoid directives)")
    rendered = "\n".join(lines)
    residual = _residual_unsafe(rendered)
    if residual is not None:
        raise _UnsafeContextPack(f"residual unsafe pattern: {residual}")
    return rendered


def build_active_context(
    *,
    agent: Any,
    original_user_message: str | None,
    config: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    """Build the active ContextOps cognitive-hydration injection for one turn.

    Returns a ``(injection_text, health)`` tuple. ``injection_text`` is
    ``None`` whenever the feature is disabled, not allowlisted for this
    channel, or any safety check fails — in which case the caller MUST leave
    the API-call context unchanged. ``health`` is always a dict suitable for
    structured logging.
    """

    section = _cfg_section(config)
    enabled = bool(section and section.get("enabled"))
    channel = _channel_identity(agent)

    if not enabled:
        return None, _health(
            enabled=False, channel=channel, allowlisted=False, skipped_reason="disabled"
        )

    message = original_user_message if isinstance(original_user_message, str) else ""
    if not message.strip():
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=False,
            skipped_reason="empty_message",
        )

    if channel is None:
        return None, _health(
            enabled=True,
            channel=None,
            allowlisted=False,
            skipped_reason="no_channel",
        )

    raw_allowlist = section.get("channel_allowlist") if section else None
    if not isinstance(raw_allowlist, list) or not raw_allowlist:
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=False,
            skipped_reason="non_allowlisted",
        )
    allowlist = {str(entry).strip() for entry in raw_allowlist if str(entry).strip()}
    if channel not in allowlist:
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=False,
            skipped_reason="non_allowlisted",
        )

    seed_path = section.get("seed_path") if section else None
    if not isinstance(seed_path, str) or not seed_path.strip():
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=True,
            skipped_reason="no_seed",
        )

    pack_id = section.get("pack_id") if section else None
    if not isinstance(pack_id, str) or not pack_id.strip():
        pack_id = _DEFAULT_PACK_ID

    try:
        state = build_hydration_preview(
            channel, message, seed_path, pack_id=pack_id
        )
    except FileNotFoundError:
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=True,
            skipped_reason="seed_unavailable",
        )
    except OSError:
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=True,
            skipped_reason="seed_unavailable",
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on any builder failure
        logger.warning("ContextOps active hydration build failed: %s", exc)
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=True,
            skipped_reason="build_failed",
        )

    try:
        text = _render_active_block(state, channel)
    except _UnsafeContextPack as exc:
        # ContextPack contained a payload-shaped or otherwise irredeemable
        # item (e.g. transcript JSON, provider payload, oversize snippet).
        # Drop the entire active block instead of injecting half-sanitized
        # content; the read-only preview path is unaffected.
        logger.warning("ContextOps active hydration rejected unsafe pack: %s", exc)
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=True,
            skipped_reason="unsafe_context_pack",
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on render failure
        logger.warning("ContextOps active hydration render failed: %s", exc)
        return None, _health(
            enabled=True,
            channel=channel,
            allowlisted=True,
            skipped_reason="render_failed",
        )

    return text, _health(
        enabled=True,
        channel=channel,
        allowlisted=True,
        skipped_reason=None,
    )
