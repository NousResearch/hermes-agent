"""Provider configuration readers for the claude-agent-sdk session.

``agent.claude_agent_sdk`` config/flag access, permission-mode and setting-source
readers, the child environment (scrubbed of metered billing vectors), stdout
framing and timeout knobs, the HTTP MCP entries and the stdio hermes-tools MCP
config. Extracted from ``claude_agent_sdk_session.py``; ``_provider_config``,
``_provider_flag`` and ``_sdk_env_overrides`` are re-exported by the facade for
the runtime and the auxiliary client.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# HERMES_TERMINAL_SECURITY_MODE → SDK permission_mode, read at session
# construction (default "auto"). Precedence: explicit constructor arg, then
# the agent.claude_agent_sdk.permission_mode config key (an SDK mode literal
# — see _configured_permission_mode), then this env mapping.
#
# SDK default posture is intentionally stricter than generic terminal `auto`:
# `default` preserves Hermes' per-tool approval bridge. Mapping `auto` to
# `acceptEdits` skips that bridge entirely, so even the fixed bounded MCP read
# surface is denied/unguarded depending on CLI state. Hermes YOLO requests are
# represented internally as ``bypassPermissions`` but normalized back to
# ``default`` before SDK option construction; the audited callback performs the
# bypass only after immutable floors.
_HERMES_TO_SDK_PERMISSION_MODE = {
    "auto": "default",
    "approval-required": "default",
    "unrestricted": "bypassPermissions",
    "yolo": "bypassPermissions",
}


# The SDK's own permission_mode literals (verified against the installed
# claude-agent-sdk 0.2.120 ClaudeAgentOptions type).
_SDK_PERMISSION_MODES = (
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
    "dontAsk",
    "auto",
)


def _configured_sdk_env() -> dict:
    """agent.claude_agent_sdk.env — extra environment for the CLI subprocess.

    The Claude Code CLI reads operational knobs from its environment that the
    SDK exposes no typed option for. Measured on 0.2.120: only
    ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` moves the context ceiling and the
    autocompact threshold (300000 -> maxTokens 300000, threshold 267000);
    ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` and ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``
    are inert. That ratio is exactly why this is a generic config surface and
    not a named option per knob — the knobs are undocumented and shift.

    Values are stringified; a non-mapping or unreadable config yields {} so a
    bad edit cannot strip the scrubbed env that ships alongside it.
    """
    raw = _provider_config().get("env")
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, value in raw.items():
        if value is None:
            continue
        try:
            out[str(key)] = str(value)
        except Exception:
            logger.warning(
                "agent.claude_agent_sdk.env[%r] is not stringifiable — ignoring", key
            )
    return out


def _sdk_env_overrides(
    *, metered_allowed: Optional[bool] = None
) -> dict[str, str]:
    """The full env override set handed to the spawned CLI.

    Metered-vector scrub first (see _METERED_ENV_DENYLIST).
    agent.claude_agent_sdk.allow_metered_key: true is the operator's explicit
    "bill me metered" opt-in (the same flag the startup guard honors), so it
    disables the scrub too — otherwise the documented escape hatch would hand
    the CLI an environment with the key blanked.

    Operator-configured env is applied last so deliberate knobs win over
    defaults, but it must NOT win over the scrub: a plain update() would let
    ``env: {ANTHROPIC_API_KEY: ...}`` overwrite the scrub's "" and silently
    re-arm metered billing behind allow_metered_key: false. Denylisted keys
    are therefore dropped (loudly) unless the metered opt-in is set.
    """
    if metered_allowed is None:
        metered_allowed = _provider_flag("allow_metered_key")
    overrides: dict[str, str] = {} if metered_allowed else _scrubbed_sdk_env()
    for key, value in _configured_sdk_env().items():
        if not metered_allowed and _is_metered_sdk_env_value(key, value):
            logger.warning(
                "agent.claude_agent_sdk.env[%s] is a metered billing vector — "
                "ignoring (set allow_metered_key: true to permit it)",
                key,
            )
            continue
        overrides[key] = value
    return overrides


def _configured_permission_mode() -> Optional[str]:
    """agent.claude_agent_sdk.permission_mode from config.yaml, validated.

    Takes a validated SDK permission-mode intent (one of
    _SDK_PERMISSION_MODES — note "auto" here is the SDK's own mode, NOT the
    HERMES_TERMINAL_SECURITY_MODE value of the same name). The literal
    ``bypassPermissions`` intent is emitted as callback-capable ``default`` and
    emulated inside Hermes after immutable approval floors. Empty/absent —
    the default — keeps current behavior: the HERMES_TERMINAL_SECURITY_MODE
    mapping stands, so existing deployments harden without env archaeology
    only when they opt in. Unknown values are ignored with a warning:
    permissions must never silently loosen (or tighten into an unusable
    mode) on a typo."""
    raw = str(_provider_config().get("permission_mode") or "").strip()
    if not raw:
        return None
    if raw not in _SDK_PERMISSION_MODES:
        logger.warning(
            "agent.claude_agent_sdk.permission_mode=%r is not a valid SDK "
            "permission mode (one of %s) — ignoring it; the "
            "HERMES_TERMINAL_SECURITY_MODE mapping stands.",
            raw,
            ", ".join(_SDK_PERMISSION_MODES),
        )
        return None
    return raw


# The SDK's own setting-source literals (verified against the installed
# claude-agent-sdk 0.2.120 SettingSource type).
_SDK_SETTING_SOURCES = ("user", "project", "local")


def _configured_setting_sources() -> list:
    """agent.claude_agent_sdk.setting_sources from config.yaml, validated.

    Default (absent/empty) is FULL ISOLATION — the SDK loads no filesystem
    settings, so ambient ``~/.claude`` or project files cannot re-permission
    tools or install hooks underneath the configured posture. Deployments
    whose operating model deliberately stores tool grants in the operator's
    own ``~/.claude/settings.json`` — an unattended box whose cron turns
    must pre-approve WebSearch/MCP tools with no human to answer a prompt —
    opt back in explicitly (``setting_sources: ["user"]``). Unknown entries
    are dropped with a warning: a typo must neither silently widen isolation
    nor quietly load an unintended source."""
    raw = _provider_config().get("setting_sources")
    if not isinstance(raw, (list, tuple)):
        return []
    sources: list = []
    for entry in raw:
        name = str(entry or "").strip()
        if name in _SDK_SETTING_SOURCES:
            if name not in sources:
                sources.append(name)
        elif name:
            logger.warning(
                "agent.claude_agent_sdk.setting_sources entry %r is not a "
                "valid SDK setting source (one of %s) — dropping it.",
                name,
                ", ".join(_SDK_SETTING_SOURCES),
            )
    return sources


# ---------- stdout framing ----------
# The SDK reads the CLI's NDJSON stdout through a line framer and kills the
# whole message reader when one message exceeds max_buffer_size — a FATAL
# CLIJSONDecodeError, not a skipped message, so the turn dies mid-flight and
# the session is retired (see the retire matrix in claude_sdk_runtime). The
# SDK's own default is 1 MiB (subprocess_cli._DEFAULT_MAX_BUFFER_SIZE), which
# a single large tool result clears easily: production forensics on
# 2026-08-17 21:03 EDT caught a turn killed this way right after an Edit,
# with the CLI transcript's largest persisted line at 347 KB — the oversized
# message never reached disk.
#
# Upstream treats the option, not the default, as the fix: PR #416 proposed
# raising the default to 10 MiB and was withdrawn ("the existing
# max_buffer_size parameter already provides the needed functionality"), and
# issue #98 was closed by adding the option. So Hermes sets it explicitly
# rather than carrying an SDK patch. 10 MiB matches the figure that
# discussion converged on.
#
# NOTE: the SDK measures this with len() on a str, so the unit is Unicode
# CODE POINTS despite the "bytes" wording in its error text (upstream issue
# #1165, open). Worst-case real memory for a multibyte-heavy message is
# therefore ~4x this number — still bounded, and the point of the limit is to
# stop an unterminated line growing without end, not to be exact.
_DEFAULT_MAX_BUFFER_SIZE = 10 * 1024 * 1024


def _configured_max_buffer_size() -> int:
    """agent.claude_agent_sdk.max_buffer_size, validated.

    Same warn-and-fall-back contract as the timeout validators: bools are
    rejected before int() (YAML `true` must not become 1), and non-numeric,
    zero or negative values warn and yield the built-in default. There is
    deliberately no `0 = unlimited`: the limit is the only backstop against a
    CLI that never terminates a line, and removing it trades a killed turn
    for an OOM on a memory-constrained host."""
    raw = _provider_config().get("max_buffer_size")
    if raw is None:
        return _DEFAULT_MAX_BUFFER_SIZE
    if isinstance(raw, bool):
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is a boolean, not a "
            "size — ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is not a number — "
            "ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    if isinstance(raw, float) and not raw.is_integer():
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is not a whole number "
            "— ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    if value <= 0:
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is out of range — "
            "ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    return value


def _configured_timeout_seconds(key: str, *, allow_zero: bool) -> Optional[float]:
    """Numeric seconds from `agent.claude_agent_sdk.<key>`, validated.

    Same warn-and-fall-back contract as _configured_max_budget_usd: bools are
    rejected before float() (YAML `true` must not become 1.0), non-numeric and
    negative values warn and yield None (= use the built-in default). `0` is
    key-specific: allowed only where "disabled" is a documented meaning."""
    raw = _provider_config().get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is a boolean, not seconds — "
            "ignoring it (using the built-in default).", key, raw,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is not a number of seconds — "
            "ignoring it (using the built-in default).", key, raw,
        )
        return None
    if value < 0 or (value == 0 and not allow_zero):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is out of range — ignoring it "
            "(using the built-in default).", key, raw,
        )
        return None
    return value


def _configured_turn_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.turn_timeout (seconds). Positive only — there
    is deliberately no `0 = unlimited`: the gateway's inactivity monitor
    (`agent.gateway_timeout`, 1800s) is the outer ceiling for SDK turns and
    an unlimited soft budget under it would leave the quiet watchdog as the
    only transport-level bound."""
    return _configured_timeout_seconds("turn_timeout", allow_zero=False)


def _configured_post_tool_quiet_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.post_tool_quiet_timeout (seconds).
    `0` = explicitly disabled. Absent/None = streaming-dependent default
    (90s with streaming on, disabled with streaming off)."""
    return _configured_timeout_seconds("post_tool_quiet_timeout", allow_zero=True)


def _mcp_registry_tool_specs(agent, snapshot: list) -> list:
    """MCP schemas the agent's gating allows that its snapshot is missing.

    The snapshot is missing them whenever Hermes' own tool-search deferral is
    active: ``get_tool_definitions`` collapses MCP tools behind
    ``tool_search``/``describe``/``call`` before the bridge is built, so they
    never reach the subprocess at all — no bucket carries them, and direct
    registration in ``mcp_servers`` covers neither stdio servers (no URL) nor
    servers whose URL is refused.

    That deferral is redundant on this lane. The spawned CLI runs its own tool
    search over whatever it is handed (``ToolSearch`` returning
    ``tool_reference`` entries), and unlike ours it keeps every tool
    reachable — it only decides what to put in context, which is where the
    token cost is actually paid. Ours removes the tool outright, so the
    subprocess cannot surface what it never received. Reading the
    pre-assembly schemas hands it the full surface and lets its own deferral
    do the job.

    This is a diff, not a concatenation: only ``mcp__``-prefixed names the
    snapshot does not already carry are returned, so a server the snapshot
    already covers is never registered twice.

    Gating is the agent's own — the same ``enabled_toolsets`` /
    ``disabled_toolsets`` it was built with, which is what
    ``refresh_agent_mcp_tools`` uses for the same rebuild. Reading the
    registry unfiltered would resurrect a server the operator disabled for
    this platform, and registry dispatch is not allowlist-gated.

    Never raises: any failure yields ``[]`` and leaves the bridge exactly as
    it would have been.
    """
    # Honour the same opt-out the snapshot refresh honours: background review
    # sets `_skip_mcp_refresh` deliberately, and reading the registry here
    # would walk straight past it.
    if getattr(agent, "_skip_mcp_refresh", False):
        return []

    try:
        from tools.mcp_tool_discovery import has_registered_mcp_tools

        if not has_registered_mcp_tools():
            return []
    except Exception:
        logger.debug("MCP registry probe failed", exc_info=True)
        return []

    try:
        from model_tools import get_tool_definitions

        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        specs = get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True,
            # Pre-assembly schemas: tool_search would otherwise hand back its
            # tier-2 placeholder and the bridge would register that instead
            # of the real tool.
            skip_tool_search_assembly=True,
        )
    except Exception:
        logger.debug("MCP registry tool-definition read failed", exc_info=True)
        return []

    def _name(spec: Any) -> Optional[str]:
        try:
            normalized = normalize_tool_spec(spec, strip_mcp_prefix=False)
        except Exception:
            return None
        return normalized[0] if normalized else None

    seen = {n for n in (_name(s) for s in (snapshot or [])) if n}
    out: list = []
    for spec in specs or []:
        name = _name(spec)
        if not name or not name.startswith("mcp__") or name in seen:
            continue
        seen.add(name)
        out.append(spec)
    return out


def _http_mcp_entries_from_config() -> dict[str, dict]:
    """Return explicitly opted-in, credential-free HTTP MCPs for the SDK.

    Direct registration shares the ``hybrid_mcp_bridge`` opt-in and exclusion
    list. Header-bearing, templated, userinfo-bearing, and credential-like
    query or fragment URLs are refused because the SDK serializes this config
    into the Claude CLI's ``--mcp-config`` process argument. Returns ``{}`` on
    any read/parse failure and never logs a URL or credential value.
    """
    import os as _os
    import re as _re
    from urllib.parse import unquote as _unquote, urlsplit as _urlsplit

    try:
        import yaml as _yaml  # type: ignore
    except Exception:
        return {}

    home = _os.environ.get("HERMES_HOME") or _os.path.expanduser("~/.hermes")
    profile = _os.environ.get("HERMES_PROFILE") or "default"
    paths = [
        _os.path.join(home, "config.yaml"),
        _os.path.join(home, "profiles", profile, "config.yaml"),
    ]

    merged: dict[str, Any] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
            block = (data.get("mcp_servers") or {}) if isinstance(data, dict) else {}
            if isinstance(block, dict):
                merged.update(block)
        except FileNotFoundError:
            continue
        except Exception:
            logger.debug("failed to read %s for HTTP MCP discovery", p, exc_info=True)
            continue

    if not _provider_flag("hybrid_mcp_bridge", default=False):
        return {}

    _env_re = _re.compile(r"\$\{[^}]*\}")
    _secret_field_re = _re.compile(
        r"(?:api[_-]?key|token|secret|password|credential|authorization|auth)",
        _re.I,
    )
    excluded_names = set(_configured_hybrid_exclude())

    out: dict[str, dict] = {}
    for raw_name, cfg in merged.items():
        if not isinstance(cfg, dict):
            continue
        url = cfg.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        name = str(raw_name)
        url = url.strip()
        if name in excluded_names:
            logger.info("claude-agent-sdk: HTTP MCP %r excluded by config", name)
            continue
        if cfg.get("headers"):
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because it has headers",
                name,
            )
            continue
        if _env_re.search(url):
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is templated",
                name,
            )
            continue
        try:
            parsed = _urlsplit(url)
        except ValueError:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is malformed",
                name,
            )
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is not HTTP(S)",
                name,
            )
            continue
        # Direct HTTP MCP configuration is passed to the SDK/CLI process.
        # Treat every query, fragment, or userinfo component as secret-bearing:
        # a denylist cannot safely distinguish signed/public query parameters
        # from encoded credentials (including nested percent encoding).
        if parsed.username is not None or parsed.query or parsed.fragment:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL has userinfo, query, or fragment",
                name,
            )
            continue
        out[name] = {"type": "http", "url": url}
    return out


def _hermes_repo_root() -> str:
    """Repo root for the hermes-tools MCP subprocess (PYTHONPATH)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


# The SDK spawns the Claude Code CLI with the FULL inherited parent env and
# merges ``ClaudeAgentOptions.env`` ON TOP of it (subprocess_cli.py builds
# ``{**os.environ, ..., **options.env, ...}``), so a key can never be REMOVED
# from the options side — the only lever is an explicit override. Every
# metered billing vector below is overridden to "" (the CLI and the AWS/GCP
# SDKs treat an empty value as unset), so a credentialed gateway environment
# cannot silently re-route this provider's billing off the Claude
# subscription. Why each class of key is stripped:
#
#   ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN — the CLI prefers these over
#     subscription OAuth: the exact silent-rebilling the fail-closed startup
#     guard exists to stop. The guard covers Hermes' own process at startup;
#     this scrub covers the spawned CLI, which otherwise inherits them.
#   CLAUDE_CODE_USE_BEDROCK + AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
#     AWS_SESSION_TOKEN — flips the CLI onto AWS Bedrock, billing the AWS
#     account (metered) instead of the subscription; the AWS static
#     credentials are the vector that makes the flip actually authenticate.
#   CLAUDE_CODE_USE_VERTEX + GOOGLE_APPLICATION_CREDENTIALS — the same
#     takeover via Google Vertex, billing the GCP project.
#
# Deliberately NOT stripped: HOME/PATH (the CLI needs them to run and to find
# its credential store) and the subscription token flow itself —
# CLAUDE_CODE_OAUTH_TOKEN / ~/.claude — which is what this provider runs on.
_METERED_ENV_DENYLIST = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _is_subscription_oauth_token(value: str) -> bool:
    """True when an ANTHROPIC_TOKEN value is OAuth/setup-token shaped — the
    subscription lane itself, not a metered vector. Unknown shapes count as
    metered (fail closed), including when the classifier cannot be imported."""
    try:
        from agent.anthropic_adapter import _is_oauth_token
    except Exception:
        return False
    try:
        return bool(_is_oauth_token(value))
    except Exception:
        return False


def _is_metered_sdk_env_value(key: str, value: str) -> bool:
    """Whether an SDK child env value can switch billing off subscription.

    ``ANTHROPIC_TOKEN`` is ambiguous in Hermes: a setup/OAuth token is the
    desired subscription credential, while every unrecognised shape is
    treated as metered.  The other denylisted variables are always metered
    routing vectors when non-empty.
    """
    if key not in _METERED_ENV_DENYLIST or not value:
        return False
    if key == "ANTHROPIC_TOKEN":
        return not _is_subscription_oauth_token(value)
    return True


def _scrubbed_sdk_env() -> dict[str, str]:
    """Empty-string overrides for every metered billing vector currently set
    in the parent environment. Only PRESENT keys are overridden — writing
    ``""`` for absent ones would introduce empty vars the child never had
    (an empty AWS_ACCESS_KEY_ID can itself confuse AWS credential chains)."""
    return {
        key: ""
        for key in _METERED_ENV_DENYLIST
        if _is_metered_sdk_env_value(key, os.environ.get(key, ""))
    }


# The SDK serializes the stdio MCP config — env INCLUDED — into the claude
# CLI's --mcp-config argument, i.e. onto the subprocess argv, which any local
# user can read via ps. Nothing secret may ever ride this dict: the env is a
# minimal ALLOWLIST, never a copy of the credentialed environment. Keyed
# Hermes tools inside the server degrade via their own check_fns — the
# subscription lane's fail-closed posture.
_MCP_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PYTHONUTF8",
    "HERMES_HOME",
    "HERMES_KANBAN_TASK",
    "HERMES_MCP_STATE_DB",  # the shims' documented state-DB override — a path, not a secret
    "HERMES_QUIET",
    "HERMES_REDACT_SECRETS",
)


def _provider_config() -> dict:
    """The `agent.claude_agent_sdk` config block ({} when absent/unreadable)."""
    try:
        from hermes_cli.config import load_config_readonly

        block = ((load_config_readonly() or {}).get("agent", {}) or {}).get(
            "claude_agent_sdk", {}
        )
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _provider_flag(config_key: str, default: bool = False) -> bool:
    """Behavioural flag read from `agent.claude_agent_sdk.<key>` in config.yaml.

    config.yaml is the ONLY interface. AGENTS.md keeps non-secret behavioural
    settings out of `HERMES_*` environment variables, so there is deliberately
    no env override here — a deployment sets the key in config.yaml.
    Canonical defaults live in `hermes_cli/config.py::DEFAULT_CONFIG`.
    """
    value = _provider_config().get(config_key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _configured_hybrid_exclude() -> list:
    """agent.claude_agent_sdk.hybrid_mcp_bridge_exclude from config.yaml.

    Names to drop from the hybrid bridge (both buckets). Match on the raw
    Hermes registry name, no ``mcp__`` prefix. Non-string entries are
    dropped silently — a typo is a config error the operator will notice
    when the tool doesn't disappear, not a reason to widen exposure.
    """
    raw = _provider_config().get("hybrid_mcp_bridge_exclude")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        name = entry.strip()
        if name and name not in out:
            out.append(name)
    return out


def _build_hermes_tools_mcp_config(
    hermes_session_id: Optional[str] = None,
) -> dict[str, Any]:
    """The stdio MCP server exposing Hermes tools into the SDK agent loop —
    the exact server the codex runtime uses (backend-agnostic), launched with
    this venv's interpreter. McpStdioServerConfig has no cwd field, so the
    repo root rides PYTHONPATH."""
    env = {
        key: os.environ[key]
        for key in _MCP_ENV_ALLOWLIST
        if os.environ.get(key)
    }
    env["PYTHONPATH"] = _hermes_repo_root() + os.pathsep + os.environ.get("PYTHONPATH", "")
    if hermes_session_id:
        # Lets the stateless session_search shim exclude the calling
        # session's own lineage from recall results (#26567). The shim reads
        # the canonical HERMES_SESSION_ID — set explicitly with THIS
        # session's id rather than allowlisting the ambient variable, so a
        # multi-session host can never leak a sibling session's id into the
        # subprocess.
        env["HERMES_SESSION_ID"] = str(hermes_session_id)
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            "-m",
            "agent.transports.hermes_tools_mcp_server",
            "--profile",
            "claude-agent-sdk",
        ],
        "env": env,
    }
