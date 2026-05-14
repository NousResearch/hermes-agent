"""Migrate Hermes' MCP server config and Codex's installed curated plugins
to the format Codex expects in ~/.codex/config.toml.

When the user enables the codex_app_server runtime, the codex subprocess
runs its own MCP client and its own plugin runtime (Linear, Atlassian,
Asana, plus per-account ChatGPT apps via app/list). For both of those to
be useful, the user's choices need to be visible to codex too. This
module:

  1. Reads Hermes' YAML and writes equivalent [mcp_servers.<name>]
     entries to ~/.codex/config.toml.
  2. Queries codex's `plugin/list` for the openai-curated marketplace
     and writes [plugins."<name>@<marketplace>"] entries for any plugin
     the user has installed=true on their codex CLI. (This is what
     OpenClaw calls "migrate native codex plugins" — the YouTube-video-
     worthy bit Pash highlighted: Canva, GitHub, Calendar, Gmail
     pre-configured.)
  3. Writes a [permissions] default profile so users on this runtime
     don't get an approval prompt on every write attempt.

What translates (MCP servers):
  Hermes mcp_servers.<n>.command/args/env  → codex stdio transport
  Hermes mcp_servers.<n>.url/headers       → codex streamable_http transport
  Hermes mcp_servers.<n>.timeout           → codex tool_timeout_sec
  Hermes mcp_servers.<n>.connect_timeout   → codex startup_timeout_sec

What does NOT translate (warned + skipped):
  Hermes-specific keys (sampling, etc.) — codex's MCP client has no
  equivalent. Listed in the per-server skipped[] field of the report.

What's NOT migrated (intentional):
  AGENTS.md — codex respects this file natively in its cwd. Hermes' own
  AGENTS.md (project-level) is already in the worktree, so codex picks
  it up without translation. No code needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Marker comments wrapping the managed section so re-runs can detect
# what's ours and what's user-edited. Both must appear or strip is a no-op.
MIGRATION_MARKER = (
    "# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section"
)
MIGRATION_END_MARKER = (
    "# end hermes-agent managed section"
)


@dataclass
class MigrationReport:
    """Outcome of a migration pass."""

    target_path: Optional[Path] = None
    migrated: list[str] = field(default_factory=list)
    skipped_keys_per_server: dict[str, list[str]] = field(default_factory=dict)
    migrated_plugins: list[str] = field(default_factory=list)
    plugin_query_error: Optional[str] = None
    wrote_permissions_default: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    written: bool = False
    dry_run: bool = False

    def summary(self) -> str:
        lines = []
        if self.dry_run:
            lines.append(f"(dry run) Would write {self.target_path}")
        elif self.written:
            lines.append(f"Wrote {self.target_path}")
        if self.migrated:
            lines.append(f"Migrated {len(self.migrated)} MCP server(s):")
            for name in self.migrated:
                skipped = self.skipped_keys_per_server.get(name, [])
                note = (
                    f" (skipped: {', '.join(skipped)})" if skipped else ""
                )
                lines.append(f"  - {name}{note}")
        else:
            lines.append("No MCP servers found in Hermes config.")
        if self.migrated_plugins:
            lines.append(
                f"Migrated {len(self.migrated_plugins)} native Codex plugin(s):"
            )
            for name in self.migrated_plugins:
                lines.append(f"  - {name}")
        elif self.plugin_query_error:
            lines.append(f"Codex plugin discovery skipped: {self.plugin_query_error}")
        if self.wrote_permissions_default:
            lines.append(
                f"Wrote default_permissions = "
                f"{self.wrote_permissions_default!r}"
            )
        for err in self.errors:
            lines.append(f"⚠ {err}")
        return "\n".join(lines)


# Hermes keys that codex's MCP schema doesn't support — dropped during
# migration with a warning. Anything not on the keep list AND not the
# transport keys is added to skipped.
_KNOWN_HERMES_KEYS = {
    # transport — stdio
    "command", "args", "env", "cwd",
    # transport — http
    "url", "headers", "transport",
    # timeouts
    "timeout", "connect_timeout",
    # general
    "enabled", "description",
}

# Subset that have a direct codex equivalent.
_KEYS_DROPPED_WITH_WARNING = {
    # Hermes' sampling subsection — codex MCP has no equivalent
    "sampling",
}


def _translate_one_server(
    name: str, hermes_cfg: dict
) -> tuple[Optional[dict], list[str]]:
    """Translate one Hermes MCP server config to the codex inline-table dict
    representation. Returns (codex_entry, skipped_keys).

    codex_entry is a dict ready for TOML serialization, or None when the
    server can't be translated (e.g. neither command nor url present)."""
    if not isinstance(hermes_cfg, dict):
        return None, []

    skipped: list[str] = []
    out: dict[str, Any] = {}

    has_command = bool(hermes_cfg.get("command"))
    has_url = bool(hermes_cfg.get("url"))

    if has_command and has_url:
        skipped.append("url (both command and url set; preferring stdio)")
        has_url = False

    if has_command:
        # Stdio transport
        out["command"] = str(hermes_cfg["command"])
        args = hermes_cfg.get("args") or []
        if args:
            out["args"] = [str(a) for a in args]
        env = hermes_cfg.get("env") or {}
        if env:
            # Codex expects string values
            out["env"] = {str(k): str(v) for k, v in env.items()}
        cwd = hermes_cfg.get("cwd")
        if cwd:
            out["cwd"] = str(cwd)
    elif has_url:
        # streamable_http transport (codex covers both http and SSE here)
        out["url"] = str(hermes_cfg["url"])
        headers = hermes_cfg.get("headers") or {}
        if headers:
            out["http_headers"] = {str(k): str(v) for k, v in headers.items()}
        # Hermes' transport: sse hint is informational; codex auto-negotiates
        if hermes_cfg.get("transport") == "sse":
            skipped.append("transport=sse (codex auto-negotiates)")
    else:
        return None, ["no command or url field"]

    # Timeouts
    if "timeout" in hermes_cfg:
        try:
            out["tool_timeout_sec"] = float(hermes_cfg["timeout"])
        except (TypeError, ValueError):
            skipped.append("timeout (not numeric)")
    if "connect_timeout" in hermes_cfg:
        try:
            out["startup_timeout_sec"] = float(hermes_cfg["connect_timeout"])
        except (TypeError, ValueError):
            skipped.append("connect_timeout (not numeric)")

    # Enabled flag (codex defaults to true so we only emit when explicitly false)
    if hermes_cfg.get("enabled") is False:
        out["enabled"] = False

    # Detect keys we explicitly drop with warning
    for key in hermes_cfg:
        if key in _KEYS_DROPPED_WITH_WARNING:
            skipped.append(f"{key} (no codex equivalent)")
        elif key not in _KNOWN_HERMES_KEYS:
            skipped.append(f"{key} (unknown Hermes key)")

    return out, skipped


def _format_toml_value(value: Any) -> str:
    """Minimal TOML value formatter for the value types we emit.

    We only emit strings, numbers, booleans, and tables of those — no nested
    arrays of tables. This covers everything codex's MCP schema accepts."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # Escape per TOML basic-string rules. Order matters: backslash
        # first so the other escapes don't get re-escaped.
        # Control characters (newline, tab, etc.) must use \-escapes
        # because TOML basic strings don't allow literal control chars
        # — passing them through would produce invalid TOML that codex
        # would refuse to load. Paths usually don't contain control
        # chars but env-var passthrough (HERMES_HOME, PYTHONPATH) could
        # in pathological cases.
        escaped = (
            value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\b", "\\b")
            .replace("\t", "\\t")
            .replace("\n", "\\n")
            .replace("\f", "\\f")
            .replace("\r", "\\r")
        )
        return f'"{escaped}"'
    if isinstance(value, list):
        items = ", ".join(_format_toml_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, dict):
        items = ", ".join(
            f'{_quote_key(k)} = {_format_toml_value(v)}' for k, v in value.items()
        )
        return "{ " + items + " }" if items else "{}"
    raise ValueError(f"Unsupported TOML value type: {type(value).__name__}")


def _quote_key(key: str) -> str:
    """Return key bare-or-quoted depending on whether it's a valid bare key."""
    if all(c.isalnum() or c in "-_" for c in key) and key:
        return key
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'

def render_codex_toml_section(
    servers: dict[str, dict],
    plugins: Optional[list[dict]] = None,
    default_permission_profile: Optional[str] = None,
) -> str:
    """Render the managed [mcp_servers.<n>] / [plugins.<id>] / [permissions]
    block for ~/.codex/config.toml.

    Args:
        servers: dict of MCP server name → translated codex inline-table
        plugins: optional list of {name, marketplace, enabled} for native
            Codex plugins to enable. (E.g. the Linear / Atlassian / Asana
            curated plugins, or per-account ChatGPT apps.)
        default_permission_profile: when set, write `[permissions] default`
            so the user doesn't get an approval prompt on every write
            attempt. Common values: "workspace-write", "read-only",
            "full-access".
    """
    out = [MIGRATION_MARKER]
    if not servers and not plugins and not default_permission_profile:
        out.append("# (no MCP servers, plugins, or permissions configured by Hermes)")
        out.append(MIGRATION_END_MARKER)
        return "\n".join(out) + "\n"

    if default_permission_profile:
        # Codex's config schema: `default_permissions` is a top-level
        # string referencing a profile name. Built-in profile names start
        # with ":" (":workspace-write", ":read-only", ":full-access"). The
        # [permissions] table is for *user-defined* named profiles with
        # structured fields — not what we want.
        normalized = (
            default_permission_profile
            if default_permission_profile.startswith(":")
            else f":{default_permission_profile}"
        )
        out.append("")
        out.append(f"default_permissions = {_format_toml_value(normalized)}")

    if servers:
        for name in sorted(servers.keys()):
            cfg = servers[name]
            out.append("")
            out.append(f"[mcp_servers.{_quote_key(name)}]")
            for k, v in cfg.items():
                out.append(f"{_quote_key(k)} = {_format_toml_value(v)}")

    if plugins:
        for plugin in sorted(plugins, key=lambda p: f"{p.get('name','')}@{p.get('marketplace','')}"):
            name = plugin.get("name") or ""
            marketplace = plugin.get("marketplace") or "openai-curated"
            enabled = bool(plugin.get("enabled", True))
            qualified = f"{name}@{marketplace}"
            out.append("")
            out.append(f'[plugins.{_quote_key(qualified)}]')
            out.append(f"enabled = {_format_toml_value(enabled)}")

    out.append("")
    out.append(MIGRATION_END_MARKER)
    return "\n".join(out) + "\n"


def _strip_existing_managed_block(toml_text: str) -> str:
    """Remove any prior managed section so re-runs idempotently replace it.

    The managed section is everything between MIGRATION_MARKER (start) and
    MIGRATION_END_MARKER (end), inclusive of both markers. User-edited
    sections above or below are preserved verbatim.

    Backward compatibility: if the start marker is found but no end marker
    follows, we fall back to the heuristic that swallows lines until we
    hit a section that's not [mcp_servers.*]/[plugins.*]/[permissions]/
    a `default_permissions =` key. This matches what older versions of
    this code wrote so re-runs don't break configs from prior Hermes
    versions."""
    lines = toml_text.splitlines(keepends=True)
    out: list[str] = []
    in_managed = False
    saw_end_marker = False
    for line in lines:
        line_stripped_nl = line.rstrip("\n")
        if line_stripped_nl == MIGRATION_MARKER:
            in_managed = True
            saw_end_marker = False
            continue
        if in_managed:
            if line_stripped_nl == MIGRATION_END_MARKER:
                in_managed = False
                saw_end_marker = True
                continue
            stripped = line.lstrip()
            if not saw_end_marker and stripped.startswith("[") and not (
                stripped.startswith("[mcp_servers")
                or stripped.startswith("[plugins")
                or stripped.startswith("[permissions]")
                or stripped.startswith("[permissions.")
            ):
                # Old-format managed block without end marker: bail back
                # to user content as soon as we see a non-managed section.
                in_managed = False
                out.append(line)
                continue
            # Otherwise swallow the line.
            continue
        out.append(line)
    return "".join(out)


def _query_codex_plugins(
    codex_home: Optional[Path] = None,
    timeout: float = 8.0,
) -> tuple[list[dict], Optional[str]]:
    """Query codex's `plugin/list` for installed curated plugins.

    Spawns `codex app-server` briefly, sends initialize + plugin/list,
    extracts plugins where installed=true. Returns (plugins, error).
    Plugins is a list of {name, marketplace, enabled} dicts ready for
    render_codex_toml_section().

    On any failure (codex not installed, RPC error, timeout) returns
    ([], error_message). Migration treats this as non-fatal — MCP
    servers and permissions still write through.
    """
    try:
        from agent.transports.codex_app_server import CodexAppServerClient
    except Exception as exc:
        return [], f"transport unavailable: {exc}"

    try:
        with CodexAppServerClient(
            codex_home=str(codex_home) if codex_home else None
        ) as client:
            client.initialize(client_name="hermes-migration")
            resp = client.request("plugin/list", {}, timeout=timeout)
    except Exception as exc:
        return [], f"plugin/list query failed: {exc}"

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    marketplaces = resp.get("marketplaces") or []
    if not isinstance(marketplaces, list):
        return [], "plugin/list response missing 'marketplaces'"
    for marketplace in marketplaces:
        if not isinstance(marketplace, dict):
            continue
        market_name = str(marketplace.get("name") or "openai-curated")
        plugins = marketplace.get("plugins") or []
        if not isinstance(plugins, list):
            continue
        for plugin in plugins:
            if not isinstance(plugin, dict):
                continue
            installed = bool(plugin.get("installed", False))
            if not installed:
                continue
            # Skip plugins codex itself reports as unavailable (broken
            # install, missing OAuth, removed from marketplace, etc.).
            # Cf. openclaw/openclaw#80815 — OpenClaw learned to gate
            # migration on app readiness to avoid writing config that
            # would fail at activation time. Our migration writes to
            # codex's config.toml directly, so a broken plugin would
            # surface as a codex error on first use. Skipping it here
            # keeps the migrated config clean and the user's first
            # codex turn from failing.
            availability = str(plugin.get("availability") or "").upper()
            if availability and availability != "AVAILABLE":
                logger.debug(
                    "skipping plugin %s: availability=%s",
                    plugin.get("name"), availability,
                )
                continue
            name = str(plugin.get("name") or "")
            if not name:
                continue
            key = (name, market_name)
            if key in seen:
                continue
            seen.add(key)
            # Carry forward whatever 'enabled' codex reports — defaults to
            # true for installed plugins. This is the same shape OpenClaw
            # writes when migrating native codex plugins.
            out.append({
                "name": name,
                "marketplace": market_name,
                "enabled": bool(plugin.get("enabled", True)),
            })
    return out, None


def _build_hermes_tools_mcp_entry() -> dict:
    """Build the codex stdio-transport entry that launches Hermes' own
    tool surface as an MCP server. Codex's subprocess will call back into
    this for browser/web/delegate_task/vision/memory/skills tools.

    The command runs the worktree's Python via the current sys.executable
    so a hermes installed under /opt/, /usr/local/, or a venv all work.
    HERMES_HOME and PYTHONPATH are passed through so the spawned process
    sees the same config + module layout the user is running."""
    import sys

    env: dict[str, str] = {}
    # HERMES_HOME passes through if set so the MCP subprocess sees the
    # same config / auth / sessions DB as the parent CLI.
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        env["HERMES_HOME"] = hermes_home
    # PYTHONPATH passes through so a worktree-launched hermes finds the
    # branch's modules instead of the installed package.
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    # Quiet mode + redaction defaults so the MCP wire stays clean.
    env["HERMES_QUIET"] = "1"
    env["HERMES_REDACT_SECRETS"] = env.get("HERMES_REDACT_SECRETS", "true")

    out: dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
    }
    if env:
        out["env"] = env
    # Generous timeouts — browser_navigate or delegate_task can take a
    # while; we don't want codex's MCP client to give up too early.
    out["startup_timeout_sec"] = 30.0
    out["tool_timeout_sec"] = 600.0
    return out


def migrate(
    hermes_config: dict,
    *,
    codex_home: Optional[Path] = None,
    dry_run: bool = False,
    discover_plugins: bool = True,
    default_permission_profile: Optional[str] = ":workspace",
    expose_hermes_tools: bool = True,
) -> MigrationReport:
    """Translate Hermes mcp_servers config + Codex curated plugins into
    ~/.codex/config.toml.

    Args:
        hermes_config: full ~/.hermes/config.yaml dict
        codex_home: override CODEX_HOME (defaults to ~/.codex)
        dry_run: skip the actual write; report what would happen
        discover_plugins: when True (default), query `plugin/list` against
            the live codex CLI to migrate any installed curated plugins
            into [plugins."<name>@<marketplace>"] entries. Set False to
            skip the subprocess spawn (for tests or restricted environments).
        default_permission_profile: when set (default ":workspace"), write
            top-level `default_permissions = "<name>"` so users on this
            runtime don't get an approval prompt on every write attempt.
            Built-in codex profile names are ":workspace", ":read-only",
            ":danger-no-sandbox" (note the leading ":"). Also accepts a
            user-defined profile name (no leading ":") that the user has
            configured in their own [permissions.<name>] table. Set None
            to leave permissions unset and let codex use its compiled-in
            default (which is read-only).
        expose_hermes_tools: when True (default), register Hermes' own
            tool surface (web_search, browser_*, delegate_task, vision,
            memory, skills, etc.) as an MCP server in ~/.codex/config.toml
            so the codex subprocess can call back into Hermes for tools
            codex doesn't have built in. Set False to opt out.
    """
    report = MigrationReport(dry_run=dry_run)
    codex_home = codex_home or Path.home() / ".codex"
    target = codex_home / "config.toml"
    report.target_path = target

    hermes_servers = (hermes_config or {}).get("mcp_servers") or {}
    if not isinstance(hermes_servers, dict):
        report.errors.append(
            "mcp_servers in Hermes config is not a dict; cannot migrate."
        )
        return report

    translated: dict[str, dict] = {}
    for name, cfg in hermes_servers.items():
        out, skipped = _translate_one_server(str(name), cfg or {})
        if out is None:
            report.errors.append(
                f"server {name!r} skipped: {', '.join(skipped) or 'no transport configured'}"
            )
            continue
        translated[str(name)] = out
        if skipped:
            report.skipped_keys_per_server[str(name)] = skipped
        report.migrated.append(str(name))

    # Discover installed Codex curated plugins. Best-effort — never blocks
    # the migration if codex is unreachable or the RPC fails.
    plugins: list[dict] = []
    if discover_plugins and not dry_run:
        plugins, plugin_err = _query_codex_plugins(codex_home=codex_home)
        if plugin_err:
            report.plugin_query_error = plugin_err
        for p in plugins:
            report.migrated_plugins.append(f"{p['name']}@{p['marketplace']}")

    # Track whether we wrote a default permission profile so the report
    # surfaces it to the user.
    if default_permission_profile:
        report.wrote_permissions_default = default_permission_profile

    # Inject Hermes' own tool surface as an MCP server so the spawned
    # codex subprocess can call back into Hermes for the tools codex
    # doesn't ship with — web_search, browser_*, delegate_task, vision,
    # memory, skills, session_search, image_generate, text_to_speech.
    # The server itself is agent/transports/hermes_tools_mcp_server.py
    # and is launched on demand by codex (stdio MCP).
    if expose_hermes_tools:
        translated["hermes-tools"] = _build_hermes_tools_mcp_entry()
        if "hermes-tools" not in report.migrated:
            report.migrated.append("hermes-tools")

    # Build the new managed block
    managed_block = render_codex_toml_section(
        translated, plugins=plugins,
        default_permission_profile=default_permission_profile,
    )

    # Read existing codex config if any, strip the prior managed block,
    # append the new one.
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except Exception as exc:
            report.errors.append(f"could not read {target}: {exc}")
            return report
        without_managed = _strip_existing_managed_block(existing)
        # Ensure exactly one blank line between user content and managed block
        if without_managed and not without_managed.endswith("\n"):
            without_managed += "\n"
        new_text = (
            without_managed.rstrip("\n") + "\n\n" + managed_block
            if without_managed.strip()
            else managed_block
        )
    else:
        new_text = managed_block

    if dry_run:
        return report

    try:
        codex_home.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a temp file in the same directory then
        # rename. Same-directory rename is atomic on POSIX and ReplaceFile
        # on Windows. Avoids leaving a half-written config.toml that
        # codex would refuse to load if we crash mid-write.
        import tempfile
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=".config.toml.", dir=str(codex_home)
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(new_text)
            tmp_path.replace(target)
        except Exception:
            # Clean up the temp file if the rename didn't happen.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise
        report.written = True
    except Exception as exc:
        report.errors.append(f"could not write {target}: {exc}")
    return report
