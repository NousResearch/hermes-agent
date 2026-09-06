"""Migrate Hermes MCP server config and Codex's installed curated plugins into ~/.codex/config.toml.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Marker comments wrapping the managed section so re-runs can detect what's ours and what's
# user-edited. Both must appear or strip is a no-op.
MIGRATION_MARKER = (
    "# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section")
MIGRATION_END_MARKER = "# end hermes-agent managed section"


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
                note = f" (skipped: {', '.join(skipped)})" if skipped else ""
                lines.append(f"  - {name}{note}")
        else:
            lines.append("No MCP servers found in Hermes config.")
        if self.migrated_plugins:
            lines.append(f"Migrated {len(self.migrated_plugins)} native Codex plugin(s):")
            lines.extend(f"  - {name}" for name in self.migrated_plugins)
        elif self.plugin_query_error:
            lines.append(f"Codex plugin discovery skipped: {self.plugin_query_error}")
        if self.wrote_permissions_default:
            lines.append(f"Wrote default_permissions = {self.wrote_permissions_default!r}")
        lines.extend(f"⚠ {err}" for err in self.errors)
        return "\n".join(lines)


# Hermes MCP keys codex understands (transport stdio/http, timeouts, general). Any other key is
# dropped with a warning: ``sampling`` has no codex equivalent; the rest are unknown Hermes keys.
_KNOWN_HERMES_KEYS = {
    "command", "args", "env", "cwd",
    "url", "headers", "transport",
    "timeout", "connect_timeout",
    "enabled", "description"}
_KEYS_DROPPED_WITH_WARNING = {"sampling"}

# (hermes key, codex key, skip note) — timeouts are emitted as floats or skipped when non-numeric.
_TIMEOUT_KEYS = (
    ("timeout", "tool_timeout_sec", "timeout (not numeric)"),
    ("connect_timeout", "startup_timeout_sec", "connect_timeout (not numeric)"))


def _str_map(d: dict) -> dict[str, str]:
    """Codex expects string keys and values in env / header tables."""
    return {str(k): str(v) for k, v in d.items()}


def _translate_one_server(name: str, hermes_cfg: dict) -> tuple[Optional[dict], list[str]]:
    """Translate one Hermes MCP server config to codex's inline-table dict.

    Returns ``(codex_entry, skipped_keys)``; ``codex_entry`` is None when the config is unusable.
    stdio (``command``) wins over ``url`` when both are set. Hermes' ``transport: sse`` hint is
    informational only — codex auto-negotiates. ``enabled`` is emitted only when explicitly false
    (codex defaults to true).
    """
    if not isinstance(hermes_cfg, dict):
        return None, []
    skipped: list[str] = []
    out: dict[str, Any] = {}
    if hermes_cfg.get("command"):
        if hermes_cfg.get("url"):
            skipped.append("url (both command and url set; preferring stdio)")
        out["command"] = str(hermes_cfg["command"])
        if hermes_cfg.get("args"):
            out["args"] = [str(a) for a in hermes_cfg["args"]]
        if hermes_cfg.get("env"):
            out["env"] = _str_map(hermes_cfg["env"])
        if hermes_cfg.get("cwd"):
            out["cwd"] = str(hermes_cfg["cwd"])
    elif hermes_cfg.get("url"):
        out["url"] = str(hermes_cfg["url"])
        if hermes_cfg.get("headers"):
            out["http_headers"] = _str_map(hermes_cfg["headers"])
        if hermes_cfg.get("transport") == "sse":
            skipped.append("transport=sse (codex auto-negotiates)")
    else:
        return None, ["no command or url field"]
    for hermes_key, codex_key, note in _TIMEOUT_KEYS:
        if hermes_key in hermes_cfg:
            try:
                out[codex_key] = float(hermes_cfg[hermes_key])
            except (TypeError, ValueError):
                skipped.append(note)
    if hermes_cfg.get("enabled") is False:
        out["enabled"] = False
    for key in hermes_cfg:
        if key in _KEYS_DROPPED_WITH_WARNING:
            skipped.append(f"{key} (no codex equivalent)")
        elif key not in _KNOWN_HERMES_KEYS:
            skipped.append(f"{key} (unknown Hermes key)")
    return out, skipped


# TOML basic-string escapes. Order matters: backslash first so the others aren't re-escaped.
# Control chars must be \-escaped — a literal one is invalid TOML that codex refuses to load;
# env-var passthrough (HERMES_HOME, PYTHONPATH) could carry one in pathological cases.
_TOML_ESCAPES = (
    ("\\", "\\\\"), ('"', '\\"'), ("\b", "\\b"), ("\t", "\\t"),
    ("\n", "\\n"), ("\f", "\\f"), ("\r", "\\r"))


def _escape_toml_string(value: str) -> str:
    for raw, esc in _TOML_ESCAPES:
        value = value.replace(raw, esc)
    return f'"{value}"'


def _format_toml_value(value: Any) -> str:
    """Minimal TOML formatter for what we emit: strings, numbers, bools, and flat lists/tables of
    those (everything codex's MCP schema accepts; no nested arrays of tables)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return _escape_toml_string(value)
    if isinstance(value, list):
        return f"[{', '.join(_format_toml_value(v) for v in value)}]"
    if isinstance(value, dict):
        items = ", ".join(f'{_quote_key(k)} = {_format_toml_value(v)}' for k, v in value.items())
        return "{ " + items + " }" if items else "{}"
    raise ValueError(f"Unsupported TOML value type: {type(value).__name__}")


def _quote_key(key: str) -> str:
    """Return key bare-or-quoted depending on whether it's a valid bare key."""
    if key and all(c.isalnum() or c in "-_" for c in key):
        return key
    return '"' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_codex_toml_section(
    servers: dict[str, dict], plugins: Optional[list[dict]] = None,
    default_permission_profile: Optional[str] = None) -> str:
    """Render the managed [mcp_servers.<n>] / [plugins.<id>] / default_permissions block.

    ``default_permission_profile`` (e.g. "workspace-write", "read-only", "full-access") is written
    as the top-level ``default_permissions`` string so the user is not prompted on every write.
    Built-in profile names start with ":" (":workspace-write", ...) and are normalized to that
    form; the ``[permissions]`` table is for *user-defined* named profiles, not what we want.
    """
    out = [MIGRATION_MARKER]
    if not servers and not plugins and not default_permission_profile:
        out += ["# (no MCP servers, plugins, or permissions configured by Hermes)", MIGRATION_END_MARKER]
        return "\n".join(out) + "\n"
    if default_permission_profile:
        profile = default_permission_profile
        normalized = profile if profile.startswith(":") else f":{profile}"
        out += ["", f"default_permissions = {_format_toml_value(normalized)}"]
    for name in sorted(servers or ()):
        out += ["", f"[mcp_servers.{_quote_key(name)}]"]
        out += [f"{_quote_key(k)} = {_format_toml_value(v)}" for k, v in servers[name].items()]
    plugin_sort_key = lambda p: f"{p.get('name','')}@{p.get('marketplace','')}"  # noqa: E731
    for plugin in sorted(plugins or (), key=plugin_sort_key):
        qualified = f"{plugin.get('name') or ''}@{plugin.get('marketplace') or 'openai-curated'}"
        out += ["", f'[plugins.{_quote_key(qualified)}]',
                f"enabled = {_format_toml_value(bool(plugin.get('enabled', True)))}"]
    out += ["", MIGRATION_END_MARKER]
    return "\n".join(out) + "\n"


def _insert_managed_block_at_top_level(user_text: str, managed_block: str) -> str:
    """Insert the managed block before the user's first table header.

    TOML has no syntax to return to the document root after a table header, so appending a root
    key like ``default_permissions = ...`` after a user ``[features]`` table would actually create
    ``features.default_permissions``, which codex rejects.
    """
    if not user_text.strip():
        return managed_block
    lines = user_text.splitlines(keepends=True)
    first_table_idx = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("[")), None)
    if first_table_idx is None:
        prefix = user_text.rstrip("\n")
        return f"{prefix}\n\n{managed_block}" if prefix else managed_block
    prefix = "".join(lines[:first_table_idx]).rstrip("\n")
    suffix = "".join(lines[first_table_idx:]).lstrip("\n")
    return f"{prefix}\n\n{managed_block}\n{suffix}" if prefix else f"{managed_block}\n{suffix}"


def _strip_unmanaged_plugin_tables(toml_text: str) -> str:
    """Remove ``[plugins."<name>@<marketplace>"]`` tables that live OUTSIDE the managed block.

    Codex writes these when the user runs ``codex plugins enable`` directly, before migrate has
    ever touched the file. Once migrate runs, ``plugin/list`` is the source of truth for what is
    installed and we own the ``[plugins.*]`` namespace, so dropping pre-existing tables is safe —
    and necessary, since re-emitting them inside the managed block would otherwise produce
    duplicate-table-header parse errors on codex's next startup.
    """
    out: list[str] = []
    in_plugin_table = False
    for line in toml_text.splitlines(keepends=True):
        stripped = line.lstrip()
        # Only a real ``[...]`` header flips state: multi-line array continuations like
        # ``["nested"],`` also start with ``[`` but must not end the table early and leak
        # array fragments into the output.
        if _looks_like_table_header(stripped):
            in_plugin_table = stripped.startswith("[plugins.")
        if not in_plugin_table:  # swallow keys/comments/blanks until the next table header
            out.append(line)
    return "".join(out)


def _looks_like_table_header(stripped_line: str) -> bool:
    """True for ``[name]`` / ``[[name]]`` headers (optional trailing comment); the closing ``]``
    must be on the same line and no ``=`` may precede it (``key = [x]`` is not a header)."""
    if not stripped_line.startswith("["):
        return False
    head = stripped_line.split("#", 1)[0].rstrip()
    return head.endswith("]") and "=" not in head[: head.index("]") + 1]


# Section headers an old-format managed block (no end marker) may contain.
_LEGACY_MANAGED_HEADERS = ("[mcp_servers", "[plugins", "[permissions]", "[permissions.")


def _strip_existing_managed_block(toml_text: str) -> str:
    """Remove any prior managed section so re-runs idempotently replace it.

    The managed section spans MIGRATION_MARKER through MIGRATION_END_MARKER inclusive; user text
    outside it is preserved verbatim. If the start marker exists without an end marker (older
    writers), swallow lines until a section that is not [mcp_servers.*]/[plugins.*]/[permissions]
    so prior-version configs still migrate.
    """
    out: list[str] = []
    in_managed = False
    for line in toml_text.splitlines(keepends=True):
        bare = line.rstrip("\n")
        if bare == MIGRATION_MARKER:
            in_managed = True
            continue
        if in_managed:
            if bare == MIGRATION_END_MARKER:
                in_managed = False
                continue
            stripped = line.lstrip()
            if not stripped.startswith("[") or stripped.startswith(_LEGACY_MANAGED_HEADERS):
                continue
            in_managed = False  # legacy block: first non-managed section ends it
        out.append(line)
    return "".join(out)


def _query_codex_plugins(
    codex_home: Optional[Path] = None, timeout: float = 8.0) -> tuple[list[dict], Optional[str]]:
    """Spawn ``codex app-server`` briefly and return ``(installed plugins, error)`` from
    ``plugin/list``. Any failure yields ``([], error)`` and is non-fatal (servers and
    permissions still write). Plugins codex reports unavailable (broken install, missing OAuth,
    delisted) are skipped — we write config.toml directly, so they would surface as a codex
    error on the first turn. ``enabled`` is carried forward as reported.
    """
    try:
        from agent.transports.codex_app_server import CodexAppServerClient
    except Exception as exc:
        return [], f"transport unavailable: {exc}"
    try:
        with CodexAppServerClient(codex_home=str(codex_home) if codex_home else None) as client:
            client.initialize(client_name="hermes-migration")
            resp = client.request("plugin/list", {}, timeout=timeout)
    except Exception as exc:
        return [], f"plugin/list query failed: {exc}"
    marketplaces = resp.get("marketplaces") or []
    if not isinstance(marketplaces, list):
        return [], "plugin/list response missing 'marketplaces'"
    out: dict[tuple[str, str], dict] = {}  # (name, marketplace) -> entry; first wins
    for marketplace in marketplaces:
        if not isinstance(marketplace, dict):
            continue
        market_name = str(marketplace.get("name") or "openai-curated")
        plugins = marketplace.get("plugins") or []
        for plugin in plugins if isinstance(plugins, list) else ():
            if not isinstance(plugin, dict) or not plugin.get("installed", False):
                continue
            # Skip plugins codex itself reports as unavailable (broken install, missing OAuth, removed from
            # marketplace, etc.). Cf. openclaw/openclaw#80815 — OpenClaw learned to gate migration on app
            # readiness to avoid writing config that would fail at activation time. Our migration writes to
            # codex's config.toml directly, so a broken plugin would surface as a codex error on first use.
            # Skipping it here keeps the migrated config clean and the user's first codex turn from failing.
            availability = str(plugin.get("availability") or "").upper()
            if availability and availability != "AVAILABLE":
                logger.debug("skipping plugin %s: availability=%s", plugin.get("name"),
                             availability)
                continue
            name = str(plugin.get("name") or "")
            if name:
                out.setdefault((name, market_name), {
                    "name": name, "marketplace": market_name,
                    "enabled": bool(plugin.get("enabled", True))})
    return list(out.values()), None


# pytest tempdir shapes: ``pytest-of-<user>/pytest-<n>/``, macOS ``/private/var/folders/…/T``.
_TEST_TEMPDIR_NEEDLES = ("pytest-of-", "/pytest-", "/tmp/pytest", "/private/var/folders/")


def _looks_like_test_tempdir(path: str) -> bool:
    """Heuristic: does ``path`` look like a pytest/transient tempdir?

    Such dirs are reaped between sessions; a HERMES_HOME pointing there burned into
    ``~/.codex/config.toml`` makes every codex-routed call fail silently once GC'd. Err on
    refusing: a false positive is far less harmful than silently bricking codex's tool surface.
    """
    return bool(path) and any(needle in path.lower() for needle in _TEST_TEMPDIR_NEEDLES)


def _build_hermes_tools_mcp_entry() -> dict:
    """Codex stdio entry launching Hermes' own tool surface as an MCP server (browser/web/
    delegate_task/vision/memory/skills call-backs).

    HERMES_HOME passes through only IF SET, read from os.environ (not get_hermes_home()): when
    unset the codex subprocess must inherit its launcher's runtime HERMES_HOME (systemd, gateway,
    kanban), not a migrate-time default burned into config.toml that pins the wrong profile. The
    pytest-tempdir guard keeps a sibling test's monkeypatched HERMES_HOME out of the user's real
    config. PYTHONPATH passes through so a worktree-launched hermes finds the branch's modules.
    """
    import sys
    env: dict[str, str] = {}
    # HERMES_HOME passes through IF SET so the MCP subprocess sees the same config / auth / sessions DB as
    # the parent CLI. Read from os.environ (not get_hermes_home()) on purpose: when the env var is unset we
    # want codex's subprocess to inherit whatever HERMES_HOME its launcher sets at runtime (systemd unit,
    # gateway, kanban dispatcher, custom shell), rather than burning the migrate-time resolved default into
    # config.toml — that would override the launcher's HERMES_HOME and pin the subprocess to the wrong
    # profile. The pytest-tempdir guard below catches the issue #26250 Bug C scenario: a sibling test's
    # monkeypatch.setenv("HERMES_HOME", tmp_path) would otherwise leak a transient pytest tempdir into the
    # user's real ~/.codex/config.toml and silently brick codex once the tempdir is GC'd.
    hermes_home = os.environ.get("HERMES_HOME") or ""
    if hermes_home and not _looks_like_test_tempdir(hermes_home):
        env["HERMES_HOME"] = hermes_home
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    # Quiet mode + redaction defaults so the MCP wire stays clean.
    env["HERMES_QUIET"] = "1"
    env["HERMES_REDACT_SECRETS"] = env.get("HERMES_REDACT_SECRETS", "true")
    return {
        "command": sys.executable,
        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
        "env": env,
        # Generous timeouts — browser_navigate or delegate_task can take a while.
        "startup_timeout_sec": 30.0,
        "tool_timeout_sec": 600.0}


def _write_atomic(target: Path, text: str) -> None:
    """Write via a same-directory temp file + rename (atomic on POSIX, ReplaceFile on Windows) so a
    crash mid-write never leaves a half-written config.toml that codex would refuse to load."""
    import tempfile
    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix=".config.toml.", dir=str(target.parent))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        tmp_path.replace(target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def migrate(
    hermes_config: dict, *, codex_home: Optional[Path] = None, dry_run: bool = False,
    discover_plugins: bool = True, default_permission_profile: Optional[str] = ":workspace",
    expose_hermes_tools: bool = True) -> MigrationReport:
    """Translate Hermes mcp_servers config + Codex curated plugins into ~/.codex/config.toml.

    ``discover_plugins`` spawns the live codex CLI (set False in tests); discovery is best-effort
    and never blocks the migration. ``default_permission_profile`` (default ":workspace"; built-ins
    carry a leading ":", user profiles do not; None leaves codex's read-only default) avoids an
    approval prompt on every write. ``expose_hermes_tools`` registers Hermes' own tool surface
    (agent/transports/hermes_tools_mcp_server.py, launched on demand by codex over stdio) as an MCP
    server so the codex subprocess can call back for tools it lacks.
    """
    report = MigrationReport(dry_run=dry_run)
    codex_home = codex_home or Path.home() / ".codex"
    target = codex_home / "config.toml"
    report.target_path = target
    hermes_servers = (hermes_config or {}).get("mcp_servers") or {}
    if not isinstance(hermes_servers, dict):
        report.errors.append("mcp_servers in Hermes config is not a dict; cannot migrate.")
        return report
    translated: dict[str, dict] = {}
    for raw_name, cfg in hermes_servers.items():
        name = str(raw_name)
        out, skipped = _translate_one_server(name, cfg or {})
        if out is None:
            reason = ", ".join(skipped) or "no transport configured"
            report.errors.append(f"server {raw_name!r} skipped: {reason}")
            continue
        translated[name] = out
        if skipped:
            report.skipped_keys_per_server[name] = skipped
        report.migrated.append(name)
    plugins: list[dict] = []
    plugin_query_succeeded = False
    if discover_plugins and not dry_run:
        plugins, plugin_err = _query_codex_plugins(codex_home=codex_home)
        if plugin_err:
            report.plugin_query_error = plugin_err
        # An authoritative plugin/list (even an empty one) means we own [plugins.*] for this
        # re-render and may strip pre-existing tables outside the managed block.
        plugin_query_succeeded = not plugin_err
        report.migrated_plugins += [f"{p['name']}@{p['marketplace']}" for p in plugins]
    if default_permission_profile:
        report.wrote_permissions_default = default_permission_profile
    if expose_hermes_tools:
        translated["hermes-tools"] = _build_hermes_tools_mcp_entry()
        if "hermes-tools" not in report.migrated:
            report.migrated.append("hermes-tools")
    managed_block = render_codex_toml_section(
        translated, plugins=plugins, default_permission_profile=default_permission_profile)
    new_text = managed_block
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except Exception as exc:
            report.errors.append(f"could not read {target}: {exc}")
            return report
        without_managed = _strip_existing_managed_block(existing)
        if plugin_query_succeeded:
            without_managed = _strip_unmanaged_plugin_tables(without_managed)
        new_text = _insert_managed_block_at_top_level(without_managed, managed_block)
    if dry_run:
        return report
    try:
        codex_home.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, new_text)
        report.written = True
    except Exception as exc:
        report.errors.append(f"could not write {target}: {exc}")
    return report
