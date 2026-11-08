"""``hermes plugins export``: package local skills and MCP servers as an Agent Plugins v1 directory.

The export half of Agent Plugins support; :mod:`hermes_cli.agent_plugins` is the import half and
validates every package written here. Only portable facts leave the profile: credential-shaped env
values become ``${NAME}`` pass-throughs, credential headers and OAuth are dropped (v1 has no portable
auth), and absolute paths are refused instead of being shipped as dead references.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from urllib.parse import urlsplit

from hermes_cli.agent_plugins import (
    MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1, _HERMES_EXTENSION, _PLUGIN_NAME_RE, _SKILL_NAME_RE,
    _parse_skill_frontmatter, load_agent_plugin)

_PLUGIN_PLACEHOLDER_RE = re.compile(r"\$\{PLUGIN_(?:ROOT|DATA)\}")
_ENV_TEMPLATE_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
_SECRET_NAME_RE = re.compile(r"(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION)",
                             re.IGNORECASE)


class ExportError(ValueError):
    """The requested package cannot be written."""


@dataclass
class ExportReport:
    root: Path
    skills: list[str] = field(default_factory=list)
    servers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _looks_secret(name: str, value: str) -> bool:
    from agent.redact import redact_sensitive_text
    return bool(_SECRET_NAME_RE.search(name)) or redact_sensitive_text(value, force=True) != value


def _portable_env(server: str, env: Mapping[str, Any], notes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in env.items():
        text = str(value)
        if _ENV_TEMPLATE_RE.match(text) or not _looks_secret(str(key), text):
            out[str(key)] = text
        else:
            out[str(key)] = f"${{{key}}}"
            notes.append(f"mcp:{server}: env {key} exported as ${{{key}}} (the value stays in your profile)")
    return out


def _stdio_entry(server: str, cfg: Mapping[str, Any], notes: list[str]) -> dict[str, Any]:
    command = str(cfg["command"])
    if "/" in command or "\\" in command or any(c.isspace() for c in command):
        raise ExportError(f"mcp:{server}: command {command!r} is a path; a portable command is one bare "
                          "executable name resolved on the user's PATH")
    args = cfg.get("args") or []
    if not isinstance(args, list):
        raise ExportError(f"mcp:{server}: args must be a list")
    args = [str(a) for a in args]
    for arg in args:
        value = arg.split("=", 1)[-1]
        if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ExportError(f"mcp:{server}: arg {arg!r} is a path on this machine; it will not exist "
                              "for the people you share with. Use a relative path or ${PLUGIN_DATA}")
    # The MCP launch-card masker knows which argv positions hold secrets (``--api-key X``, ``-H
    # 'Authorization: …'``, ``KEY=VALUE``) and which only look like it (``--auth-type``, ``--max-tokens``).
    from tools.connectors.mcp_launch import _mask_flag_values
    templated = [_PLUGIN_PLACEHOLDER_RE.sub("", arg) for arg in args]  # ${PLUGIN_ROOT}/${PLUGIN_DATA} are portable
    if any("${" in arg for arg in templated) or _mask_flag_values(command, args) != args:
        raise ExportError(f"mcp:{server}: args carry a credential or ${{VAR}} template; pass it through "
                          "env instead so the package ships only the variable name")
    entry: dict[str, Any] = {"type": "stdio", "command": command, "args": args}
    if isinstance(cfg.get("env"), Mapping) and cfg["env"]:
        entry["env"] = _portable_env(server, cfg["env"], notes)
    if cfg.get("cwd"):
        notes.append(f"mcp:{server}: cwd dropped (a profile path does not exist on other machines)")
    return entry


def _remote_entry(server: str, cfg: Mapping[str, Any], notes: list[str]) -> dict[str, Any]:
    url = str(cfg["url"])
    if "${" in url:
        raise ExportError(f"mcp:{server}: url uses a ${{VAR}} template; portable URLs are literal")
    from tools.connectors.mcp_launch import scrub
    if urlsplit(url).username or scrub(url) != url:
        raise ExportError(f"mcp:{server}: url carries a credential; v1 remote URLs and headers are "
                          "visible package data")
    entry: dict[str, Any] = {"type": "sse" if cfg.get("transport") == "sse" else "streamable-http", "url": url}
    headers = {}
    for key, value in (cfg.get("headers") or {}).items():
        if "${" in str(value) or _looks_secret(str(key), str(value)):
            notes.append(f"mcp:{server}: header {key} dropped (credentials stay client-managed in v1)")
        else:
            headers[str(key)] = str(value)
    if headers:
        entry["headers"] = headers
    if cfg.get("auth"):
        notes.append(f"mcp:{server}: auth: {cfg['auth']} dropped (each client signs in on its own)")
    return entry


def _hermes_server_extension(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ext: dict[str, Any] = {}
    if cfg.get("trust") == "untrusted":
        ext["trust"] = "untrusted"
    if isinstance(cfg.get("timeout"), (int, float)) and not isinstance(cfg.get("timeout"), bool):
        ext["timeout"] = cfg["timeout"]
    for native, portable in (("tool_costs", "costs"), ("spend_limit", "spend_limit"), ("cost_unit", "cost_unit")):
        if native in cfg:
            ext[portable] = cfg[native]
    return ext


def _mcp_entries(names: list[str], notes: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Portable ``mcpServers`` plus the Hermes extension block, from config.yaml AS WRITTEN (never the
    env-interpolated view, which would bake the user's secret values into the package)."""
    from hermes_cli.config import read_raw_config_readonly
    configured = read_raw_config_readonly().get("mcp_servers") or {}
    servers: dict[str, Any] = {}
    extensions: dict[str, Any] = {}
    for name in names:
        cfg = configured.get(name)
        if not isinstance(cfg, Mapping):
            raise ExportError(f"mcp:{name}: no such server in mcp_servers")
        if cfg.get("command"):
            servers[name] = _stdio_entry(name, cfg, notes)
        elif cfg.get("url"):
            servers[name] = _remote_entry(name, cfg, notes)
        else:
            raise ExportError(f"mcp:{name}: neither command nor url is set")
        if (ext := _hermes_server_extension(cfg)) and servers[name]["type"] == "sse":
            notes.append(f"mcp:{name}: Hermes trust/timeout/cost settings not exported for a legacy sse "
                         "server (Hermes' importer skips sse, so they would have no server to attach to)")
        elif ext:
            extensions[name] = ext
    return servers, extensions


def _copy_skill(name: str, skills_root: Path, notes: list[str]) -> str:
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(name)
    if not found:
        raise ExportError(f"skill:{name}: not found in this profile")
    source = Path(found["path"])
    frontmatter = _parse_skill_frontmatter(source / "SKILL.md")
    skill_name = str(frontmatter.get("name") or source.name)
    if not 1 <= len(skill_name) <= 64 or _SKILL_NAME_RE.fullmatch(skill_name) is None:
        raise ExportError(f"skill:{name}: name {skill_name!r} is not a valid Agent Skills name "
                          "(lowercase letters, digits and single hyphens)")
    dest = skills_root / skill_name
    if dest.exists():
        raise ExportError(f"skill:{name}: two requested skills share the name {skill_name!r}")
    root = source.resolve()
    for link in (p for p in source.rglob("*") if p.is_symlink()):
        if not link.resolve().is_relative_to(root):
            raise ExportError(f"skill:{name}: {link.relative_to(source)} links outside the skill folder; "
                              "copy the file in or remove the link")
    # symlinks=False copies link targets, so the package carries files, never links out of it.
    shutil.copytree(source, dest, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    metadata = frontmatter.get("metadata")
    if metadata is not None and not (isinstance(metadata, dict)
                                     and all(isinstance(v, str) for v in metadata.values())):
        _rewrite_frontmatter(dest / "SKILL.md", frontmatter)
        notes.append(f"skill:{skill_name}: nested metadata (e.g. metadata.hermes) dropped from the exported "
                     "copy; Agent Skills v1 metadata maps strings to strings")
    return skill_name


def _rewrite_frontmatter(skill_md: Path, frontmatter: Mapping[str, Any]) -> None:
    from agent.skill_utils import parse_frontmatter
    from hermes_yaml import safe_dump
    _old, body = parse_frontmatter(skill_md.read_text(encoding="utf-8-sig"))
    kept = {k: v for k, v in frontmatter.items() if k != "metadata"}
    if isinstance(frontmatter.get("metadata"), dict):
        strings = {str(k): v for k, v in frontmatter["metadata"].items() if isinstance(v, str)}
        if strings:
            kept["metadata"] = strings
    head = safe_dump(kept, sort_keys=False)
    skill_md.write_text(f"---\n{head}---\n{body}", encoding="utf-8")


def _refuse_output_inside_skill(output: Path, skills: list[str]) -> None:
    """Writing into a selected skill would copy the package into itself until the disk fills."""
    from tools.skill_manager_tool import _find_skill
    target = output.resolve()
    for name in skills:
        found = _find_skill(name)
        if found and target.is_relative_to(Path(found["path"]).resolve()):
            raise ExportError(f"--output {output} is inside the skill {name!r} being exported; "
                              "choose a directory outside it")


def export_agent_plugin(name: str, output: Path, *, skills: list[str], servers: list[str],
                        version: str = "0.1.0", description: str = "") -> ExportReport:
    """Write an Agent Plugins v1 package to *output* and validate it with the importer."""
    if not _PLUGIN_NAME_RE.fullmatch(name) or len(name) > 64:
        raise ExportError(f"plugin name {name!r} must be lowercase letters, digits, '.' and '-'")
    if not skills and not servers:
        raise ExportError("nothing to export: pass at least one --skill or --mcp")
    if output.exists() and any(output.iterdir()):
        raise ExportError(f"{output} already exists and is not empty")
    report = ExportReport(root=output)
    mcp_servers, extensions = _mcp_entries(servers, report.notes)  # fail before writing anything
    _refuse_output_inside_skill(output, skills)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for skill in skills:
            report.skills.append(_copy_skill(skill, output / "skills", report.notes))
        manifest: dict[str, Any] = {"$schema": PLUGIN_SCHEMA_V1, "name": name, "version": version}
        if description:
            manifest["description"] = description
        if extensions:
            manifest["extensions"] = {_HERMES_EXTENSION: {"servers": extensions}}
        (output / "plugin.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if mcp_servers:
            (output / "mcp.json").write_text(json.dumps(
                {"$schema": MCP_SCHEMA_V1, "mcpServers": mcp_servers}, indent=2) + "\n", encoding="utf-8")
        report.servers = list(mcp_servers)
        _validate_export(output, report)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return report


_SSE_SKIP = "portable sse transport is not supported"


def _validate_export(output: Path, report: ExportReport) -> None:
    """Load the package with the real importer (every manifest/extension check it runs at install);
    any skill or server it would skip fails the export. The one allowed skip is a legacy ``sse``
    entry: v1 makes sse optional and Hermes' importer declines it, but other clients load it."""
    data = output / ".export-check-data"
    try:
        package = load_agent_plugin(output, data)
    finally:
        shutil.rmtree(data, ignore_errors=True)
    diagnostics = package.diagnostics
    problems = [f"{d.scope}: {d.message}" for d in diagnostics if d.message != _SSE_SKIP]
    sse = {d.scope.removeprefix("mcp:") for d in diagnostics if d.message == _SSE_SKIP}
    loaded = {skill.name for skill in package.skills} | set(package.mcp_servers)
    missing = (set(report.skills) | (set(report.servers) - sse)) - loaded
    if problems or missing:
        raise ExportError("the exported package does not validate: "
                          + "; ".join(problems or [f"not loadable: {sorted(missing)}"]))
    if sse:
        report.notes.append(f"legacy sse servers {sorted(sse)} are valid v1 but Hermes' importer skips them")


def cmd_export(args) -> None:
    from hermes_cli.plugins_cmd import _console, _fail
    console = _console()
    output = Path(args.output or args.name).expanduser()
    try:
        report = export_agent_plugin(args.name, output, skills=args.skill or [], servers=args.mcp or [],
                                     version=args.package_version, description=args.description or "")
    except (ExportError, OSError, ValueError) as exc:
        _fail(console, f"[red]Export failed:[/red] {exc}")
    console.print(f"[green]✓[/green] Wrote Agent Plugins v1 package [bold]{args.name}[/bold] to {report.root}")
    if report.skills:
        console.print(f"  skills: {', '.join(report.skills)}")
    if report.servers:
        console.print(f"  MCP servers: {', '.join(report.servers)}")
    for note in report.notes:
        console.print(f"  [yellow]note[/yellow] {note}")
