"""MCP catalog — curated, Nous-approved MCP servers shipped with the repo.

Entries are added only by merging a PR into hermes-agent; presence in ``optional-mcps/`` = Nous
approval (no community tier, no other trust signals). Manifests pin transport details.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home, get_optional_mcps_dir
from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config, save_config, get_env_value, save_env_value
from hermes_cli.cli_output import prompt as _prompt_input

_MANIFEST_VERSION = 1

# Substituted at install time inside `transport.command` / `transport.args`.
_INSTALL_DIR_VAR = "${INSTALL_DIR}"


@dataclass
class EnvVarSpec:
    name: str
    prompt: str
    required: bool = True
    secret: bool = True
    default: str = ""


@dataclass
class AuthSpec:
    type: str  # "api_key" | "oauth" | "none"
    env: List[EnvVarSpec] = field(default_factory=list)
    provider: Optional[str] = None  # OAuth-specific (third-party provider like Google)
    scopes: List[str] = field(default_factory=list)
    env_var: Optional[str] = None


@dataclass
class TransportSpec:
    type: str  # "stdio" | "http"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    version: Optional[str] = None  # informational, pinned
    # Static env for the stdio subprocess (telemetry opt-outs, mode flags). NOT for secrets — those
    # go through auth.env so they are prompted for and land in ~/.hermes/.env.
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class InstallSpec:
    """Optional bootstrap step (git clone + dep install)."""
    type: str  # "git"
    url: str
    ref: str  # commit/tag/branch — pinned, never floats
    bootstrap: List[str] = field(default_factory=list)


@dataclass
class ToolsSpec:
    """Manifest-side tool-selection hints (see _apply_tool_selection()).

    ``default_enabled``: pre-checked in the install checklist / applied directly on probe failure;
    None => all pre-checked (no filter written on failure). ``default_excluded``: exclude-mode
    counterpart written to ``tools.exclude`` — everything NOT matching stays enabled, including tools
    the server adds later (for huge OpenAPI-derived surfaces). Mutually exclusive.
    """

    default_enabled: Optional[List[str]] = None
    default_excluded: Optional[List[str]] = None


@dataclass
class SuggestSpec:
    """Composer-suggestion metadata (desktop "brand pill" triggers).

    GitHub is intentionally NOT in the catalog and must not be suggested here: its hosted MCP needs a
    per-host OAuth app (generic DCR 404s) and the bundled github/* skills are far more capable.
    """

    keywords: List[str] = field(default_factory=list)  # lowercase whole-word/phrase triggers
    hosts: List[str] = field(default_factory=list)  # hostname suffixes ("atlassian.net")


@dataclass
class SetupStep:
    """One thing only the user can do before the connector can work (create a Google Cloud project, generate a
    token in someone's admin console) — work on another site that no automation on our side removes; the most a
    client can do is name it precisely and link to the page. ``text`` carries inline markdown links because a real
    step routinely points at more than one page. Distinct from ``post_install`` (notes for once it works)."""

    text: str


@dataclass
class CatalogEntry:
    name: str
    description: str
    source: str
    transport: TransportSpec
    auth: AuthSpec
    tools: ToolsSpec = field(default_factory=ToolsSpec)
    install: Optional[InstallSpec] = None
    post_install: str = ""
    suggest: Optional[SuggestSpec] = None
    manifest_path: Path = field(default_factory=Path)
    # The product's own site ("https://n8n.io"): an identity hint only — UIs read its favicon so a connector
    # without a bundled brand glyph still shows its own mark. `source` can't stand in (it points at the bridge's
    # repo, so a third-party bridge to n8n would wear GitHub's logo).
    homepage: str = ""
    setup: List[SetupStep] = field(default_factory=list)


class CatalogError(Exception):
    """Manifest parse/validation failure or install error."""


def _catalog_root() -> Path:
    """The optional-mcps/ dir: env-var override / packaged location, else the source checkout's."""
    return get_optional_mcps_dir(Path(__file__).parent.parent / "optional-mcps")


def _parse_env_spec(raw: Any) -> EnvVarSpec:
    if not isinstance(raw, dict):
        raise CatalogError(f"env entry must be a mapping, got {type(raw).__name__}")
    name = raw.get("name") or ""
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise CatalogError(f"invalid env var name: {name!r}")
    return EnvVarSpec(
        name=name, prompt=raw.get("prompt") or name, required=bool(raw.get("required", True)),
        secret=bool(raw.get("secret", True)), default=str(raw.get("default") or ""))


def _require_mapping(path: Path, key: str, raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise CatalogError(f"{path}: '{key}' must be a mapping")
    return raw


def _require_list(path: Path, field: str, raw: Any) -> list:
    if not isinstance(raw, list):
        raise CatalogError(f"{path}: {field} must be a list")
    return raw


def _require_str_list(path: Path, field: str, raw: Any, *, non_empty: bool = False) -> None:
    ok = isinstance(raw, list) and all(isinstance(t, str) and (t.strip() if non_empty else True) for t in raw)
    if not ok:
        kind = "non-empty strings" if non_empty else "strings"
        raise CatalogError(f"{path}: {field} must be a list of {kind}")


def _parse_transport(path: Path, raw: Any) -> TransportSpec:
    transport_raw = _require_mapping(path, "transport", raw or {})
    t_type = transport_raw.get("type")
    if t_type not in ("stdio", "http"):
        raise CatalogError(f"{path}: transport.type must be 'stdio' or 'http'")
    args = _require_list(path, "transport.args", transport_raw.get("args") or [])
    env_raw = transport_raw.get("env") or {}
    if not isinstance(env_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
    ):
        raise CatalogError(f"{path}: transport.env must be a mapping of string to string")
    transport = TransportSpec(
        type=t_type, command=transport_raw.get("command"), args=[str(a) for a in args],
        url=transport_raw.get("url"), version=transport_raw.get("version"), env=dict(env_raw))
    if t_type == "stdio" and not transport.command:
        raise CatalogError(f"{path}: stdio transport requires 'command'")
    if t_type == "http" and not transport.url:
        raise CatalogError(f"{path}: http transport requires 'url'")
    return transport


def _parse_auth(path: Path, raw: Any, name: str, http: bool) -> AuthSpec:
    auth_raw = _require_mapping(path, "auth", raw or {"type": "none"})
    a_type = auth_raw.get("type") or "none"
    if a_type not in ("api_key", "oauth", "none"):
        raise CatalogError(f"{path}: auth.type must be 'api_key'|'oauth'|'none'")
    env_list = [_parse_env_spec(e) for e in _require_list(path, "auth.env", auth_raw.get("env") or [])]
    if http and a_type == "api_key":
        # _build_server_config emits an Authorization header referencing ${MCP_<NAME>_API_KEY}, but
        # install_entry only persists the env vars DECLARED in auth.env. Enforce the naming contract
        # here, or a manifest declaring e.g. N8N_API_KEY would send a literal-placeholder header (401).
        from hermes_cli.mcp_config import _env_key_for_server

        _required_key = _env_key_for_server(name)
        if all(spec.name != _required_key for spec in env_list):
            raise CatalogError(
                f"{path}: http + api_key auth requires auth.env to declare "
                f"'{_required_key}' (the key the Authorization header references)"
            )
    return AuthSpec(
        type=a_type, env=env_list, provider=auth_raw.get("provider"),
        scopes=list(auth_raw.get("scopes") or []), env_var=auth_raw.get("env_var"))


def _parse_tools(path: Path, raw: Any) -> ToolsSpec:
    tools_raw = _require_mapping(path, "tools", raw or {})
    default_enabled = tools_raw.get("default_enabled")
    default_excluded = tools_raw.get("default_excluded")
    for key, val in (("default_enabled", default_enabled), ("default_excluded", default_excluded)):
        if val is not None:
            _require_str_list(path, f"tools.{key}", val)
    if default_enabled is not None and default_excluded is not None:
        raise CatalogError(f"{path}: tools.default_enabled and tools.default_excluded are mutually exclusive")
    return ToolsSpec(default_enabled=default_enabled, default_excluded=default_excluded)


def _parse_suggest(path: Path, suggest_raw: Any) -> Optional[SuggestSpec]:
    if suggest_raw is None:
        return None
    _require_mapping(path, "suggest", suggest_raw)
    kw_raw = suggest_raw.get("keywords") or []
    hosts_raw = suggest_raw.get("hosts") or []
    _require_str_list(path, "suggest.keywords", kw_raw, non_empty=True)
    _require_str_list(path, "suggest.hosts", hosts_raw, non_empty=True)
    if not kw_raw and not hosts_raw:
        raise CatalogError(f"{path}: 'suggest' requires at least one keyword or host")
    # Matching is case-insensitive whole-word / host-suffix: store lowercase so UIs needn't re-normalize.
    return SuggestSpec(
        keywords=[k.strip().lower() for k in kw_raw],
        hosts=[h.strip().lower().lstrip(".") for h in hosts_raw])


def _parse_install(path: Path, install_raw: Any) -> Optional[InstallSpec]:
    if install_raw is None:
        return None
    _require_mapping(path, "install", install_raw)
    i_type = install_raw.get("type")
    if i_type != "git":
        raise CatalogError(f"{path}: install.type must be 'git' (got {i_type!r})")
    url, ref = install_raw.get("url") or "", install_raw.get("ref") or ""
    if not url or not ref:
        raise CatalogError(f"{path}: install.url and install.ref are required")
    bootstrap = _require_list(path, "install.bootstrap", install_raw.get("bootstrap") or [])
    return InstallSpec(type=i_type, url=url, ref=ref, bootstrap=[str(c) for c in bootstrap])


def _parse_manifest(path: Path) -> CatalogEntry:
    """Read and validate a manifest.yaml. Raise CatalogError on any problem."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        raise CatalogError(f"failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: manifest must be a mapping")

    mv = data.get("manifest_version")
    if mv != _MANIFEST_VERSION:
        raise CatalogError(
            f"{path}: manifest_version {mv!r} unsupported "
            f"(this Hermes understands version {_MANIFEST_VERSION})"
        )
    name = data.get("name") or ""
    if not name or not re.match(r"^[A-Za-z0-9_-]+$", name):
        raise CatalogError(f"{path}: invalid or missing 'name'")
    description = str(data.get("description") or "").strip()
    if not description:
        raise CatalogError(f"{path}: 'description' required")
    homepage = str(data.get("homepage") or "").strip()
    if homepage and not homepage.startswith(("http://", "https://")):
        raise CatalogError(f"{path}: 'homepage' must be an http(s) URL")
    setup_raw = data.get("setup") or []
    if not isinstance(setup_raw, list):
        raise CatalogError(f"{path}: 'setup' must be a list of steps")
    setup = [SetupStep(text=text) for text in (str(s).strip() for s in setup_raw) if text]

    # Validation order (transport, auth, tools, suggest, install) determines which error surfaces.
    transport = _parse_transport(path, data.get("transport"))
    auth = _parse_auth(path, data.get("auth"), name, transport.type == "http")
    tools = _parse_tools(path, data.get("tools"))
    suggest = _parse_suggest(path, data.get("suggest"))
    install = _parse_install(path, data.get("install"))
    return CatalogEntry(
        name=name, description=description, source=str(data.get("source") or "").strip(),
        transport=transport, auth=auth, tools=tools, install=install,
        post_install=str(data.get("post_install") or ""), suggest=suggest, manifest_path=path,
        homepage=homepage, setup=setup,
    )


# Populated by list_catalog(); inspected by the picker / catalog UIs so the user gets actionable
# feedback instead of a silently-shorter list.
_CATALOG_DIAGNOSTICS: List[tuple] = []


def list_catalog() -> List[CatalogEntry]:
    """Return all valid catalog entries, sorted by name.

    Invalid manifests are skipped silently (CI catches them); future ``manifest_version`` ones are
    skipped too but surfaced via :func:`catalog_diagnostics` so UIs can say "update Hermes".
    """
    root = _catalog_root()
    if not root.exists():
        return []
    entries: List[CatalogEntry] = []
    _CATALOG_DIAGNOSTICS.clear()
    for child in sorted(root.iterdir()):
        manifest = child / "manifest.yaml"
        if not manifest.is_file():
            continue
        try:
            entries.append(_parse_manifest(manifest))
        except CatalogError as exc:
            msg = str(exc)
            future = "manifest_version" in msg and "unsupported" in msg
            _CATALOG_DIAGNOSTICS.append((child.name, "future_manifest" if future else "invalid", msg))
    return entries


def catalog_diagnostics() -> List[tuple]:
    """``(entry_name, kind, message)`` tuples from the most recent :func:`list_catalog` call;
    ``kind`` is ``future_manifest`` (newer than this Hermes) or ``invalid`` (malformed)."""
    return list(_CATALOG_DIAGNOSTICS)


def get_entry(name: str) -> Optional[CatalogEntry]:
    """Look up a single entry by name. ``official/<name>`` prefix accepted."""
    if name.startswith("official/"):
        name = name[len("official/"):]
    return next((e for e in list_catalog() if e.name == name), None)


def installed_servers() -> Dict[str, dict]:
    """Return current ``mcp_servers`` block from config.yaml."""
    from hermes_cli.mcp_config import _get_mcp_servers

    return _get_mcp_servers()


def is_installed(name: str) -> bool:
    return name in installed_servers()


def server_enabled(cfg: dict) -> bool:
    """Interpret a server block's ``enabled`` flag (bools, and yes/true/1 strings)."""
    enabled = cfg.get("enabled", True)
    if isinstance(enabled, str):
        return enabled.lower() in {"true", "1", "yes"}
    return bool(enabled)


def is_enabled(name: str) -> bool:
    cfg = installed_servers().get(name)
    return bool(cfg) and server_enabled(cfg)


def remove_server(name: str) -> bool:
    """Drop ``mcp_servers.<name>`` from config.yaml (pruning an empty block). True if it existed."""
    from hermes_cli.mcp_config import _remove_mcp_server

    return _remove_mcp_server(name)


def _say(msg: str, colour: str = Colors.GREEN) -> None:
    print(color(msg, colour))


def _install_root() -> Path:
    """Where git-bootstrapped MCPs are cloned. Per-user, profile-aware."""
    root = get_hermes_home() / "mcp-installs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_bootstrap(cwd: Path, commands: List[str]) -> None:
    """Execute bootstrap commands in *cwd*. Raise CatalogError on first failure."""
    for cmd in commands:
        _say(f"  $ {cmd}", Colors.DIM)
        rc = subprocess.run(cmd, cwd=str(cwd), shell=True).returncode
        if rc != 0:
            raise CatalogError(f"bootstrap step failed (exit {rc}): {cmd}")


def _do_git_install(entry: CatalogEntry) -> Path:
    """Clone the entry's repo into ``~/.hermes/mcp-installs/<name>`` and run bootstrap. Returns the dir."""
    assert entry.install is not None and entry.install.type == "git"
    install = entry.install
    dest = _install_root() / entry.name

    git = shutil.which("git")
    if not git:
        raise CatalogError("git is required to install this MCP but was not found on PATH")
    if dest.exists():
        # Fresh checkout each install — the manifest ref is the source of truth.
        _say(f"  Removing existing install at {dest}", Colors.DIM)
        shutil.rmtree(dest)
    _say(f"  Cloning {install.url} ({install.ref}) → {dest}", Colors.CYAN)

    # `git clone --branch` only accepts branches/tags, NOT commit SHAs; detect SHA-shaped refs
    # upfront so the fast path doesn't always fail noisily before the full-clone fallback.
    is_sha_ref = bool(re.fullmatch(r"[0-9a-f]{7,40}", install.ref))
    # Never hang on a credential prompt: installs run from CLI/dashboard flows nobody can answer.
    _git_env = noninteractive_git_env()

    def _git(*args: str) -> int:
        return subprocess.run([git, *args], stdin=subprocess.DEVNULL, env=_git_env).returncode

    if not is_sha_ref and _git("clone", "--depth", "1", "--branch", install.ref, install.url, str(dest)) != 0:
        # Branch/tag form failed (e.g. ref deleted upstream): fall through to full-clone path.
        if dest.exists():
            shutil.rmtree(dest)
        is_sha_ref = True
    if is_sha_ref:
        if _git("clone", install.url, str(dest)) != 0:
            raise CatalogError(f"git clone failed for {install.url}")
        if _git("-C", str(dest), "checkout", install.ref) != 0:
            raise CatalogError(f"git checkout {install.ref} failed")

    if install.bootstrap:
        _run_bootstrap(dest, install.bootstrap)
    return dest


def _expand_install_dir(value: str, install_dir: Optional[Path]) -> str:
    if _INSTALL_DIR_VAR not in value:
        return value
    if install_dir is None:
        raise CatalogError(f"manifest references {_INSTALL_DIR_VAR} but no install block exists")
    return value.replace(_INSTALL_DIR_VAR, str(install_dir))


def _prompt_env_vars(specs: List[EnvVarSpec]) -> Dict[str, str]:
    """Prompt for each env spec; secrets and non-secrets alike go to ~/.hermes/.env."""
    collected: Dict[str, str] = {}
    for spec in specs:
        existing = get_env_value(spec.name)
        if existing:
            _say(f"  ✓ {spec.name} already set in .env")
            collected[spec.name] = existing
            continue
        value = _prompt_input(spec.prompt, default=spec.default or None, password=spec.secret)
        if value:
            save_env_value(spec.name, value)
            collected[spec.name] = value
        elif spec.required:
            raise CatalogError(f"{spec.name} is required but no value was provided")
    return collected


def _build_server_config(entry: CatalogEntry, install_dir: Optional[Path]) -> dict:
    """Translate a manifest into the ``mcp_servers.<name>`` block format used by hermes_cli/mcp_config.py."""
    cfg: dict = {}
    t = entry.transport
    if t.type == "stdio":
        cfg["command"] = _expand_install_dir(t.command or "", install_dir)
        if t.args:
            cfg["args"] = [_expand_install_dir(a, install_dir) for a in t.args]
        env = dict(t.env) if t.env else {}
        if entry.auth.type == "api_key":
            # A stdio child's environment is allowlist-filtered (_build_safe_env), so a credential in .env reaches
            # it only if the stanza names it. Referenced, not inlined: ${NAME} resolves at connect time.
            for var in entry.auth.env:
                env.setdefault(var.name, f"${{{var.name}}}")
        if env:
            cfg["env"] = env
    elif t.type == "http":
        cfg["url"] = t.url
        if entry.auth.type == "oauth":
            cfg["auth"] = "oauth"
        elif entry.auth.type == "api_key":
            from hermes_cli.mcp_config import _bearer_auth_headers

            cfg["headers"] = _bearer_auth_headers(entry.name)
    return cfg


def _read_prior_tool_list(name: str, key: str) -> Optional[List[str]]:
    """The user's prior ``tools.<key>`` (``include``/``exclude``) for *name*, if well-formed.

    Read BEFORE a reinstall overwrites the entry: a prior include list pre-checks the checklist and a
    user-edited exclude list survives instead of being clobbered by the manifest's ``default_excluded``.
    """
    tools_cfg = (installed_servers().get(name) or {}).get("tools") or {}
    if not isinstance(tools_cfg, dict):
        return None
    value = tools_cfg.get(key)
    ok = isinstance(value, list) and all(isinstance(t, str) for t in value)
    return list(value) if ok else None


def _probe_tools(name: str) -> Optional[List[tuple]]:
    """Connect to a freshly-configured MCP and list its tools.

    ``(tool_name, description)`` tuples on success, ``None`` on any failure (unreachable, OAuth not
    yet completed, ...). Failures are swallowed here; :func:`_apply_tool_selection` handles them.
    """
    server_cfg = installed_servers().get(name)
    if not server_cfg:
        return None
    try:
        from hermes_cli.mcp_config import _probe_single_server  # lazy: keep this module cheap

        tools = _probe_single_server(name, server_cfg)
        return list(tools) if tools is not None else []
    except Exception as exc:
        _say(f"  Probe failed: {exc}", Colors.YELLOW)
        return None


def _write_tools_filter(name: str, mode: str, values: Optional[List[str]]) -> None:
    """Persist ``mcp_servers.<name>.tools.<mode>`` (``include``/``exclude``), clearing the other
    mode; ``values=None`` drops the whole tools block (no filter)."""
    cfg = load_config()
    servers = cfg.setdefault("mcp_servers", {})
    server_entry = servers.get(name) or {}
    if values is None:
        server_entry.pop("tools", None)
    else:
        tools_block = server_entry.get("tools") or {}
        if not isinstance(tools_block, dict):
            tools_block = {}
        tools_block[mode] = list(values)
        tools_block.pop("exclude" if mode == "include" else "include", None)
        server_entry["tools"] = tools_block
    servers[name] = server_entry
    cfg["mcp_servers"] = servers
    save_config(cfg)


def _apply_tool_selection(
    entry: CatalogEntry,
    *,
    prior_selection: Optional[List[str]],
    prior_exclude: Optional[List[str]] = None) -> None:
    """Probe the server and let the user pick which tools to enable.

    Probe-success: curses checklist; pre-check priority *prior_selection* (reinstall) > manifest
    ``tools.default_enabled`` > all; all-on clears any filter. Probe-fail: keep the prior filter,
    else apply ``default_enabled``, else no filter; point the user at ``hermes mcp configure``.
    """
    print()
    name = entry.name
    configure_hint = f"`hermes mcp configure {name}`"

    # Exclude-mode manifests never probe: the curated exclude list (names or globs) is written as-is
    # and everything else stays enabled, including tools the server adds later. A prior include
    # selection falls through to the checklist; a prior user-edited exclude list is kept verbatim.
    if entry.tools.default_excluded and prior_selection is None:
        edit_hint = f"Edit mcp_servers.{name}.tools.exclude in config.yaml or run {configure_hint} to change."
        if prior_exclude is not None:
            _write_tools_filter(name, "exclude", prior_exclude)
            _say(f"  Kept your existing exclude list ({len(prior_exclude)} entries). {edit_hint}")
            return
        _write_tools_filter(name, "exclude", entry.tools.default_excluded)
        _say(
            f"  Applied manifest exclude list ({len(entry.tools.default_excluded)} entries); "
            f"everything else stays enabled. {edit_hint}"
        )
        return

    _say(f"  Probing '{name}' for available tools...", Colors.CYAN)
    probed = _probe_tools(name)

    # Probe failure. Order matters: a reinstall must keep the user's previous filter intact (common
    # for OAuth entries — the entry rewrite precedes first auth, so the server is unreachable here).
    if probed is None:
        manifest_default = entry.tools.default_enabled
        refine_hint = f"Run {configure_hint} after the server is reachable to refine."
        if prior_selection is not None:
            mode, values = "include", prior_selection
            msg = f"Kept your previous tool selection ({len(prior_selection)} tools). {refine_hint}"
        elif prior_exclude is not None:
            mode, values = "exclude", prior_exclude
            msg = f"Kept your existing exclude list ({len(prior_exclude)} entries)."
        elif manifest_default:
            mode, values = "include", manifest_default
            msg = f"Applied manifest default ({len(manifest_default)} tools). {refine_hint}"
        else:
            mode, values = "include", None
            msg = (
                "installed with no tool filter (all tools enabled when "
                f"reachable). Run {configure_hint} after first connect to prune."
            )
        _write_tools_filter(name, mode, values)
        sep = ";" if values is None else "."
        _say(f"  Couldn't probe server{sep} {msg}", Colors.YELLOW)
        return

    if not probed:
        _write_tools_filter(name, "include", None)
        _say("  Server reported no tools.", Colors.YELLOW)
        return

    tool_names = [t[0] for t in probed]

    # Non-TTY: skip the checklist; same priority as the interactive pre-check.
    import sys as _sys
    if not _sys.stdin.isatty():
        preferred = prior_selection if prior_selection is not None else (entry.tools.default_enabled or None)
        _write_tools_filter(
            name, "include", None if preferred is None else [n for n in preferred if n in tool_names]
        )
        return

    pre_set = {n for n in (prior_selection or entry.tools.default_enabled or tool_names) if n in tool_names}
    pre_indices = {i for i, n in enumerate(tool_names) if n in pre_set}
    _say(f"  Found {len(probed)} tool(s). Pre-checked: {len(pre_indices)}.")

    from hermes_cli.curses_ui import curses_checklist

    labels = [f"{n}  —  {(d[:60] + '...') if len(d) > 60 else d}" for n, d in probed]
    chosen_indices = curses_checklist(
        f"Select tools for '{name}' (SPACE toggle, ENTER confirm)", labels, pre_indices)
    if not chosen_indices:
        # Everything unchecked: write an empty include so the server is installed but contributes
        # nothing until reconfigured.
        _write_tools_filter(name, "include", [])
        _say(f"  No tools selected. Run {configure_hint} to change.", Colors.YELLOW)
        return
    if len(chosen_indices) == len(probed):
        # Clear the filter: tools the server adds later are auto-enabled too. To pin the current set,
        # re-run `hermes mcp configure <name>` and unselect a tool (switches to include-mode).
        _write_tools_filter(name, "include", None)
        _say(
            f"  ✓ All {len(probed)} tools enabled (no filter — new tools "
            "the server adds later will be auto-enabled)."
        )
        return
    chosen_names = [tool_names[i] for i in sorted(chosen_indices)]
    _write_tools_filter(name, "include", chosen_names)
    _say(f"  ✓ {len(chosen_names)}/{len(probed)} tools enabled.")


def install_entry(entry: CatalogEntry, *, enable: bool = True) -> None:
    """Install a catalog entry end-to-end.

    Order: git clone + bootstrap (if any); API-key prompt to .env or the ``auth: oauth`` marker;
    write ``mcp_servers.<name>``; probe + tool checklist (falling back per
    :func:`_apply_tool_selection`); print post_install notes.
    """
    print()
    _say(f"  Installing MCP '{entry.name}'", Colors.CYAN + Colors.BOLD)
    if entry.description:
        _say(f"  {entry.description}", Colors.DIM)
    if entry.source:
        _say(f"  Source: {entry.source}", Colors.DIM)
    print()

    install_dir = _do_git_install(entry) if entry.install is not None else None

    if entry.auth.type == "api_key":
        print()
        _say("  Configure credentials:", Colors.CYAN)
        _prompt_env_vars(entry.auth.env)
    elif entry.auth.type == "oauth" and entry.auth.provider:
        # Provider-mediated OAuth relies on the existing `hermes auth <provider>` flow; surface
        # guidance rather than auto-running it to keep install decoupled from provider-auth lifecycle.
        _say(
            f"  This MCP uses {entry.auth.provider} OAuth. Run "
            f"`hermes auth {entry.auth.provider}` if you have not "
            "already authenticated.",
            Colors.YELLOW)
    elif entry.auth.type == "oauth":
        _say(
            "  This MCP uses native OAuth 2.1; tokens will be acquired "
            "on first connection (browser flow).",
            Colors.DIM)

    # Read prior user selection BEFORE overwriting the entry so a reinstall preserves it.
    prior_selection = _read_prior_tool_list(entry.name, "include")
    prior_exclude = _read_prior_tool_list(entry.name, "exclude")

    server_cfg = _build_server_config(entry, install_dir)
    server_cfg["enabled"] = enable

    from hermes_cli.mcp_config import _save_mcp_server

    if not _save_mcp_server(entry.name, server_cfg):
        raise CatalogError(f"catalog entry '{entry.name}' rejected: suspicious command/args configuration")

    _apply_tool_selection(entry, prior_selection=prior_selection, prior_exclude=prior_exclude)

    print()
    _say(
        f"  ✓ Installed '{entry.name}' "
        f"({'enabled' if enable else 'disabled'}). "
        f"Start a new Hermes session to load its tools."
    )
    if entry.post_install:
        print()
        for line in entry.post_install.strip().splitlines():
            _say(f"  {line}", Colors.DIM)
    print()


def uninstall_entry(name: str, *, purge_install_dir: bool = True) -> bool:
    """Remove a catalog-installed MCP from config and (optionally) its clone dir. True if anything removed."""
    removed = remove_server(name)
    if purge_install_dir:
        clone = _install_root() / name
        if clone.exists():
            shutil.rmtree(clone)
            removed = True
    return removed
