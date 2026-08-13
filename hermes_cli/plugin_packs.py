"""Plugin packs — declarative, shareable plugin sets (#64166).

A pack is a single YAML file (``hermes-pack.yaml``) that pins a set of
plugins (source + exact commit SHA + optional non-secret config seeds).
Installing a pack is nothing new at runtime: it fans out to N ordinary
plugin installs through the existing pinned-ref install path, then seeds
``plugins.entries.<id>`` config keys.

Format (canonical)::

    name: voice-assistant-pack
    description: STT + streaming TTS + approval relay
    author: hyper
    version: 1.0.0
    plugins:
      - name: hermes-media-studio          # bare community-index name…
        ref: e8d59971d2b7901405b39dac7b03bdd616272d0d
      - repo: owner/approval-relay         # …or explicit owner/repo / git URL
        ref: 8f3c2d1a9b4e5f6071829304a5b6c7d8e9f00112
        subdir: plugins/relay              # optional path within the repo
    config:                                # optional plugins.entries seeds
      hermes-media-studio:
        default_model: flux-3
    skills: []                             # declared seam — NOT auto-installed

Supply-chain posture:

* Every plugin entry MUST pin an exact 40-character commit SHA in ``ref``.
  Tags and branch names are rejected with an error naming the entry.
* ``config`` seeds are limited to ``plugins.entries.<id>.*`` keys and may
  never carry secrets (secret-shaped key names are rejected) nor
  capability-grant keys (a pack cannot pre-consent capabilities).
* Capability consent is NEVER bulk-granted: after each plugin installs,
  its declared capabilities ride the exact same per-plugin consent flow
  as a normal ``hermes plugins install`` (#64228).

``skills:`` is parsed and displayed but not installed — wiring skill-hub
ids into the skills installer is a documented follow-up seam.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = __import__("logging").getLogger(__name__)

_EXACT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

# Key names that look like secrets are refused in pack config seeds and
# stripped from exports. Packs declare needed secrets via each plugin's own
# ``requires_env`` manifest field, which prompts at install time.
_SECRET_KEY_RE = re.compile(
    r"(?i)(token|secret|passw(or)?d|api[_-]?key|private[_-]?key|credential|auth)"
)

# plugins.entries.<id> keys a pack may never set: consent/capability state
# and the deprecated allow_* trust gates. A pack cannot pre-grant anything.
_RESERVED_ENTRY_KEYS = frozenset({"granted_capabilities", "capabilities_consent"})

_MAX_PACK_BYTES = 1 * 1024 * 1024  # a pack is a small manifest, not a payload
_FETCH_TIMEOUT = 15.0


class PackError(Exception):
    """Pack parse/validation/fetch failure (CLI exits non-zero)."""


_CYCLIC_CONFIG = object()


def _apply_config_key_policy(
    value: Any,
    *,
    plugin_id: Optional[str] = None,
    _active: Optional[set[int]] = None,
) -> Any:
    """Recursively validate or sanitize plugin config values.

    Passing ``plugin_id`` validates an imported seed and raises on forbidden
    keys. Omitting it produces an export-safe copy with those keys removed.
    """
    if not isinstance(value, (dict, list, tuple)):
        return value

    active = _active if _active is not None else set()
    value_id = id(value)
    if value_id in active:
        if plugin_id is not None:
            raise PackError(
                f"Pack config for plugin '{plugin_id}' contains a cyclic YAML alias. "
                "Pack config must be acyclic."
            )
        return _CYCLIC_CONFIG

    active.add(value_id)
    try:
        if isinstance(value, (list, tuple)):
            out_list = []
            for item in value:
                sanitized = _apply_config_key_policy(
                    item, plugin_id=plugin_id, _active=active
                )
                if sanitized is _CYCLIC_CONFIG:
                    return _CYCLIC_CONFIG
                out_list.append(sanitized)
            return out_list

        out: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key.strip():
                if plugin_id is not None:
                    raise PackError(
                        f"Pack config for plugin '{plugin_id}' has an invalid key: {key!r}."
                    )
                continue
            if key in _RESERVED_ENTRY_KEYS or key.startswith("allow_"):
                if plugin_id is not None:
                    raise PackError(
                        f"Pack config for plugin '{plugin_id}' sets reserved key "
                        f"'{key}': packs cannot pre-grant capabilities or trust gates. "
                        "Capability consent happens interactively at install time."
                    )
                continue
            if _SECRET_KEY_RE.search(key):
                if plugin_id is not None:
                    raise PackError(
                        f"Pack config for plugin '{plugin_id}' sets secret-shaped key "
                        f"'{key}': secrets never travel in packs. Declare the secret in "
                        "the plugin's requires_env instead — it is prompted at install."
                    )
                continue
            sanitized = _apply_config_key_policy(
                child, plugin_id=plugin_id, _active=active
            )
            if sanitized is _CYCLIC_CONFIG:
                continue
            out[key] = sanitized
        return out
    finally:
        active.remove(value_id)


@dataclass
class PackPluginEntry:
    """One pinned plugin in a pack."""

    ref: str                          # exact 40-char commit SHA (lowercased)
    name: Optional[str] = None        # bare community-index name…
    repo: Optional[str] = None        # …or owner/repo shorthand / git URL
    subdir: Optional[str] = None      # path within the repo

    @property
    def display(self) -> str:
        base = self.name or self.repo or "?"
        return f"{base}/{self.subdir}" if (self.repo and self.subdir) else base

    @property
    def install_identifier(self) -> Optional[str]:
        """Identifier for the existing install path (None for bare names,
        which must first resolve through the community index)."""
        if self.repo:
            return f"{self.repo}/{self.subdir}" if self.subdir else self.repo
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.name:
            d["name"] = self.name
        if self.repo:
            d["repo"] = self.repo
        d["ref"] = self.ref
        if self.subdir:
            d["subdir"] = self.subdir
        return d


@dataclass
class PluginPack:
    """A parsed, validated pack manifest."""

    name: str
    description: str = ""
    author: str = ""
    version: str = ""
    plugins: List[PackPluginEntry] = field(default_factory=list)
    # plugin id → {entry-key: seed-value}; validated non-secret, non-reserved.
    config: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Skill-hub ids. Parsed + displayed, NOT installed (documented seam).
    skills: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.author:
            d["author"] = self.author
        if self.version:
            d["version"] = self.version
        d["plugins"] = [p.to_dict() for p in self.plugins]
        if self.config:
            d["config"] = self.config
        if self.skills:
            d["skills"] = list(self.skills)
        return d


# ---------------------------------------------------------------------------
# Parse + validate
# ---------------------------------------------------------------------------

def _entry_label(item: Any, index: int) -> str:
    if isinstance(item, dict):
        label = item.get("name") or item.get("repo")
        if isinstance(label, str) and label.strip():
            return f"'{label.strip()}'"
    return f"#{index + 1}"


def validate_config_seed(plugin_id: str, seed: Any) -> dict[str, Any]:
    """Validate one plugin's config seed mapping.

    Rejects non-dict seeds, reserved consent/capability keys, deprecated
    ``allow_*`` trust gates, and secret-shaped key names. Returns the
    validated dict.
    """
    if not isinstance(seed, dict):
        raise PackError(
            f"Pack config for plugin '{plugin_id}' must be a mapping of "
            f"plugins.entries.{plugin_id} keys."
        )
    return _apply_config_key_policy(seed, plugin_id=plugin_id)


def parse_pack(text: str, *, source: str = "<pack>") -> PluginPack:
    """Parse and validate a pack YAML document.

    Raises :class:`PackError` with an actionable message on any problem —
    including refs that are not exact 40-character commit SHAs.
    """
    import yaml

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PackError(f"Pack {source} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PackError(f"Pack {source} must be a YAML mapping.")

    # Accept the issue-sketch nested form (pack: {name: ...}) as sugar.
    meta = raw.get("pack") if isinstance(raw.get("pack"), dict) else raw

    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PackError(f"Pack {source} is missing a 'name'.")

    plugins_raw = raw.get("plugins")
    if not isinstance(plugins_raw, list) or not plugins_raw:
        raise PackError(f"Pack {source} must declare a non-empty 'plugins' list.")

    entries: List[PackPluginEntry] = []
    for i, item in enumerate(plugins_raw):
        label = _entry_label(item, i)
        if not isinstance(item, dict):
            raise PackError(f"Pack plugin entry {label} must be a mapping.")
        entry_name = item.get("name")
        entry_repo = item.get("repo") or item.get("source")
        if isinstance(entry_repo, str):
            entry_repo = entry_repo.removeprefix("github:").strip() or None
        if entry_name is not None and (
            not isinstance(entry_name, str) or not entry_name.strip()
        ):
            raise PackError(f"Pack plugin entry {label} has an invalid 'name'.")
        entry_name = entry_name.strip() if isinstance(entry_name, str) else None
        if not entry_name and not entry_repo:
            raise PackError(
                f"Pack plugin entry {label} needs either 'name' (community "
                "index) or 'repo' (owner/repo or git URL)."
            )
        ref = item.get("ref") or item.get("version")
        if not isinstance(ref, str) or not _EXACT_SHA_RE.fullmatch(ref.strip()):
            raise PackError(
                f"Pack plugin entry {label} has ref {ref!r}: refs must be "
                "exact 40-character commit SHAs (tags and branch names are "
                "rejected — pin the commit for reproducible installs)."
            )
        subdir = item.get("subdir")
        if subdir is not None and (not isinstance(subdir, str) or not subdir.strip("/")):
            raise PackError(f"Pack plugin entry {label} has an invalid 'subdir'.")
        entries.append(
            PackPluginEntry(
                ref=ref.strip().lower(),
                name=entry_name,
                repo=entry_repo,
                subdir=subdir.strip("/") if isinstance(subdir, str) else None,
            )
        )

    config_raw = raw.get("config") or {}
    if not isinstance(config_raw, dict):
        raise PackError(f"Pack {source} 'config' must be a mapping of plugin id → settings.")
    config: dict[str, dict[str, Any]] = {}
    for plugin_id, seed in config_raw.items():
        config[str(plugin_id)] = validate_config_seed(str(plugin_id), seed)

    skills_raw = raw.get("skills") or []
    if not isinstance(skills_raw, list):
        raise PackError(f"Pack {source} 'skills' must be a list of skill ids.")
    skills = [str(s).strip() for s in skills_raw if str(s).strip()]

    return PluginPack(
        name=name.strip(),
        description=str(meta.get("description") or ""),
        author=str(meta.get("author") or ""),
        version=str(meta.get("version") or ""),
        plugins=entries,
        config=config,
        skills=skills,
    )


def load_pack(path_or_url: str) -> PluginPack:
    """Load a pack from a local file path or an ``https://`` URL."""
    if path_or_url.startswith("https://"):
        try:
            import httpx

            resp = httpx.get(path_or_url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:
            raise PackError(f"Could not fetch pack from {path_or_url}: {exc}") from exc
        if len(text.encode("utf-8", errors="ignore")) > _MAX_PACK_BYTES:
            raise PackError("Pack payload exceeds the 1 MiB size limit.")
        return parse_pack(text, source=path_or_url)
    if path_or_url.startswith(("http://", "file://", "ftp://")):
        raise PackError(
            "Pack URLs must use https:// (or pass a local file path)."
        )
    path = Path(path_or_url).expanduser()
    if not path.is_file():
        raise PackError(f"Pack file not found: {path}")
    if path.stat().st_size > _MAX_PACK_BYTES:
        raise PackError("Pack file exceeds the 1 MiB size limit.")
    return parse_pack(path.read_text(encoding="utf-8"), source=str(path))


# ---------------------------------------------------------------------------
# Resolution (bare index names → owner/repo) + review screen
# ---------------------------------------------------------------------------

@dataclass
class ResolvedPackPlugin:
    """A pack entry resolved to an installable identifier."""

    entry: PackPluginEntry
    identifier: Optional[str]        # None when index resolution failed
    index_capabilities: List[str] = field(default_factory=list)
    resolve_error: Optional[str] = None


def resolve_pack_plugins(pack: PluginPack) -> List[ResolvedPackPlugin]:
    """Resolve every entry; bare names go through the community index.

    Resolution failures do not raise — they are carried per-entry so the
    review screen can show them and install can report partial failure.
    """
    resolved: List[ResolvedPackPlugin] = []
    index_entries = None
    for entry in pack.plugins:
        ident = entry.install_identifier
        if ident is not None:
            resolved.append(ResolvedPackPlugin(entry=entry, identifier=ident))
            continue
        # Bare community-index name.
        try:
            if index_entries is None:
                from hermes_cli.plugin_index import load_index

                index_entries, _src = load_index()
            from hermes_cli.plugin_index import resolve_name

            match, candidates = resolve_name(index_entries, entry.name or "")
        except Exception as exc:  # index load must not crash pack handling
            resolved.append(
                ResolvedPackPlugin(
                    entry=entry, identifier=None,
                    resolve_error=f"community index unavailable: {exc}",
                )
            )
            continue
        if match is None:
            detail = (
                "ambiguous in the community index"
                if len(candidates) > 1
                else "not found in the community index"
            )
            resolved.append(
                ResolvedPackPlugin(entry=entry, identifier=None, resolve_error=detail)
            )
            continue
        resolved.append(
            ResolvedPackPlugin(
                entry=entry,
                identifier=match.install_identifier,
                index_capabilities=list(match.capabilities),
            )
        )
    return resolved


def render_pack_review(console, pack: PluginPack, resolved: List[ResolvedPackPlugin]) -> None:
    """Print the full pack review screen (mandatory before install)."""
    from rich.table import Table

    header = f"[bold]{pack.name}[/bold]"
    if pack.version:
        header += f" v{pack.version}"
    if pack.author:
        header += f" — by {pack.author}"
    console.print(f"\n{header}")
    if pack.description:
        console.print(f"[dim]{pack.description}[/dim]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Plugin")
    table.add_column("Source")
    table.add_column("Pinned ref")
    table.add_column("Capabilities (declared)")
    for rp in resolved:
        caps = ", ".join(rp.index_capabilities) if rp.index_capabilities else "(shown at install)"
        source = rp.identifier or f"[red]unresolved: {rp.resolve_error}[/red]"
        table.add_row(rp.entry.display, source, rp.entry.ref[:12], caps)
    console.print(table)

    if pack.config:
        console.print("[bold]Config seeds[/bold] (plugins.entries.<id>, non-secret):")
        for plugin_id, seed in pack.config.items():
            for key, value in seed.items():
                console.print(f"  {plugin_id}.{key} = {value!r}")
    if pack.skills:
        console.print(
            "[yellow]Pack lists skills (NOT auto-installed yet):[/yellow] "
            + ", ".join(pack.skills)
        )
        console.print(
            "[dim]Install them manually, e.g. `hermes skills install <id>`.[/dim]"
        )
    console.print(
        "\n[dim]Installing a pack runs third-party code × "
        f"{len(resolved)} plugins. Each plugin's declared capabilities still "
        "require individual consent after install — a pack never bulk-grants.[/dim]"
    )


# ---------------------------------------------------------------------------
# Install fan-out
# ---------------------------------------------------------------------------

@dataclass
class PackInstallResult:
    """Outcome of one plugin install within a pack."""

    display: str
    ok: bool
    installed_name: Optional[str] = None
    error: Optional[str] = None


def _seed_plugin_config(plugin_id: str, seed: dict[str, Any], console) -> None:
    """Seed plugins.entries.<plugin_id> keys that are not already set.

    Existing user values always win; the pack only fills blanks.
    """
    from hermes_cli.config import load_config, save_config

    seed = validate_config_seed(plugin_id, seed)
    config = load_config()
    plugins_cfg = config.setdefault("plugins", {})
    entries = plugins_cfg.setdefault("entries", {})
    entry = entries.setdefault(plugin_id, {})
    if not isinstance(entry, dict):
        console.print(
            f"[yellow]Warning:[/yellow] plugins.entries.{plugin_id} is not a "
            "mapping; skipping pack config seed."
        )
        return
    wrote = False
    for key, value in seed.items():
        if key in entry:
            console.print(
                f"[dim]  plugins.entries.{plugin_id}.{key} already set — keeping "
                "your value.[/dim]"
            )
            continue
        entry[key] = value
        wrote = True
    if wrote:
        save_config(config)
        console.print(f"[dim]  Seeded plugins.entries.{plugin_id} from pack.[/dim]")


def install_pack_plugins(
    pack: PluginPack,
    resolved: List[ResolvedPackPlugin],
    console,
    *,
    force: bool = False,
) -> List[PackInstallResult]:
    """Fan a pack out to N ordinary pinned installs; never raises per-plugin.

    Each plugin goes through the existing exact-ref install path, then its
    declared capabilities go through the SAME per-plugin consent flow as a
    single install (:func:`hermes_cli.plugins_cmd._run_capability_consent`).
    Successful installs are enabled (the user consented via the review
    screen) and their pack config seed is applied.
    """
    from hermes_cli.plugins_cmd import (
        PluginOperationError,
        _declared_capabilities_from_manifest,
        _get_disabled_set,
        _get_enabled_set,
        _install_plugin_core,
        _prompt_plugin_env_vars,
        _run_capability_consent,
        _save_disabled_set,
        _save_enabled_set,
    )

    results: List[PackInstallResult] = []
    for rp in resolved:
        display = rp.entry.display
        if rp.identifier is None:
            results.append(
                PackInstallResult(display=display, ok=False, error=rp.resolve_error)
            )
            console.print(f"[red]✗[/red] {display}: {rp.resolve_error}")
            continue
        console.print(f"[dim]Installing {display} @ {rp.entry.ref[:12]}...[/dim]")
        try:
            target, manifest, installed_name = _install_plugin_core(
                rp.identifier, force=force, ref=rp.entry.ref
            )
        except PluginOperationError as exc:
            results.append(PackInstallResult(display=display, ok=False, error=str(exc)))
            console.print(f"[red]✗[/red] {display}: {exc}")
            continue
        except Exception as exc:  # keep the fan-out alive on unexpected errors
            logger.exception("pack install failed for %s", display)
            results.append(PackInstallResult(display=display, ok=False, error=str(exc)))
            console.print(f"[red]✗[/red] {display}: {exc}")
            continue

        # Secrets are prompted (requires_env), never carried by the pack.
        try:
            _prompt_plugin_env_vars(manifest, console)
        except Exception:
            logger.debug("requires_env prompt failed for %s", installed_name, exc_info=True)

        # Enable: the user confirmed the mandatory review screen.
        enabled = _get_enabled_set()
        disabled = _get_disabled_set()
        enabled.add(installed_name)
        disabled.discard(installed_name)
        _save_enabled_set(enabled)
        _save_disabled_set(disabled)

        # Per-plugin capability consent — the SAME flow as a single
        # install (#64228). A pack never bulk-grants capabilities.
        declared = _declared_capabilities_from_manifest(manifest, installed_name)
        if declared:
            _run_capability_consent(console, installed_name, declared, context="install")

        seed = pack.config.get(installed_name)
        if seed is None and rp.entry.name and rp.entry.name != installed_name:
            seed = pack.config.get(rp.entry.name)
        if seed:
            try:
                _seed_plugin_config(installed_name, seed, console)
            except PackError as exc:
                console.print(f"[yellow]Warning:[/yellow] {exc}")

        console.print(f"[green]✓[/green] {display} installed as [bold]{installed_name}[/bold].")
        results.append(
            PackInstallResult(display=display, ok=True, installed_name=installed_name)
        )
    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

_GITHUB_HTTPS_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s#]+?)(?:\.git)?$"
)


def _source_to_repo_subdir(source: str) -> tuple[Optional[str], Optional[str]]:
    """Turn recorded install-metadata source into (repo-or-url, subdir)."""
    if not source:
        return None, None
    base, _, subdir = source.partition("#")
    subdir = subdir.strip("/") or None
    m = _GITHUB_HTTPS_RE.match(base.strip())
    if m:
        return f"{m.group('owner')}/{m.group('repo')}", subdir
    return base.strip() or None, subdir


def _sanitized_entry_config(plugin_id: str) -> dict[str, Any]:
    """Return an export-safe copy of ``plugins.entries.<id>`` config."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return {}
    entries = (config.get("plugins") or {}).get("entries") or {}
    entry = entries.get(plugin_id)
    if not isinstance(entry, dict):
        return {}
    sanitized = _apply_config_key_policy(entry)
    return sanitized if isinstance(sanitized, dict) else {}


def export_pack(*, enabled_only: bool = False, pack_name: str = "my-hermes-pack") -> tuple[str, List[str]]:
    """Build pack YAML from the current install.

    Returns ``(yaml_text, warnings)``. Plugins whose Git provenance is
    unknown (local-only, no install metadata) are listed in the warnings
    and included as comments in the YAML, never as installable entries.
    """
    import yaml

    from hermes_cli.plugins_cmd import (
        _get_enabled_set,
        _plugins_dir,
        _read_install_metadata,
    )

    metadata = _read_install_metadata()
    enabled = _get_enabled_set()
    plugins_dir = _plugins_dir()

    installed = sorted(
        d.name for d in plugins_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if enabled_only:
        installed = [n for n in installed if n in enabled]

    entries: List[dict[str, Any]] = []
    config: dict[str, dict[str, Any]] = {}
    warnings: List[str] = []
    for plugin_id in installed:
        record = metadata.get(plugin_id) or {}
        source = record.get("source")
        revision = record.get("revision")
        repo, subdir = _source_to_repo_subdir(str(source or ""))
        if not repo or not isinstance(revision, str) or not _EXACT_SHA_RE.fullmatch(revision):
            warnings.append(
                f"{plugin_id}: no Git provenance (local-only or pre-metadata "
                "install) — listed as a comment, not installable from this pack."
            )
            continue
        entry: dict[str, Any] = {"repo": repo, "ref": revision.lower()}
        if subdir:
            entry["subdir"] = subdir
        entries.append(entry)
        seed = _sanitized_entry_config(plugin_id)
        if seed:
            config[plugin_id] = seed

    doc: dict[str, Any] = {
        "name": pack_name,
        "description": "Exported by `hermes plugins pack export`.",
        "version": "1.0.0",
        "plugins": entries,
    }
    if config:
        doc["config"] = config

    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    if warnings:
        comment_lines = "\n".join(
            f"# WARNING (not exported): {w}" for w in warnings
        )
        text = f"{comment_lines}\n{text}"
    return text, warnings


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_pack_show(source: str) -> None:
    """``hermes plugins pack show <path-or-url>`` — dry-run review."""
    from rich.console import Console

    console = Console()
    try:
        pack = load_pack(source)
    except PackError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    resolved = resolve_pack_plugins(pack)
    render_pack_review(console, pack, resolved)
    unresolved = [rp for rp in resolved if rp.identifier is None]
    if unresolved:
        console.print(
            f"\n[yellow]{len(unresolved)} entr{'y' if len(unresolved) == 1 else 'ies'} "
            "could not be resolved — install would skip them and exit non-zero.[/yellow]"
        )
    console.print("\n[dim]Dry run only. Install with `hermes plugins pack install ...`.[/dim]")


def cmd_pack_install(source: str, *, force: bool = False) -> None:
    """``hermes plugins pack install <path-or-url>``.

    Mandatory review screen → one summary consent for the pack contents →
    fan-out installs with pinned refs → per-plugin capability consent via
    the standard flow. Partial failures are reported per plugin; exits
    non-zero when any plugin failed.
    """
    from rich.console import Console

    console = Console()
    try:
        pack = load_pack(source)
    except PackError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    resolved = resolve_pack_plugins(pack)
    render_pack_review(console, pack, resolved)

    # Mandatory review confirmation — no --yes in v1 (a pack is arbitrary
    # third-party code × N). Non-interactive sessions cannot consent.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print(
            "[red]Error:[/red] Pack install requires an interactive terminal "
            "to review and confirm the pack contents (no --yes in v1)."
        )
        sys.exit(1)
    try:
        answer = console.input(
            f"\nInstall {len(resolved)} plugin(s) from pack "
            f"'{pack.name}'? [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in {"y", "yes"}:
        console.print("[dim]Aborted — nothing installed.[/dim]")
        sys.exit(1)

    results = install_pack_plugins(pack, resolved, console, force=force)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    console.print(
        f"\n[bold]Pack '{pack.name}':[/bold] {len(ok)} installed, {len(failed)} failed."
    )
    for r in failed:
        console.print(f"  [red]✗[/red] {r.display}: {r.error}")
    if ok:
        console.print("[dim]Restart the gateway for the plugins to take effect:[/dim]")
        console.print("[dim]  hermes gateway restart[/dim]")
    if failed:
        sys.exit(1)


def cmd_pack_export(*, enabled_only: bool = False, name: str = "my-hermes-pack") -> None:
    """``hermes plugins pack export [--enabled-only]`` — pack YAML on stdout."""
    from rich.console import Console

    console = Console(stderr=True)
    try:
        text, warnings = export_pack(enabled_only=enabled_only, pack_name=name)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Could not export pack: {exc}")
        sys.exit(1)
    for w in warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")
    sys.stdout.write(text)


def pack_command(args) -> None:
    """Dispatch ``hermes plugins pack <action>``."""
    action = getattr(args, "pack_action", None)
    if action == "install":
        cmd_pack_install(args.source, force=getattr(args, "force", False))
    elif action == "export":
        cmd_pack_export(
            enabled_only=getattr(args, "enabled_only", False),
            name=getattr(args, "name", None) or "my-hermes-pack",
        )
    elif action == "show":
        cmd_pack_show(args.source)
    else:
        from rich.console import Console

        Console().print(
            "[red]Error:[/red] Usage: hermes plugins pack {install|export|show}"
        )
        sys.exit(1)
