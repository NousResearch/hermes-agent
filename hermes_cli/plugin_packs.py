"""Plugin packs — declarative, shareable plugin sets.

Every plugin entry MUST pin an exact 40-character commit SHA in ``ref``; tags and branch names are
rejected with an error naming the entry.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_EXACT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
# Secret-shaped keys are refused in config seeds and stripped from exports; packs declare secrets
# via each plugin's ``requires_env``, which prompts at install time.
_SECRET_KEY_RE = re.compile(r"(?i)(token|secret|passw(or)?d|api[_-]?key|private[_-]?key|credential|auth)")
# plugins.entries.<id> keys a pack may never set (consent state; allow_* gates are also refused).
_RESERVED_ENTRY_KEYS = frozenset({"granted_capabilities", "capabilities_consent"})
_MAX_PACK_BYTES = 1 * 1024 * 1024  # a pack is a small manifest, not a payload
_MAX_PACK_REDIRECTS = 5
_FETCH_TIMEOUT = 15.0


class PackError(Exception):
    """Pack parse/validation/fetch failure (CLI exits non-zero)."""


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
        """Identifier for the install path; None for bare names (resolved via the community index)."""
        if self.repo:
            return f"{self.repo}/{self.subdir}" if self.subdir else self.repo
        return None


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


# ── Parse + validate ────────────────────────────────────────────────────────────────────────

def _entry_label(item: Any, index: int) -> str:
    if isinstance(item, dict):
        label = item.get("name") or item.get("repo")
        if isinstance(label, str) and label.strip():
            return f"'{label.strip()}'"
    return f"#{index + 1}"


def validate_config_seed(plugin_id: str, seed: Any) -> dict[str, Any]:
    """Validate one plugin's config seed mapping and return a copy. Rejects non-dict seeds,
    reserved consent keys, ``allow_*`` trust gates, and secret-shaped keys."""
    if not isinstance(seed, dict):
        raise PackError(
            f"Pack config for plugin '{plugin_id}' must be a mapping of plugins.entries.{plugin_id} keys.")
    for key in seed:
        if not isinstance(key, str) or not key.strip():
            raise PackError(f"Pack config for plugin '{plugin_id}' has an invalid key: {key!r}.")
        if key in _RESERVED_ENTRY_KEYS or key.startswith("allow_"):
            raise PackError(
                f"Pack config for plugin '{plugin_id}' sets reserved key "
                f"'{key}': packs cannot pre-grant capabilities or trust gates. "
                "Capability consent happens interactively at install time.")
        if _SECRET_KEY_RE.search(key):
            raise PackError(
                f"Pack config for plugin '{plugin_id}' sets secret-shaped key "
                f"'{key}': secrets never travel in packs. Declare the secret in "
                "the plugin's requires_env instead — it is prompted at install.")
    return dict(seed)


def parse_pack(text: str, *, source: str = "<pack>") -> PluginPack:
    """Parse and validate a pack YAML document."""
    import yaml
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PackError(f"Pack {source} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PackError(f"Pack {source} must be a YAML mapping.")

    # Accept the nested form (pack: {name: ...}) as sugar.
    meta = raw.get("pack") if isinstance(raw.get("pack"), dict) else raw
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PackError(f"Pack {source} is missing a 'name'.")
    plugins_raw = raw.get("plugins")
    if not isinstance(plugins_raw, list) or not plugins_raw:
        raise PackError(f"Pack {source} must declare a non-empty 'plugins' list.")
    entries = [_parse_pack_entry(item, _entry_label(item, i)) for i, item in enumerate(plugins_raw)]

    config_raw = raw.get("config") or {}
    if not isinstance(config_raw, dict):
        raise PackError(f"Pack {source} 'config' must be a mapping of plugin id → settings.")
    config = {str(pid): validate_config_seed(str(pid), seed) for pid, seed in config_raw.items()}

    skills_raw = raw.get("skills") or []
    if not isinstance(skills_raw, list):
        raise PackError(f"Pack {source} 'skills' must be a list of skill ids.")

    return PluginPack(
        name=name.strip(), description=str(meta.get("description") or ""),
        author=str(meta.get("author") or ""), version=str(meta.get("version") or ""), plugins=entries,
        config=config, skills=[str(s).strip() for s in skills_raw if str(s).strip()])


def _parse_pack_entry(item: Any, label: str) -> PackPluginEntry:
    """Validate one ``plugins:`` item (``name`` and/or ``repo``, exact-SHA ``ref``, ``subdir``)."""
    if not isinstance(item, dict):
        raise PackError(f"Pack plugin entry {label} must be a mapping.")
    entry_name = item.get("name")
    entry_repo = item.get("repo") or item.get("source")
    if isinstance(entry_repo, str):
        entry_repo = entry_repo.removeprefix("github:").strip() or None
    if entry_name is not None and (not isinstance(entry_name, str) or not entry_name.strip()):
        raise PackError(f"Pack plugin entry {label} has an invalid 'name'.")
    entry_name = entry_name.strip() if isinstance(entry_name, str) else None
    if not entry_name and not entry_repo:
        raise PackError(
            f"Pack plugin entry {label} needs either 'name' (community "
            "index) or 'repo' (owner/repo or git URL).")
    ref = item.get("ref") or item.get("version")
    if not isinstance(ref, str) or not _EXACT_SHA_RE.fullmatch(ref.strip()):
        raise PackError(
            f"Pack plugin entry {label} has ref {ref!r}: refs must be "
            "exact 40-character commit SHAs (tags and branch names are "
            "rejected — pin the commit for reproducible installs).")
    subdir = item.get("subdir")
    if subdir is not None and (not isinstance(subdir, str) or not subdir.strip("/")):
        raise PackError(f"Pack plugin entry {label} has an invalid 'subdir'.")
    return PackPluginEntry(
        ref=ref.strip().lower(), name=entry_name, repo=entry_repo,
        subdir=subdir.strip("/") if isinstance(subdir, str) else None)


# ---------------------------------------------------------------------------
# Remote fetch
# ---------------------------------------------------------------------------


def _validate_pack_url(url: str) -> None:
    """Require an HTTPS URL that passes the shared SSRF policy."""
    from tools.url_safety import is_safe_url

    if (urlparse(url).scheme or "").lower() != "https":
        raise PackError("Pack URLs and redirects must use https://.")
    if not is_safe_url(url):
        raise PackError("Pack URL blocked by URL safety policy.")


def _pack_redirect_guard(response: Any) -> None:
    """Validate each redirect target before httpx follows it."""
    from tools.url_safety import redirect_target_from_response

    redirect_url = redirect_target_from_response(response)
    if redirect_url:
        _validate_pack_url(redirect_url)


def load_pack(path_or_url: str) -> PluginPack:
    """Load a pack from a local file path or an ``https://`` URL."""
    if path_or_url.startswith("https://"):
        try:
            from tools.url_safety import create_ssrf_safe_client

            _validate_pack_url(path_or_url)
            with create_ssrf_safe_client(
                timeout=_FETCH_TIMEOUT,
                follow_redirects=True,
                max_redirects=_MAX_PACK_REDIRECTS,
                event_hooks={"response": [_pack_redirect_guard]},
            ) as client:
                resp = client.get(path_or_url)
                resp.raise_for_status()
                text = resp.text
        except PackError:
            raise
        except Exception as exc:
            # Do not echo a caller-supplied URL or transport exception: either
            # may contain credentials from a userinfo/query-bearing URL.
            raise PackError(
                f"Could not fetch pack ({type(exc).__name__})."
            ) from exc
        if len(text.encode("utf-8", errors="ignore")) > _MAX_PACK_BYTES:
            raise PackError("Pack payload exceeds the 1 MiB size limit.")
        return parse_pack(text, source=path_or_url)
    if path_or_url.startswith(("http://", "file://", "ftp://")):
        raise PackError("Pack URLs must use https:// (or pass a local file path).")
    path = Path(path_or_url).expanduser()
    if not path.is_file():
        raise PackError(f"Pack file not found: {path}")
    if path.stat().st_size > _MAX_PACK_BYTES:
        raise PackError("Pack file exceeds the 1 MiB size limit.")
    return parse_pack(path.read_text(encoding="utf-8"), source=str(path))


# ── Resolution (bare index names → owner/repo) + review screen ──────────────────────────────


@dataclass
class ResolvedPackPlugin:
    """A pack entry resolved to an installable identifier."""

    entry: PackPluginEntry
    identifier: Optional[str]        # None when index resolution failed
    index_capabilities: List[str] = field(default_factory=list)
    resolve_error: Optional[str] = None


def resolve_pack_plugins(pack: PluginPack) -> List[ResolvedPackPlugin]:
    """Resolve every entry; bare names go through the community index. Failures do not raise —
    they are carried per-entry so the review screen shows them and install reports partial failure."""
    resolved: List[ResolvedPackPlugin] = []
    index_entries = None
    for entry in pack.plugins:
        if entry.install_identifier is not None:
            resolved.append(ResolvedPackPlugin(entry=entry, identifier=entry.install_identifier))
            continue
        try:
            from hermes_cli.plugin_index import load_index, resolve_name
            if index_entries is None:
                index_entries, _src = load_index()
            match, candidates = resolve_name(index_entries, entry.name or "")
        except Exception as exc:  # index load must not crash pack handling
            resolved.append(ResolvedPackPlugin(
                entry=entry, identifier=None, resolve_error=f"community index unavailable: {exc}"))
            continue
        if match is None:
            detail = "ambiguous" if len(candidates) > 1 else "not found"
            resolved.append(ResolvedPackPlugin(
                entry=entry, identifier=None, resolve_error=f"{detail} in the community index"))
            continue
        resolved.append(ResolvedPackPlugin(
            entry=entry, identifier=match.install_identifier,
            index_capabilities=list(match.capabilities)))
    return resolved


def render_pack_review(console, pack: PluginPack, resolved: List[ResolvedPackPlugin]) -> None:
    """Print the full pack review screen (mandatory before install)."""
    from rich.table import Table
    header = f"[bold]{pack.name}[/bold]" + (f" v{pack.version}" if pack.version else "")
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
            "[yellow]Pack lists skills (NOT auto-installed yet):[/yellow] " + ", ".join(pack.skills))
        console.print("[dim]Install them manually, e.g. `hermes skills install <id>`.[/dim]")
    console.print(
        "\n[dim]Installing a pack runs third-party code × "
        f"{len(resolved)} plugins. Each plugin's declared capabilities still "
        "require individual consent after install — a pack never bulk-grants.[/dim]")


# ── Install fan-out ─────────────────────────────────────────────────────────────────────────

@dataclass
class PackInstallResult:
    """Outcome of one plugin install within a pack."""

    display: str
    ok: bool
    installed_name: Optional[str] = None
    error: Optional[str] = None


def _seed_plugin_config(plugin_id: str, seed: dict[str, Any], console) -> None:
    """Seed plugins.entries.<plugin_id> keys that are not already set (user values always win)."""
    from hermes_cli.config import load_config, save_config
    seed = validate_config_seed(plugin_id, seed)
    config = load_config()
    entry = config.setdefault("plugins", {}).setdefault("entries", {}).setdefault(plugin_id, {})
    if not isinstance(entry, dict):
        console.print(
            f"[yellow]Warning:[/yellow] plugins.entries.{plugin_id} is not a "
            "mapping; skipping pack config seed.")
        return
    wrote = False
    for key, value in seed.items():
        if key in entry:
            console.print(f"[dim]  plugins.entries.{plugin_id}.{key} already set — keeping your value.[/dim]")
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

    Each plugin goes through the exact-ref install path, then the SAME per-plugin capability
    consent flow as a single install (a pack never bulk-grants). Successful installs are enabled
    (the user consented via the review screen) and their config seed applied.
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

    def _fail(display: str, error: str) -> None:
        results.append(PackInstallResult(display=display, ok=False, error=error))
        console.print(f"[red]✗[/red] {display}: {error}")

    for rp in resolved:
        display = rp.entry.display
        if rp.identifier is None:
            _fail(display, rp.resolve_error)
            continue
        console.print(f"[dim]Installing {display} @ {rp.entry.ref[:12]}...[/dim]")
        try:
            target, manifest, installed_name = _install_plugin_core(
                rp.identifier, force=force, ref=rp.entry.ref)
        except PluginOperationError as exc:
            _fail(display, str(exc))
            continue
        except Exception as exc:  # keep the fan-out alive on unexpected errors
            logger.exception("pack install failed for %s", display)
            _fail(display, str(exc))
            continue

        # Secrets are prompted (requires_env), never carried by the pack.
        try:
            _prompt_plugin_env_vars(manifest, console)
        except Exception:
            logger.debug("requires_env prompt failed for %s", installed_name, exc_info=True)

        enabled = _get_enabled_set()
        disabled = _get_disabled_set()
        enabled.add(installed_name)
        disabled.discard(installed_name)
        _save_enabled_set(enabled)
        _save_disabled_set(disabled)

        # Per-plugin capability consent — the SAME flow as a single install (#64228). A pack never
        # bulk-grants capabilities.
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
        results.append(PackInstallResult(display=display, ok=True, installed_name=installed_name))
    return results


# ── Export ──────────────────────────────────────────────────────────────────────────────────

_GITHUB_HTTPS_RE = re.compile(r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s#]+?)(?:\.git)?$")


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
    """Exportable plugins.entries.<id> keys: scalars only, secrets stripped."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return {}
    entry = ((config.get("plugins") or {}).get("entries") or {}).get(plugin_id)
    if not isinstance(entry, dict):
        return {}
    return {
        key: value for key, value in entry.items()
        if isinstance(key, str)
        and key not in _RESERVED_ENTRY_KEYS
        and not key.startswith("allow_")
        and not _SECRET_KEY_RE.search(key)
        and (value is None or isinstance(value, (str, int, float, bool, list, dict)))
    }


def export_pack(*, enabled_only: bool = False, pack_name: str = "my-hermes-pack") -> tuple[str, List[str]]:
    """Build pack YAML from the current install; returns ``(yaml_text, warnings)``. Plugins with
    unknown Git provenance (no install metadata) become warnings + YAML comments, never entries."""
    import yaml
    from hermes_cli.plugins_cmd import _get_enabled_set, _plugins_dir, _read_install_metadata
    metadata = _read_install_metadata()
    enabled = _get_enabled_set()
    installed = sorted(d.name for d in _plugins_dir().iterdir() if d.is_dir() and not d.name.startswith("."))
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
                "install) — listed as a comment, not installable from this pack.")
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
        text = "\n".join(f"# WARNING (not exported): {w}" for w in warnings) + f"\n{text}"
    return text, warnings


# ── CLI commands ────────────────────────────────────────────────────────────────────────────

def _load_and_review(console, source: str):
    """Load *source* (exit 1 on PackError), resolve it, print the review screen."""
    from hermes_cli.plugins_cmd import _fail
    try:
        pack = load_pack(source)
    except PackError as exc:
        _fail(console, f"[red]Error:[/red] {exc}")
    resolved = resolve_pack_plugins(pack)
    render_pack_review(console, pack, resolved)
    return pack, resolved


def cmd_pack_show(source: str) -> None:
    """``hermes plugins pack show <path-or-url>`` — dry-run review."""
    from hermes_cli.plugins_cmd import _console
    console = _console()
    pack, resolved = _load_and_review(console, source)
    unresolved = [rp for rp in resolved if rp.identifier is None]
    if unresolved:
        console.print(
            f"\n[yellow]{len(unresolved)} entr{'y' if len(unresolved) == 1 else 'ies'} "
            "could not be resolved — install would skip them and exit non-zero.[/yellow]")
    console.print("\n[dim]Dry run only. Install with `hermes plugins pack install ...`.[/dim]")


def cmd_pack_install(source: str, *, force: bool = False) -> None:
    """``hermes plugins pack install <path-or-url>``: mandatory review screen -> one pack-level
    consent -> pinned fan-out installs -> per-plugin capability consent. Exit 1 if any failed."""
    from hermes_cli.plugins_cmd import _ask_yes, _console, _fail, _is_tty
    console = _console()
    pack, resolved = _load_and_review(console, source)

    # Mandatory review confirmation — no --yes in v1 (arbitrary third-party code × N).
    if not _is_tty():
        _fail(console, (
            "[red]Error:[/red] Pack install requires an interactive terminal "
            "to review and confirm the pack contents (no --yes in v1)."))
    if not _ask_yes(f"\nInstall {len(resolved)} plugin(s) from pack '{pack.name}'? [y/N] ", console.input):
        _fail(console, "[dim]Aborted — nothing installed.[/dim]")

    results = install_pack_plugins(pack, resolved, console, force=force)
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    console.print(f"\n[bold]Pack '{pack.name}':[/bold] {len(ok)} installed, {len(failed)} failed.")
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
    handler = _PACK_ACTIONS.get(getattr(args, "pack_action", None))
    if handler is None:
        from hermes_cli.plugins_cmd import _console, _fail
        _fail(_console(), "[red]Error:[/red] Usage: hermes plugins pack {install|export|show}")
    handler(args)


_PACK_ACTIONS = {
    "install": lambda args: cmd_pack_install(args.source, force=getattr(args, "force", False)),
    "export": lambda args: cmd_pack_export(
        enabled_only=getattr(args, "enabled_only", False),
        name=getattr(args, "name", None) or "my-hermes-pack"),
    "show": lambda args: cmd_pack_show(args.source),
}
