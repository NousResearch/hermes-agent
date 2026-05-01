"""``hermes plugins`` CLI subcommand — install, update, remove, and list plugins.

Plugins are installed from Git repositories into ``~/.hermes/plugins/``.
Supports full URLs and ``owner/repo`` shorthand (resolves to GitHub).

After install, if the plugin ships an ``after-install.md`` file it is
rendered with Rich Markdown.  Otherwise a default confirmation is shown.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

# Minimum manifest version this installer understands.
# Plugins may declare ``manifest_version: 1`` in plugin.yaml;
# future breaking changes to the manifest schema bump this.
_SUPPORTED_MANIFEST_VERSION = 1


def _plugins_dir() -> Path:
    """Return the user plugins directory, creating it if needed."""
    plugins = get_hermes_home() / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins


def _sanitize_plugin_name(name: str, plugins_dir: Path) -> Path:
    """Validate a plugin name and return the safe target path inside *plugins_dir*.

    Raises ``ValueError`` if the name contains path-traversal sequences or would
    resolve outside the plugins directory.
    """
    if not name:
        raise ValueError("Plugin name must not be empty.")

    if name in (".", ".."):
        raise ValueError(
            f"Invalid plugin name '{name}': must not reference the plugins directory itself."
        )

    # Reject obvious traversal characters
    for bad in ("/", "\\", ".."):
        if bad in name:
            raise ValueError(f"Invalid plugin name '{name}': must not contain '{bad}'.")

    target = (plugins_dir / name).resolve()
    plugins_resolved = plugins_dir.resolve()

    if target == plugins_resolved:
        raise ValueError(
            f"Invalid plugin name '{name}': resolves to the plugins directory itself."
        )

    try:
        target.relative_to(plugins_resolved)
    except ValueError:
        raise ValueError(
            f"Invalid plugin name '{name}': resolves outside the plugins directory."
        )

    return target


def _resolve_git_url(identifier: str) -> str:
    """Turn an identifier into a cloneable Git URL.

    Accepted formats:
    - Full URL: https://github.com/owner/repo.git
    - Full URL: git@github.com:owner/repo.git
    - Full URL: ssh://git@github.com/owner/repo.git
    - Shorthand: owner/repo  →  https://github.com/owner/repo.git

    NOTE: ``http://`` and ``file://`` schemes are accepted but will trigger a
    security warning at install time.
    """
    # Already a URL
    if identifier.startswith(("https://", "http://", "git@", "ssh://", "file://")):
        return identifier

    # owner/repo shorthand
    parts = identifier.strip("/").split("/")
    if len(parts) == 2:
        owner, repo = parts
        return f"https://github.com/{owner}/{repo}.git"

    raise ValueError(
        f"Invalid plugin identifier: '{identifier}'. "
        "Use a Git URL or owner/repo shorthand."
    )


def _repo_name_from_url(url: str) -> str:
    """Extract the repo name from a Git URL for the plugin directory name."""
    # Strip trailing .git and slashes
    name = url.rstrip("/")
    if name.endswith(".git"):
        name = name[:-4]
    # Get last path component
    name = name.rsplit("/", 1)[-1]
    # Handle ssh-style urls: git@github.com:owner/repo
    if ":" in name:
        name = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return name


def _read_manifest(plugin_dir: Path) -> dict:
    """Read plugin.yaml and return the parsed dict, or empty dict."""
    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.exists():
        return {}
    try:
        import yaml

        with open(manifest_file) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to read plugin.yaml in %s: %s", plugin_dir, e)
        return {}


def _copy_example_files(plugin_dir: Path, console) -> None:
    """Copy any .example files to their real names if they don't already exist.

    For example, ``config.yaml.example`` becomes ``config.yaml``.
    Skips files that already exist to avoid overwriting user config on reinstall.
    """
    for example_file in plugin_dir.glob("*.example"):
        real_name = example_file.stem  # e.g. "config.yaml" from "config.yaml.example"
        real_path = plugin_dir / real_name
        if not real_path.exists():
            try:
                shutil.copy2(example_file, real_path)
                console.print(
                    f"[dim]  Created {real_name} from {example_file.name}[/dim]"
                )
            except OSError as e:
                console.print(
                    f"[yellow]Warning:[/yellow] Failed to copy {example_file.name}: {e}"
                )


def _prompt_plugin_env_vars(manifest: dict, console) -> None:
    """Prompt for required environment variables declared in plugin.yaml.

    ``requires_env`` accepts two formats:

    Simple list (backwards-compatible)::

        requires_env:
          - MY_API_KEY

    Rich list with metadata::

        requires_env:
          - name: MY_API_KEY
            description: "API key for Acme service"
            url: "https://acme.com/keys"
            secret: true

    Already-set variables are skipped.  Values are saved to the user's ``.env``.
    """
    requires_env = manifest.get("requires_env") or []
    if not requires_env:
        return

    from hermes_cli.config import get_env_value, save_env_value  # noqa: F811
    from hermes_constants import display_hermes_home

    # Normalise to list-of-dicts
    env_specs: list[dict] = []
    for entry in requires_env:
        if isinstance(entry, str):
            env_specs.append({"name": entry})
        elif isinstance(entry, dict) and entry.get("name"):
            env_specs.append(entry)

    # Filter to only vars that aren't already set
    missing = [s for s in env_specs if not get_env_value(s["name"])]
    if not missing:
        return

    plugin_name = manifest.get("name", "this plugin")
    console.print(f"\n[bold]{plugin_name}[/bold] requires the following environment variables:\n")

    for spec in missing:
        name = spec["name"]
        desc = spec.get("description", "")
        url = spec.get("url", "")
        secret = spec.get("secret", False)

        label = f"  {name}"
        if desc:
            label += f" — {desc}"
        console.print(label)
        if url:
            console.print(f"  [dim]Get yours at: {url}[/dim]")

        try:
            if secret:
                import getpass
                value = getpass.getpass(f"  {name}: ").strip()
            else:
                value = input(f"  {name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[dim]  Skipped (you can set these later in {display_hermes_home()}/.env)[/dim]")
            return

        if value:
            save_env_value(name, value)
            os.environ[name] = value
            console.print(f"  [green]✓[/green] Saved to {display_hermes_home()}/.env")
        else:
            console.print(f"  [dim]  Skipped (set {name} in {display_hermes_home()}/.env later)[/dim]")

    console.print()


def _display_after_install(plugin_dir: Path, identifier: str) -> None:
    """Show after-install.md if it exists, otherwise a default message."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    after_install = plugin_dir / "after-install.md"

    if after_install.exists():
        content = after_install.read_text(encoding="utf-8")
        md = Markdown(content)
        console.print()
        console.print(Panel(md, border_style="green", expand=False))
        console.print()
    else:
        console.print()
        console.print(
            Panel(
                f"[green bold]Plugin installed:[/] {identifier}\n"
                f"[dim]Location:[/] {plugin_dir}",
                border_style="green",
                title="✓ Installed",
                expand=False,
            )
        )
        console.print()


def _display_removed(name: str, plugins_dir: Path) -> None:
    """Show confirmation after removing a plugin."""
    from rich.console import Console

    console = Console()
    console.print()
    console.print(f"[red]✗[/red] Plugin [bold]{name}[/bold] removed from {plugins_dir}")
    console.print()


def _require_installed_plugin(name: str, plugins_dir: Path, console) -> Path:
    """Return the plugin path if it exists, or exit with an error listing installed plugins."""
    target = _sanitize_plugin_name(name, plugins_dir)
    if not target.exists():
        installed = ", ".join(d.name for d in plugins_dir.iterdir() if d.is_dir()) or "(none)"
        console.print(
            f"[red]Error:[/red] Plugin '{name}' not found in {plugins_dir}.\n"
            f"Installed plugins: {installed}"
        )
        sys.exit(1)
    return target


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_install(
    identifier: str,
    force: bool = False,
    enable: Optional[bool] = None,
) -> None:
    """Install a plugin from a Git URL or owner/repo shorthand.

    After install, prompt "Enable now? [y/N]" unless *enable* is provided
    (True = auto-enable without prompting, False = install disabled).
    """
    import tempfile
    from rich.console import Console

    console = Console()

    try:
        git_url = _resolve_git_url(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Warn about insecure / local URL schemes
    if git_url.startswith(("http://", "file://")):
        console.print(
            "[yellow]Warning:[/yellow] Using insecure/local URL scheme. "
            "Consider using https:// or git@ for production installs."
        )

    plugins_dir = _plugins_dir()

    # Clone into a temp directory first so we can read plugin.yaml for the name
    with tempfile.TemporaryDirectory() as tmp:
        tmp_target = Path(tmp) / "plugin"
        console.print(f"[dim]Cloning {git_url}...[/dim]")

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", git_url, str(tmp_target)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            console.print("[red]Error:[/red] git is not installed or not in PATH.")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            console.print("[red]Error:[/red] Git clone timed out after 60 seconds.")
            sys.exit(1)

        if result.returncode != 0:
            console.print(
                f"[red]Error:[/red] Git clone failed:\n{result.stderr.strip()}"
            )
            sys.exit(1)

        # Read manifest
        manifest = _read_manifest(tmp_target)
        plugin_name = manifest.get("name") or _repo_name_from_url(git_url)

        # Sanitize plugin name against path traversal
        try:
            target = _sanitize_plugin_name(plugin_name, plugins_dir)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

        # Check manifest_version compatibility
        mv = manifest.get("manifest_version")
        if mv is not None:
            try:
                mv_int = int(mv)
            except (ValueError, TypeError):
                console.print(
                    f"[red]Error:[/red] Plugin '{plugin_name}' has invalid "
                    f"manifest_version '{mv}' (expected an integer)."
                )
                sys.exit(1)
            if mv_int > _SUPPORTED_MANIFEST_VERSION:
                from hermes_cli.config import recommended_update_command
                console.print(
                    f"[red]Error:[/red] Plugin '{plugin_name}' requires manifest_version "
                    f"{mv}, but this installer only supports up to {_SUPPORTED_MANIFEST_VERSION}.\n"
                    f"Run [bold]{recommended_update_command()}[/bold] to get a newer installer."
                )
                sys.exit(1)

        if target.exists():
            if not force:
                console.print(
                    f"[red]Error:[/red] Plugin '{plugin_name}' already exists at {target}.\n"
                    f"Use [bold]--force[/bold] to remove and reinstall, or "
                    f"[bold]hermes plugins update {plugin_name}[/bold] to pull latest."
                )
                sys.exit(1)
            console.print(f"[dim]  Removing existing {plugin_name}...[/dim]")
            shutil.rmtree(target)

        # Move from temp to final location
        shutil.move(str(tmp_target), str(target))

    # Validate it looks like a plugin
    if not (target / "plugin.yaml").exists() and not (target / "__init__.py").exists():
        console.print(
            f"[yellow]Warning:[/yellow] {plugin_name} doesn't contain plugin.yaml "
            f"or __init__.py. It may not be a valid Hermes plugin."
        )

    # Copy .example files to their real names (e.g. config.yaml.example → config.yaml)
    _copy_example_files(target, console)

    # Re-read manifest from installed location (for env var prompting)
    installed_manifest = _read_manifest(target)

    # Prompt for required environment variables before showing after-install docs
    _prompt_plugin_env_vars(installed_manifest, console)

    _display_after_install(target, identifier)

    # Determine the canonical plugin name for enable-list bookkeeping.
    installed_name = installed_manifest.get("name") or target.name

    # Decide whether to enable: explicit flag > interactive prompt > default off
    should_enable = enable
    if should_enable is None:
        # Interactive prompt unless stdin isn't a TTY (scripted install).
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                answer = input(
                    f"  Enable '{installed_name}' now? [y/N]: "
                ).strip().lower()
                should_enable = answer in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                should_enable = False
        else:
            should_enable = False

    if should_enable:
        enabled = _get_enabled_set()
        disabled = _get_disabled_set()
        enabled.add(installed_name)
        disabled.discard(installed_name)
        _save_enabled_set(enabled)
        _save_disabled_set(disabled)
        console.print(
            f"[green]✓[/green] Plugin [bold]{installed_name}[/bold] enabled."
        )
    else:
        console.print(
            f"[dim]Plugin installed but not enabled. "
            f"Run `hermes plugins enable {installed_name}` to activate.[/dim]"
        )

    console.print("[dim]Restart the gateway for the plugin to take effect:[/dim]")
    console.print("[dim]  hermes gateway restart[/dim]")
    console.print()


def cmd_update(name: str) -> None:
    """Update an installed plugin by pulling latest from its git remote."""
    from rich.console import Console

    console = Console()
    plugins_dir = _plugins_dir()

    try:
        target = _require_installed_plugin(name, plugins_dir, console)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not (target / ".git").exists():
        console.print(
            f"[red]Error:[/red] Plugin '{name}' was not installed from git "
            f"(no .git directory). Cannot update."
        )
        sys.exit(1)

    console.print(f"[dim]Updating {name}...[/dim]")

    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(target),
        )
    except FileNotFoundError:
        console.print("[red]Error:[/red] git is not installed or not in PATH.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        console.print("[red]Error:[/red] Git pull timed out after 60 seconds.")
        sys.exit(1)

    if result.returncode != 0:
        console.print(f"[red]Error:[/red] Git pull failed:\n{result.stderr.strip()}")
        sys.exit(1)

    # Copy any new .example files
    _copy_example_files(target, console)

    output = result.stdout.strip()
    if "Already up to date" in output:
        console.print(
            f"[green]✓[/green] Plugin [bold]{name}[/bold] is already up to date."
        )
    else:
        console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] updated.")
        console.print(f"[dim]{output}[/dim]")


def cmd_remove(name: str) -> None:
    """Remove an installed plugin by name."""
    from rich.console import Console

    console = Console()
    plugins_dir = _plugins_dir()

    try:
        target = _require_installed_plugin(name, plugins_dir, console)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    shutil.rmtree(target)
    _display_removed(name, plugins_dir)


def _get_disabled_set() -> set:
    """Read the disabled plugins set from config.yaml.

    An explicit deny-list. A plugin name here never loads, even if also
    listed in ``plugins.enabled``.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _save_disabled_set(disabled: set) -> None:
    """Write the disabled plugins list to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "plugins" not in config:
        config["plugins"] = {}
    config["plugins"]["disabled"] = sorted(disabled)
    save_config(config)


def _get_enabled_set() -> set:
    """Read the enabled plugins allow-list from config.yaml.

    Plugins are opt-in: only names here are loaded. Returns ``set()`` if
    the key is missing (same behaviour as "nothing enabled yet").
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        plugins_cfg = config.get("plugins", {})
        if not isinstance(plugins_cfg, dict):
            return set()
        enabled = plugins_cfg.get("enabled", [])
        return set(enabled) if isinstance(enabled, list) else set()
    except Exception:
        return set()


def _save_enabled_set(enabled: set) -> None:
    """Write the enabled plugins list to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "plugins" not in config:
        config["plugins"] = {}
    config["plugins"]["enabled"] = sorted(enabled)
    save_config(config)


def cmd_enable(name: str) -> None:
    """Add a plugin to the enabled allow-list (and remove it from disabled)."""
    from rich.console import Console

    console = Console()
    # Discover the plugin — check installed (user) AND bundled.
    if not _plugin_exists(name):
        console.print(f"[red]Plugin '{name}' is not installed or bundled.[/red]")
        sys.exit(1)

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()

    if name in enabled and name not in disabled:
        console.print(f"[dim]Plugin '{name}' is already enabled.[/dim]")
        return

    enabled.add(name)
    disabled.discard(name)
    _save_enabled_set(enabled)
    _save_disabled_set(disabled)
    console.print(
        f"[green]✓[/green] Plugin [bold]{name}[/bold] enabled. "
        "Takes effect on next session."
    )


def cmd_disable(name: str) -> None:
    """Remove a plugin from the enabled allow-list (and add to disabled)."""
    from rich.console import Console

    console = Console()
    if not _plugin_exists(name):
        console.print(f"[red]Plugin '{name}' is not installed or bundled.[/red]")
        sys.exit(1)

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()

    if name not in enabled and name in disabled:
        console.print(f"[dim]Plugin '{name}' is already disabled.[/dim]")
        return

    enabled.discard(name)
    disabled.add(name)
    _save_enabled_set(enabled)
    _save_disabled_set(disabled)
    console.print(
        f"[yellow]\u2298[/yellow] Plugin [bold]{name}[/bold] disabled. "
        "Takes effect on next session."
    )


def _plugin_exists(name: str) -> bool:
    """Return True if a plugin with *name* is installed (user) or bundled."""
    # Installed: directory name or manifest name match in user plugins dir
    user_dir = _plugins_dir()
    if user_dir.is_dir():
        if (user_dir / name).is_dir():
            return True
        for child in user_dir.iterdir():
            if not child.is_dir():
                continue
            manifest = _read_manifest(child)
            if manifest.get("name") == name:
                return True
    # Bundled: <repo>/plugins/<name>/ (or HERMES_BUNDLED_PLUGINS on Nix).
    from hermes_cli.plugins import get_bundled_plugins_dir
    repo_plugins = get_bundled_plugins_dir()
    if repo_plugins.is_dir():
        candidate = repo_plugins / name
        if candidate.is_dir() and (
            (candidate / "plugin.yaml").exists()
            or (candidate / "plugin.yml").exists()
        ):
            return True
    return False


def _discover_all_plugins() -> list:
    """Return a list of (name, version, description, source, dir_path) for
    every plugin the loader can see — user + bundled + project.

    Matches the ordering/dedup of ``PluginManager.discover_and_load``:
    bundled first, then user, then project; user overrides bundled on
    name collision.
    """
    try:
        import yaml
    except ImportError:
        yaml = None

    seen: dict = {}  # name -> (name, version, description, source, path)

    # Bundled (<repo>/plugins/<name>/), excluding memory/ and context_engine/
    from hermes_cli.plugins import get_bundled_plugins_dir
    repo_plugins = get_bundled_plugins_dir()
    for base, source in ((repo_plugins, "bundled"), (_plugins_dir(), "user")):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            if source == "bundled" and d.name in ("memory", "context_engine"):
                continue
            manifest_file = d / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = d / "plugin.yml"
            if not manifest_file.exists():
                continue
            name = d.name
            version = ""
            description = ""
            if yaml:
                try:
                    with open(manifest_file) as f:
                        manifest = yaml.safe_load(f) or {}
                    name = manifest.get("name", d.name)
                    version = manifest.get("version", "")
                    description = manifest.get("description", "")
                except Exception:
                    pass
            # User plugins override bundled on name collision.
            if name in seen and source == "bundled":
                continue
            src_label = source
            if source == "user" and (d / ".git").exists():
                src_label = "git"
            seen[name] = (name, version, description, src_label, d)
    return list(seen.values())


def cmd_list() -> None:
    """List all plugins (bundled + user) with enabled/disabled state."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    entries = _discover_all_plugins()
    if not entries:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()

    table = Table(title="Plugins", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Version", style="dim")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for name, version, description, source, _dir in entries:
        if name in disabled:
            status = "[red]disabled[/red]"
        elif name in enabled:
            status = "[green]enabled[/green]"
        else:
            status = "[yellow]not enabled[/yellow]"
        table.add_row(name, status, str(version), description, source)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Interactive toggle:[/dim] hermes plugins")
    console.print("[dim]Enable/disable:[/dim] hermes plugins enable/disable <name>")
    console.print("[dim]Plugins are opt-in by default — only 'enabled' plugins load.[/dim]")


# ---------------------------------------------------------------------------
# Provider plugin discovery helpers
# ---------------------------------------------------------------------------


def _discover_memory_providers() -> list[tuple[str, str]]:
    """Return [(name, description), ...] for available memory providers."""
    try:
        from plugins.memory import discover_memory_providers
        return [(name, desc) for name, desc, _avail in discover_memory_providers()]
    except Exception:
        return []


def _discover_context_engines() -> list[tuple[str, str]]:
    """Return [(name, description), ...] for available context engines."""
    try:
        from plugins.context_engine import discover_context_engines
        return [(name, desc) for name, desc, _avail in discover_context_engines()]
    except Exception:
        return []


def _get_current_memory_provider() -> str:
    """Return the current memory.provider from config (empty = built-in)."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "memory", "provider", default="") or ""
    except Exception:
        return ""


def _get_current_context_engine() -> str:
    """Return the current context.engine from config."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "context", "engine", default="compressor") or "compressor"
    except Exception:
        return "compressor"


def _save_memory_provider(name: str) -> None:
    """Persist memory.provider to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "memory" not in config:
        config["memory"] = {}
    config["memory"]["provider"] = name
    save_config(config)


def _save_context_engine(name: str) -> None:
    """Persist context.engine to config.yaml."""
    from hermes_cli.config import load_config, save_config
    config = load_config()
    if "context" not in config:
        config["context"] = {}
    config["context"]["engine"] = name
    save_config(config)


def _configure_memory_provider() -> bool:
    """Launch a radio picker for memory providers. Returns True if changed."""
    from hermes_cli.curses_ui import curses_radiolist

    current = _get_current_memory_provider()
    providers = _discover_memory_providers()

    # Build items: "built-in" first, then discovered providers
    items = ["built-in (default)"]
    names = [""]  # empty string = built-in
    selected = 0

    for name, desc in providers:
        names.append(name)
        label = f"{name} \u2014 {desc}" if desc else name
        items.append(label)
        if name == current:
            selected = len(items) - 1

    # If current provider isn't in discovered list, add it
    if current and current not in names:
        names.append(current)
        items.append(f"{current} (not found)")
        selected = len(items) - 1

    choice = curses_radiolist(
        title="Memory Provider (select one)",
        items=items,
        selected=selected,
    )

    new_provider = names[choice]
    if new_provider != current:
        _save_memory_provider(new_provider)
        return True
    return False


def _configure_context_engine() -> bool:
    """Launch a radio picker for context engines. Returns True if changed."""
    from hermes_cli.curses_ui import curses_radiolist

    current = _get_current_context_engine()
    engines = _discover_context_engines()

    # Build items: "compressor" first (built-in), then discovered engines
    items = ["compressor (default)"]
    names = ["compressor"]
    selected = 0

    for name, desc in engines:
        names.append(name)
        label = f"{name} \u2014 {desc}" if desc else name
        items.append(label)
        if name == current:
            selected = len(items) - 1

    # If current engine isn't in discovered list and isn't compressor, add it
    if current != "compressor" and current not in names:
        names.append(current)
        items.append(f"{current} (not found)")
        selected = len(items) - 1

    choice = curses_radiolist(
        title="Context Engine (select one)",
        items=items,
        selected=selected,
    )

    new_engine = names[choice]
    if new_engine != current:
        _save_context_engine(new_engine)
        return True
    return False


# ---------------------------------------------------------------------------
# Composite plugins UI
# ---------------------------------------------------------------------------


def cmd_toggle() -> None:
    """Interactive composite UI — general plugins + provider plugin categories."""
    from rich.console import Console

    console = Console()

    # -- General plugins discovery (bundled + user) --
    entries = _discover_all_plugins()
    enabled_set = _get_enabled_set()
    disabled_set = _get_disabled_set()

    plugin_names = []
    plugin_labels = []
    plugin_selected = set()

    for i, (name, _version, description, source, _d) in enumerate(entries):
        label = f"{name} \u2014 {description}" if description else name
        if source == "bundled":
            label = f"{label} [bundled]"
        plugin_names.append(name)
        plugin_labels.append(label)
        # Selected (enabled) when in enabled-set AND not in disabled-set
        if name in enabled_set and name not in disabled_set:
            plugin_selected.add(i)

    # -- Provider categories --
    current_memory = _get_current_memory_provider() or "built-in"
    current_context = _get_current_context_engine()
    categories = [
        ("Memory Provider", current_memory, _configure_memory_provider),
        ("Context Engine", current_context, _configure_context_engine),
    ]

    has_plugins = bool(plugin_names)
    has_categories = bool(categories)

    if not has_plugins and not has_categories:
        console.print("[dim]No plugins installed and no provider categories available.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    # Non-TTY fallback
    if not sys.stdin.isatty():
        console.print("[dim]Interactive mode requires a terminal.[/dim]")
        return

    # Launch the composite curses UI
    try:
        import curses
        _run_composite_ui(curses, plugin_names, plugin_labels, plugin_selected,
                          disabled_set, categories, console)
    except ImportError:
        _run_composite_fallback(plugin_names, plugin_labels, plugin_selected,
                                disabled_set, categories, console)


def _run_composite_ui(curses, plugin_names, plugin_labels, plugin_selected,
                      disabled, categories, console):
    """Custom curses screen with checkboxes + category action rows."""
    from hermes_cli.curses_ui import flush_stdin

    chosen = set(plugin_selected)
    n_plugins = len(plugin_names)
    # Total rows: plugins + separator + categories
    # separator is not navigable
    n_categories = len(categories)
    total_items = n_plugins + n_categories  # navigable items

    result_holder = {"plugins_changed": False, "providers_changed": False}

    def _draw(stdscr):
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, 8, -1)  # dim gray
        cursor = 0
        scroll_offset = 0

        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            # Header
            try:
                hattr = curses.A_BOLD
                if curses.has_colors():
                    hattr |= curses.color_pair(2)
                stdscr.addnstr(0, 0, "Plugins", max_x - 1, hattr)
                stdscr.addnstr(
                    1, 0,
                    "  \u2191\u2193 navigate  SPACE toggle  ENTER configure/confirm  ESC done",
                    max_x - 1, curses.A_DIM,
                )
            except curses.error:
                pass

            # Build display rows
            # Row layout:
            #   [plugins section header] (not navigable, skipped in scroll math)
            #   plugin checkboxes (navigable, indices 0..n_plugins-1)
            #   [separator] (not navigable)
            #   [categories section header] (not navigable)
            #   category action rows (navigable, indices n_plugins..total_items-1)

            visible_rows = max_y - 4
            if cursor < scroll_offset:
                scroll_offset = cursor
            elif cursor >= scroll_offset + visible_rows:
                scroll_offset = cursor - visible_rows + 1

            y = 3  # start drawing after header

            # Determine which items are visible based on scroll
            # We need to map logical cursor positions to screen rows
            # accounting for non-navigable separator/headers


            # --- General Plugins section ---
            if n_plugins > 0:
                # Section header
                if y < max_y - 1:
                    try:
                        sattr = curses.A_BOLD
                        if curses.has_colors():
                            sattr |= curses.color_pair(2)
                        stdscr.addnstr(y, 0, "  General Plugins", max_x - 1, sattr)
                    except curses.error:
                        pass
                    y += 1

                for i in range(n_plugins):
                    if y >= max_y - 1:
                        break
                    check = "\u2713" if i in chosen else " "
                    arrow = "\u2192" if i == cursor else " "
                    line = f" {arrow} [{check}] {plugin_labels[i]}"
                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(1)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass
                    y += 1

            # --- Separator ---
            if y < max_y - 1:
                y += 1  # blank line

            # --- Provider Plugins section ---
            if n_categories > 0 and y < max_y - 1:
                try:
                    sattr = curses.A_BOLD
                    if curses.has_colors():
                        sattr |= curses.color_pair(2)
                    stdscr.addnstr(y, 0, "  Provider Plugins", max_x - 1, sattr)
                except curses.error:
                    pass
                y += 1

                for ci, (cat_name, cat_current, _cat_fn) in enumerate(categories):
                    if y >= max_y - 1:
                        break
                    cat_idx = n_plugins + ci
                    arrow = "\u2192" if cat_idx == cursor else " "
                    line = f" {arrow}   {cat_name:<24} \u25b8 {cat_current}"
                    attr = curses.A_NORMAL
                    if cat_idx == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(3)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass
                    y += 1

            stdscr.refresh()
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k")):
                if total_items > 0:
                    cursor = (cursor - 1) % total_items
            elif key in (curses.KEY_DOWN, ord("j")):
                if total_items > 0:
                    cursor = (cursor + 1) % total_items
            elif key == ord(" "):
                if cursor < n_plugins:
                    # Toggle general plugin
                    chosen.symmetric_difference_update({cursor})
                else:
                    # Provider category — launch sub-screen
                    ci = cursor - n_plugins
                    if 0 <= ci < n_categories:
                        curses.endwin()
                        _cat_name, _cat_cur, cat_fn = categories[ci]
                        changed = cat_fn()
                        if changed:
                            result_holder["providers_changed"] = True
                            # Refresh current values
                            categories[ci] = (
                                _cat_name,
                                _get_current_memory_provider() or "built-in" if ci == 0
                                else _get_current_context_engine(),
                                cat_fn,
                            )
                        # Re-enter curses
                        stdscr = curses.initscr()
                        curses.noecho()
                        curses.cbreak()
                        stdscr.keypad(True)
                        if curses.has_colors():
                            curses.start_color()
                            curses.use_default_colors()
                            curses.init_pair(1, curses.COLOR_GREEN, -1)
                            curses.init_pair(2, curses.COLOR_YELLOW, -1)
                            curses.init_pair(3, curses.COLOR_CYAN, -1)
                            curses.init_pair(4, 8, -1)
                        curses.curs_set(0)
            elif key in (curses.KEY_ENTER, 10, 13):
                if cursor < n_plugins:
                    # ENTER on a plugin checkbox — confirm and exit
                    result_holder["plugins_changed"] = True
                    return
                else:
                    # ENTER on a category — same as SPACE, launch sub-screen
                    ci = cursor - n_plugins
                    if 0 <= ci < n_categories:
                        curses.endwin()
                        _cat_name, _cat_cur, cat_fn = categories[ci]
                        changed = cat_fn()
                        if changed:
                            result_holder["providers_changed"] = True
                            categories[ci] = (
                                _cat_name,
                                _get_current_memory_provider() or "built-in" if ci == 0
                                else _get_current_context_engine(),
                                cat_fn,
                            )
                        stdscr = curses.initscr()
                        curses.noecho()
                        curses.cbreak()
                        stdscr.keypad(True)
                        if curses.has_colors():
                            curses.start_color()
                            curses.use_default_colors()
                            curses.init_pair(1, curses.COLOR_GREEN, -1)
                            curses.init_pair(2, curses.COLOR_YELLOW, -1)
                            curses.init_pair(3, curses.COLOR_CYAN, -1)
                            curses.init_pair(4, 8, -1)
                        curses.curs_set(0)
            elif key in (27, ord("q")):
                # Save plugin changes on exit
                result_holder["plugins_changed"] = True
                return

    curses.wrapper(_draw)
    flush_stdin()

    # Persist general plugin changes. The new allow-list is the set of
    # plugin names that were checked; anything not checked is explicitly
    # disabled (written to disabled-list) so it remains off even if the
    # plugin code does something clever like auto-enable in the future.
    new_enabled: set = set()
    new_disabled: set = set(disabled)  # preserve existing disabled state for unseen plugins
    for i, name in enumerate(plugin_names):
        if i in chosen:
            new_enabled.add(name)
            new_disabled.discard(name)
        else:
            new_disabled.add(name)

    prev_enabled = _get_enabled_set()
    enabled_changed = new_enabled != prev_enabled
    disabled_changed = new_disabled != disabled

    if enabled_changed or disabled_changed:
        _save_enabled_set(new_enabled)
        _save_disabled_set(new_disabled)
        console.print(
            f"\n[green]\u2713[/green] General plugins: {len(new_enabled)} enabled, "
            f"{len(plugin_names) - len(new_enabled)} disabled."
        )
    elif n_plugins > 0:
        console.print("\n[dim]General plugins unchanged.[/dim]")

    if result_holder["providers_changed"]:
        new_memory = _get_current_memory_provider() or "built-in"
        new_context = _get_current_context_engine()
        console.print(
            f"[green]\u2713[/green] Memory provider: [bold]{new_memory}[/bold]  "
            f"Context engine: [bold]{new_context}[/bold]"
        )

    if n_plugins > 0 or result_holder["providers_changed"]:
        console.print("[dim]Changes take effect on next session.[/dim]")
    console.print()


def _run_composite_fallback(plugin_names, plugin_labels, plugin_selected,
                            disabled, categories, console):
    """Text-based fallback for the composite plugins UI."""
    from hermes_cli.colors import Colors, color

    print(color("\n  Plugins", Colors.YELLOW))

    # General plugins
    if plugin_names:
        chosen = set(plugin_selected)
        print(color("\n  General Plugins", Colors.YELLOW))
        print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))

        while True:
            for i, label in enumerate(plugin_labels):
                marker = color("[\u2713]", Colors.GREEN) if i in chosen else "[ ]"
                print(f"  {marker} {i + 1:>2}. {label}")
            print()
            try:
                val = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
                if not val:
                    break
                idx = int(val) - 1
                if 0 <= idx < len(plugin_names):
                    chosen.symmetric_difference_update({idx})
            except (ValueError, KeyboardInterrupt, EOFError):
                return
            print()

        new_enabled: set = set()
        new_disabled: set = set(disabled)
        for i, name in enumerate(plugin_names):
            if i in chosen:
                new_enabled.add(name)
                new_disabled.discard(name)
            else:
                new_disabled.add(name)
        prev_enabled = _get_enabled_set()
        if new_enabled != prev_enabled or new_disabled != disabled:
            _save_enabled_set(new_enabled)
            _save_disabled_set(new_disabled)

    # Provider categories
    if categories:
        print(color("\n  Provider Plugins", Colors.YELLOW))
        for ci, (cat_name, cat_current, cat_fn) in enumerate(categories):
            print(f"  {ci + 1}. {cat_name} [{cat_current}]")
        print()
        try:
            val = input(color("  Configure # (or Enter to skip): ", Colors.DIM)).strip()
            if val:
                ci = int(val) - 1
                if 0 <= ci < len(categories):
                    categories[ci][2]()  # call the configure function
        except (ValueError, KeyboardInterrupt, EOFError):
            pass

    print()


def plugins_command(args) -> None:
    """Dispatch hermes plugins subcommands."""
    action = getattr(args, "plugins_action", None)

    if action == "install":
        # Map argparse tri-state: --enable=True, --no-enable=False, neither=None (prompt)
        enable_arg = None
        if getattr(args, "enable", False):
            enable_arg = True
        elif getattr(args, "no_enable", False):
            enable_arg = False
        cmd_install(
            args.identifier,
            force=getattr(args, "force", False),
            enable=enable_arg,
        )
    elif action == "update":
        cmd_update(args.name)
    elif action in ("remove", "rm", "uninstall"):
        cmd_remove(args.name)
    elif action == "enable":
        cmd_enable(args.name)
    elif action == "disable":
        cmd_disable(args.name)
    elif action in ("list", "ls"):
        cmd_list()
    elif action is None:
        cmd_toggle()
    else:
        from rich.console import Console

        Console().print(f"[red]Unknown plugins action: {action}[/red]")
        sys.exit(1)
