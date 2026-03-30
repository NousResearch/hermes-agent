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

logger = logging.getLogger(__name__)

# Minimum manifest version this installer understands.
# Plugins may declare ``manifest_version: 1`` in plugin.yaml;
# future breaking changes to the manifest schema bump this.
_SUPPORTED_MANIFEST_VERSION = 1


def _plugins_dir() -> Path:
    """Return the user plugins directory, creating it if needed."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    plugins = Path(hermes_home) / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins


def _sanitize_plugin_name(name: str, plugins_dir: Path) -> Path:
    """Validate a plugin name and return the safe target path inside *plugins_dir*.

    Raises ``ValueError`` if the name contains path-traversal sequences or would
    resolve outside the plugins directory.
    """
    if not name:
        raise ValueError("Plugin name must not be empty.")

    # Reject obvious traversal characters
    for bad in ("/", "\\", ".."):
        if bad in name:
            raise ValueError(f"Invalid plugin name '{name}': must not contain '{bad}'.")

    target = (plugins_dir / name).resolve()
    plugins_resolved = plugins_dir.resolve()

    if (
        not str(target).startswith(str(plugins_resolved) + os.sep)
        and target != plugins_resolved
    ):
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


def cmd_install(identifier: str, force: bool = False) -> None:
    """Install a plugin from a Git URL or owner/repo shorthand."""
    import tempfile
    from rich.console import Console

    console = Console()

    try:
        git_url = _resolve_git_url(identifier)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Warn about insecure / local URL schemes
    if git_url.startswith("http://") or git_url.startswith("file://"):
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
                console.print(
                    f"[red]Error:[/red] Plugin '{plugin_name}' requires manifest_version "
                    f"{mv}, but this installer only supports up to {_SUPPORTED_MANIFEST_VERSION}.\n"
                    f"Run [bold]hermes update[/bold] to get a newer installer."
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

    _display_after_install(target, identifier)

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
    """Read the disabled plugins set from config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        disabled = config.get("plugins", {}).get("disabled", [])
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


def cmd_enable(name: str) -> None:
    """Enable a previously disabled plugin."""
    from rich.console import Console

    console = Console()
    plugins_dir = _plugins_dir()

    # Verify the plugin exists
    target = plugins_dir / name
    if not target.is_dir():
        console.print(f"[red]Plugin '{name}' is not installed.[/red]")
        sys.exit(1)

    disabled = _get_disabled_set()
    if name not in disabled:
        console.print(f"[dim]Plugin '{name}' is already enabled.[/dim]")
        return

    disabled.discard(name)
    _save_disabled_set(disabled)
    console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] enabled. Takes effect on next session.")


def cmd_disable(name: str) -> None:
    """Disable a plugin without removing it."""
    from rich.console import Console

    console = Console()
    plugins_dir = _plugins_dir()

    # Verify the plugin exists
    target = plugins_dir / name
    if not target.is_dir():
        console.print(f"[red]Plugin '{name}' is not installed.[/red]")
        sys.exit(1)

    disabled = _get_disabled_set()
    if name in disabled:
        console.print(f"[dim]Plugin '{name}' is already disabled.[/dim]")
        return

    disabled.add(name)
    _save_disabled_set(disabled)
    console.print(f"[yellow]⊘[/yellow] Plugin [bold]{name}[/bold] disabled. Takes effect on next session.")


def cmd_list() -> None:
    """List installed plugins."""
    from rich.console import Console
    from rich.table import Table

    try:
        import yaml
    except ImportError:
        yaml = None

    console = Console()
    plugins_dir = _plugins_dir()

    dirs = sorted(d for d in plugins_dir.iterdir() if d.is_dir())
    if not dirs:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    disabled = _get_disabled_set()

    table = Table(title="Installed Plugins", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Version", style="dim")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for d in dirs:
        manifest_file = d / "plugin.yaml"
        name = d.name
        version = ""
        description = ""
        source = "local"

        if manifest_file.exists() and yaml:
            try:
                with open(manifest_file) as f:
                    manifest = yaml.safe_load(f) or {}
                name = manifest.get("name", d.name)
                version = manifest.get("version", "")
                description = manifest.get("description", "")
            except Exception:
                pass

        # Check if it's a git repo (installed via hermes plugins install)
        if (d / ".git").exists():
            source = "git"

        is_disabled = name in disabled or d.name in disabled
        status = "[red]disabled[/red]" if is_disabled else "[green]enabled[/green]"
        table.add_row(name, status, str(version), description, source)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Interactive toggle:[/dim] hermes plugins")
    console.print("[dim]Enable/disable:[/dim] hermes plugins enable/disable <name>")


def cmd_toggle() -> None:
    """Interactive curses checklist to enable/disable installed plugins."""
    from rich.console import Console

    try:
        import yaml
    except ImportError:
        yaml = None

    console = Console()
    plugins_dir = _plugins_dir()

    dirs = sorted(d for d in plugins_dir.iterdir() if d.is_dir())
    if not dirs:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    disabled = _get_disabled_set()

    # Build items list: "name — description" for display
    names = []
    labels = []
    selected = set()

    for i, d in enumerate(dirs):
        manifest_file = d / "plugin.yaml"
        name = d.name
        description = ""

        if manifest_file.exists() and yaml:
            try:
                with open(manifest_file) as f:
                    manifest = yaml.safe_load(f) or {}
                name = manifest.get("name", d.name)
                description = manifest.get("description", "")
            except Exception:
                pass

        names.append(name)
        label = f"{name} — {description}" if description else name
        labels.append(label)

        if name not in disabled and d.name not in disabled:
            selected.add(i)

    from hermes_cli.curses_ui import curses_checklist

    result = curses_checklist(
        title="Plugins — toggle enabled/disabled",
        items=labels,
        selected=selected,
    )

    # Compute new disabled set from deselected items
    new_disabled = set()
    for i, name in enumerate(names):
        if i not in result:
            new_disabled.add(name)

    if new_disabled != disabled:
        _save_disabled_set(new_disabled)
        enabled_count = len(names) - len(new_disabled)
        console.print(
            f"\n[green]✓[/green] {enabled_count} enabled, {len(new_disabled)} disabled. "
            f"Takes effect on next session."
        )
    else:
        console.print("\n[dim]No changes.[/dim]")


def plugins_command(args) -> None:
    """Dispatch hermes plugins subcommands."""
    action = getattr(args, "plugins_action", None)

    if action == "install":
        cmd_install(args.identifier, force=getattr(args, "force", False))
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
