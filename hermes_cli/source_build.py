"""Source launch/update composition over the shared JavaScript builders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def source_product_current(project_root: Path, product: str, out: Path) -> bool:
    """Read the compiler's receipt without acquiring tools or dependencies."""
    from pm import env_for

    env = env_for("node")
    node = shutil.which("node", path=env.get("PATH", ""))
    if not node:
        return False
    try:
        result = subprocess.run(
            [node, str(project_root / "scripts/build/freshness.mjs"),
             "--source", str(project_root), "--product", product, "--out", str(out)],
            cwd=project_root, env=env, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def source_build_env(base_env: dict | None = None, *, explicit: bool = False) -> dict[str, str]:
    from pm import ensure
    from pm.environments import project_python, running_from_selected_environment
    from pm.paths import repo_root
    from hermes_constants import get_hermes_home

    # The historical update runs on store Python with the selected environment
    # activated in-process. Icon generation starts an isolated child, which needs
    # the selected venv executable rather than the store interpreter.
    root = repo_root()
    python = str(project_python(root)) if running_from_selected_environment(root) else sys.executable
    env = {**os.environ, **(base_env or {}), "CI": "1", "HERMES_PYTHON": python,
           "PYTHON": python}
    env.pop("ESBUILD_BINARY_PATH", None)
    npmrc = get_hermes_home() / "npmrc"
    if npmrc.is_file():
        env.setdefault("NPM_CONFIG_USERCONFIG", str(npmrc))
    return ensure("npm", base_env=env, explicit=explicit).env


def run_source_script(project_root: Path, script: str, *args: str, env: dict, label: str) -> None:
    from pm.progress import run_contained

    # npm's deprecation warnings are the loudest lines and never actionable
    # here; they still land in the failure tail.
    run_contained(
        [shutil.which("node", path=env["PATH"]), str(project_root / script), *args],
        label, hide=lambda line: line.lower().startswith("npm warn"), indent="  ",
        cwd=project_root, env=env,
    )


def prepare_source_dependencies(project_root: Path, workspaces: tuple[str, ...], *, env: dict,
                                explicit: bool = False) -> None:
    from pm import lazy_installs_allowed

    run_source_script(
        project_root, "scripts/build/node-deps.mjs", "--source", str(project_root), "--reuse",
        *(() if explicit or lazy_installs_allowed() else ("--no-install",)),
        *(arg for workspace in workspaces for arg in ("--workspace", workspace)), env=env,
        label="Preparing Node dependencies",
    )


def prepare_launch_dependencies(project_root: Path, *, env: dict) -> None:
    """A launch rebuild must not prune another installed source frontend."""
    from hermes_cli.main_desktop import _desktop_dist_exists, _desktop_packaged_executable

    desktop_dir = project_root / "apps/desktop"
    desktop = _desktop_dist_exists(desktop_dir) or _desktop_packaged_executable(desktop_dir) is not None
    workspaces = ("ui-tui", "web") + (("apps/desktop",) if desktop else ())
    prepare_source_dependencies(project_root, workspaces, env=env)


def build_source_tui(project_root: Path, *, env: dict) -> None:
    run_source_script(project_root, "scripts/build/tui.mjs", env=env, label="Building the TUI")


def build_source_web(project_root: Path, *, env: dict, icons: Path | None = None) -> None:
    # Default-brand icons are committed; installs never render them.
    icons = icons or project_root
    run_source_script(project_root, "scripts/build/web.mjs", "--source", str(project_root),
                      "--icons", str(icons), "--out", str(project_root / "hermes_cli/web_dist"), env=env,
                      label="Building the web UI")


def source_frontends(project_root: Path) -> tuple[str, ...]:
    """The frontend workspaces this checkout carries. A source slice without them
    (python-only installs, the installer's acceptance fixture) has no products
    to build; it still publishes commands and runs the maintenance tail."""
    return tuple(name for name in ("ui-tui", "web") if (project_root / name / "package.json").is_file())


def build_update_products(project_root: Path, *, desktop: bool) -> None:
    """Prepare the selected union once; a failed product aborts the update."""
    # Both current updates and historical takeover reach this in a fresh target
    # interpreter, never in the updater's pre-sync import graph.
    from hermes_cli.main_install_repair import _warn_configured_features_missing_deps
    from hermes_cli.update_stage import publish_stage

    _warn_configured_features_missing_deps()
    frontends = source_frontends(project_root)
    if not frontends:
        return
    env = source_build_env(explicit=True)
    workspaces = frontends + (("apps/desktop",) if desktop else ())
    publish_stage("Updating Node dependencies")
    prepare_source_dependencies(project_root, workspaces, env=env, explicit=True)
    if "ui-tui" in frontends:
        publish_stage("Building the TUI")
        build_source_tui(project_root, env=env)
    if "web" in frontends:
        publish_stage("Building the web UI")
        build_source_web(project_root, env=env)
    if desktop:
        from hermes_cli.main_desktop import _install_rebuilt_desktop_app, build_prepared_desktop

        publish_stage("Building the desktop app")
        build_prepared_desktop(
            project_root / "apps/desktop", source_mode=False,
            npm=shutil.which("npm", path=env["PATH"]), env=env, icons=project_root,
        )
        # A current release/ can still sit beside a stale installed copy (an earlier
        # update rebuilt but never installed); healing must not wait for the next build.
        installed, problems = _install_rebuilt_desktop_app(project_root / "apps/desktop")
        for app in installed:
            print(f"  ✓ Installed the rebuilt Desktop app at {app}")
        for problem in problems:
            print(f"  ⚠ {problem}")
    # A configured memory provider that no longer ships in core is installed from the
    # catalog for every profile home sharing this venv (config, data and tool names
    # unchanged). The update must finish even if the migration blows up.
    try:
        from hermes_cli.memory_provider_migration import migrate_all_homes

        migrate_all_homes()
    except Exception as exc:
        print(f"  ⚠ Memory provider migration skipped: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build source-install frontends")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    build_update_products(args.source.resolve(), desktop=args.desktop)
