"""Source launch/update composition over the shared JavaScript builders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def source_build_env(base_env: dict | None = None) -> dict[str, str]:
    from pm import ensure
    from hermes_constants import get_hermes_home

    env = {**os.environ, **(base_env or {}), "CI": "1", "HERMES_PYTHON": sys.executable,
           "PYTHON": sys.executable}
    env.pop("ESBUILD_BINARY_PATH", None)
    npmrc = get_hermes_home() / "npmrc"
    if npmrc.is_file():
        env.setdefault("NPM_CONFIG_USERCONFIG", str(npmrc))
    return ensure("npm", base_env=env, explicit=True).env


def run_source_script(project_root: Path, script: str, *args: str, env: dict) -> None:
    subprocess.run(
        [shutil.which("node", path=env["PATH"]), str(project_root / script), *args],
        cwd=project_root, env=env, check=True,
    )


def prepare_source_dependencies(project_root: Path, workspaces: tuple[str, ...], *, env: dict) -> None:
    run_source_script(
        project_root, "scripts/build/node-deps.mjs", "--source", str(project_root), "--reuse",
        *(arg for workspace in workspaces for arg in ("--workspace", workspace)), env=env,
    )


def prepare_launch_dependencies(project_root: Path, *, env: dict) -> None:
    """A launch rebuild must not prune another installed source frontend."""
    from hermes_cli.main_desktop import _desktop_dist_exists, _desktop_packaged_executable

    desktop_dir = project_root / "apps/desktop"
    desktop = _desktop_dist_exists(desktop_dir) or _desktop_packaged_executable(desktop_dir) is not None
    workspaces = ("ui-tui", "web") + (("apps/desktop",) if desktop else ())
    prepare_source_dependencies(project_root, workspaces, env=env)


def build_source_tui(project_root: Path, *, env: dict) -> None:
    run_source_script(project_root, "scripts/build/tui.mjs", env=env)


def build_source_web(project_root: Path, *, env: dict) -> None:
    from hermes_cli.main_web_build import _write_web_ui_build_stamp

    run_source_script(project_root, "scripts/generate-icons.mjs", env=env)
    run_source_script(project_root, "scripts/build/web.mjs", env=env)
    _write_web_ui_build_stamp(project_root, project_root / "web")


def build_update_products(project_root: Path, *, desktop: bool) -> None:
    """Prepare the selected union once; a failed product aborts the update."""
    env = source_build_env()
    workspaces = ("ui-tui", "web") + (("apps/desktop",) if desktop else ())
    prepare_source_dependencies(project_root, workspaces, env=env)
    build_source_tui(project_root, env=env)
    build_source_web(project_root, env=env)
    if desktop:
        from hermes_cli.main_desktop import build_prepared_desktop

        build_prepared_desktop(
            project_root / "apps/desktop", source_mode=False,
            npm=shutil.which("npm", path=env["PATH"]), env=env,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build source-install frontends")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    build_update_products(args.source.resolve(), desktop=args.desktop)
