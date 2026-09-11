"""Shared payload layout, source snapshots and generated launchers."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_manifest(root: Path, *, target: str, repo: str, ref: str | None = None) -> dict:
    manifest = {"schema": 1, "target": target, "repo": repo, "venv": "venv", "store": "tools"}
    if ref is not None:
        manifest["ref"] = ref
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def snapshot(repo: Path, ref: str, destination: Path) -> None:
    """Archive a resolved git revision without carrying checkout metadata."""
    repo, destination = repo.resolve(), destination.resolve()
    if repo == destination or repo.is_relative_to(destination):
        raise ValueError("the snapshot destination must not contain the source checkout")
    with tempfile.TemporaryDirectory(prefix="hermes-archive-") as temp:
        archive = Path(temp) / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), ref], cwd=repo, check=True)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")


def project_entries(repo: Path) -> dict[str, str]:
    return tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8-sig"))["project"]["scripts"]


def record_tools(root: Path, lock_path: Path, target: str, entries: dict[str, str]) -> None:
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import tree_digest

    store = root / "tools"
    facts, lock = Facts(store / "facts.json"), Lockfile(lock_path)
    for name, entry_name in entries.items():
        entry = store / entry_name
        version, artifacts = lock.version(name), lock.artifacts(name, target)
        if not entry.is_dir() or not version or not artifacts:
            raise ValueError(f"incomplete payload tool: {name}")
        facts.record(name, version, entry_name, get_package(name).env(entry, target), store,
                     target=target, artifacts=[a["sha256"] for a in artifacts], digest=tree_digest(entry))


def rehash_tools(root: Path) -> int:
    """Record final tool bytes before the enclosing package is signed."""
    from pm.lock import Facts

    store = root / "tools"
    return Facts(store / "facts.json", strict=True).refresh_digests(store)


def plant_surfaces(repo: Path, source: Path, *, dashboard: bool = True) -> None:
    tui = source / "ui-tui/dist/entry.js"
    if not tui.is_file():
        raise FileNotFoundError(tui)
    destination = repo / "hermes_cli/tui_dist"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tui, destination / "entry.js")
    if dashboard:
        web = source / "hermes_cli/web_dist"
        if not (web / "index.html").is_file():
            raise FileNotFoundError(web / "index.html")
        shutil.rmtree(repo / "hermes_cli/web_dist", ignore_errors=True)
        shutil.copytree(web, repo / "hermes_cli/web_dist")


def seal_pm_runtime(root: Path, python: Path) -> dict:
    """Record a resident PM runtime without Windows' CWD-bound redirector.

    Sealed workers execute the base interpreter with -I -S and add only the
    recorded site directory. Its paths remain valid after the payload moves.
    """
    root, python = root.resolve(), python.resolve()
    if not python.is_relative_to(root) or not python.is_file():
        raise ValueError(f"PM interpreter must belong to the payload: {python}")
    runtime = root / "pm-runtime"
    sites = list(runtime.glob("lib/python*/site-packages")) + list(runtime.glob("Lib/site-packages"))
    if len(sites) != 1:
        raise ValueError(f"PM dependency directory missing or ambiguous: {runtime}")
    marker = {
        "python": Path(os.path.relpath(python, runtime)).as_posix(),
        "sitePackages": sites[0].relative_to(runtime).as_posix(),
    }
    cfg = runtime / "pyvenv.cfg"
    lines = cfg.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.partition("=")[0].strip() not in
             {"home", "executable", "base-executable", "base-prefix", "base-exec-prefix", "command"}]
    lines.insert(0, f"home = {os.path.relpath(python.parent, runtime)}")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # A sealed runtime is not activated, and copied Windows redirectors cannot
    # follow its relative home from arbitrary working directories.
    bindir = runtime / ("Scripts" if os.name == "nt" else "bin")
    for entry in bindir.iterdir():
        if os.name == "nt" or not entry.is_symlink():
            if entry.is_file():
                entry.unlink()
    _relativize_bin_links(root, runtime / "bin")
    (runtime / "pm-runtime.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def relativize_links(root: Path) -> int:
    """Only dependency-venv links move; framework links belong to codesign."""
    root = root.resolve()
    return sum(_relativize_bin_links(root, root / name / "bin") for name in ("venv", "pm-runtime"))


def _relativize_bin_links(root: Path, directory: Path) -> int:
    count = 0
    if not directory.is_dir():
        return count
    for link in directory.iterdir():
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        if not os.path.isabs(target):
            # Keep sibling chains intact; their absolute store link is rewritten separately.
            if not Path(os.path.abspath(directory / target)).is_relative_to(root):
                raise ValueError(f"link escapes payload: {link} -> {target}")
            continue
        resolved = (directory / target).resolve()
        if not resolved.is_relative_to(root):
            parts = Path(target).parts
            if "tools" not in parts:
                raise ValueError(f"link escapes payload: {link} -> {target}")
            resolved = root.joinpath(*parts[parts.index("tools"):]).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            raise ValueError(f"missing payload link target: {link} -> {target}")
        relative = os.path.relpath(resolved, directory)
        if relative != target:
            link.unlink()
            link.symlink_to(relative)
            count += 1
    for link in directory.iterdir():
        if link.is_symlink() and (not link.resolve().is_relative_to(root) or not link.exists()):
            raise ValueError(f"invalid relative payload link: {link}")
    return count


def render_wrapper(entry: str, repo: str, site: str) -> str:
    module, func = entry.split(":", 1)
    text = (Path(__file__).with_name("launcher_wrapper.py")).read_text(encoding="utf-8-sig")
    for key, value in {"ENTRY_MODULE": module, "ENTRY_FUNC": func, "REPO_REL": repo, "SITE_REL": site}.items():
        if '"' in value or "\n" in value or "__" in value:
            raise ValueError(f"invalid launcher value: {key}")
        text = text.replace(f"__HERMES_{key}__", value)
    unresolved = re.search(r"__HERMES_\w+?__", text)
    if unresolved:
        raise ValueError(f"unresolved launcher placeholder: {unresolved.group()}")
    return text


def posix_launcher(name: str, entry: str, *, python: str, repo: str, site: str, target: str) -> str:
    import shlex

    module, func = entry.split(":", 1)
    bionic = target.endswith("-bionic")
    header = "#!/data/data/com.termux/files/usr/bin/sh" if bionic else "#!/usr/bin/env bash"
    extra = ""
    if bionic:
        extra = '''PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
export PREFIX
export LD_LIBRARY_PATH="$root/tools/python/data/data/com.termux/files/usr/lib:$root/tools/node/data/data/com.termux/files/usr/lib:$root/tools/ffmpeg/data/data/com.termux/files/usr/lib:$root/runtime-libs/lib:$PREFIX/lib"
export HERMES_PYTHON_SRC_ROOT="$REPO"
export HERMES_PYTHON="$PYTHON"
export HERMES_NODE="$root/tools/node/data/data/com.termux/files/usr/bin/node"
export HERMES_RUNTIME_DIR="$root/tools"
export PATH="$root/tools/npm/bin:$root/tools/node/data/data/com.termux/files/usr/bin:$root/tools/ffmpeg/data/data/com.termux/files/usr/bin:$root/tools/ripgrep:$PATH"
'''
    code = f"import sys; sys.argv[0]={name!r}; from {module} import {func}; sys.exit({func}())"
    return f'''{header}
set -eu
self="$0"
while [ -L "$self" ]; do
    target="$(readlink "$self")"
    case "$target" in
        /*) self="$target" ;;
        *) self="$(dirname "$self")/$target" ;;
    esac
done
root="$(cd "$(dirname "$self")/.." && pwd)"
PYTHON="$root/{python}"
REPO="$root/{repo}"
SITE="$root/{site}"
[ -x "$PYTHON" ] || {{ printf '%s\\n' 'Bundled interpreter missing; reinstall Hermes.' >&2; exit 2; }}
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="$REPO:$SITE"
export PYTHONPYCACHEPREFIX="${{PYTHONPYCACHEPREFIX:-${{XDG_CACHE_HOME:-$HOME/.cache}}/hermes-pycache}}"
{extra}exec "$PYTHON" -P -c {shlex.quote(code)} "$@"
'''


def stage_launchers(root: Path, manifest: dict, *, run=subprocess.run) -> list[str]:
    from pm.lock import Facts
    from pm.registry import get_package

    target = manifest["target"]
    repo = root / manifest["repo"]
    entries = project_entries(repo)
    facts = Facts(root / manifest["store"] / "facts.json")
    python_fact = facts.get("python")
    if not python_fact:
        raise ValueError("payload has no Python fact")
    python = get_package("python").binary(root / manifest["store"] / python_fact["entry"], target)
    if python is None or not python.is_file():
        raise FileNotFoundError("payload interpreter missing")
    windows = target.startswith("win32")
    minor = python_fact["version"].split("+")[0].rsplit(".", 1)[0]
    site = f'{manifest["venv"]}/' + ("Lib/site-packages" if windows else f"lib/python{minor}/site-packages")
    bindir = root / "bin"
    if bindir.exists():
        shutil.rmtree(bindir)
    bindir.mkdir(parents=True, exist_ok=True)
    relative_python = python.relative_to(root).as_posix()
    for name, entry in entries.items():
        if windows:
            with tempfile.TemporaryDirectory(prefix="hermes-mint-") as temp:
                wrapper = Path(temp) / "wrapper.py"
                wrapper.write_text(render_wrapper(entry, f'../{manifest["repo"]}', f"../{site}"), encoding="utf-8")
                module, func = entry.split(":", 1)
                env = {**os.environ, "HERMES_MINT_BIN_DIR": str(bindir),
                       "HERMES_MINT_SPECS": json.dumps([{"name": name, "module": module, "func": func}]),
                       "HERMES_MINT_WRAPPER": str(wrapper),
                       "HERMES_MINT_PYTHON": "<launcher_dir>\\..\\" + relative_python.replace("/", "\\")}
                run([str(python), str(Path(__file__).with_name("mint_launchers.py"))], env=env, check=True)
        else:
            # Termux's prefix is contractual; its venv interpreter is built at that prefix.
            launch_python = f'{manifest["venv"]}/bin/python' if target.endswith("-bionic") else relative_python
            script = posix_launcher(name, entry, python=launch_python, repo=manifest["repo"], site=site, target=target)
            output = bindir / name
            output.write_text(script, encoding="utf-8")
            output.chmod(0o755)
    # Consumers receive a completed launch contract, not a Python-layout puzzle.
    if not (root / site).is_dir():
        raise FileNotFoundError(f"payload dependency tree missing: {site}")
    commands = {name: f"bin/{name}{'.exe' if windows else ''}" for name in entries}
    for command in commands.values():
        if not (root / command).is_file():
            raise FileNotFoundError(f"payload launcher missing: {command}")
    manifest["launchers"] = list(entries)
    manifest["runtime"] = {
        "repoDir": manifest["repo"], "toolsDir": manifest["store"],
        "storePython": relative_python, "sitePackages": site, "commands": commands,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return list(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launchers", "relocate", "surfaces", "rehash"])
    parser.add_argument("payload", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--tui-only", action="store_true")
    parser.add_argument("--repo-dir", help="staged repository directory when no manifest exists yet")
    args = parser.parse_args()
    if args.action == "rehash":
        print(f"rehashed {rehash_tools(args.payload)} payload tools")
        return
    if args.action == "relocate":
        relativize_links(args.payload)
        return
    if args.action == "surfaces" and args.repo_dir:
        plant_surfaces(args.payload / args.repo_dir, args.source, dashboard=not args.tui_only)
        return
    manifest = json.loads((args.payload / "manifest.json").read_text(encoding="utf-8-sig"))
    if args.action == "launchers":
        stage_launchers(args.payload.resolve(), manifest)
    else:
        plant_surfaces(args.payload / manifest["repo"], args.source, dashboard=not args.tui_only)


if __name__ == "__main__":
    main()
