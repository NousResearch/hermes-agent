"""Shared payload launcher rendering; all paths are supplied by the assembler."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import subprocess
import tempfile


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

    def shell_path(value: str) -> str:
        return shlex.quote(value) if Path(value).is_absolute() else '"$root"/' + shlex.quote(value)

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
PYTHON={shell_path(python)}
REPO={shell_path(repo)}
SITE={shell_path(site)}
[ -x "$PYTHON" ] || {{ printf '%s\\n' 'Bundled interpreter missing; reinstall Hermes.' >&2; exit 2; }}
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="$REPO:$SITE"
export PYTHONPYCACHEPREFIX="${{PYTHONPYCACHEPREFIX:-${{XDG_CACHE_HOME:-$HOME/.cache}}/hermes-pycache}}"
{extra}exec "$PYTHON" -P -c {shlex.quote(code)} "$@"
'''


def write_launchers(root: Path, entries: dict[str, str], *, python: str,
                    repo: str, site: str, target: str, bin_dir: str = "bin",
                    run=subprocess.run) -> dict[str, str]:
    """Mint only declared commands; the caller owns completion publication."""
    bindir = root / bin_dir
    bindir.mkdir(parents=True, exist_ok=True)
    windows = target.startswith("win32")
    for name, entry in entries.items():
        if windows:
            with tempfile.TemporaryDirectory(prefix="hermes-mint-") as temp:
                wrapper = Path(temp) / "wrapper.py"
                wrapper.write_text(render_wrapper(entry, f"../{repo}", f"../{site}"), encoding="utf-8")
                module, func = entry.split(":", 1)
                env = {**os.environ, "HERMES_MINT_BIN_DIR": str(bindir),
                       "HERMES_MINT_SPECS": json.dumps([{"name": name, "module": module, "func": func}]),
                       "HERMES_MINT_WRAPPER": str(wrapper),
                       "HERMES_MINT_PYTHON": "<launcher_dir>\\..\\" + python.replace("/", "\\")}
                run([str(root / python), str(Path(__file__).with_name("mint_launchers.py"))], env=env, check=True)
        else:
            script = posix_launcher(name, entry, python=python, repo=repo, site=site, target=target)
            output = bindir / name
            output.write_text(script, encoding="utf-8")
            output.chmod(0o755)
    commands = {name: f"{bin_dir}/{name}{'.exe' if windows else ''}" for name in entries}
    for command in commands.values():
        if not (root / command).is_file():
            raise FileNotFoundError(f"payload launcher missing: {command}")
    return commands
