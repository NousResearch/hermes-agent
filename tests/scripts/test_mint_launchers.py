"""Live tests for the win32 launcher mint (scripts/desktop-cli/mint-launchers.py).

These RUN the real mint on the current interpreter and then execute the
minted launcher — the strongest proof the mechanism is intact (the research
experiments proved it once; these tests keep proving it on every run):

  * the minted exe is a real PE with a `#!<launcher_dir>\\..` shebang,
  * the <launcher_dir> placeholder resolves against the launcher's OWN
    directory (proved by moving the tree before running),
  * the wrapper puts the payload repo + venv site-packages on sys.path and
    dispatches to the entry module with argv forwarded and the exit code
    returned.

Skipped off-windows (POSIX payloads ship $0-relative bash trampolines
instead) and when neither distlib nor pip is importable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MINT = _REPO / "scripts" / "desktop-cli" / "mint-launchers.py"

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="distlib launcher minting is win32-only"),
]


def _distlib_available() -> bool:
    try:
        import distlib.scripts  # noqa: F401
        return True
    except ImportError:
        try:
            from pip._vendor import distlib  # noqa: F401
            return True
        except ImportError:
            return False


pytestmark.append(pytest.mark.skipif(not _distlib_available(), reason="no distlib (nor pip's vendored copy)"))


@pytest.fixture()
def payload_tree(tmp_path: Path):
    """A miniature payload: bin/, repo snapshot with a stub hermes_cli,
    venv site-packages, and a REAL interpreter at the fake store path."""
    bin_dir = tmp_path / "stage" / "bin"
    py_dir = tmp_path / "stage" / "tools" / "py-1"
    repo = tmp_path / "stage" / "repo"
    site = tmp_path / "stage" / "venv" / "Lib" / "site-packages"
    for d in (bin_dir, py_dir, repo / "hermes_cli", site):
        d.mkdir(parents=True, exist_ok=True)

    # A real interpreter at the shebang's target: the launcher will create
    # the process from <launcher_dir>\..\tools\py-1\python.exe, so the exe
    # must actually be able to boot there. Copy from the BASE interpreter —
    # a venv python.exe is a launcher stub whose DLLs/Lib live beside the
    # real one.
    base = Path(sys.base_prefix)
    for name in os.listdir(base):
        if name in ("Lib", "Scripts", "share", "include"):
            continue
        src = base / name
        if src.is_dir():
            shutil.copytree(src, py_dir / name)
        else:
            shutil.copy2(src, py_dir / name)
    shutil.copytree(
        base / "Lib",
        py_dir / "Lib",
        ignore=shutil.ignore_patterns("__pycache__", "site-packages", "test", "idlelib", "tkinter", "turtledemo", "config-*"),
    )

    (repo / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "hermes_cli" / "main.py").write_text(
        "import os, sys\n"
        "def main():\n"
        "    import hermes_cli, stubdep\n"
        "    assert sys.argv[0].lower().endswith('hermes.exe'), sys.argv[0]\n"
        "    assert sys.argv[1:] == ['--version'], sys.argv\n"
        "    print('OK', os.environ.get('PYTHONHOME'))\n"
        "    return 7\n",
        encoding="utf-8",
    )
    (site / "stubdep.py").write_text("X = 1\n", encoding="utf-8")

    return {"bin": bin_dir, "root": tmp_path / "stage", "tmp": tmp_path, "site": site, "repo": repo}


def _mint(bin_dir: Path, wrapper: Path, specs) -> list[str]:
    env = dict(
        os.environ,
        HERMES_MINT_BIN_DIR=str(bin_dir),
        HERMES_MINT_SPECS=json.dumps(specs),
        HERMES_MINT_WRAPPER=str(wrapper),
        HERMES_MINT_PYTHON=r"<launcher_dir>\..\tools\py-1\python.exe",
    )
    proc = subprocess.run([sys.executable, str(_MINT)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()


def _render_wrapper(tmp_path: Path) -> Path:
    """cli-entrypoints.mjs's renderWinWrapper, replicated (the .mjs cannot
    be imported from python) — same placeholders, same substitution."""
    text = (_REPO / "scripts" / "desktop-cli" / "launcher-wrapper.py").read_text(encoding="utf-8")
    for placeholder, value in {
        "__HERMES_ENTRY_MODULE__": "hermes_cli.main",
        "__HERMES_ENTRY_FUNC__": "main",
        "__HERMES_REPO_REL__": "../repo",
        "__HERMES_SITE_REL__": "../venv/Lib/site-packages",
    }.items():
        text = text.replace(placeholder, value)
    out = tmp_path / "wrapper.py"
    out.write_text(text, encoding="utf-8")
    return out


def test_mint_writes_exactly_one_named_exe_per_spec(payload_tree, tmp_path):
    wrapper = _render_wrapper(tmp_path)
    out = _mint(payload_tree["bin"], wrapper, [{"name": "hermes", "module": "hermes_cli.main", "func": "main"}])
    assert out == ["hermes.exe"] or out == [str(payload_tree["bin"] / "hermes.exe")]
    assert sorted(p.name for p in payload_tree["bin"].iterdir()) == ["hermes.exe"], "no versioned twin may remain"


def test_minted_launcher_shebang_carries_the_launcher_dir_placeholder(payload_tree, tmp_path):
    wrapper = _render_wrapper(tmp_path)
    _mint(payload_tree["bin"], wrapper, [{"name": "hermes", "module": "hermes_cli.main", "func": "main"}])
    exe = payload_tree["bin"] / "hermes.exe"
    blob = exe.read_bytes()
    shebang = blob[blob.rfind(b"#!"):]
    assert shebang.startswith(b"#!<launcher_dir>\\..\\tools\\py-1\\python.exe\n"), shebang[:60]


def test_minted_launcher_runs_relocated_and_forwards_exit_code(payload_tree, tmp_path):
    wrapper = _render_wrapper(tmp_path)
    _mint(payload_tree["bin"], wrapper, [{"name": "hermes", "module": "hermes_cli.main", "func": "main"}])

    # RELOCATE the whole tree before running: the launcher must resolve the
    # interpreter and the payload paths relative to its own dir, wherever
    # the install lands.
    moved = payload_tree["tmp"] / "moved-install"
    shutil.move(str(payload_tree["root"]), str(moved))
    exe = moved / "bin" / "hermes.exe"

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX")}
    env["LOCALAPPDATA"] = str(payload_tree["tmp"] / "lad")
    proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True, cwd=os.path.expanduser("~"), env=env)
    assert proc.returncode == 7, proc.stderr[-800:]
    assert proc.stdout.strip() == "OK None"  # PYTHONHOME was dropped

    # The default pycache prefix kept bytecode out of the (sealed) repo.
    assert (payload_tree["tmp"] / "lad" / "hermes" / "pycache").exists()
    assert not (moved / "repo" / "hermes_cli" / "__pycache__").exists()


def test_mint_rejects_a_non_launcher_dir_shebang(tmp_path):
    env = dict(
        os.environ,
        HERMES_MINT_BIN_DIR=str(tmp_path),
        HERMES_MINT_SPECS=json.dumps([{"name": "hermes", "module": "m", "func": "main"}]),
        HERMES_MINT_PYTHON=r"C:\abs\python.exe",
    )
    wrapper = tmp_path / "w.py"
    wrapper.write_text("x = 1\n", encoding="utf-8")
    env["HERMES_MINT_WRAPPER"] = str(wrapper)
    proc = subprocess.run([sys.executable, str(_MINT)], capture_output=True, text=True, env=env)
    assert proc.returncode != 0
    assert "launcher_dir" in proc.stderr
