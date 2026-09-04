"""Unit tests for the bundled payload's win32 launcher wrapper.

scripts/desktop-cli/launcher-wrapper.py is the zip overlay a distlib
ScriptMaker packs into every minted CLI launcher exe (the rust shim's
replacement). The wrapper is import-safe on purpose, so these tests drive
its real logic — path resolution, sys.path order, the pycache_prefix
default — plus a real subprocess dispatch through a rendered copy.

The mechanism itself (distlib minting a PE whose shebang carries the
<launcher_dir> placeholder) is exercised end-to-end by
test_mint_launchers.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_WRAPPER = _REPO / "scripts" / "desktop-cli" / "launcher-wrapper.py"


def _load(env=None):
    """Import the wrapper with its placeholders substituted, like the build
    script does, and return its module namespace."""
    text = _WRAPPER.read_text(encoding="utf-8")
    for placeholder, value in {
        "__HERMES_ENTRY_MODULE__": "stubmod.entry",
        "__HERMES_ENTRY_FUNC__": "main",
        "__HERMES_REPO_REL__": "../repo",
        "__HERMES_SITE_REL__": "../venv/Lib/site-packages",
    }.items():
        text = text.replace(placeholder, value)
    namespace: dict = {"__name__": "launcher_wrapper_under_test"}
    exec(compile(text, str(_WRAPPER), "exec"), namespace)  # noqa: S102 - test fixture
    return namespace


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_launcher_dir_resolves_from_argv0(tmp_path):
    ns = _load()
    argv0 = tmp_path / "bin" / "hermes.exe"
    assert ns["launcher_dir"](str(argv0)) == str(tmp_path / "bin")


def test_payload_sys_paths_order_repo_first(tmp_path):
    ns = _load()
    here = str(tmp_path / "bin")
    entries = ns["payload_sys_paths"](here)
    # Repo first — its hermes_cli wins over anything stale in the venv
    # (the sealed payload has no working editable install). Same order the
    # old rust shim composed PYTHONPATH in.
    assert entries == [
        os.path.join(here, "..", "repo"),
        os.path.join(here, "..", "venv", "Lib", "site-packages"),
    ]


def test_payload_sys_paths_accepts_forward_slash_rels_on_any_platform(tmp_path):
    """The layout facts ride in the wrapper as forward-slash strings (the
    payload manifest convention); splitting must not depend on os.sep."""
    ns = _load()
    here = str(tmp_path / "bin")
    for entry in ns["payload_sys_paths"](here):
        assert os.path.isabs(os.path.normpath(entry))


# ---------------------------------------------------------------------------
# configure(): env hygiene, sys.path, pycache_prefix default
# ---------------------------------------------------------------------------


def test_configure_prepends_repo_then_site_packages(tmp_path):
    ns = _load()
    here = str(tmp_path / "bin")
    original = list(sys.path)
    try:
        entries = ns["configure"](here, environ={})
        assert sys.path[:2] == entries
        assert sys.path[0] == os.path.join(here, "..", "repo")
        assert sys.path[1] == os.path.join(here, "..", "venv", "Lib", "site-packages")
    finally:
        sys.path[:] = original


def test_configure_drops_inherited_pythonpath_and_pythonhome(tmp_path):
    ns = _load()
    original = list(sys.path)
    environ = {"PYTHONPATH": "/foreign/install", "PYTHONHOME": "/foreign/python"}
    try:
        ns["configure"](str(tmp_path), environ=environ)
        assert "PYTHONPATH" not in environ
        assert "PYTHONHOME" not in environ
    finally:
        sys.path[:] = original


def test_configure_defaults_pycache_prefix_to_localappdata(tmp_path, monkeypatch):
    ns = _load()
    environ = {"LOCALAPPDATA": str(tmp_path / "lad")}
    original = sys.pycache_prefix
    try:
        ns["configure"](str(tmp_path), environ=environ)
        expected = os.path.join(str(tmp_path / "lad"), "hermes", "pycache")
        assert environ["PYTHONPYCACHEPREFIX"] == expected
        # The env var alone cannot retro-activate; sys.pycache_prefix is
        # the live switch (the rust shim set it on the CHILD's environment).
        assert str(sys.pycache_prefix) == expected
    finally:
        sys.pycache_prefix = original


def test_configure_keeps_a_user_set_pycache_prefix(tmp_path, monkeypatch):
    ns = _load()
    environ = {"LOCALAPPDATA": str(tmp_path / "lad"), "PYTHONPYCACHEPREFIX": str(tmp_path / "mine")}
    original = sys.pycache_prefix
    try:
        ns["configure"](str(tmp_path), environ=environ)
        assert environ["PYTHONPYCACHEPREFIX"] == str(tmp_path / "mine")
        assert sys.pycache_prefix == original  # untouched
    finally:
        sys.pycache_prefix = original


def test_configure_without_localappdata_leaves_pycache_alone(tmp_path):
    ns = _load()
    environ: dict = {}
    original = sys.pycache_prefix
    try:
        ns["configure"](str(tmp_path), environ=environ)
        assert "PYTHONPYCACHEPREFIX" not in environ
        assert sys.pycache_prefix == original
    finally:
        sys.pycache_prefix = original


# ---------------------------------------------------------------------------
# Real dispatch: run a rendered copy as a subprocess against a stub module
# ---------------------------------------------------------------------------


def _render(tmp_path: Path) -> Path:
    text = _WRAPPER.read_text(encoding="utf-8")
    for placeholder, value in {
        "__HERMES_ENTRY_MODULE__": "hermes_cli.main",
        "__HERMES_ENTRY_FUNC__": "main",
        "__HERMES_REPO_REL__": "../repo",
        "__HERMES_SITE_REL__": "../venv/Lib/site-packages",
    }.items():
        text = text.replace(placeholder, value)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    out = bin_dir / "hermes.exe"
    out.write_text(text, encoding="utf-8")
    return out


@pytest.mark.skipif(sys.platform != "win32", reason="minted launchers only exist on win32 payloads")
def test_rendered_wrapper_dispatches_to_the_entry_module(tmp_path):
    """The full wrapper contract without distlib: a rendered copy placed in
    bin/ next to a stub repo + venv resolves its own dir from sys.argv[0],
    imports hermes_cli.main off the payload paths, and returns main()'s
    exit code."""
    bin_dir = tmp_path / "bin"
    (bin_dir / ".." / "repo" / "hermes_cli").mkdir(parents=True, exist_ok=True)
    (bin_dir / ".." / "venv" / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (bin_dir / ".." / "repo" / "hermes_cli" / "__init__.py").write_text("")
    (bin_dir / ".." / "repo" / "hermes_cli" / "main.py").write_text(
        "import sys\n"
        "def main():\n"
        "    assert sys.argv[0].endswith('hermes.exe'), sys.argv[0]\n"
        "    assert sys.argv[1:] == ['--version']\n"
        "    import stubdep  # importable only via the venv site-packages entry\n"
        "    return 7\n"
    )
    (bin_dir / ".." / "venv" / "Lib" / "site-packages" / "stubdep.py").write_text("X = 1\n")

    wrapper = _render(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX")}
    env["LOCALAPPDATA"] = str(tmp_path / "lad")
    python = Path(sys.base_prefix) / "python.exe" if sys.platform == "win32" else sys.executable
    proc = subprocess.run([str(python), str(wrapper), "--version"], capture_output=True, text=True, env=env)
    assert proc.returncode == 7, proc.stderr
