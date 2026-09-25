"""Contract tests for gateway/run.py::_ensure_windows_gateway_venv_imports.

The regression these pin: on a PM install (store python + committed generation, VIRTUAL_ENV
popped by activation) the gateway adopted the pre-PM in-tree ``<root>/venv`` — built by a
DIFFERENT interpreter — as its site-packages and put it at ``sys.path[0]``. Its pure-Python
packages imported fine while their compiled submodules could not load, so every gateway
turn died with "Failed to initialize OpenAI client: No module named 'pydantic_core'".
"""

import sys
from pathlib import Path

import pytest

from gateway import run as gateway_run


def _foreign_version() -> str:
    """A version_info guaranteed to differ from the running interpreter's minor version."""
    major, minor = sys.version_info[:2]
    return f"{major}.{minor + 1}.0"


def _running_version() -> str:
    return ".".join(str(p) for p in sys.version_info[:3])


def _write_venv(venv_dir: Path, version_info: str | None) -> Path:
    """Create a minimal venv tree; returns the venv dir (the value the resolver consumes)."""
    venv_dir.mkdir(parents=True, exist_ok=True)
    (venv_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    if version_info is not None:
        (venv_dir / "pyvenv.cfg").write_text(
            f"home = C:\\python\nimplementation = CPython\nversion_info = {version_info}\n",
            encoding="utf-8",
        )
    return venv_dir


def _site(venv_dir: Path) -> str:
    return str(venv_dir / "Lib" / "site-packages")


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    """A PM-style install: a committed generation plus a foreign-ABI in-tree ``venv``."""
    root = tmp_path / "hermes-agent"
    (root / "gateway").mkdir(parents=True)
    generation = _write_venv(
        tmp_path / "installs" / "abc" / "environments" / "gen1" / "venv", _running_version()
    )
    # The pre-PM tree: unusable for this interpreter, so it must never reach sys.path.
    stale = _write_venv(root / "venv", _foreign_version())
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    return root, generation, stale


def _run(monkeypatch, committed):
    """Invoke the repair with the committed selection stubbed; return the sys.path entries added."""
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: committed)
    saved = list(sys.path)
    try:
        gateway_run._ensure_windows_gateway_venv_imports()
        return [p for p in sys.path if p not in saved]
    finally:
        sys.path[:] = saved


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only sys.path repair")
class TestEnsureWindowsGatewayVenvImports:
    def test_committed_generation_wins_over_foreign_in_tree_venv(self, fake_install, monkeypatch):
        """The committed environment is used; the foreign-ABI tree never reaches sys.path."""
        _root, generation, stale = fake_install
        monkeypatch.setenv("VIRTUAL_ENV", str(stale))
        added = _run(monkeypatch, generation)
        assert _site(generation) in added
        assert _site(stale) not in sys.path

    def test_foreign_in_tree_venv_is_refused_without_a_selection(self, fake_install, monkeypatch):
        """No committed selection and nothing usable in VIRTUAL_ENV: the foreign tree stays out."""
        _root, _generation, stale = fake_install
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        _run(monkeypatch, None)
        assert _site(stale) not in sys.path

    def test_legacy_venv_is_adopted_when_it_matches_the_running_python(
        self, fake_install, monkeypatch
    ):
        """A same-version in-tree venv is still honoured (the pre-PM contract)."""
        root, _generation, _stale = fake_install
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        _write_venv(root / "venv", _running_version())
        added = _run(monkeypatch, None)
        assert _site(root / "venv") in added

    def test_foreign_venv_in_the_environment_is_not_adopted(self, fake_install, monkeypatch):
        """A VIRTUAL_ENV naming a foreign-ABI tree is refused, not blindly trusted.

        The generated Windows launchers set VIRTUAL_ENV themselves, so it can name a tree
        this interpreter cannot load; trusting it unqualified re-creates the same crash.
        """
        _root, _generation, stale = fake_install
        monkeypatch.setenv("VIRTUAL_ENV", str(stale))
        _run(monkeypatch, None)
        assert _site(stale) not in sys.path


class TestVenvMatchesRunningPython:
    def test_unreadable_manifest_is_allowed(self, tmp_path):
        """A missing pyvenv.cfg cannot be judged — never a new refusal."""
        assert gateway_run._windows_venv_matches_running_python(tmp_path / "nope") is True

    def test_manifest_without_version_info_is_allowed(self, tmp_path):
        venv = tmp_path / "v"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = C:\\python\n", encoding="utf-8")
        assert gateway_run._windows_venv_matches_running_python(venv) is True

    def test_foreign_minor_version_is_refused(self, tmp_path):
        venv = tmp_path / "v"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text(f"version_info = {_foreign_version()}\n", encoding="utf-8")
        assert gateway_run._windows_venv_matches_running_python(venv) is False

    def test_matching_minor_version_is_accepted(self, tmp_path):
        venv = tmp_path / "v"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text(f"version_info = {_running_version()}\n", encoding="utf-8")
        assert gateway_run._windows_venv_matches_running_python(venv) is True