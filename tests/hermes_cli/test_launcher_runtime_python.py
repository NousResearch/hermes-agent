"""A published launcher must embed an interpreter its children can import from.

2026-09-25: the PM publish of 13:32 embedded the *store* Python, whose
site-packages holds pip and nothing else. Every supervisor path re-spawns the
commands that launcher publishes as ``sys.executable -m hermes_cli.main``
(kanban workers, external cron workers) and such a child runs no launcher body,
so it enters neither the install root nor ``activate_dependencies``: workers died
on ``ModuleNotFoundError: No module named 'hermes_cli'`` for twelve minutes while
the gateway itself kept working, because the gateway booted before the publish.

These tests drive the real publication against a fabricated install state, so the
interpreter is resolved the way a sync resolves it -- from the selection record,
never from a literal path in the launcher's own writer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import _launchers
from pm.environments import install_key

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A stand-in CLI whose only job is to report the interpreter that executed it.
STUB_MAIN = (
    "import json\n"
    "import sys\n"
    "def main():\n"
    "    print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix}))\n"
    "    return 0\n"
)


class InstallFixture:
    """A source install with a committed dependency generation, laid out for real."""

    def __init__(self, tmp_path: Path, monkeypatch, *, generation: bool = True,
                 generation_has_interpreter: bool = True) -> None:
        home, store = tmp_path / "home", tmp_path / "store"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
        monkeypatch.setattr(Path, "home", lambda: home)

        self.root = tmp_path / "hermes-agent"
        self.home, self.store = home, store
        self.out = self.root / ".hermes" / "bin"
        self.out.mkdir(parents=True)

        # The application stand-in: real launcher production, an observable entry point.
        package = self.root / "hermes_cli"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "main.py").write_text(STUB_MAIN, encoding="utf-8")
        (self.root / "hermes_bootstrap.py").write_text("", encoding="utf-8")
        (self.root / "hermes_constants.py").write_text(
            "from pathlib import Path\n"
            "def get_default_hermes_root(home=None):\n"
            "    return Path(home or __import__('os').environ.get('HERMES_HOME', Path.home()))\n",
            encoding="utf-8",
        )
        (self.root / "pm").mkdir(exist_ok=True)
        (self.root / "pm" / "__init__.py").touch()

        version = f"{sys.version_info[0]}.{sys.version_info[1]}"
        self.generation_python: Path | None = None
        if generation:
            environment = home / "installs" / install_key(self.root) / "environments" / "gen" / "venv"
            (environment / "lib" / f"python{version}" / "site-packages").mkdir(parents=True)
            (environment / "pyvenv.cfg").write_text(
                f"home = {Path(sys._base_executable).parent}\n"
                f"version_info = {version}.0\n",
                encoding="utf-8",
            )
            if generation_has_interpreter:
                (environment / "bin").mkdir()
                self.generation_python = environment / "bin" / "python"
                self.generation_python.symlink_to(sys._base_executable)
            (home / "installs" / install_key(self.root) / "facts.json").write_text(
                json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8",
            )

        # The store interpreter, as _launchers resolves it: from the store's own facts.
        self.store_python = store / "python-managed" / "bin" / "python3"
        self.store_python.parent.mkdir(parents=True)
        self.store_python.symlink_to(sys._base_executable)
        (store / "facts.json").write_text(
            json.dumps({"packages": {"python": {"entry": "python-managed"}}}), encoding="utf-8",
        )

    def publish(self) -> Path:
        launcher = _launchers.stage_launcher("hermes", self.root, self.out)
        assert launcher is not None
        return launcher

    def run(self, launcher: Path) -> dict:
        result = subprocess.run([str(launcher), "--version"], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheEmbeddedInterpreter:
    def test_a_committed_generation_is_what_gets_embedded(self, tmp_path, monkeypatch):
        install = InstallFixture(tmp_path, monkeypatch)

        launcher = install.publish()

        body = launcher.read_text(encoding="utf-8")
        assert str(install.generation_python) in body
        assert _launchers.resolve_launcher_python(install.root) == install.generation_python

    def test_the_store_interpreter_is_only_the_fallback(self, tmp_path, monkeypatch):
        """The store interpreter is a complete runtime for the process, not for a child."""
        install = InstallFixture(tmp_path, monkeypatch)

        body = install.publish().read_text(encoding="utf-8")

        primary = body.index(str(install.generation_python))
        fallback = body.index(str(install.store_python))
        assert primary < fallback, body

    def test_nothing_committed_keeps_the_store_interpreter(self, tmp_path, monkeypatch):
        """A first install has no generation yet; the launcher body completes it."""
        install = InstallFixture(tmp_path, monkeypatch, generation=False)

        assert _launchers.resolve_launcher_python(install.root) == install.store_python
        assert str(install.store_python) in install.publish().read_text(encoding="utf-8")

    def test_a_generation_without_an_interpreter_keeps_the_store_interpreter(self, tmp_path, monkeypatch):
        """A half-built generation must not be published as the runtime."""
        install = InstallFixture(tmp_path, monkeypatch, generation_has_interpreter=False)

        assert _launchers.resolve_launcher_python(install.root) == install.store_python

    def test_an_unreadable_record_keeps_the_store_interpreter(self, tmp_path, monkeypatch):
        install = InstallFixture(tmp_path, monkeypatch)
        (install.home / "installs" / install_key(install.root) / "facts.json").write_text(
            "{not json", encoding="utf-8",
        )

        assert _launchers.resolve_launcher_python(install.root) == install.store_python

    def test_no_managed_interpreter_at_all_is_not_published(self, tmp_path, monkeypatch):
        install = InstallFixture(tmp_path, monkeypatch, generation=False)
        (install.store / "facts.json").unlink()

        assert _launchers.resolve_launcher_python(install.root) is None
        assert _launchers.stage_launcher("hermes", install.root, install.out) is None


@pytest.mark.platforms("posix")
class TestTheExecTimeGuard:
    def test_the_generation_interpreter_is_the_one_that_runs(self, tmp_path, monkeypatch):
        install = InstallFixture(tmp_path, monkeypatch)
        interpreter = install.generation_python
        assert interpreter is not None

        observed = install.run(install.publish())

        assert Path(observed["executable"]).resolve() == interpreter.resolve(), observed
        assert Path(observed["prefix"]).resolve() == interpreter.parent.parent.resolve()

    def test_a_collected_generation_still_runs_the_install(self, tmp_path, monkeypatch):
        """A pruned generation must not turn the launcher into 'command not found'.

        ``exec`` of a path that is no longer there dies before any of our code runs, so
        the fallback has to be in the shell body -- and it has to be a real interpreter,
        with a working child, not a plain exit.
        """
        install = InstallFixture(tmp_path, monkeypatch)
        launcher = install.publish()
        assert install.generation_python is not None
        install.generation_python.unlink()

        observed = install.run(launcher)

        assert Path(observed["executable"]).resolve() == install.store_python.resolve(), observed

    def test_neither_interpreter_is_a_legible_failure(self, tmp_path, monkeypatch):
        install = InstallFixture(tmp_path, monkeypatch)
        launcher = install.publish()
        assert install.generation_python is not None
        install.generation_python.unlink()
        install.store_python.unlink()

        result = subprocess.run([str(launcher)], capture_output=True, text=True, timeout=120)

        assert result.returncode != 0
        assert "no usable interpreter" in result.stderr, result

    def test_an_unchanged_launcher_is_not_rewritten(self, tmp_path, monkeypatch):
        """Publication runs at every boot path; a no-op must stay a no-op."""
        install = InstallFixture(tmp_path, monkeypatch)
        launcher = install.publish()
        os.utime(launcher, (0, 0))

        assert install.publish() == launcher
        assert launcher.stat().st_mtime == 0
