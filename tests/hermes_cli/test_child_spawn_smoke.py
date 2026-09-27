"""The post-publish child-spawn smoke detector.

The detector exists because of 2026-09-25: a publish embedded the
dependency-less store interpreter, the gateway kept working (hermes_bootstrap
patches ``sys.path`` in-process), and every child -- kanban workers, cron
workers -- died with ``ModuleNotFoundError`` for twelve minutes before a human
read the cron error stream.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from hermes_cli import child_spawn_smoke, venv_sync
from hermes_cli import _launchers
from pm.package import InstallError


def _stub_interpreter(directory: Path, *, body: str) -> Path:
    """A POSIX 'interpreter' the probe can spawn and read the answer from."""
    path = directory / "python-stub"
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.platforms("posix")
def test_probe_reports_the_missing_module(tmp_path):
    interpreter = _stub_interpreter(
        tmp_path,
        body="echo \"ModuleNotFoundError: No module named 'ruamel'\" >&2\nexit 1",
    )
    result = child_spawn_smoke.probe_child_spawn(interpreter, timeout=30)

    assert not result.ok
    assert result.missing_module == "ruamel"
    assert [check.returncode for check in result.checks] == [1, 1]
    # Both children the supervisor spawns are exercised: the module entry point
    # and the import probe.
    assert result.checks[0].argv[-3:] == ("-m", "hermes_cli.main", "--version")
    assert ", ".join(child_spawn_smoke.PROBE_MODULES) in result.checks[1].argv[-1]
    # The two entry paths a supervisor re-spawns as `sys.executable -m <module>`:
    # kanban workers, and external cron workers (cron/scheduler.py).
    assert {"hermes_cli.main", "cron.scheduler"} <= set(child_spawn_smoke.PROBE_MODULES)


@pytest.mark.platforms("posix")
def test_probe_passes_when_the_child_runs(tmp_path):
    interpreter = _stub_interpreter(tmp_path, body="echo hermes 0.0.0-fixture\nexit 0")
    result = child_spawn_smoke.probe_child_spawn(interpreter, timeout=30)

    assert result.ok
    assert result.missing_module is None
    assert not result.timed_out


@pytest.mark.platforms("posix")
def test_probe_treats_a_hanging_child_as_broken(tmp_path):
    interpreter = _stub_interpreter(tmp_path, body="sleep 30")
    result = child_spawn_smoke.probe_child_spawn(interpreter, timeout=0.5)

    assert not result.ok
    assert result.timed_out


@pytest.mark.platforms("posix")
def test_probe_child_runs_from_a_bare_cwd(tmp_path):
    """A child run from the checkout would import hermes_cli from the cwd."""
    interpreter = _stub_interpreter(tmp_path, body="pwd\nexit 0")
    result = child_spawn_smoke.probe_child_spawn(interpreter, timeout=30)

    cwd = Path(result.checks[0].summary)
    assert cwd.is_dir()
    assert cwd != Path.cwd()
    assert not (cwd / "hermes_cli").exists()


@pytest.mark.platforms("posix")
def test_failure_message_names_the_interpreter_and_the_module(tmp_path):
    interpreter = _stub_interpreter(
        tmp_path,
        body="echo \"ModuleNotFoundError: No module named 'hermes_cli'\" >&2\nexit 1",
    )
    launcher = tmp_path / "hermes"
    result = child_spawn_smoke.probe_child_spawn(interpreter, timeout=30)

    cause, remedy = child_spawn_smoke.render_failure(result, launcher=launcher)
    assert str(interpreter) in cause
    assert "missing module: hermes_cli" in cause
    assert str(launcher) in cause
    assert "-m hermes_cli.main" in cause
    assert "hermes pm repair" in remedy


@pytest.mark.platforms("posix")
def test_verify_raises_when_the_published_runtime_cannot_spawn_a_child(tmp_path, monkeypatch):
    interpreter = _stub_interpreter(
        tmp_path,
        body="echo \"ModuleNotFoundError: No module named 'cron'\" >&2\nexit 1",
    )
    monkeypatch.setattr(child_spawn_smoke, "committed_generation_python", lambda root: interpreter)
    monkeypatch.setattr(child_spawn_smoke, "published_runtime", lambda root, launcher=None: interpreter)
    launcher = tmp_path / ".hermes" / "bin" / "hermes"

    with pytest.raises(InstallError) as raised:
        child_spawn_smoke.verify_published_runtime(tmp_path)
    message = str(raised.value)
    assert str(interpreter) in message
    assert "cron" in message
    assert str(launcher) in message


@pytest.mark.platforms("posix")
def test_verify_probes_the_store_interpreter_row_that_used_to_be_skipped(tmp_path, monkeypatch, caplog):
    """No committed generation IS the state that publishes the store interpreter.

    The probe is the standing detector for 2026-09-25, so it has to run on the row
    that fault lived in: an install with no usable generation embeds the
    dependency-less store Python. Raising there would fail installs that are
    behaving correctly, so the failure is a logged health signal -- measured, not
    skipped.
    """
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    # No select_generation(): the fixture tree's own state, and the gate that used
    # to bail out of the check before the probe could run.
    assert child_spawn_smoke.committed_generation_python(repo) is None
    dep_less = _stub_interpreter(
        tmp_path,
        body="echo \"ModuleNotFoundError: No module named 'hermes_cli'\" >&2\nexit 1",
    )
    out = repo / ".hermes" / "bin"
    out.mkdir(parents=True, exist_ok=True)
    launcher = _launchers.mint_launcher("hermes", repo, out, dep_less, None)
    assert launcher is not None

    with caplog.at_level(logging.WARNING):
        result = child_spawn_smoke.verify_published_runtime(repo)  # must not raise

    assert result is not None, "the check skipped the row it exists for"
    assert not result.ok
    assert result.missing_module == "hermes_cli"
    # The runtime under test was read from the published launcher file, not guessed.
    assert result.python == dep_less
    assert "health signal, not a failed install" in caplog.text


@pytest.mark.platforms("posix")
def test_verify_raises_when_a_committed_generations_runtime_cannot_spawn(tmp_path, monkeypatch):
    """A generation IS committed: its interpreter owning the dependencies is the promise."""
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    dep_less = _stub_interpreter(
        tmp_path,
        body="echo \"ModuleNotFoundError: No module named 'hermes_cli'\" >&2\nexit 1",
    )
    monkeypatch.setattr(child_spawn_smoke, "committed_generation_python", lambda root: dep_less)
    out = repo / ".hermes" / "bin"
    out.mkdir(parents=True, exist_ok=True)
    launcher = _launchers.mint_launcher("hermes", repo, out, dep_less, None)
    assert launcher is not None

    with pytest.raises(InstallError) as raised:
        child_spawn_smoke.verify_published_runtime(repo)

    message = str(raised.value)
    assert str(dep_less) in message
    assert "hermes_cli" in message
    assert str(launcher) in message


@pytest.mark.platforms("posix")
def test_verify_passes_a_launcher_whose_runtime_spawns_a_child(tmp_path, monkeypatch):
    """The passing verdict has to survive the read-back: a good runtime stays good."""
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    working = _stub_interpreter(tmp_path, body="echo hermes 0.0.0-fixture\nexit 0")
    monkeypatch.setattr(child_spawn_smoke, "committed_generation_python", lambda root: working)
    out = repo / ".hermes" / "bin"
    out.mkdir(parents=True, exist_ok=True)
    assert _launchers.mint_launcher("hermes", repo, out, working, None) is not None

    result = child_spawn_smoke.verify_published_runtime(repo)

    assert result is not None and result.ok
    assert result.python == working
    assert not result.timed_out


@pytest.mark.platforms("posix")
def test_verify_never_answers_from_the_selection_record(tmp_path, monkeypatch, caplog):
    """A readable record is not a read-back: with no launcher file, report that."""
    monkeypatch.setattr(child_spawn_smoke, "committed_generation_python",
                        lambda root: Path(sys.executable))
    monkeypatch.setattr(_launchers, "resolve_launcher_python", lambda root: Path(sys.executable))

    with caplog.at_level(logging.WARNING):
        result = child_spawn_smoke.verify_published_runtime(tmp_path)

    assert result is None
    assert "could not read an interpreter" in caplog.text


@pytest.mark.platforms("posix")
def test_published_runtime_python_reads_the_launcher_body(tmp_path):
    embedded = tmp_path / "gen" / "bin" / "python"
    embedded.parent.mkdir(parents=True)
    embedded.write_text("", encoding="utf-8")

    guarded = tmp_path / "hermes"
    guarded.write_text(
        _launchers._guarded_shell_body(embedded, Path("/store/python3"),
                                       _launchers._launcher_script("hermes", tmp_path, None)),
        encoding="utf-8",
    )
    assert _launchers.published_runtime_python(guarded) == embedded

    plain = tmp_path / "hermes-plain"
    plain.write_text(f'#!/bin/sh\nexec {embedded} -I -c "x" "$@"\n', encoding="utf-8")
    assert _launchers.published_runtime_python(plain) == embedded

    # A guarded body whose runtime is gone answers with the fallback it would
    # exec, so a probe still tests the interpreter children would really get.
    absent = tmp_path / "collected" / "bin" / "python"
    fallback = tmp_path / "store" / "bin" / "python3"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("", encoding="utf-8")
    degraded = tmp_path / "hermes-degraded"
    degraded.write_text(
        _launchers._guarded_shell_body(absent, fallback,
                                       _launchers._launcher_script("hermes", tmp_path, None)),
        encoding="utf-8",
    )
    assert _launchers.published_runtime_python(degraded) == fallback

    assert _launchers.published_runtime_python(tmp_path / "missing") is None


def test_published_launcher_resolves_the_name_each_platform_writes(tmp_path):
    """Windows publishes `hermes.exe` or `hermes.cmd`; there is no suffix-less one."""
    out = tmp_path / ".hermes" / "bin"
    out.mkdir(parents=True)

    assert _launchers.published_launcher(tmp_path, windows=False) == out / "hermes"
    assert _launchers.published_launcher(tmp_path, windows=True) == out / "hermes.exe"

    (out / "hermes.cmd").write_text("@echo off\r\n", encoding="utf-8")
    assert _launchers.published_launcher(tmp_path, windows=True) == out / "hermes.cmd"

    (out / "hermes.exe").write_bytes(b"MZ")
    assert _launchers.published_launcher(tmp_path, windows=True) == out / "hermes.exe"
    assert _launchers.published_launcher(tmp_path, windows=False) == out / "hermes"


def test_published_runtime_python_reads_a_windows_trampoline(tmp_path):
    """distlib writes ``launcher + shebang + zip payload``; the shebang is the runtime."""
    interpreter = tmp_path / "gen" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"MZ")

    launcher = tmp_path / "hermes.exe"
    launcher.write_bytes(
        b"MZ\x90\x00" + b"\x00" * 64
        + f'#!"{interpreter}" -I'.encode("utf-8")
        # A payload is compressed bytes that could match anything: it is never read.
        + b"PK\x03\x04" + b"not the loader: #!C:\\wrong\\python.exe -I"
    )

    assert _launchers.published_runtime_python(launcher) == interpreter


def test_published_runtime_python_reads_a_windows_cmd_launcher(tmp_path):
    """The no-distlib fallback quotes its interpreter on the first command line."""
    interpreter = tmp_path / "gen" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"MZ")

    launcher = tmp_path / "hermes.cmd"
    launcher.write_text(
        f'@echo off\r\n"{interpreter}" -I -c "import base64" %*\r\n', encoding="utf-8",
    )

    assert _launchers.published_runtime_python(launcher) == interpreter
    assert _launchers.published_runtime_python(tmp_path / "absent.cmd") is None


@pytest.mark.platforms("posix")
def test_publish_launchers_runs_the_child_probe(tmp_path, monkeypatch):
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(child_spawn_smoke, "verify_published_runtime",
                        lambda root, **kw: seen.append(Path(root)))

    venv_sync.publish_launchers(repo)

    assert seen == [repo]
    assert (repo / ".hermes" / "bin" / "hermes").is_file()


@pytest.mark.platforms("posix")
def test_publish_launchers_passes_an_install_that_never_committed_a_generation(tmp_path, monkeypatch):
    """The real verdict path, unpatched: a fixture tree has no usable generation.

    The probe now runs here (it used to bail before it) and its failure is a health
    warning, so a publish that is behaving correctly still must not raise.
    """
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)

    assert child_spawn_smoke.committed_generation_python(repo) is None
    venv_sync.publish_launchers(repo)  # must not raise


@pytest.mark.platforms("posix")
def test_cli_reports_the_verdict_on_a_launcher_that_embeds_a_dep_less_python(tmp_path, monkeypatch, capsys):
    """Acceptance shape: point the launcher at a runtime that cannot import."""
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree, select_generation

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    select_generation(repo, "fixture", "ready")
    dep_less = _stub_interpreter(
        tmp_path,
        body="echo \"ModuleNotFoundError: No module named 'hermes_cli'\" >&2\nexit 1",
    )
    # A committed, usable generation is what makes the check applicable (the
    # fixture's stand-in); the runtime under test is the one the LAUNCHER embeds.
    monkeypatch.setattr(child_spawn_smoke, "committed_generation_python", lambda root: dep_less)
    out = repo / ".hermes" / "bin"
    out.mkdir(parents=True)
    _launchers.mint_launcher("hermes", repo, out, dep_less, None)

    assert child_spawn_smoke.main([str(repo), "--json"]) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["ok"] is False
    assert verdict["missing_module"] == "hermes_cli"
    assert verdict["python"] == str(dep_less)
    assert verdict["launcher"] == str(out / "hermes")
