"""Legacy pythonw launcher normalization + post-update launcher refresh.

Covers the two halves of the "legacy pythonw gateways survive updates
forever" gap:

1. ``gateway_windows._resolve_detached_python`` — normalizes a legacy
   ``pythonw.exe`` interpreter (pre-aa2ae36c3f launchers / argv snapshots)
   to the sibling console ``python.exe`` so respawns and regenerated
   launchers use the hidden-console design (#54220/#56747) and don't die
   with ``RuntimeError: sys.stderr is None`` (#71671).
2. ``hermes_cli.main._refresh_windows_gateway_launchers`` — ``hermes
   update`` regenerates the installed Scheduled Task / Startup launcher
   scripts instead of leaving install-time artifacts stale forever.

``_resolve_detached_python`` is a pure path helper and runs on any host.
``windowless_gateway_restart_spec`` returns its argv unchanged off Windows,
so the test that exercises the rewrite is ``windows_only`` rather than run
against a faked ``sys.platform``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.gateway  # noqa: F401  (see below)
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as cli_main

# ``hermes_cli.gateway`` binds ``get_hermes_home`` from ``hermes_cli.config`` at
# module level, so whichever test imports it first decides that binding for the
# whole session.  A test below patches ``hermes_cli.config.get_hermes_home``
# with a ``mock.patch`` context manager; if ``hermes_cli.gateway`` were first
# imported inside that ``with`` block it would keep the Mock after the block
# exits, and every later caller of ``_profile_suffix`` would get a ``str``
# instead of a ``Path``.  Importing it here makes the binding deterministic.


# ---------------------------------------------------------------------------
# _resolve_detached_python: legacy pythonw normalization
# ---------------------------------------------------------------------------


def _make_venv(tmp_path: Path, *, with_console_python: bool) -> tuple[Path, Path]:
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    pythonw = scripts / "pythonw.exe"
    pythonw.write_text("", encoding="utf-8")
    python = scripts / "python.exe"
    if with_console_python:
        python.write_text("", encoding="utf-8")
    return pythonw, python


def test_resolve_detached_python_swaps_legacy_pythonw_for_console_sibling(tmp_path):
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    exe, venv_dir, extra = gateway_windows._resolve_detached_python(str(pythonw))

    assert exe == str(python)
    assert venv_dir == tmp_path / "venv"
    assert extra == []




@pytest.mark.windows_only
def test_restart_spec_normalizes_legacy_pythonw_argv(tmp_path):
    """A pre-rework Scheduled Task argv snapshot (leading pythonw.exe) must be
    respawned through the console python + hidden-console launch, with every
    argument after the interpreter preserved verbatim.

    ``windows_only``: ``windowless_gateway_restart_spec`` returns the argv
    untouched off Windows, so the fake was the only thing making the rewrite
    (and its ``Scripts/``-layout venv derivation) run at all.
    """
    pythonw, python = _make_venv(tmp_path, with_console_python=True)

    argv = [str(pythonw), "-m", "hermes_cli.main", "gateway", "run"]
    with mock.patch.object(
        gateway_windows, "_stable_gateway_working_dir", return_value=str(tmp_path)
    ), mock.patch("hermes_cli.config.get_hermes_home", return_value=str(tmp_path)):
        new_argv, cwd, env = gateway_windows.windowless_gateway_restart_spec(list(argv))

    assert new_argv[0] == str(python)
    assert new_argv[1:] == argv[1:]
    assert cwd == str(tmp_path)
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")


# ---------------------------------------------------------------------------
# _refresh_windows_gateway_launchers: hermes update regenerates launchers
# ---------------------------------------------------------------------------


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    """A fake Hermes root whose ``profiles/`` dir the real scan can walk.

    ``list_profile_names`` is a plain directory scan, so pointing
    ``get_default_hermes_root`` at ``tmp_path`` exercises the real enumeration
    rather than a stubbed profile list.
    """
    import hermes_constants

    (tmp_path / "profiles" / "arthur_tutor").mkdir(parents=True)
    (tmp_path / "profiles" / "joao_pessoal").mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    return tmp_path


def _install_only(monkeypatch, installed_homes):
    """Make ``is_installed()`` answer for the home the CALLER scoped it to.

    It reads ``get_hermes_home()`` itself, so a test using this fails if the
    context-local override is not actually installed around the call — not
    merely if the loop is missing.
    """
    from hermes_constants import get_hermes_home

    wanted = {str(Path(h)) for h in installed_homes}
    monkeypatch.setattr(
        gateway_windows, "is_installed", lambda: str(get_hermes_home()) in wanted
    )


def test_enumeration_keeps_only_profiles_with_an_installed_entry(
    profile_root, monkeypatch
):
    _install_only(monkeypatch, [profile_root, profile_root / "profiles" / "arthur_tutor"])

    homes = cli_main._installed_gateway_profile_homes()

    assert [str(h) for h in homes] == [
        str(profile_root),
        str(profile_root / "profiles" / "arthur_tutor"),
    ]


def test_enumeration_restores_the_previous_home_afterwards(profile_root, monkeypatch):
    """The override is context-local; leaking one would retarget the rest of
    the update at a profile it was never asked to touch."""
    from hermes_constants import get_hermes_home

    _install_only(monkeypatch, [profile_root])
    before = str(get_hermes_home())

    cli_main._installed_gateway_profile_homes()

    assert str(get_hermes_home()) == before


def test_enumeration_survives_a_profile_that_raises(profile_root, monkeypatch):
    from hermes_constants import get_hermes_home

    bad = str(profile_root / "profiles" / "arthur_tutor")

    def flaky():
        if str(get_hermes_home()) == bad:
            raise OSError("schtasks exploded")
        return True

    monkeypatch.setattr(gateway_windows, "is_installed", flaky)

    homes = [str(h) for h in cli_main._installed_gateway_profile_homes()]

    assert bad not in homes
    assert str(profile_root / "profiles" / "joao_pessoal") in homes


def test_refresh_writes_a_launcher_for_every_installed_profile(
    profile_root, monkeypatch, capsys
):
    """The regression #91675 is about: only the active profile was refreshed,
    so a non-active profile's launcher stayed stale forever.

    Asserts the home each ``_write_task_script`` call OBSERVES, because that
    is what decides which profile's ``.cmd`` gets rewritten — iterating
    without scoping would write the same file N times.
    """
    from hermes_constants import get_hermes_home

    installed = [profile_root, profile_root / "profiles" / "joao_pessoal"]
    _install_only(monkeypatch, installed)
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)

    seen = []
    monkeypatch.setattr(
        gateway_windows, "_write_task_script", lambda: seen.append(str(get_hermes_home()))
    )

    cli_main._refresh_windows_gateway_launchers()

    assert seen == [str(h) for h in installed]
    assert "2 profiles" in capsys.readouterr().out


def test_refresh_reports_a_single_profile_the_way_it_always_did(
    profile_root, monkeypatch, capsys
):
    _install_only(monkeypatch, [profile_root])
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway_windows, "_write_task_script", lambda: None)

    cli_main._refresh_windows_gateway_launchers()

    out = capsys.readouterr().out
    assert "Refreshed Windows gateway launcher scripts" in out
    assert "profiles)" not in out


def test_one_failing_profile_does_not_cost_the_others_their_refresh(
    profile_root, monkeypatch
):
    from hermes_constants import get_hermes_home

    bad = str(profile_root / "profiles" / "arthur_tutor")
    good = str(profile_root / "profiles" / "joao_pessoal")
    _install_only(monkeypatch, [profile_root, Path(bad), Path(good)])
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)

    seen = []

    def flaky():
        home = str(get_hermes_home())
        if home == bad:
            raise PermissionError("locked by AV")
        seen.append(home)

    monkeypatch.setattr(gateway_windows, "_write_task_script", flaky)

    cli_main._refresh_windows_gateway_launchers()

    assert seen == [str(profile_root), good]


def test_refresh_prints_nothing_when_no_profile_has_a_launcher(
    profile_root, monkeypatch, capsys
):
    _install_only(monkeypatch, [])
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(
        gateway_windows,
        "_write_task_script",
        lambda: pytest.fail("nothing is installed; there is no launcher to write"),
    )

    cli_main._refresh_windows_gateway_launchers()

    assert capsys.readouterr().out == ""


def test_refresh_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: False)
    monkeypatch.setattr(
        cli_main,
        "_installed_gateway_profile_homes",
        lambda: pytest.fail("no profile enumeration may run off Windows"),
    )

    cli_main._refresh_windows_gateway_launchers()


# ---------------------------------------------------------------------------
# The refresh is a content replace, not a touch (#91956 review)
# ---------------------------------------------------------------------------

STOCK_CMD = b"@echo off\r\npython.exe -m hermes_cli.main gateway run\r\n"
STOCK_VBS = b'CreateObject("WScript.Shell").Run "...", 0, False\r\n'


def _stub_renderer(monkeypatch):
    """Stand in for ``_write_task_script`` with something that really writes.

    The real renderer resolves ``get_python_path()``/``PROJECT_ROOT`` and a
    profile argv, none of which this behaviour depends on. What it does depend
    on is that the render *replaces file content*, which is exactly what a stub
    that writes a fixed stock pair reproduces.
    """
    def _render():
        path = gateway_windows.get_task_script_path()
        path.write_bytes(STOCK_CMD)
        path.with_suffix(".vbs").write_bytes(STOCK_VBS)
        return path

    monkeypatch.setattr(gateway_windows, "_write_task_script", _render)


@pytest.fixture
def windows_refresh(profile_root, monkeypatch):
    """A refresh that runs its real path resolution on any host.

    ``get_task_script_path`` calls ``_assert_windows``, so the platform has to
    be faked for the path derivation (and therefore the backup) to run at all
    off Windows.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    _stub_renderer(monkeypatch)
    return profile_root


def _script_path_for(home):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        return gateway_windows.get_task_script_path()
    finally:
        reset_hermes_home_override(token)


def test_a_wrapped_launcher_is_saved_before_it_is_replaced(
    windows_refresh, monkeypatch, capsys
):
    """A host that wrapped the generated .cmd (local supervisor / watchdog)
    gets its wrapper flattened by the refresh. Healing a legacy launcher means
    we cannot decline to rewrite, so the previous copy has to survive."""
    _install_only(monkeypatch, [windows_refresh])
    script = _script_path_for(windows_refresh)
    wrapper = b'wscript.exe //B //Nologo "Hermes_Gateway_hidden.vbs"\r\n'
    script.write_bytes(wrapper)

    cli_main._refresh_windows_gateway_launchers()

    backup = script.with_name(script.name + ".pre-refresh.bak")
    assert script.read_bytes() == STOCK_CMD
    assert backup.read_bytes() == wrapper
    assert str(backup) in capsys.readouterr().out


def test_a_second_update_does_not_overwrite_the_saved_wrapper(
    windows_refresh, monkeypatch
):
    """The trap in a naive backup: update #2 finds a stock script, copies THAT
    over the backup, and the user's wrapper is gone for good. Only a launcher
    whose content actually changed is preserved, so #2 copies nothing."""
    _install_only(monkeypatch, [windows_refresh])
    script = _script_path_for(windows_refresh)
    wrapper = b"REM the only copy of this that will ever exist\r\n"
    script.write_bytes(wrapper)

    cli_main._refresh_windows_gateway_launchers()
    cli_main._refresh_windows_gateway_launchers()

    assert script.with_name(script.name + ".pre-refresh.bak").read_bytes() == wrapper


def test_an_unchanged_launcher_is_not_backed_up(windows_refresh, monkeypatch, capsys):
    _install_only(monkeypatch, [windows_refresh])
    script = _script_path_for(windows_refresh)
    script.write_bytes(STOCK_CMD)
    script.with_suffix(".vbs").write_bytes(STOCK_VBS)

    cli_main._refresh_windows_gateway_launchers()

    assert not script.with_name(script.name + ".pre-refresh.bak").exists()
    assert "previous launcher saved" not in capsys.readouterr().out


def test_a_launcher_that_did_not_exist_is_created_without_a_backup(
    windows_refresh, monkeypatch
):
    """Creating a file is not destructive, so there is nothing to preserve."""
    _install_only(monkeypatch, [windows_refresh])
    script = _script_path_for(windows_refresh)
    vbs = script.with_suffix(".vbs")
    assert not vbs.exists()

    cli_main._refresh_windows_gateway_launchers()

    assert vbs.read_bytes() == STOCK_VBS
    assert not vbs.with_name(vbs.name + ".pre-refresh.bak").exists()


def test_each_profile_keeps_its_backup_in_its_own_home(windows_refresh, monkeypatch):
    """The reported host had a wrapper on one profile only; a backup written to
    the wrong home would be as lost as no backup at all."""
    other = windows_refresh / "profiles" / "arthur_tutor"
    _install_only(monkeypatch, [windows_refresh, other])
    wrappers = {}
    for home in (windows_refresh, other):
        script = _script_path_for(home)
        wrappers[script] = f"REM wrapper for {home.name}\r\n".encode()
        script.write_bytes(wrappers[script])

    cli_main._refresh_windows_gateway_launchers()

    for script, original in wrappers.items():
        assert script.with_name(script.name + ".pre-refresh.bak").read_bytes() == original
