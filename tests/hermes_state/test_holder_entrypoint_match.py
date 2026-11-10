"""Product-written entry points must count as Hermes holders (#152715).

``hermes_state_holders._looks_like_hermes`` is the fallback identity proof the state-maintenance
authority uses when a process's descriptors cannot be read (another user, a race). A miss there is
unsafe: ``sessions repair`` / ``optimize-storage`` / ``sessions backup`` then treat a live Hermes
process as a foreign one and may operate on the store it is holding.

The module table only carried the two spellings the earliest launchers used, so a gateway started as
``python -m gateway.run``, the stdio backend the Ink TUI and the Desktop drive
(``python -m tui_gateway.entry``), the ACP entry module from ``[project.scripts]``
(``python -m acp_adapter.entry``) and ``python -m run_agent`` all read as foreign processes.
Matching is boundary-aware on the module path: a submodule of a Hermes entry module is Hermes, a
sibling module whose name merely starts with the same characters is not.

The behavioural case drives the real holder scan: a second user's ``python -m gateway.run`` has an
unreadable ``/proc/<pid>/fd`` and a cmdline with no absolute paths, so the scan must keep it
fail-closed as an uninspectable holder of our store.
"""

import errno
import os

import pytest

import hermes_state_holders

# Pristine stdlib handles captured at import: the fake /proc projection below re-enters them.
_REAL_LISTDIR = os.listdir

# The module spellings the product itself launches, each with the source that writes it.
PRODUCT_MODULE_ENTRY_POINTS = {
    "hermes_cli.main": "pyproject [project.scripts] hermes, and every CLI relaunch",
    "acp_adapter": "acp_adapter/__main__.py",
    "acp_adapter.entry": "pyproject [project.scripts] hermes-acp",
    "tui_gateway.entry": "the stdio backend the Ink TUI and the Desktop drive",
    "gateway.run": "gateway/run.py -- 'Run via ``python -m gateway.run``'",
    "run_agent": "run_agent.py __main__ guard",
}


@pytest.mark.parametrize("module", sorted(PRODUCT_MODULE_ENTRY_POINTS))
def test_product_module_entry_point_is_a_holder(module):
    """Every module the product launches identifies a Hermes holder, with or without arguments."""
    assert hermes_state_holders._looks_like_hermes(["python", "-m", module])
    assert hermes_state_holders._looks_like_hermes(
        ["/opt/hermes-agent/.venv/bin/python", "-X", "faulthandler", "-u", "-m", module, "--flag"]
    )
    assert hermes_state_holders._looks_like_hermes(["python", f"-m{module}"])
    assert hermes_state_holders._looks_like_hermes(["python.exe", "-m", module.upper()])


@pytest.mark.parametrize(
    "argv",
    (
        ("python", "-m", "gateway.runner"),
        ("python", "-m", "gateway.runner_extra"),
        ("python", "-m", "acp_adapterx"),
        ("python", "-m", "hermes_cli.mainx"),
        ("python", "-m", "run_agents"),
        ("python", "-m", "tui_gateway.entrypoint"),
        ("python", "-m", "tools.gateway.run"),
        ("python", "-c", "import gateway.run"),
        ("python", "-m", "other.module", "gateway.run"),
    ),
)
def test_sibling_module_names_are_not_holders(argv):
    """Boundary-aware matching must not adopt another project's similarly named module."""
    assert not hermes_state_holders._looks_like_hermes(argv)


def test_console_script_entry_points_are_holders():
    """A new ``[project.scripts]`` entry must not need a second edit to count as a holder.

    ``hermes_cli._launchers.ENTRY_POINTS`` is the in-tree copy of ``[project.scripts]`` (hermes,
    hermes-agent, hermes-acp); the scan has to recognise every module it names.
    """
    launchers = pytest.importorskip("hermes_cli._launchers")
    for _command, (module, _callable) in launchers.ENTRY_POINTS.items():
        assert hermes_state_holders._looks_like_hermes(["python", "-m", module]), module


# PIDs the fake /proc tree exposes: 111 is "us", 222 is the other user's gateway.
_OTHER_USER_GATEWAY_PID = 222


def _fake_proc(monkeypatch, tmp_path):
    """Project ``/proc`` onto ``tmp_path/proc`` with one cross-user, unlistable ``fd`` dir.

    Local copy of the fake in ``test_fts_holder_instance_scope.py`` rather than an import: the health
    gate treats a test module's shared names as redefinitions, and this case needs only the
    unreadable-``fd`` branch.
    """
    proc_root = tmp_path / "proc"
    (proc_root / str(_OTHER_USER_GATEWAY_PID)).mkdir(parents=True)
    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)

    def _listdir(path):
        if isinstance(path, str):
            if path == f"/proc/{_OTHER_USER_GATEWAY_PID}/fd":
                raise PermissionError(errno.EACCES, path)
            path = path.replace("/proc", str(proc_root))
        return _REAL_LISTDIR(path)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
    monkeypatch.setattr(
        hermes_state_holders,
        "_read_proc_argv",
        lambda pid: ["python", "-m", "gateway.run"] if pid == _OTHER_USER_GATEWAY_PID else None,
    )


@pytest.mark.platforms("linux")
def test_uninspectable_gateway_run_process_is_a_holder(tmp_path, monkeypatch):
    """A gateway this scan cannot inspect must still be reported, not treated as quiescence."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    _fake_proc(monkeypatch, tmp_path)

    holders = hermes_state_holders.foreign_state_db_holders(db_path)

    assert [_pid for _pid, _detail in holders] == [_OTHER_USER_GATEWAY_PID]
    assert holders[0][1].startswith("uninspectable holder:")
