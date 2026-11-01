"""Class invariant: a spawn site's child must reach the install's third-party dependencies.

WHY (measured, do not weaken) — on a PM-managed install ``sys.executable`` is the PM
*store* Python: its site-packages holds pip and nothing else, while the ~200 application
dependencies live in the committed environment
(``~/.hermes/installs/<id>/environments/<id>/venv``) and are put on ``sys.path`` by
``hermes_bootstrap`` / the published launcher — never by the interpreter itself. The
subprocess env sanitizer strips Hermes-owned PYTHONPATH entries, so a child spawned as
``[sys.executable, "-m", <app module>]`` cannot inherit the path either, and it dies at its
first dependency import ("No module named ruamel" in ``hermes_yaml.py:10``, then "No module
named dotenv" in ``hermes_cli/env_loader.py:17`` — a late import inside the worker payload,
which is why a bare ``import cron.scheduler`` succeeds while real dispatch dies). Live
effect: 41+ consecutive failed agent-kind cron dispatches, docs mirror stalled, watchdog
dead. Tracking: upstream issue #122222.

This is the behavioral counterpart of ``tests/agent/test_subprocess_env_guard.py`` (the lint-level
guard deciding WHICH env factory a spawn site may use): here every covered site is actually run,
through its own seam, and asked whether the child can import a third-party dependency. The
store-interpreter stand-in is the house pattern from
``tests/hermes_cli/test_gateway_restart_watcher_bare_python.py`` (shell wrapper that unsets PYTHONPATH
and execs the real interpreter with ``-S``, premise asserted); the PM install shape (store pin +
committed generation selected on facts) reuses ``fixture_tree`` / ``select_generation`` from
``tests/hermes_cli/test_source_launcher_publication.py``.

Covered sites — each probe argv comes from the site's own seam (helper/builder), never from a
literal copied out of the source, so the test moves when the site moves:
``cron/scheduler.py:3569`` (worker Popen; command at :3473), ``kanban_db_dispatch.py:2534``
``_resolve_hermes_argv()`` (shim-less dispatch fallback), ``codex_runtime_plugin_migration.py:380``
``_build_hermes_tools_mcp_entry()`` (its ``sys.executable`` is PERSISTED into
``~/.codex/config.toml``), ``tools/tts_tool_local.py:60`` ``_generate_neutts`` (script-shaped
child) — all four broken on this tip. Deliberately NOT covered, with the reason:
``tools/code_execution_env.py:238`` ``_resolve_child_python`` (the child runs the caller's
snippet against a stdlib-only RPC stub, so it needs no app/third-party import; strict mode
returning ``sys.executable`` is the documented isolation design, not a spawn-dependency bug);
``tools/environments/local.py`` shim resolution (returns a launcher path to consumers, spawns
nothing); ``tui_gateway/server.py``'s slash worker (its entry module imports
``hermes_bootstrap`` itself, so swapping the entry for a non-bootstrapping probe would report a
break that does not exist); ``tools/tts_tool_local.py:99`` ``-m piper.download_voices`` (same
interpreter shape as the covered neutts spawn — the class is covered once).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE_MODULE = "tests.installation.spawn_dep_probe"
PROBE_SCRIPT = Path(__file__).with_name("spawn_dep_probe.py")
ISSUE = "#122222"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell wrapper stands in for the store interpreter"
)


def _bare_store_python(tmp_path, monkeypatch) -> Path:
    """A working interpreter that sees the checkout but no third-party package, like the store Python."""
    bare = tmp_path / "bare-python"
    bare.write_text(f'#!/bin/sh\nunset PYTHONPATH\nexec "{sys.executable}" -S "$@"\n', encoding="utf-8")
    bare.chmod(0o755)
    probe = subprocess.run([str(bare), "-c", "import ruamel.yaml"], cwd=REPO, capture_output=True, text=True)
    assert probe.returncode != 0, "premise: the store stand-in must not see site-packages"
    # Production shape: the parent pins the checkout; the sanitizer decides what survives. Keeping the
    # test runner's own paths out is what makes the probe honest.
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setattr(sys, "executable", str(bare))
    return bare


def _probe_argv(argv, shape: str, module: str) -> list[str]:
    """Swap the site's entry module/script for the probe, keeping interpreter/launcher prefix + args."""
    for flag in ("-m", "--run-module"):
        if flag in argv:
            cut = argv.index(flag) + 1
            return [*argv[:cut], module, *argv[cut + 1:]]
    if shape == "script":  # interpreter + script path: same shape, the probe file stands in
        return [argv[0], str(PROBE_SCRIPT), *argv[2:]]
    if "-c" in argv:  # store-Python bootstrap: the payload the site composed already names the probe
        payload = argv[argv.index("-c") + 1]
        assert f"run_module({module!r}" in payload, f"the -c payload does not run {module}: {payload}"
        return argv
    raise AssertionError(f"unrecognized site argv shape: {argv!r}")


def _cron_external_worker(tmp_path, monkeypatch, bare):
    """``cron/scheduler.py:3569`` — Popen of the restart-safe worker (command at :3473)."""
    import cron.scheduler as scheduler
    import tools.process_registry as process_registry

    captured: dict = {}

    class _Captured(Exception):
        pass

    real_popen = subprocess.Popen

    def _fake_popen(argv, **kwargs):
        if "cron.scheduler" not in argv:
            return real_popen(argv, **kwargs)
        captured.update(argv=list(argv), env=dict(kwargs.get("env") or {}))
        raise _Captured

    with monkeypatch.context() as ctx:
        # The site's own spawn seams only: scope-argv builder, handoff claim, Popen. The worker
        # env build, tree pin and presence-var scrub all run for real.
        ctx.setattr(process_registry, "restart_safe_gateway_child_argv",
                    lambda command, **kw: process_registry.GatewayChildDispatch("degraded", list(command)))
        ctx.setattr(scheduler, "mark_execution_handoff_pending", lambda execution_id: True)
        ctx.setattr(subprocess, "Popen", _fake_popen)
        with pytest.raises(_Captured):
            scheduler._launch_external_cron_worker(
                {"id": "spawn-probe", "execution_id": "spawn-probe", "name": "spawn-probe"}
            )
    assert captured, "the cron worker Popen seam never ran"
    return captured["argv"], captured["env"], "module", None, PROBE_MODULE


def _kanban_worker(tmp_path, monkeypatch, bare):
    """``hermes_cli/kanban_db_dispatch.py:2534`` — the shim-less worker argv; env as built at :2825."""
    from hermes_cli import kanban_db_dispatch as dispatch
    from tools.environments.local import build_subprocess_env

    monkeypatch.delenv("HERMES_BIN", raising=False)
    argv = dispatch._resolve_hermes_argv()
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    return argv, env, "module", None, PROBE_MODULE


def _codex_mcp_server(tmp_path, monkeypatch, bare):
    """``hermes_cli/codex_runtime_plugin_migration.py:380`` — the MCP command persisted for codex."""
    from hermes_cli import codex_runtime_plugin_migration as migration

    entry = migration._build_hermes_tools_mcp_entry()
    argv = [entry["command"], *entry["args"]]
    env = {**os.environ, **entry["env"]}  # what codex launches the server with
    return argv, env, "module", None, PROBE_MODULE


def _neutts_synth(tmp_path, monkeypatch, bare):
    """``tools/tts_tool_local.py:60`` — the out-of-process NeuTTS synthesis spawn (script shape)."""
    from tools import tts_tool_local
    from tools.environments.local import build_subprocess_env

    captured: dict = {}

    class _Captured(Exception):
        pass

    def _capture(cmd, timeout=None):
        captured["argv"] = list(cmd)
        raise _Captured

    with monkeypatch.context() as ctx:
        ctx.setattr(tts_tool_local, "_run_helper", _capture)
        with pytest.raises(_Captured):
            tts_tool_local._generate_neutts("spawn-probe", str(tmp_path / "spawn-probe.wav"), {})
    assert captured, "the neutts spawn seam never ran"
    return (captured["argv"], build_subprocess_env(scrub_secrets=False, inherit_profile_home=True),
            "script", None, PROBE_MODULE)


def _fake_pm_install(tmp_path, monkeypatch, bare):
    """A PM install whose committed generation — and nothing else — provides ruamel.yaml."""
    import ruamel.yaml

    from hermes_cli import _launchers
    from pm.environments import site_packages
    from tests.hermes_cli.test_source_launcher_publication import fixture_tree, select_generation
    from tools.environments.local import build_subprocess_env

    repo, home, _ = fixture_tree(tmp_path, monkeypatch)
    store_entry = tmp_path / "store-python-entry"  # stand-in store interpreter pin
    (store_entry / "bin").mkdir(parents=True)
    (store_entry / "bin" / "python3").symlink_to(bare)
    (home / "tools" / "facts.json").write_text(
        json.dumps({"schema": 1, "packages": {"python": {"version": "probe", "entry": str(store_entry)}}}),
        encoding="utf-8",
    )
    site = site_packages(select_generation(repo, "current", "probe"))
    shutil.copytree(Path(ruamel.yaml.__file__).parents[1], site / "ruamel")
    # tests/ is not part of an install: the probe entry ships as a top-level module there.
    shutil.copyfile(PROBE_SCRIPT, repo / "spawn_dep_probe.py")
    (repo / ".hermes" / "bin").mkdir(parents=True, exist_ok=True)
    launcher = _launchers.mint_launcher("hermes", repo, repo / ".hermes" / "bin", store_entry / "bin" / "python3", None)
    assert launcher is not None, "the fixture install must publish its launcher"
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    return repo, site, env


def _launcher_runtime_command(tmp_path, monkeypatch, bare):
    """``hermes_cli/_launchers.py:26`` runtime_command — store Python + bootstrap (gateway/dashboard spawners)."""
    from hermes_cli import _launchers

    repo, site, env = _fake_pm_install(tmp_path, monkeypatch, bare)
    # tests/ does not exist inside an install, so the probe entry is the copied top-level module.
    return _launchers.runtime_command(repo, module="spawn_dep_probe"), env, "module", site, "spawn_dep_probe"


def _launcher_installation_command(tmp_path, monkeypatch, bare):
    """``hermes_cli/_launchers.py:65`` installation_command — the published source launcher form."""
    from hermes_cli import _launchers

    repo, site, env = _fake_pm_install(tmp_path, monkeypatch, bare)
    return _launchers.installation_command(repo, module="spawn_dep_probe"), env, "module", site, "spawn_dep_probe"


# Seam builders; bootstrapped sites carry no marker and must PASS.
def _site(builder, site_id, reason):
    return pytest.param(builder, marks=pytest.mark.xfail(strict=False, reason=f"{reason} ({ISSUE})"), id=site_id)


SITES = [
    _site(_cron_external_worker, "cron-external-worker",
          "the restart-safe cron worker spawns sys.executable -m cron.scheduler, and cron/ has no "
          "hermes_cli.main bootstrap: the child dies at its first dependency import on the store Python"),
    _site(_kanban_worker, "kanban-worker",
          "kanban dispatch's shim-less fallback spawns sys.executable -m hermes_cli.main from the worker "
          "workspace with the sanitized env, so the store Python reaches neither the app tree nor its deps"),
    _site(_codex_mcp_server, "codex-mcp-server",
          "codex's Hermes-tools MCP entry PERSISTS sys.executable -m agent.transports."
          "hermes_tools_mcp_server into config.toml: a store Python bound to an ABI that changes with "
          "the selected generation, owning no dependencies"),
    _site(_neutts_synth, "neutts-synth",
          "NeuTTS synthesis spawns sys.executable <tools/neutts_synth.py>: on the store Python the "
          "script's first dependency import fails"),
    pytest.param(_launcher_runtime_command, id="launcher-runtime-command"),
    pytest.param(_launcher_installation_command, id="launcher-installation-command"),
]


@pytest.mark.parametrize("builder", SITES)
def test_child_reaches_the_installs_third_party_dependencies(builder, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    bare = _bare_store_python(tmp_path, monkeypatch)
    argv, env, shape, ruamel_root, module = builder(tmp_path, monkeypatch, bare)
    assert Path(argv[0]) == bare or Path(argv[0]).is_relative_to(tmp_path), (
        f"{builder.__name__} did not launch from the store stand-in or a launcher minted for it: {argv[0]}")
    result = subprocess.run(
        _probe_argv(argv, shape, module), cwd=REPO, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, (
        f"the child spawned by {builder.__name__} cannot import the install's dependencies "
        f"({argv[0]} {' '.join(argv[1:3])}):\nstdout: {result.stdout}\nstderr: {result.stderr}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    if ruamel_root is not None:
        # Bootstrapped: the dependency must come from the install's committed generation, not from
        # whatever the store interpreter happens to see.
        assert Path(payload["ruamel"]).is_relative_to(ruamel_root)
