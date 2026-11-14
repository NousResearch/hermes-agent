"""Runtime wiring of the ``LEFT_CORE`` provider rows: the real ``resolve_runtime_provider`` installs the
plugin of a provider that left core for a home that uses it, once per process, honouring the failure
backoff and ``security.allow_lazy_installs``.

Real config files under a temp ``HERMES_HOME``; the catalog lookup and the network install are stand-ins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_terminal_front_end", False)
    for name in ("HERMES_INFERENCE_PROVIDER", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)  # never the network


@pytest.fixture
def installs(monkeypatch):
    calls = []

    def fake_install(identifier, *, force, enable, catalog_name=None, assume_deps_consent=None, **kw):
        calls.append((catalog_name, assume_deps_consent))
        return {"ok": False, "error": "fatal: unable to access the catalog repo"}
    import hermes_cli.plugins_cmd as pc
    monkeypatch.setattr(pc, "dashboard_install_plugin", fake_install)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)  # the harness's bridge; config decides
    return calls


def _copilot_home(tmp_path: Path, monkeypatch, config: str = "model:\n  provider: copilot\n") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _resolve(**kw):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    with pytest.raises(AuthError):  # the install failed: still an unknown provider, with the hint
        resolve_runtime_provider(**kw)


def test_runtime_resolution_installs_once_per_process_and_honours_the_backoff(tmp_path, monkeypatch, installs):
    _copilot_home(tmp_path, monkeypatch)
    for _ in range(3):
        _resolve()
    assert installs == [("github-copilot", True)]
    monkeypatch.setattr(lcm, "_attempted", set())  # a second process inside the failure backoff
    _resolve()
    assert len(installs) == 1


def test_runtime_resolution_without_lazy_installs_fetches_nothing(tmp_path, monkeypatch, installs):
    _copilot_home(tmp_path, monkeypatch,
                  "model:\n  provider: copilot\nsecurity:\n  allow_lazy_installs: false\n")
    _resolve()
    assert installs == []


def test_a_failed_attempt_is_not_rechecked_on_every_resolution(tmp_path, monkeypatch, installs):
    """After this process's attempt, a resolution of the provider costs a set lookup, not a re-read of the
    home's config, jobs, auth store and .env."""
    _copilot_home(tmp_path, monkeypatch)
    feature = next(f for f in lcm.LEFT_CORE if f.plugin == "github-copilot")
    calls = []

    def counted(home, **kw):
        calls.append(home)
        return lcm.copilot_in_use(home, **kw)
    from dataclasses import replace
    monkeypatch.setattr(lcm, "LEFT_CORE", tuple(replace(f, in_use=counted) if f is feature else f
                                                for f in lcm.LEFT_CORE))
    lcm.recover_for_provider("copilot")
    first = len(calls)
    for _ in range(5):
        lcm.recover_for_provider("copilot")
    assert len(calls) == first



def test_an_auto_route_following_the_stored_active_provider_installs_it(tmp_path, monkeypatch, installs):
    """``provider: auto`` follows auth.json's ``active_provider``: a core-era copilot choice is use."""
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home = _copilot_home(tmp_path, monkeypatch, "model:\n  provider: auto\n")
    (home / "auth.json").write_text('{"active_provider": "copilot"}', encoding="utf-8")
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_fixture")
    try:
        resolve_runtime_provider()
    except AuthError:
        pass  # the stand-in install failed: what matters is that it was attempted
    assert installs == [("github-copilot", True)]


@pytest.mark.parametrize("config", [
    "model:\n  provider: nous\ncron:\n  model_provider: copilot\n",
    "model:\n  provider: nous\ndelegation:\n  backends:\n    model_map:\n      options:\n"
    "        - name: fast\n          provider: copilot\n          model: gpt-5.4\n",
])
def test_a_route_configured_outside_the_main_model_installs_when_resolved(tmp_path, monkeypatch, installs, config):
    """A cron run or a routed delegate resolves the provider its config slot names: that is use."""
    _copilot_home(tmp_path, monkeypatch, config)
    _resolve(requested="copilot")
    assert installs == [("github-copilot", True)]


def test_doctor_installs_nothing_and_names_the_install(tmp_path, monkeypatch, installs, capsys):
    """`hermes doctor` is read-only: its auxiliary routing check resolves through the runtime resolver,
    which must not fetch the plugin there; the failed check names the install command instead."""
    from types import SimpleNamespace
    from hermes_cli import doctor
    from hermes_cli.doctor_config import _check_config_file
    home = _copilot_home(tmp_path, monkeypatch, "model:\n  provider: nous\nauxiliary:\n  vision:\n"
                                                "    provider: copilot\n    model: gpt-4.1\n")
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, _check_config_file),))
    doctor.run_doctor(SimpleNamespace(fix=False, ack=None, live=False))
    assert installs == []
    assert not (home / "plugins" / "github-copilot").exists()
    assert "hermes plugins install github-copilot" in capsys.readouterr().out


def test_an_append_racing_a_claim_is_delivered_not_lost(tmp_path, monkeypatch):
    """A writer that opened the notice file before a claimant took it: its line still reaches a claimant."""
    import builtins
    import threading
    home = _copilot_home(tmp_path, monkeypatch)
    lcm._queue_notice(home, "first")
    opened, go = threading.Event(), threading.Event()

    def held_open(*args, **kwargs):  # the writer has its handle and stalls before writing
        handle = builtins.open(*args, **kwargs)
        opened.set()
        go.wait(10)
        return handle
    monkeypatch.setattr(lcm, "open", held_open, raising=False)
    writer = threading.Thread(target=lcm._queue_notice, args=(home, "second"))
    writer.start()
    assert opened.wait(10)
    claimed: list[str] = []
    claimant = threading.Thread(target=lambda: claimed.extend(lcm._claim_notices(home)))
    claimant.start()
    claimant.join(0.5)  # a claim that can run now (no lock) finishes here, before the writer writes
    go.set()
    writer.join(10)
    claimant.join(10)
    assert sorted(claimed + lcm._claim_notices(home)) == ["first", "second"]


def test_repeated_starts_keep_one_copy_of_the_same_notice(tmp_path, monkeypatch, installs):
    """With allow_lazy_installs off every gateway start reports the same row: the chat agent gets it once."""
    home = _copilot_home(tmp_path, monkeypatch, "model:\n  provider: copilot\nsecurity:\n  allow_lazy_installs: false\n")
    for _ in range(3):
        monkeypatch.setattr(lcm, "_attempted", set())  # three gateway starts, no agent in between
        lcm.recover_at_startup()
    assert len(lcm._notice_file(home).read_text(encoding="utf-8").splitlines()) == 1
    monkeypatch.setattr(lcm, "_attempted", set())
    said: list[str] = []
    lcm.recover_at_startup(say=said.append)  # a terminal's start: the kept copy, then its own report
    assert len(said) == 1 and "allow_lazy_installs is off" in said[0]


def test_doctor_in_a_terminal_front_end_leaves_kept_notices_for_the_next_agent(tmp_path, monkeypatch, installs,
                                                                               capsys):
    """The classic CLI runs doctor in-process (console engine): a read-only diagnostic neither prints nor
    consumes what a migration kept for the home's next agent."""
    from types import SimpleNamespace
    from hermes_cli import doctor
    from hermes_cli.doctor_config import _check_config_file
    home = _copilot_home(tmp_path, monkeypatch, "model:\n  provider: nous\nauxiliary:\n  vision:\n"
                                                "    provider: copilot\n    model: gpt-4.1\n")
    lcm._queue_notice(home, "kept by the gateway's start")
    lcm.serve_notices_on_stderr()
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, _check_config_file),))
    doctor.run_doctor(SimpleNamespace(fix=False, ack=None, live=False))
    assert "kept by the gateway's start" not in capsys.readouterr().err
    assert lcm._claim_notices(home) == ["kept by the gateway's start"]


@pytest.fixture
def installs_ok(monkeypatch):
    calls = []

    def fake_install(identifier, *, force, enable, catalog_name=None, assume_deps_consent=None, **kw):
        from hermes_constants import get_hermes_home
        calls.append(catalog_name)
        (Path(get_hermes_home()) / "plugins" / catalog_name).mkdir(parents=True)
        return {"ok": True}
    import hermes_cli.plugins_cmd as pc
    monkeypatch.setattr(pc, "dashboard_install_plugin", fake_install)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    return calls


def _agent_start_in_another_process() -> str:
    """What a fresh process's agent start (its startup-warning sink) delivers for ``HERMES_HOME``."""
    import os
    import subprocess
    import sys
    code = ("import hermes_cli.left_core_migration as m; "
            "m.recover_at_startup(say=lambda message: print('NOTICE', message))")
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
                         capture_output=True, text=True, timeout=120, check=True)
    return "".join(line for line in out.stdout.splitlines(keepends=True) if line.startswith("NOTICE"))


def test_a_resolution_outcome_reaches_the_first_agent_of_any_process_once(tmp_path, monkeypatch, installs_ok,
                                                                          capsys):
    """A detached turn: the CLI process resolves the provider (and installs), the agent runs in another
    process. The outcome is kept in the home and the first agent anywhere delivers it, exactly once."""
    _copilot_home(tmp_path, monkeypatch)
    monkeypatch.setattr(lcm, "_interactive", lambda: False)
    lcm.recover_for_provider("copilot")  # process 1: no agent here
    assert installs_ok == ["github-copilot"]
    assert capsys.readouterr().err == ""
    first, second = _agent_start_in_another_process(), _agent_start_in_another_process()
    assert first.count("GitHub Copilot moved out of core") == 1
    assert "run `hermes auth add copilot`" in first  # no token the plugin reads: the gh-only hint rides along
    assert "moved out of core" not in second


def test_a_tty_without_a_declared_front_end_keeps_the_outcome_for_the_chat_agent(tmp_path, monkeypatch, installs_ok,
                                                                                 capsys):
    """A foreground `hermes gateway run` has a TTY, but its resolutions serve the chat user: the outcome (and
    its "run `hermes auth add copilot`" instruction) stays kept for the chat agent, not the gateway terminal."""
    _copilot_home(tmp_path, monkeypatch)
    monkeypatch.setattr(lcm, "_interactive", lambda: True)
    lcm.recover_for_provider("copilot")
    assert installs_ok == ["github-copilot"]
    assert capsys.readouterr().err == ""
    chat = _agent_start_in_another_process()
    assert chat.count("GitHub Copilot moved out of core") == 1 and "hermes auth add copilot" in chat


def test_a_piped_cli_front_end_prints_the_outcome_and_what_another_process_kept(tmp_path, monkeypatch,
                                                                               installs_ok, capsys):
    """``hermes chat -q`` with output piped: its turn runs in the gateway, and a resolution that fails
    before any agent exists (no credentials) leaves none to deliver what the migration kept. The classic
    CLI is a terminal front end: it prints outcomes on stderr, also one its gateway kept, exactly once."""
    home = _copilot_home(tmp_path, monkeypatch)
    monkeypatch.setattr(lcm, "_interactive", lambda: False)
    lcm._queue_notice(home, "kept by the gateway's start")
    lcm.serve_notices_on_stderr()
    lcm.recover_for_provider("copilot")
    lcm.recover_for_provider("copilot")
    err = capsys.readouterr().err
    assert err.count("kept by the gateway's start") == 1
    assert err.count("GitHub Copilot moved out of core") == 1
    assert "moved out of core" not in _agent_start_in_another_process()


_INSTALLED_PROVIDER = """
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(name="copilot", display_name="Stand-in", auth_type="api_key",
                                  env_vars=("COPILOT_GITHUB_TOKEN",), base_url="https://copilot.stand-in.test/v1"))
"""


def test_a_provider_installed_during_resolution_resolves_in_that_same_call(tmp_path, monkeypatch):
    """A fast (cached) install lands inside the plugin-dir stamp TTL: the resolution that triggered it must
    see the new profile, not answer "installed but not loaded: run hermes plugins enable"."""
    import sys

    import providers
    home = _copilot_home(tmp_path, monkeypatch)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "stand-in-token")
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})

    def install(identifier, *, force, enable, catalog_name=None, assume_deps_consent=None, **kw):
        plugin = home / "plugins" / "github-copilot"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: github-copilot\nkind: model-provider\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(_INSTALLED_PROVIDER, encoding="utf-8")
        return {"ok": True}
    import hermes_cli.plugins_cmd as pc
    monkeypatch.setattr(pc, "dashboard_install_plugin", install)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        runtime = resolve_runtime_provider()
        assert (runtime["provider"], runtime["base_url"]) == ("copilot", "https://copilot.stand-in.test/v1")
    finally:
        for name in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
            del sys.modules[name]
