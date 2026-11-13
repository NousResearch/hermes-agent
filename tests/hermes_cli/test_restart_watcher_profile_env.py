"""``hermes update`` respawns a sibling profile's gateway with that profile's credentials, never the updater's.

The updater has its own profile's ``.env`` in ``os.environ``. The restart watcher hands the env it was
spawned with to the gateway it respawns, so a profile that declares no bot token of its own (an
API-only profile) used to start the ROOT bot and steal its polling lock (#135792).
"""

import os
import subprocess

import pytest

_ROOT_TOKEN = "root-bot-token-111"
_OWN_TOKEN = "api1-own-token-222"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    api1 = root / "profiles" / "api1"
    api1.mkdir(parents=True)
    for home in (root, api1):
        (home / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (root / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_ROOT_TOKEN}\n", encoding="utf-8")
    (api1 / ".env").write_text("API_SERVER_ENABLED=true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    # The updater runs on the default profile: its .env is loaded into the process env.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _ROOT_TOKEN)
    # Which names a dotenv load put into os.environ is process-wide state: isolate it per test.
    from hermes_agent.hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    return root, api1


def _watcher_env(monkeypatch, profile: str) -> dict:
    """Env of the restart watcher ``launch_detached_profile_gateway_restart`` spawns (recorded, not run)."""
    from hermes_agent.hermes_cli import gateway

    spawned = []

    def _record(argv, **kwargs):
        if len(argv) > 2 and argv[1] == "-c" and "pid_exists_stdlib" in argv[2]:
            spawned.append(kwargs.get("env"))

    monkeypatch.setattr(subprocess, "Popen", _record)
    assert gateway.launch_detached_profile_gateway_restart(profile, 424242)
    assert len(spawned) == 1
    return spawned[0] if spawned[0] is not None else dict(os.environ)


@pytest.mark.platforms("posix")
def test_sibling_profile_without_a_token_never_gets_the_updaters(homes, monkeypatch):
    _root, api1 = homes
    env = _watcher_env(monkeypatch, "api1")
    assert env.get("HERMES_HOME") == str(api1)
    assert env.get("TELEGRAM_BOT_TOKEN") != _ROOT_TOKEN
    assert env.get("API_SERVER_ENABLED") == "true"

    # The profile's own token still reaches it; the host restart still carries the root's.
    (api1 / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_OWN_TOKEN}\n", encoding="utf-8")
    from hermes_agent.hermes_cli.env_loader import reset_secret_source_cache
    reset_secret_source_cache()
    assert _watcher_env(monkeypatch, "api1").get("TELEGRAM_BOT_TOKEN") == _OWN_TOKEN
    assert _watcher_env(monkeypatch, "default").get("TELEGRAM_BOT_TOKEN") == _ROOT_TOKEN


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("unit", [False, True], ids=["shell", "unit"])
def test_updater_on_the_same_profile_keeps_its_exported_token(homes, monkeypatch, unit):
    _root, api1 = homes
    # `hermes -p api1 update` with a token exported in the shell (or a unit's Environment=), api1's .env declaring none.
    if unit:
        monkeypatch.setenv("INVOCATION_ID", "unit-invocation")
    monkeypatch.setenv("HERMES_HOME", str(api1))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "exported-api1-token-333")
    assert _watcher_env(monkeypatch, "api1").get("TELEGRAM_BOT_TOKEN") == "exported-api1-token-333"


@pytest.mark.platforms("posix")
def test_shell_exported_keys_reach_a_sibling_but_the_root_bot_token_does_not(homes, monkeypatch):
    """Only the launch profile's own bot tokens are withheld. A model key or a platform token that
    lives only in the user's shell reached the sibling on its manual launch, so the update restart
    must keep it, or every turn fails until someone restarts the gateway by hand."""
    root, _api1 = homes
    from hermes_agent.hermes_cli.env_loader import load_hermes_dotenv
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-or-key")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "shell-discord-token")
    load_hermes_dotenv(hermes_home=root)  # the updater loads its own profile's .env
    env = _watcher_env(monkeypatch, "api1")
    assert env.get("TELEGRAM_BOT_TOKEN") is None
    assert (env.get("OPENROUTER_API_KEY"), env.get("DISCORD_BOT_TOKEN")) == ("shell-or-key", "shell-discord-token")


@pytest.mark.platforms("posix")
def test_a_unit_injected_root_token_never_follows_a_sibling(homes, monkeypatch):
    """``/update`` sent to a systemd gateway: its token came from ``Environment=``, not any .env."""
    root, _api1 = homes
    (root / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("INVOCATION_ID", "unit-invocation")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "unit-injected-token")
    assert _watcher_env(monkeypatch, "api1").get("TELEGRAM_BOT_TOKEN") is None


@pytest.mark.platforms("posix")
def test_the_root_dotenv_model_key_never_follows_a_sibling(homes, monkeypatch):
    """A key in the root's ``.env`` is the root's, not the shell's: a manual ``hermes -p api1`` never had it."""
    root, _api1 = homes
    from hermes_agent.hermes_cli.env_loader import load_hermes_dotenv
    (root / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_ROOT_TOKEN}\nOPENROUTER_API_KEY=root-or-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")  # recorded, so teardown also removes what the load sets
    load_hermes_dotenv(hermes_home=root)
    assert _watcher_env(monkeypatch, "api1").get("OPENROUTER_API_KEY") is None


@pytest.mark.platforms("posix")
def test_abort_recovery_restarts_a_sibling_without_the_updaters_token(homes, monkeypatch):
    """An aborted restart phase relaunches every profile from a fresh interpreter that inherited the
    updater's env: the sibling's ``gateway restart`` child must get the same env as the normal path."""
    _root, api1 = homes
    from hermes_agent.hermes_cli import update_restart_recovery as recovery

    seen = []

    def _run(argv, **kwargs):
        seen.append((argv, kwargs["env"]))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(recovery, "_host_served_profiles", set)
    recovery.restart_profiles(["api1"], run=_run)
    (argv, env), = seen
    assert argv[argv.index("-p") + 1] == "api1"
    assert env.get("HERMES_HOME") == str(api1)
    assert env.get("TELEGRAM_BOT_TOKEN") != _ROOT_TOKEN
    assert env.get("API_SERVER_ENABLED") == "true"
    assert env[recovery._RECOVERY_ENV] == "1"


@pytest.mark.platforms("posix")
def test_fresh_abort_recovery_withholds_what_the_updaters_secret_source_supplied(homes, monkeypatch, tmp_path):
    """The abort recovery runs in a NEW interpreter: it inherits the updater's values but not which of
    them a secret source supplied, so a source-only key reached the sibling as if the shell exported it.
    Real fresh process; a ``sitecustomize`` shim records the ``gateway restart`` grandchild's env."""
    import json
    import sys
    import textwrap
    from hermes_agent.hermes_cli import env_loader
    from hermes_agent.hermes_cli import update_abort_recovery as abort
    monkeypatch.setattr(env_loader, "_SOURCE_SUPPLIED_NAMES", {"GITHUB_TOKEN"})
    monkeypatch.setenv("GITHUB_TOKEN", "root-source-key")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")
    ledger, shim = tmp_path / "grandchild.jsonl", tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(textwrap.dedent(f"""
        import json, subprocess
        _real = subprocess.run
        def _run(argv, *a, **k):
            if "hermes_cli.main" in list(argv):
                with open({str(ledger)!r}, "a") as fh:
                    fh.write(json.dumps(k.get("env")) + "\\n")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return _real(argv, *a, **k)
        subprocess.run = _run
    """), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(shim), *sys.path]))
    result = abort._run_fresh_recovery_process(
        ["api1"], {"api1": "launchd"}, gateway_mode=False, recover_serve=False, skip_units=[])
    assert result is not None and result.returncode == 0, result and result.stderr
    env, = (json.loads(line) for line in ledger.read_text().splitlines())
    assert (env.get("GITHUB_TOKEN"), env.get("OPENAI_API_KEY")) == (None, "shell-key")
