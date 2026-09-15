"""Real startup paths share one passive PM verdict, including fast dispatch."""
import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("surface", ["serve", "version", "gateway"])
def test_each_start_checks_pm_once_before_dispatch(surface, tmp_path, monkeypatch, capsys, caplog):
    import pm
    from hermes_cli import boot_bootstrap

    home = tmp_path / "home"
    home.mkdir()
    runtime = tmp_path / "tools"
    runtime.mkdir()
    # A recorded but empty store is genuinely drifted against the shipped lock.
    (runtime / "facts.json").write_text(json.dumps({"schema": 1, "packages": {}}))
    root = tmp_path / "payload"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({
        "commit": "abcdef012345", "payload": "bundled", "updateMechanism": "electron-updater",
    }))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_DISABLE_FAST_SERVE_LAUNCH", raising=False)
    checked = []
    real_check = pm.check

    def check():
        problems = real_check()
        assert problems
        checked.append(problems)
        return problems

    def no_install(*args, **kwargs):
        pytest.fail("startup must not install or sync")

    monkeypatch.setattr(importlib.import_module("pm.ensure"), "check", check)
    monkeypatch.setattr(pm, "ensure", no_install)
    monkeypatch.setattr(pm, "sync_venv", no_install)
    dispatched = []
    if surface == "gateway":
        import gateway.run as gr

        async def start(*args, **kwargs):
            dispatched.append(len(checked))
            return True

        monkeypatch.setattr(gr, "start_gateway", start)
        monkeypatch.setattr(gr, "_exit_after_graceful_shutdown", lambda code: None)
        monkeypatch.setattr(sys, "argv", ["gateway"])
        run = gr.main
    else:
        from hermes_cli import main

        monkeypatch.setattr(main, "cmd_dashboard", lambda args: dispatched.append(len(checked)))
        monkeypatch.setattr(main, "cmd_version", lambda args: dispatched.append(len(checked)))
        monkeypatch.setattr(sys, "argv", ["hermes", "serve" if surface == "serve" else "--version"])
        run = main.main
    for _ in range(2):
        checked.clear()
        capsys.readouterr()
        caplog.clear()
        run()
        assert len(checked) == 1
        assert dispatched[-1] == 1
        output = capsys.readouterr().err
        warnings = [r for r in caplog.records if "install out of sync" in r.message]
        if surface == "gateway":
            assert len(warnings) == 1
        else:
            assert output.count("install out of sync") == 1
        assert "rebuild the artifact" in (warnings[0].message if surface == "gateway" else output)
    assert not list(home.rglob("machine.json"))
    assert boot_bootstrap.current_install_identity(root) == "abcdef012345"


@pytest.mark.parametrize("sibling_profile", [False, True])
def test_concurrent_boots_bound_real_home_migration(tmp_path, monkeypatch, sibling_profile):
    import os
    import subprocess
    import sqlite3
    import time
    from hermes_cli.config import DEFAULT_CONFIG

    root = tmp_path / "payload"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({
        "commit": "abcdef012345", "updateMechanism": "external", "distribution": "nix",
    }))
    home = tmp_path / "home"
    other = home / "profiles/coder" if sibling_profile else home
    for directory in {home, other}:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(f"_config_version: {DEFAULT_CONFIG['_config_version'] - 1}\n")
        (directory / ".env").write_text("EXAMPLE_KEY=test\n")
        with sqlite3.connect(directory / "state.db") as db:
            db.execute("CREATE TABLE retained (value)")
            db.execute("INSERT INTO retained VALUES ('before')")
    entered = tmp_path / "entered"
    release = tmp_path / "release"
    script = tmp_path / "boot.py"
    script.write_text('''import json, sys, time
from pathlib import Path
from hermes_cli import boot_bootstrap, post_update
root, entered, release = map(Path, sys.argv[1:4])
def migrate():
    if sys.argv[4] == 'hold':
        entered.touch()
        deadline = time.monotonic() + 30
        while not release.exists():
            if time.monotonic() > deadline: raise RuntimeError('release timeout')
            time.sleep(0.02)
    return post_update.step_migrate_config()
post_update.BOOT_HOME_STEPS = (('migrate', migrate), ('db', post_update.step_state_db_guard))
print(json.dumps(boot_bootstrap.run_boot_bootstrap(root)))
''')
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(home),
               HERMES_RUNTIME_DIR=str(tmp_path / "tools"),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    command = [sys.executable, str(script), str(root), str(entered), str(release)]
    first = subprocess.Popen([*command, "hold"], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not entered.exists():
            assert first.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        second = subprocess.run([*command, "go"], env=dict(env, HERMES_HOME=str(other)),
                                capture_output=True, text=True, timeout=30)
        assert second.returncode == 0, second.stderr
        result = json.loads(second.stdout.splitlines()[-1])
        if sibling_profile:
            assert result["home"]["migrate"]["ok"]
            assert result["home"]["db"]["ok"]
        else:
            assert result == {"home": "lost-race"}
        release.touch()
        stdout, stderr = first.communicate(timeout=30)
        assert first.returncode == 0, stderr
        assert json.loads(stdout.splitlines()[-1])["home"]["db"]["ok"]
        again = subprocess.run([*command, "go"], env=env, capture_output=True, text=True, timeout=30)
        assert json.loads(again.stdout.splitlines()[-1]) == {"home": "skipped"}
        for directory in {home, other}:
            assert len(list(directory.glob("config.yaml.bak-*"))) == 1
            with sqlite3.connect(directory / "state.db") as db:
                assert db.execute("SELECT value FROM retained").fetchall() == [("before",)]
        # A changed artifact or Git identity must run the real migration again.
        for revision in ('sealed', 'git'):
            (home / 'config.yaml').write_text(f"_config_version: {DEFAULT_CONFIG['_config_version'] - 1}\n")
            if revision == 'sealed':
                (root / 'install-stamp.json').write_text(json.dumps({
                    'commit': 'fedcba987654', 'updateMechanism': 'external'}))
            else:
                subprocess.run(['git', 'init', '-q', str(root)], check=True)
                subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=t@example.invalid',
                                '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'revision'],
                               cwd=root, check=True)
            changed = subprocess.run([*command, 'go'], env=env, capture_output=True, text=True, timeout=30)
            assert changed.returncode == 0, changed.stderr
            assert json.loads(changed.stdout.splitlines()[-1])['home']['migrate']['ok']
            import hermes_yaml
            assert hermes_yaml.safe_load((home / 'config.yaml').read_text())['_config_version'] == DEFAULT_CONFIG['_config_version']
    finally:
        release.touch()
        if first.poll() is None:
            first.kill()
        first.communicate(timeout=30)


def test_failed_migration_is_restored_and_not_retried(tmp_path, monkeypatch):
    from hermes_cli import boot_bootstrap, config, post_update

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = tmp_path / "payload"
    root.mkdir()
    (root / "install-stamp.json").write_text(json.dumps({
        "commit": "abcdef012345", "updateMechanism": "external", "distribution": "nix",
    }))
    config_file = home / "config.yaml"
    config_file.write_text(f"_config_version: {config.DEFAULT_CONFIG['_config_version'] - 1}\n")
    env_file = home / ".env"
    env_file.write_text("EXAMPLE_KEY=keep\n")
    original = {p: p.read_bytes() for p in (config_file, env_file)}
    calls = []

    def interrupted(**kwargs):
        calls.append(True)
        config_file.write_text("half written")
        env_file.write_text("half written")
        raise RuntimeError("interrupted migration")

    monkeypatch.setattr(config, "migrate_config", interrupted)
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("migrate", post_update.step_migrate_config),))
    assert not boot_bootstrap.run_boot_bootstrap(root)["home"]["migrate"]["ok"]
    assert {p: p.read_bytes() for p in original} == original
    assert boot_bootstrap.run_boot_bootstrap(root) == {"home": "skipped"}
    assert calls == [True]
    record = boot_bootstrap.read_last_known(boot_bootstrap.record_path(root))
    assert record['identity'] == 'abcdef012345'
    assert record['results']['migrate']['ok'] is False
