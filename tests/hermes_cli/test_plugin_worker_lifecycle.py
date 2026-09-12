"""Command → real PM worker → offline wheels → persisted state → fresh imports."""
from __future__ import annotations

import pytest

from tests.hermes_cli.plugin_worker_support import (
    plugin_world as plugin_world,
    isolated_python as isolated_python,
)


@pytest.mark.parametrize("surface,suffix", [
    ("pyproject", "yaml"), ("python_dependencies", "yaml"), ("pip_dependencies", "yaml"),
    ("python_dependencies", "yml"), ("pip_dependencies", "yml"),
])
def test_declaration_consent_admission_and_resync(plugin_world, monkeypatch, capsys, surface, suffix):
    from hermes_cli import plugins_cmd
    from pm import client, receipt

    world = plugin_world
    origin, revision = world.origin(surface=surface, suffix=suffix)
    before = (world.home / "config.yaml").read_bytes()
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: False)
    world.command("install", identifier=origin.as_uri(), ref=revision, enable=True, allow_removed=True)
    assert world.enabled() == [], "non-interactive Python admission bypassed consent"
    assert (world.home / "config.yaml").read_bytes() == before
    assert "skipped (non-interactive)" in " ".join(capsys.readouterr().out.split())

    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    world.command("install", identifier=origin.as_uri(), force=True, enable=True, allow_removed=True)
    assert world.enabled() == []
    assert (world.home / "config.yaml").read_bytes() == before
    assert "declined" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    world.command("install", identifier=origin.as_uri(), force=True, enable=True, allow_removed=True)
    assert world.enabled() == ["plugin-worker-proof"]
    assert receipt.latest()["outcome"] == "ok"
    world.imports()
    selected = world.selected()
    client.sync_venv(explicit=True)
    assert world.selected() == selected
    world.imports()

    declaration = world.home / "plugins/plugin-worker-proof" / ("pyproject.toml" if surface == "pyproject" else f"plugin.{suffix}")
    declaration.write_text(declaration.read_text(encoding="utf-8").replace("==1.0", "==2.0"), encoding="utf-8")
    client.sync_venv(explicit=True)
    assert world.selected() != selected, "changed declaration was not stamped"
    # Code has not changed: inspect the new dependency independently.
    import os
    import subprocess
    python = world.selected() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run([str(python), "-I", "-c", "import plugin_proof_dep; assert plugin_proof_dep.__version__ == '2.0'"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_active_force_reinstall_decline_preserves_entire_selection(plugin_world, monkeypatch):
    from hermes_cli import plugins_cmd
    from pm import paths
    from tests.hermes_cli.plugin_worker_support import git, version

    world = plugin_world
    origin, first = world.origin()
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, "isatty", lambda: True)
    answers = []
    monkeypatch.setattr("builtins.input", lambda prompt: answers.append(prompt) or "yes")
    world.command("install", identifier=origin.as_uri(), ref=first, enable=True, allow_removed=True)
    world.imports()
    target = world.home / "plugins/plugin-worker-proof"
    watched = [world.home / "config.yaml", target / "__init__.py", target / "plugin.yaml",
               target.parent / ".install-metadata.json", paths.runtime_facts_path()]
    before = {path: path.read_bytes() for path in watched}
    selected = world.selected()
    second = version(origin, pin="2.0")
    monkeypatch.setattr("builtins.input", lambda prompt: answers.append(prompt) or "no")
    try:
        world.command("install", identifier=origin.as_uri(), ref=second, force=True, enable=True, allow_removed=True)
    except SystemExit as exc:
        assert exc.code == 1
    assert {path: path.read_bytes() for path in watched} == before
    assert git(target, "rev-parse", "HEAD") == first
    assert world.selected() == selected
    world.imports()

    answers.clear()
    monkeypatch.setattr("builtins.input", lambda prompt: answers.append(prompt) or "yes")
    world.command("install", identifier=origin.as_uri(), ref=second, force=True, enable=True, allow_removed=True)
    assert len(answers) == 1, "active replacement asked for the same consent twice"
    assert git(target, "rev-parse", "HEAD") == second
    assert world.selected() != selected
    world.imports(pin="2.0")


@pytest.mark.parametrize("flag", ["no_enable", "enable"])
def test_active_reinstall_does_not_overwrite_a_later_disable(plugin_world, monkeypatch, flag):
    from hermes_cli import plugins_cmd
    from pm import paths
    from tests.hermes_cli.plugin_worker_support import version

    world = plugin_world
    origin, first = world.origin()
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    world.command("install", identifier=origin.as_uri(), ref=first, enable=True, allow_removed=True)
    second = version(origin, pin="2.0")
    watched = [world.home / "config.yaml", paths.runtime_facts_path()]
    acknowledged = {}
    def disable_after_publication(*args):
        world.command("disable", name="plugin-worker-proof")
        acknowledged.update({path: path.read_bytes() for path in watched})
    monkeypatch.setattr(plugins_cmd, "_display_after_install", disable_after_publication)
    world.command("install", identifier=origin.as_uri(), ref=second, force=True, allow_removed=True, **{flag: True})
    assert acknowledged
    assert {path: path.read_bytes() for path in watched} == acknowledged
    assert world.enabled() == []


def test_malformed_portable_member_cannot_join_a_new_generation(plugin_world, monkeypatch):
    import json
    from hermes_cli import plugins_cmd
    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
    from pm import client, paths
    from tests.hermes_cli.plugin_worker_support import git

    world = plugin_world
    origin, _ = world.origin(surface="pyproject")
    (origin / "plugin.yaml").unlink()
    (origin / "plugin.json").write_text(json.dumps({"$schema": PLUGIN_SCHEMA_V1, "name": "plugin-worker-proof"}), encoding="utf-8")
    git(origin, "add", "--all")
    git(origin, "commit", "-qm", "portable declaration")
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    world.command("install", identifier=origin.as_uri(), enable=True, allow_removed=True)
    world.imports()
    watched = [world.home / "config.yaml", paths.runtime_facts_path()]
    before = {path: path.read_bytes() for path in watched}
    (world.home / "plugins/plugin-worker-proof/plugin.json").write_text("{malformed", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="plugin.json"):
        client.sync_venv(explicit=True)
    assert {path: path.read_bytes() for path in watched} == before


def test_pack_admits_compatible_members_and_reports_conflict(plugin_world, monkeypatch, capsys):
    import json
    import hermes_yaml as yaml
    from hermes_cli import plugins_cmd
    from tests.hermes_cli.plugin_worker_support import git, version

    world = plugin_world
    incumbent, first = world.origin()
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    world.command("install", identifier=incumbent.as_uri(), ref=first, enable=True, allow_removed=True)
    world.imports()
    good, good_sha = world.origin(name="pack-good", capabilities=["tools.override"])
    bad, bad_sha = world.origin(name="pack-conflict", pin="2.0")
    last, last_sha = world.origin(name="pack-last", capabilities=["tools.override"])
    # Moving HEAD cannot move a pack's pinned selection.
    version(good, name="pack-good", pin="2.0")
    pack = world.root / "pack.yaml"
    pack.write_text(yaml.safe_dump({"name": "proof-pack", "plugins": [
        {"repo": repo.as_uri(), "ref": sha} for repo, sha in [(good, good_sha), (bad, bad_sha), (last, last_sha)]
    ], "config": {"pack-good": {"voice": "fixture"}}}), encoding="utf-8")
    answers = iter(["yes", "yes", "no"])
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt: next(answers))
    with pytest.raises(SystemExit) as exc:
        world.command("pack", pack_action="install", source=str(pack))
    assert exc.value.code == 1
    from rich.text import Text
    assert "2 installed, 1 failed" in " ".join(Text.from_ansi(capsys.readouterr().out).plain.split())
    assert set(world.enabled()) == {"plugin-worker-proof", "pack-good", "pack-last"}
    for name, sha in [("pack-good", good_sha), ("pack-conflict", bad_sha), ("pack-last", last_sha)]:
        assert git(world.home / "plugins" / name, "rev-parse", "HEAD") == sha
    for name in ("plugin-worker-proof", "pack-good", "pack-last"):
        world.imports(name)
    config = yaml.safe_load((world.home / "config.yaml").read_text(encoding="utf-8"))
    entries = config["plugins"]["entries"]
    assert entries["pack-good"]["voice"] == "fixture"
    assert "tools.override" in entries["pack-good"]["granted_capabilities"]
    assert "tools.override" not in entries.get("pack-last", {}).get("granted_capabilities", [])
    receipts = [json.loads(path.read_text()) for path in (world.home / "logs/update_receipts").glob("pm_*.json")]
    assert any(row["outcome"] == "failed" and "plugin-proof-dep" in json.dumps(row) for row in receipts)


@pytest.mark.parametrize("sibling_profiles", [False, True], ids=["same-profile", "sibling-profiles"])
def test_concurrent_commands_keep_acknowledged_enables(plugin_world, sibling_profiles):
    import queue
    import threading
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import hermes_yaml as yaml
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tests.hermes_cli.plugin_worker_support import worker_command

    world = plugin_world
    homes = [world.home, world.home]
    if sibling_profiles:
        homes = [world.home / "profiles" / name for name in ("left", "right")]
        for home in homes:
            home.mkdir(parents=True)
            (home / "config.yaml").write_text("plugins:\n  enabled: []\n  disabled: []\n", encoding="utf-8")
    for name, home in zip(("race-left", "race-right"), homes):
        origin, revision = world.origin(name=name, dependency="plugin-proof-dep" if name == "race-left" else "plugin-proof-other")
        token = set_hermes_home_override(home)
        try:
            world.command("install", identifier=origin.as_uri(), ref=revision, no_enable=True, allow_removed=True)
        finally:
            reset_hermes_home_override(token)
    untouched = (world.home / "config.yaml").read_bytes()
    checkout = Path(__file__).resolve().parents[2]
    command = worker_command(checkout / "pm/worker.py", shutil.which("uv"), sys.executable,
                             runtime_python=world.runtime_python)
    processes = []
    try:
        for name, home in zip(("race-left", "race-right"), homes):
            script = (
                "import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(checkout)!r}); from pm import client, paths; "
                f"paths.repo_root = lambda: Path({str(world.core)!r}); "
                f"paths.lockfile_path = lambda: Path({str(world.root / 'lock.json')!r})\n"
                "def ready(worker, **kwargs):\n"
                "    print('READY', flush=True)\n"
                "    assert sys.stdin.readline().strip() == 'go'\n"
                f"    return {command!r}\n"
                "client.runtime_command = ready\n"
                "from hermes_cli.plugins_cmd import cmd_enable\n"
                f"cmd_enable({name!r}, allow_tool_override=False)\n"
            )
            processes.append(subprocess.Popen([sys.executable, "-I", "-B", "-c", script],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "HERMES_HOME": str(home)}))
        # Both commands have reached the worker launch (after the old stale
        # read). Release one all the way through publication, then the other.
        ready = queue.Queue()
        for process in processes:
            threading.Thread(target=lambda proc=process: ready.put(proc.stdout.readline().strip()), daemon=True).start()
        assert [ready.get(timeout=30) for _ in processes] == ["READY", "READY"]
        for process in processes:
            out, err = process.communicate("go\n", timeout=60)
            assert process.returncode == 0, out + err
        if sibling_profiles:
            assert (world.home / "config.yaml").read_bytes() == untouched
            for name, home in zip(("race-left", "race-right"), homes):
                assert yaml.safe_load((home / "config.yaml").read_text())["plugins"]["enabled"] == [name]
        else:
            assert set(world.enabled()) == {"race-left", "race-right"}, "acknowledged enable was lost"
        for name, home in zip(("race-left", "race-right"), homes):
            original = world.home
            world.home = home
            try:
                world.imports(name)
            finally:
                world.home = original
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)