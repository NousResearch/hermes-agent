"""A default plugin install that keeps failing is attempted once per backoff, however many Hermes
processes boot meanwhile.

Real processes against one temp home: the detached coordinator (``install_and_record``) whose install
fails, and CLI boots running the real boot step (``step_install_default_plugins``) while that pass is
in flight. Each boot used to re-request the pass before the failure backoff was recorded, and the
coordinator repeated its failing pass while requests kept arriving (13 clone attempts per plugin in
76 s at one config write a second). The clone is a stand-in that fails on a signal; the spawn of the
detached child is recorded instead of run (it would fetch the live catalog).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from hermes_cli import default_plugins
from tests.waiting import wait_until

_COORDINATOR = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    from tests.waiting import wait_until
    import hermes_cli.left_core_migration as lcm
    import hermes_cli.plugin_catalog as catalog
    from hermes_cli import default_plugins
    from hermes_cli.plugin_catalog import entry_from_mapping

    signals = Path(sys.argv[1])
    entry = entry_from_mapping({"name": "standin", "repo": "https://github.com/NousResearch/standin",
                                "sha": "a" * 40, "tier": "official", "default": True, "category": "platform"}, "x")
    default_plugins.default_entries = lambda: [entry]
    catalog.fetch_live_catalog = lambda **kwargs: {"entries": [{}]}

    def install_into(home, consent=None):
        def install(name, discard=None):
            with open(signals / "attempts", "a", encoding="utf-8") as fh:
                fh.write(name + "\\n")
            (signals / "installing").touch()
            wait_until(lambda: (signals / "go").exists(), timeout=60, what="signal 'go'")
            return {"ok": False, "error": "fatal: unable to access 'https://github.com/': Could not resolve host"}
        return install
    lcm._install_into = install_into
    print(json.dumps(default_plugins.install_and_record(say=lambda m: None, wait=30)))
""")

_BOOT = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    from hermes_cli import default_plugins

    signals = Path(sys.argv[1])

    def spawned(argv, **kwargs):  # the detached child itself would read the live catalog
        with open(signals / "spawned", "a", encoding="utf-8") as fh:
            fh.write("1\\n")
    default_plugins.subprocess.Popen = spawned
    print(json.dumps(default_plugins.step_install_default_plugins()))
""")


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def test_boots_during_a_failing_pass_never_repeat_it(tmp_path):
    home, signals = tmp_path / "home", tmp_path / "signals"
    home.mkdir()
    signals.mkdir()
    (home / "config.yaml").write_text("_config_version: 1\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "PYTHONPATH"))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home),
               PYTHONPATH=str(Path(default_plugins.__file__).parents[1]))

    def boot() -> dict:
        done = subprocess.run([sys.executable, "-c", _BOOT, str(signals)], env=env, check=False,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    coordinator = subprocess.Popen([sys.executable, "-c", _COORDINATOR, str(signals)], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_until(lambda: (signals / "installing").exists() or coordinator.poll() is not None,
                   timeout=60, what="the pass to start installing")
        assert coordinator.poll() is None, coordinator.communicate()
        for _ in range(3):  # CLI processes booting while the failing pass is in flight
            boot()
        (signals / "go").touch()
        out, err = coordinator.communicate(timeout=120)
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
    assert coordinator.returncode == 0, err
    assert json.loads(out.strip().splitlines()[-1])["owed"] == ["standin"]

    after = boot()  # a boot after the failure: inside the backoff, nothing is started

    assert _lines(signals / "attempts") == 1, "the coordinator repeated a failing pass a boot re-requested"
    assert _lines(signals / "spawned") == 0, "a boot started another installer while one was running"
    assert after.get("spawned") is False
