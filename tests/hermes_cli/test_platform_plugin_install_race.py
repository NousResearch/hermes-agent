"""Unattended platform plugin installs: one install per home, outcomes that tell the truth.

The race runs two real processes against one temp home: the detached default-plugin pass and a gateway
start's left-core migration, each installing through the real git install path from a local repo. The
other tests use real config/.env files and a stand-in row; the network install is a stand-in that places
the plugin directory (the catalog install is exercised end to end outside the unit suite).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm

# A stand-in platform row and catalog, shared by the race processes and the in-process tests.
_STANDIN = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    from tests.waiting import wait_until
    import hermes_cli.left_core_migration as lcm
    import hermes_cli.memory_provider_migration as mpm
    import hermes_cli.plugin_catalog as catalog
    from hermes_cli import plugins_cmd

    home, repo, signals = Path(os.environ["HERMES_HOME"]), sys.argv[2], Path(sys.argv[3])
    catalog.fetch_live_catalog = lambda **kwargs: None
    plugins_cmd._scan_on_install_enabled = lambda: False

    def wait_for(name):  # a handshake that never arrives fails this process (and the test) loudly
        wait_until(lambda: (signals / name).exists(), timeout=60, what=f"signal {name!r}")

    def real_install_into(home, consent=None, before=None):
        def install(name, discard=None):
            if before:
                before()
            try:  # the real clone-and-place path, whose refusal of an existing tree is the race
                plugins_cmd._install_plugin_core(f"file://{repo}", force=False)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True}
        return install
""")

_DEFAULT_PASS = _STANDIN + textwrap.dedent("""
    from hermes_cli import default_plugins
    from hermes_cli.plugin_catalog import entry_from_mapping
    entry = entry_from_mapping({"name": "standin", "repo": "https://github.com/NousResearch/standin",
                                "sha": "a" * 40, "tier": "official", "default": True, "category": "platform"}, "x")
    default_plugins.default_entries = lambda: [entry]
    catalog.fetch_live_catalog = lambda **kwargs: {"entries": [{}]}
    # Mid-pass (the installer lock held) when the left-core start decides to install; its tree lands after.
    lcm._install_into = lambda home, consent=None: real_install_into(
        home, before=lambda: ((signals / "installing").touch(), wait_for("pending")))
    out = default_plugins.install_defaults_all_homes(say=lambda m: None, wait=30)
    (signals / "placed").touch()
    print(json.dumps(out))
""")

_GATEWAY_START = _STANDIN + textwrap.dedent("""
    def in_use(home, **_):
        wait_for("installing")
        (signals / "pending").touch()
        return True
    lcm.LEFT_CORE = (lcm.LeftCoreFeature(plugin="standin", label="Stand-in", in_use=in_use,
                                         unchanged="nothing changes", platform="standin"),)
    mpm.catalog_source = lambda name: name
    # Not the bounded gateway-start budget (its own test below): this start waits for the pass to finish.
    lcm._GATEWAY_START_LOCK_WAIT = 600.0
    called = []
    lcm._install_into = lambda home, consent=None: real_install_into(
        home, before=lambda: (called.append(1), wait_for("placed")))
    installed = lcm.recover_at_startup(gateway_start=True)
    print(json.dumps({"installed": installed, "called": len(called), "notices": lcm.claim_kept_notices()}))
""")


def _plugin_repo(root: Path) -> Path:
    repo = root / "standin-repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: standin\nmanifest_version: 1\n", encoding="utf-8")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    git = ["git", "-c", "user.email=f@example.com", "-c", "user.name=F"]
    subprocess.run([*git, "init", "-qb", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run([*git, "commit", "-qm", "v1"], cwd=repo, check=True, capture_output=True)
    return repo


def test_a_gateway_start_beside_the_default_pass_installs_once_and_reports_no_failure(tmp_path):
    """The detached default-plugin child and a gateway start's left-core migration both owed the same platform
    plugin. The start used to install while the child's tree landed and recorded the child's success as its
    own failure: a failure stamp (the next starts skipped the row) and a "could not be installed" notice."""
    home, signals = tmp_path / "home", tmp_path / "signals"
    home.mkdir()
    signals.mkdir()
    (home / "config.yaml").write_text("_config_version: 1\n", encoding="utf-8")
    repo = _plugin_repo(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "PYTHONPATH"))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(Path(lcm.__file__).parents[1]))
    args = [str(repo), str(signals)]
    default_pass = subprocess.Popen([sys.executable, "-c", _DEFAULT_PASS, "a", *args], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gateway = subprocess.run([sys.executable, "-c", _GATEWAY_START, "b", *args], env=env, check=False,
                             capture_output=True, text=True, timeout=120)
    out_a, err_a = default_pass.communicate(timeout=120)
    assert gateway.returncode == 0 and default_pass.returncode == 0, (gateway.stderr, err_a)

    started = json.loads(gateway.stdout.strip().splitlines()[-1])
    assert json.loads(out_a.strip().splitlines()[-1])["installed"] == ["standin"]
    assert started["called"] == 0, "the start installed over the tree the default pass placed"
    assert started["installed"] == ["standin"]  # it is in the home: the runner reloads its platform config
    assert not any("could not be installed" in notice for notice in started["notices"]), started["notices"]
    assert not (home / "cache" / "left-core-standin.failed").exists()
    assert "standin" in lcm._read_config(home)["_left_core_installed"]


@pytest.fixture
def standin(tmp_path, monkeypatch):
    """A stand-in platform row whose predicate is its credential in the home's .env; installs place the tree."""
    import hermes_cli.memory_provider_migration as mpm
    import hermes_cli.plugin_catalog as catalog
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delenv("STANDIN_TOKEN", raising=False)
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_attempted_with", {}, raising=False)
    monkeypatch.setattr(catalog, "fetch_live_catalog", lambda **kwargs: None)
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)
    row = lcm.LeftCoreFeature(plugin="standin", label="Stand-in", unchanged="nothing changes", platform="standin",
                              in_use=lambda h, **_: lcm._home_env(h, "STANDIN_TOKEN", process_env=False) != "")
    monkeypatch.setattr(lcm, "LEFT_CORE", (row,))
    outcome = {"ok": True}
    calls = []

    def install_into(h, consent=None):
        def install(name, discard=None):
            calls.append(name)
            if outcome["ok"]:
                (h / "plugins" / name).mkdir(parents=True)
            return outcome
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    return home, calls, outcome


def test_credentials_added_after_a_start_are_picked_up_by_the_next_scan_in_that_process(standin):
    """A served profile's rescan (its .env edited while the gateway runs) re-enters the start hook, which
    tried that home once per process and so never installed the platform the new credentials configure."""
    home, calls, _ = standin
    assert lcm.recover_at_startup(gateway_start=True) == []
    (home / ".env").write_text("STANDIN_TOKEN=t\n", encoding="utf-8")
    assert lcm.recover_at_startup(gateway_start=True) == ["standin"]
    assert lcm.recover_at_startup(gateway_start=True) == [] and calls == ["standin"]


def test_credentials_added_while_a_start_installs_are_not_marked_checked(standin, monkeypatch):
    """The start records the settings it saw only once they held still: credentials for a second platform
    written while the first one installed (or waited for the installer lock) are installed in that start."""
    home, calls, _ = standin
    first = lcm.LEFT_CORE[0]
    second = lcm.LeftCoreFeature(plugin="other", label="Other", unchanged="nothing changes", platform="other",
                                 in_use=lambda h, **_: lcm._home_env(h, "OTHER_TOKEN", process_env=False) != "")
    monkeypatch.setattr(lcm, "LEFT_CORE", (first, second))
    (home / ".env").write_text("STANDIN_TOKEN=t\n", encoding="utf-8")
    install_into = lcm._install_into

    def editing_install_into(h, consent=None):
        install = install_into(h, consent)

        def run(name, discard=None):
            if name == "standin":  # the user saves the second platform's credentials meanwhile
                (h / ".env").write_text("STANDIN_TOKEN=t\nOTHER_TOKEN=o\n", encoding="utf-8")
            return install(name, discard)
        return run
    monkeypatch.setattr(lcm, "_install_into", editing_install_into)

    assert lcm.recover_at_startup(gateway_start=True) == ["standin", "other"]
    assert calls == ["standin", "other"]


def test_a_gateway_start_retries_inside_the_failure_backoff_and_agent_starts_do_not(standin):
    """Offline first start, network restored, gateway restarted: the restart used to sit out the backoff in
    silence (no install, no line); only the frequent agent starts are spared the round trip."""
    home, calls, outcome = standin
    (home / ".env").write_text("STANDIN_TOKEN=t\n", encoding="utf-8")
    outcome.update(ok=False, error="Could not resolve host: github.com")
    assert lcm.recover_at_startup(gateway_start=True) == []
    assert (home / "cache" / "left-core-standin.failed").exists()
    lcm._attempted.clear()
    assert lcm.recover_at_startup() == [] and calls == ["standin"], "an agent start ignored the backoff"
    lcm._attempted.clear()
    outcome.clear()
    outcome["ok"] = True
    assert lcm.recover_at_startup(gateway_start=True) == ["standin"]


def test_catalog_platform_credentials_install_a_default_entry_and_point_to_any_other(tmp_path, monkeypatch):
    """A catalog platform that never shipped in core, its credentials set: a default (official/verified)
    entry is installed at gateway start; a community one gets the command that installs it, never an
    unattended install."""
    import hermes_cli.plugin_catalog as catalog
    from hermes_cli import catalog_platforms, default_plugins
    from hermes_cli.plugin_catalog import entry_from_mapping

    def entry(name, tier, default):
        return entry_from_mapping({"name": name, "repo": "https://github.com/example/x", "sha": "a" * 40,
                                   "tier": tier, "default": default, "category": "platform",
                                   "capabilities": {"requires_env": [f"{name.upper()}_TOKEN"]}}, name)
    entries = [entry("chatone", "official", True), entry("chattwo", "community", False),
               entry("chatthree", "community", False)]
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("CHATONE_TOKEN=a\nCHATTWO_TOKEN=b\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: entries)
    monkeypatch.setattr(default_plugins, "default_entries", lambda: [e for e in entries if e.default])
    monkeypatch.setattr(catalog_platforms, "_attempted", {})
    calls = []

    def install_into(h, consent=None):
        def install(name, discard=None):
            calls.append(name)
            (h / "plugins" / name).mkdir(parents=True)
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    said = []

    assert catalog_platforms.recover_in(home, report=said.append) == ["chatone"]
    assert calls == ["chatone"]
    told = [line for line in said if "chattwo" in line]
    assert len(told) == 1 and "`hermes plugins install chattwo`" in told[0] and "CHATTWO_TOKEN" in told[0]
    assert not any("chatthree" in line for line in said)


def test_a_catalog_platform_whose_credentials_another_platform_serves_is_not_reported(tmp_path, monkeypatch):
    """The real catalog's community ``hermes-tag`` requires ``FEISHU_APP_ID``/``FEISHU_APP_SECRET``, which the
    Feishu platform plugin this home has also requires: those credentials are served, so a working Feishu
    user is not told on every start that 'hermes-tag' is missing. A platform nothing serves still is."""
    import hermes_cli.plugin_catalog as catalog
    from hermes_cli import catalog_platforms

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("FEISHU_APP_ID=a\nFEISHU_APP_SECRET=b\nFLUXER_BOT_TOKEN=c\n", encoding="utf-8")
    for name in ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FLUXER_BOT_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(catalog, "fetch_live_catalog", lambda **kwargs: None)
    names = {entry.name for entry in catalog.load_catalog()}
    assert {"hermes-tag", "fluxer-platform"} <= names  # the real in-tree catalog file

    pending = {entry.name for entry in catalog_platforms.pending_entries(home)}

    assert "hermes-tag" not in pending
    assert "fluxer-platform" in pending


def test_a_tree_a_refused_enable_left_is_owed_and_the_next_start_enables_it(standin, monkeypatch):
    """An enable that lost to other writers of the plugin selection leaves the plugin on disk but disabled:
    that tree used to count as installed, so no later start ever enabled it."""
    home, calls, outcome = standin
    (home / ".env").write_text("STANDIN_TOKEN=t\n", encoding="utf-8")

    def install_into(h, consent=None):
        def install(name, discard=None):  # places the tree either way, as the real install does
            calls.append(name)
            (h / "plugins" / name).mkdir(parents=True, exist_ok=True)
            return dict(outcome)
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    outcome.update(ok=False, raced=True, error="enable refused: plugin configuration changed; retry")
    assert lcm.recover_at_startup(gateway_start=True) == []
    lcm._attempted.clear()
    outcome.clear()
    outcome["ok"] = True  # the next start: the enable goes through, on the tree already there
    assert lcm.recover_at_startup(gateway_start=True) == ["standin"]
    assert calls == ["standin", "standin"]


_HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from hermes_cli import default_plugins
    with default_plugins.installer_lock(5):
        Path(sys.argv[1]).touch()
        time.sleep(120)
""")


def test_a_gateway_start_waits_one_bounded_budget_for_another_installer(standin, monkeypatch, tmp_path):
    """The detached default-plugin pass holds the installer lock for its whole pass. A gateway start whose
    profile has credentials for two platforms it owes waited up to five minutes per row for it, serving
    nothing meanwhile (Docker: ~70 s before api_server answered). It now waits one budget, then starts."""
    import threading
    from gateway.run_startup import recover_left_core_at_gateway_start
    home, calls, _ = standin
    rows = tuple(lcm.LeftCoreFeature(plugin=name, label=name.title(), unchanged="nothing changes", platform=name,
                                     in_use=lambda h, **_: True) for name in ("standin", "other"))
    monkeypatch.setattr(lcm, "LEFT_CORE", rows)
    monkeypatch.setattr(lcm, "_GATEWAY_START_LOCK_WAIT", 4.0, raising=False)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda *a, **k: home)
    from hermes_cli import default_plugins
    waits, real_lock = [], default_plugins.installer_lock

    def counted_lock(wait):
        waits.append(wait)
        return real_lock(wait)
    monkeypatch.setattr(default_plugins, "installer_lock", counted_lock)
    held = tmp_path / "held"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "PYTHONPATH"))}
    env.update(HERMES_HOME=str(home), PYTHONPATH=str(Path(lcm.__file__).parents[1]))
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(held)], env=env)
    try:
        from tests.waiting import wait_until
        wait_until(held.exists, timeout=60, what="the other installer to hold the lock")
        result = []
        worker = threading.Thread(target=lambda: result.append(recover_left_core_at_gateway_start()), daemon=True)
        worker.start()
        worker.join(timeout=120)
        assert not worker.is_alive(), "the gateway start still waits for the other installer"
    finally:
        holder.kill()
        holder.wait(timeout=30)
    # What the start was allowed to wait, not the wall clock (a loaded host adds seconds of its own):
    # both rows asked for the lock, and their waits add up to one budget, never one budget each.
    assert len(waits) == 2 and sum(waits) <= 4.0, f"each row waited its own budget: {waits}"
    assert result == [False] and calls == []
    notices = lcm.claim_kept_notices()
    assert any("another Hermes process is still installing plugins" in n for n in notices), notices


@pytest.mark.parametrize("gate", ["env", "config"])
def test_catalog_platforms_makes_no_network_fetch_when_lazy_installs_off(tmp_path, monkeypatch, gate):
    """Gateway start runs ``catalog_platforms.recover_in`` on every profile: when lazy installs are
    disabled (``HERMES_DISABLE_LAZY_INSTALLS=1`` or ``security.allow_lazy_installs: false``), neither
    ``pending_entries`` nor ``recover_in`` may fetch the live catalog over the network."""
    import hermes_cli.plugin_catalog as catalog
    from hermes_cli import catalog_platforms

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("FLUXER_BOT_TOKEN=tok\n", encoding="utf-8")
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    if gate == "env":
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    else:
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        (home / "config.yaml").write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")
    catalog_platforms._attempted.clear()

    def _forbid_fetch(**kwargs):
        raise AssertionError("fetch_live_catalog must not be called when lazy installs are off")

    monkeypatch.setattr(catalog, "fetch_live_catalog", _forbid_fetch)

    pending = [entry.name for entry in catalog_platforms.pending_entries(home)]
    assert "fluxer-platform" in pending

    notices: list[str] = []
    installed = catalog_platforms.recover_in(home, report=notices.append)
    assert installed == []
    assert any("fluxer-platform" in msg and "hermes plugins install" in msg for msg in notices), notices
