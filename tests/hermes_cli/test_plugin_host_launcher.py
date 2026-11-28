"""plugins.isolation: host — host launcher selection, env expansion, multiplexing and provider extraction."""

import json
import os
import sys

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from tests.hermes_cli.plugin_host_support import (
    PROBE_PLUGIN,
    _home_with_plugins,
    _hosted_model_provider,
)


def test_a_marked_profiles_own_launcher_is_not_used(tmp_path, monkeypatch):
    """``plugins.host.launcher`` is an argv prefix the host starts under. The backend of a profile a plugin
    runs elsewhere writes that profile's config.yaml, so only the managed overlay's launcher counts there."""
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_cli.plugin_isolation import host_launcher

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"isolation": "host", "host": {"launcher": ["profile-runner"]}}}), encoding="utf-8")
    assert host_launcher() == ["profile-runner"]
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    assert host_launcher() == []
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["operator-runner"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    assert host_launcher() == ["operator-runner"]


def test_a_managed_launcher_expands_its_env_refs_for_a_marked_profile_too(tmp_path, monkeypatch):
    """``${VAR}`` refs in the managed overlay expand against the process env. A profile a plugin runs
    elsewhere takes its launcher from that overlay alone, and gets the same expanded value as any profile."""
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_cli.plugin_isolation import host_launcher

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["${PLUGIN_RUNNER}", "--quiet"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("PLUGIN_RUNNER", "operator-runner")
    assert host_launcher() == ["operator-runner", "--quiet"]
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    assert host_launcher() == ["operator-runner", "--quiet"]


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts
def test_a_single_profile_launcher_expands_against_the_process_env_not_a_later_env_file(tmp_path, monkeypatch):
    """Outside multiplexing a launcher's ``${VAR}`` is the running process's value, as on main: an ``.env`` edited
    after the process loaded it does not pick a different launcher for the next host start."""
    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    loaded_ran, edited_ran = runner("loaded-runner"), runner("edited-runner")
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "loaded-runner"))
    (home / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'edited-runner'}\n", encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert loaded_ran.exists() and not edited_ran.exists()
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("surface", ["tui_gateway", "dashboard"])
def test_a_host_started_inside_the_launch_profiles_bound_scope_takes_its_launcher_value_from_it(
        tmp_path, monkeypatch, surface):
    """The TUI gateway and the dashboard bind the launch profile's secrets around their bodies even single-profile
    (``.env`` over the live env; the dashboard stamps that scope with the launch home). A host started inside such a
    body expands its launcher from that scope, as on main, so a value only the ``.env`` holds (set after the process
    loaded it) still reaches the launcher."""
    from hermes_cli.web_server_profiles import _config_profile_scope
    from tui_gateway import server

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.delenv("PLUGIN_RUNNER", raising=False)
    (home / ".env").write_text(f"PLUGIN_RUNNER={runner}\n", encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        body = (server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False)
                if surface == "tui_gateway" else _config_profile_scope(None))
        with body:
            host.ensure_started()
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


def _tagging_launch_host(tmp_path, monkeypatch, dotenv: str):
    """The launch profile's host with a ``[runner, ${RUNNER_TAG}]`` launcher whose runner records its tag, the home's
    ``.env`` set to *dotenv*. Returns the host, the tags file and the home."""
    tags, runner = tmp_path / "tags", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$1" >> "{tags}"\nshift\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": [str(runner), "${RUNNER_TAG}"]}}}), encoding="utf-8")
    (home / ".env").write_text(dotenv, encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    return plugins_mod.get_plugin_manager()._plugin_host(), tags, home


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("dotenv, expected", [
    ("", "${RUNNER_TAG}"),  # a name the launch scope lacks stays unresolved
    ("RUNNER_TAG=${RUNNER_OVERRIDE:-from-the-env-file}\n", "from-the-env-file"),  # a template keeps its default
], ids=["missing_name", "env_file_template"])
def test_a_launch_body_under_task_local_multiplexing_gives_its_launcher_no_value_written_to_the_live_env_since(
        tmp_path, monkeypatch, dotenv, expected):
    """A launch-profile body under task-local multiplexing (the desktop cron ticker) never reads a value from the
    live process env, which another profile's task may have written since: a name its scope lacks stays
    unresolved, and a ``.env`` template resolves against the launch env frozen when multiplexing began. The
    launcher of a host it starts reads its values the same way."""
    from agent import secret_scope as ss
    from tui_gateway import server

    for name in ("RUNNER_TAG", "RUNNER_OVERRIDE"):
        monkeypatch.delenv(name, raising=False)
    host, tags, _ = _tagging_launch_host(tmp_path, monkeypatch, dotenv)
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            for name in ("RUNNER_TAG", "RUNNER_OVERRIDE"):
                monkeypatch.setenv(name, "written-after-the-scope-was-bound")
            assert ss.get_secret("RUNNER_TAG") == (expected if dotenv else None)
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == [expected]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_launch_hosts_launcher_and_child_resolve_its_env_file_the_same_way_under_multiplexing(tmp_path, monkeypatch):
    """Under multiplexing the launch profile's launcher resolves its ``.env`` templates against the launch env frozen
    when multiplexing began. The host child's env carries that ``.env`` resolved the same way, so one start never
    gives the launcher and its child two values for one setting after a name the ``.env`` references changed in
    the live env. A process-global key is read from the process env by both, as the reload since left it, and a
    line naming one sees that value."""
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from tui_gateway import launch_profile_policy

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setenv("RUNNER_TAG", "registered-for-restore")
    monkeypatch.delenv("RUNNER_TAG")
    monkeypatch.setenv("RUNNER_BASE", "/old")
    monkeypatch.setenv("TERMINAL_RUNNER_TAG", "/old/term")
    seen, runner = tmp_path / "seen", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$1|$2|$3|${{RUNNER_TAG:-unset}}|${{TERMINAL_RUNNER_TAG:-unset}}|'
                      f'${{RUNNER_TERM:-unset}}" >> "{seen}"\nshift 3\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host", "host": {
        "launcher": [str(runner), "${RUNNER_TAG}", "${TERMINAL_RUNNER_TAG}", "${RUNNER_TERM}"]}}}), encoding="utf-8")
    (home / ".env").write_text("RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nTERMINAL_RUNNER_TAG=${RUNNER_BASE:-unset}/term\n"
                               "RUNNER_TERM=${TERMINAL_RUNNER_TAG}/sub\n", encoding="utf-8")
    ss.set_multiplex_active(True)
    launch_profile_policy.capture_launch_env()  # the launch env freezes with RUNNER_BASE=/old
    monkeypatch.setenv("RUNNER_BASE", "/new")  # written to the live env since
    monkeypatch.setenv("TERMINAL_RUNNER_TAG", "/new/term")  # and the process-global key a reload resolved from it
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert seen.read_text(encoding="utf-8").split() == [
            "/old/tag|/new/term|/new/term/sub|/old/tag|/new/term|/new/term/sub"]
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("reloaded", [False, True], ids=["loaded_once", "reloaded_since_the_freeze"])
def test_a_launch_launchers_env_file_template_resolves_as_the_process_loaded_the_file(
        tmp_path, monkeypatch, reloaded):
    """Under multiplexing the launch profile's launcher resolves its ``.env`` templates against the frozen launch
    env with the process's own dotenv output peeled back, as the process loaded the file and the body reads it: a
    line naming a key the file only sets later takes its default, not that later value. A reload since the freeze
    that changed that value does not change it."""
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from tui_gateway import server

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    for name in ("RUNNER_TAG", "RUNNER_BASE"):
        monkeypatch.setenv(name, "registered-for-restore")
        monkeypatch.delenv(name)
    host, tags, home = _tagging_launch_host(tmp_path, monkeypatch, "RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/old\n")
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["RUNNER_TAG"] == "unset/tag"
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            assert ss.get_secret("RUNNER_TAG") == "unset/tag"
            if reloaded:
                (home / ".env").write_text("RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/new\n", encoding="utf-8")
                env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
                assert os.environ["RUNNER_BASE"] == "/new"
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == ["unset/tag"]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_reload_racing_the_launch_env_freeze_cannot_split_the_frozen_env_from_its_dotenv_record(
        tmp_path, monkeypatch):
    """The launch env freezes once, with the dotenv publication record that peels its dotenv output back for the
    launcher's ``.env`` templates. A ``.env`` reload racing that freeze waits for it, so the frozen env is never
    paired with a later reload's record (which no longer recognises its values) until the process restarts."""
    import threading

    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from tui_gateway import launch_profile_policy, server

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    for name in ("RUNNER_TAG", "RUNNER_BASE"):
        monkeypatch.setenv(name, "registered-for-restore")
        monkeypatch.delenv(name)
    host, tags, home = _tagging_launch_host(tmp_path, monkeypatch, "RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/old\n")
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    (home / ".env").write_text("RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/new\n", encoding="utf-8")
    env_copied, reloaded = threading.Event(), threading.Event()
    record = env_loader.dotenv_publication_record

    def record_after_the_env_copy():
        env_copied.set()
        reloaded.wait(2)  # a reload not held off by the freeze lands here, between the two copies
        return record()

    def reload():
        env_copied.wait(10)
        env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
        reloaded.set()

    monkeypatch.setattr(env_loader, "dotenv_publication_record", record_after_the_env_copy)
    racer = threading.Thread(target=reload, daemon=True)
    racer.start()
    launch_profile_policy.capture_launch_env()
    racer.join(10)
    assert reloaded.is_set() and os.environ["RUNNER_BASE"] == "/new"
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == ["unset/tag"]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts
@pytest.mark.parametrize("caller", ["launch", "secondary"])
def test_a_caller_bound_to_another_profile_lends_its_scope_to_no_other_profiles_launcher(
        tmp_path, monkeypatch, caller):
    """A caller bound to one profile gives another profile's launcher none of its values. A body bound to the
    launch profile that starts a secondary's host leaves the secondary on its own ``.env``, and a body bound to a
    secondary that starts the launch profile's host leaves it on the env the process loaded, not on a launch
    ``.env`` edited since."""
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    ops_ran, launch_ran, edited_ran = runner("ops-runner"), runner("launch-runner"), runner("edited-runner")
    ops = _home_with_plugins(tmp_path / "ops", monkeypatch, {})
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    launcher = yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}})
    for home in (ops, launch):
        (home / "config.yaml").write_text(launcher, encoding="utf-8")
    (ops / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'ops-runner'}\n", encoding="utf-8")
    (launch / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'edited-runner'}\n", encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))
    plugins_mod._reset_plugin_managers_for_tests()
    started, bound_to = (ops, None) if caller == "launch" else (launch, str(ops))
    home_token = set_hermes_home_override(started)
    try:
        host = plugins_mod.get_plugin_manager()._plugin_host()
    finally:
        reset_hermes_home_override(home_token)
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": bound_to}, hydrate_secrets=False):
            host.ensure_started()
        ran = [path.exists() for path in (ops_ran, launch_ran, edited_ran)]
        assert ran == ([True, False, False] if started == ops else [False, True, False])
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts
def test_a_secondary_profiles_launcher_under_task_local_multiplexing_expands_from_its_own_env_file(
        tmp_path, monkeypatch):
    """A process that is not a multiplexer still serves another profile in a task (the desktop backend's cron
    ticker firing a sibling's job): that profile's launcher expands from its own ``.env``, never the launch
    profile's value for the name, whichever context the host start runs in."""
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway.launch_profile_policy import owner_secret_scope

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    own_ran, launch_ran = runner("own-runner"), runner("launch-runner")
    ops = _home_with_plugins(tmp_path / "ops", monkeypatch, {})
    _home_with_plugins(tmp_path, monkeypatch, {})
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    (ops / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'own-runner'}\n", encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))
    plugins_mod._reset_plugin_managers_for_tests()
    context_token = ss.set_multiplex_context(True)
    home_token = set_hermes_home_override(ops)
    scope_token = ss.set_secret_scope(owner_secret_scope(ops), profile_home=str(ops))
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert own_ran.exists() and not launch_ran.exists()
    finally:
        host.shutdown()
        ss.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts; SIGKILL
@pytest.mark.parametrize("crashed", ["secondary", "launch"])
def test_a_crash_restart_under_multiplexing_runs_the_profiles_own_launcher(tmp_path, monkeypatch, crashed):
    """A crash restart starts the host from a thread with no binding. A ``${VAR}`` launcher still expands from
    the crashed profile's own secrets: a secondary's ``.env`` (never the launch profile's value for the name),
    or the launch profile's env-only keys (systemd ``Environment=``), which no file holds."""
    import signal

    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.waiting import wait_until
    from tools.registry import registry
    from tui_gateway.launch_profile_policy import owner_secret_scope

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    def runs(ran):
        return ran.read_text(encoding="utf-8").count("ran") if ran.exists() else 0

    own_ran, launch_ran = runner("own-runner"), runner("launch-runner")
    ops = _home_with_plugins(tmp_path / "ops", monkeypatch, {"hostprobe": PROBE_PLUGIN})
    launch = _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN})
    home = ops if crashed == "secondary" else launch
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["hostprobe"], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    if crashed == "secondary":
        (ops / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'own-runner'}\n", encoding="utf-8")
        monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))
    else:
        monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "own-runner"))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    # The first start runs inside the profile's binding, as a gateway's load of that profile's plugins does.
    home_token = set_hermes_home_override(home)
    scope_token = ss.set_secret_scope(owner_secret_scope(home), profile_home=str(home))
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        first = manager._plugins["hostprobe"]
        assert first.error is None
        assert runs(own_ran) == 1
        crashed_pid = manager._plugin_host().pid
        os.kill(crashed_pid, signal.SIGKILL)

        def reloaded():
            loaded = manager._plugins.get("hostprobe")
            return loaded if loaded is not first else None

        # The restarted host answered hello and loaded the plugin again, so both launcher counts are final.
        after = wait_until(reloaded, timeout=15, what="the crashed host's restart and reload")
        assert after is not None and after.error is None
        answer = json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))
        assert answer["pid"] not in {crashed_pid, os.getpid()}
        assert (runs(own_ran), runs(launch_ran)) == (2, 0)
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        ss.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script; SIGKILL
@pytest.mark.parametrize("env_file, tag", [
    ("RUNNER_TAG=from-env-file\n", "from-env-file"),
    ("RUNNER_TAG=${LAUNCH_ONLY_TAG:-not-the-launch-env}\n", "from-launch-env"),  # a template of an env-only key
], ids=["plain_value", "template_of_an_env_only_key"])
def test_a_crash_restart_after_the_launch_profiles_link_is_renamed_keeps_its_env_only_launcher_keys(
        tmp_path, monkeypatch, env_file, tag):
    """The process keeps naming its own profile by the path it started with. After that link is renamed the
    manager answers to the new name, and a crash restart under multiplexing still expands the launcher from both
    of the launch profile's sources: its env-only keys (systemd ``Environment=``) and its ``.env``, read through
    the name that now leads to it, its templates resolved as the process's own against those env-only keys. A
    start inside a launch-profile body bound since (its scope read through the old name, so without the ``.env``)
    builds that scope again through the new name rather than borrowing the body's. The host child's own env is
    still the launch profile's too: a credential only the service env holds, and its ``.env`` resolved the same
    way."""
    import signal

    from agent import secret_scope as ss
    from tests.waiting import wait_until
    from tui_gateway import server

    ran, seen, runner = tmp_path / "launcher.ran", tmp_path / "child-env.seen", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$1" >> "{ran}"\n'
                      f'echo "${{RUNNER_TAG:-unset}}|${{OPENAI_API_KEY:-unset}}" >> "{seen}"\nshift\nexec "$@"\n',
                      encoding="utf-8")
    runner.chmod(0o755)
    child = f"{tag}|from-the-service"
    outside = _home_with_plugins(tmp_path / "outside", monkeypatch, {"hostprobe": PROBE_PLUGIN})
    (outside / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["hostprobe"], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}", "${RUNNER_TAG}"]}}}),
        encoding="utf-8")
    (outside / ".env").write_text(env_file, encoding="utf-8")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    monkeypatch.setenv("PLUGIN_RUNNER", str(runner))
    monkeypatch.setenv("LAUNCH_ONLY_TAG", "from-launch-env")
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-service")  # e.g. systemd Environment=, no .env line
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        first = manager._plugins["hostprobe"]
        assert first.error is None and ran.read_text(encoding="utf-8").split() == [tag]
        assert seen.read_text(encoding="utf-8").split() == [child]
        (profiles / "ops").rename(profiles / "ops2")
        os.kill(manager._plugin_host().pid, signal.SIGKILL)
        def reloaded():
            loaded = manager._plugins.get("hostprobe")
            return loaded if loaded is not first else None

        after = wait_until(reloaded, timeout=15, what="the crashed host's restart")
        assert after is not None and after.error is None
        assert ran.read_text(encoding="utf-8").split() == [tag, tag]
        assert seen.read_text(encoding="utf-8").split() == [child, child]
        manager._plugin_host().shutdown()
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            assert ss.get_secret("RUNNER_TAG") is None
            manager._plugin_host().ensure_started()
        assert ran.read_text(encoding="utf-8").split() == [tag] * 3
        assert seen.read_text(encoding="utf-8").split() == [child] * 3
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_marked_profiles_managed_launcher_expands_against_the_process_env_under_multiplexing(tmp_path, monkeypatch):
    """Under multiplexing too, a profile a plugin runs elsewhere takes its launcher from the managed overlay alone,
    its ``${VAR}``s expanded against the process env as for any profile, not through a scope built for its home."""
    from agent import secret_scope as ss
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host"}}), encoding="utf-8")
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=launch)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("PLUGIN_RUNNER", str(runner))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(ops)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        reset_hermes_home_override(token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("caller", ["none", "bound_to_the_secondary"])
def test_a_secondary_profiles_managed_launcher_expands_against_the_process_env_under_multiplexing(
        tmp_path, monkeypatch, caller):
    """The managed overlay's ``${VAR}``s expand against the process env, so a user's ``.env`` cannot stand in for
    an administrator's value. A secondary profile's launcher is read inside that profile's own secret scope, but
    a managed launcher still takes a name only the process env holds (systemd ``Environment=``)."""
    import contextlib

    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host"}}), encoding="utf-8")
    (ops / ".env").write_text("PLUGIN_RUNNER=/not/the/operators/runner\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["${SANDBOX_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("SANDBOX_RUNNER", str(runner))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(ops)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    body = (server._session_profile_runtime_scope({"profile_home": str(ops)}, hydrate_secrets=False)
            if caller == "bound_to_the_secondary" else contextlib.nullcontext())
    try:
        with body:
            host.ensure_started()
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        reset_hermes_home_override(token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_secondarys_provider_extraction_runs_under_the_launcher_its_host_start_chose(tmp_path, monkeypatch):
    """With no extraction cached, a host start extracts the profile's hosted model providers while it builds the
    child's env. That extraction child runs under the launcher the start chose for the profile, its ``${VAR}``s
    read through the profile's own files, even when only the starting task is multiplexed and the host's own
    context holds no secret scope. An extraction there once the start is over chooses again."""
    from hermes_cli.plugin_isolation import extraction_launcher
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    seen, runner = tmp_path / "seen", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$1" >> "{seen}"\nshift\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host", "host": {
        "launcher": [str(runner), "${PLUGIN_TAG}"]}}}), encoding="utf-8")
    (ops / ".env").write_text("PLUGIN_TAG=ops-tag\n", encoding="utf-8")
    _hosted_model_provider(ops)
    monkeypatch.setenv("PLUGIN_TAG", "launch-tag")  # the launch profile's value
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(ops)
    context_token = ss.set_multiplex_context(True)  # this task only (the desktop cron ticker)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert seen.read_text(encoding="utf-8").split() == ["ops-tag", "ops-tag"]  # the extraction, then the host
        (ops / ".env").write_text("PLUGIN_TAG=ops-tag-edited\n", encoding="utf-8")
        assert host._base_context.run(extraction_launcher, ops) == [str(runner), "ops-tag-edited"]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        reset_hermes_home_override(token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("env_builder", ["imported", "first_imported_by_the_start"])
def test_a_launch_body_starts_provider_extraction_under_the_launcher_it_chose_for_the_host(env_builder, tmp_path,
                                                                                         monkeypatch):
    """A host start inside a launch-profile body (the TUI gateway binds one even single-profile) reads its launcher
    through the launch scope, the ``.env`` as it is now. A cold provider extraction that start runs takes that
    same launcher, whether its ambient-credential scan or its first import of the child-env builder runs it,
    though the host's own context binds no scope and the process env still holds what the process loaded
    before the ``.env`` was edited."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from tui_gateway import server

    monkeypatch.setenv("RUNNER_TAG", "loaded-before-the-edit")
    host, tags, home = _tagging_launch_host(tmp_path, monkeypatch, "RUNNER_TAG=edited-since\n")
    _hosted_model_provider(home)
    if env_builder == "first_imported_by_the_start":  # its blocklist reads the registry, which extracts
        for module in ("tools.environments.local", "tools.environments.local_env_policy"):
            monkeypatch.delitem(sys.modules, module, raising=False)
        PROVIDER_REGISTRY.sync_before_first_read()
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == ["edited-since", "edited-since"]  # extraction, host
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_profile_marked_after_its_start_chose_does_not_lend_that_launcher_to_its_extraction(tmp_path, monkeypatch):
    """A host start lends the launcher it chose to the provider extraction it runs. When the profile is marked to
    run elsewhere between that choice and the extraction, the extraction takes the launcher a marked profile takes
    (the managed overlay's, here none), not the profile's own."""
    from hermes_cli import plugin_isolation as iso
    from hermes_cli import profiles_run_elsewhere as pre
    import tools.environments.local  # loaded, as in a running process: its import lists the providers

    seen, runner = tmp_path / "seen", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$*" >> "{seen}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host", "host": {
        "launcher": [str(runner)]}}}), encoding="utf-8")
    _hosted_model_provider(home)
    choose = iso.launcher_for

    def chosen_then_marked(*args, **kwargs):
        choice = choose(*args, **kwargs)
        if pre.run_elsewhere_refusal(home) is None:
            pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
        return choice

    monkeypatch.setattr(iso, "launcher_for", chosen_then_marked)
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()
    assert list((home / "cache" / "plugin_host" / "model-providers").glob("hostmodel-*.json"))  # it extracted
    ran = seen.read_text(encoding="utf-8").splitlines() if seen.exists() else []
    assert not [line for line in ran if "--extract-profiles" in line]


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_managed_launcher_that_expands_to_nothing_is_not_replaced_by_a_profiles_env_file_value(tmp_path, monkeypatch):
    """A managed launcher whose ``${VAR}`` the process env sets empty is still the administrator's: the host starts
    with no launcher rather than expanding the same reference again against the profile's ``.env``."""
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host"}}), encoding="utf-8")
    (ops / ".env").write_text(f"SANDBOX_RUNNER={runner}\n", encoding="utf-8")  # the user's, not the operator's
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": "${SANDBOX_RUNNER}"}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("SANDBOX_RUNNER", "")
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(ops)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert host.alive and not ran.exists()
    finally:
        host.shutdown()
        reset_hermes_home_override(token)
        plugins_mod._reset_plugin_managers_for_tests()
