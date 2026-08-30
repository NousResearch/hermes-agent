from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tomllib
from types import SimpleNamespace

import pytest

from ares_runtime.local_runtime import (
    AresLocalPaths,
    AresLocalRuntime,
    AresLocalRuntimeError,
    _desktop_launch_arguments,
    _desktop_disable_gpu_policy,
    _parser,
    main,
)


def _runtime(tmp_path: Path) -> AresLocalRuntime:
    return AresLocalRuntime(
        AresLocalPaths(
            state_root=tmp_path / "state",
            data_root=tmp_path / "data",
            agent_home=tmp_path / "ares-home",
            launcher_path=tmp_path / "bin" / "ares",
            unit_path=tmp_path / "unit" / "ares-gateway.service",
        )
    )


def _release(runtime: AresLocalRuntime, revision: str) -> Path:
    source = runtime.paths.releases_dir / revision / "source"
    source.mkdir(parents=True)
    return source


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _commit(directory: Path, message: str) -> str:
    _git(directory, "add", ".")
    _git(directory, "commit", "-m", message)
    return _git(directory, "rev-parse", "HEAD")


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "--initial-branch", "main")
    _git(path, "config", "user.name", "Ares Runtime Tests")
    _git(path, "config", "user.email", "ares-runtime-tests@example.invalid")
    return path


def test_current_link_is_the_only_active_runtime_pointer(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = "a" * 40
    second = "b" * 40
    first_source = _release(runtime, first)
    second_source = _release(runtime, second)

    runtime._activate(first)
    runtime._activate(second)

    assert runtime.active_release() == (second, second_source.resolve())
    assert runtime.previous_release() == (first, first_source.resolve())


def test_rollback_swaps_current_and_previous_without_a_worktree_fallback(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = "a" * 40
    second = "b" * 40
    first_source = _release(runtime, first)
    second_source = _release(runtime, second)
    runtime._activate(first)
    runtime._activate(second)

    runtime._atomic_link(runtime.paths.current_link, first_source.resolve())
    runtime._atomic_link(runtime.paths.previous_link, second_source.resolve())

    assert runtime.active_release() == (first, first_source.resolve())
    assert runtime.previous_release() == (second, second_source.resolve())


def test_activation_failure_restores_the_complete_pointer_pair(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    prior_revision = "0" * 40
    current_revision = "a" * 40
    candidate_revision = "b" * 40
    prior_source = _release(runtime, prior_revision)
    current_source = _release(runtime, current_revision)
    candidate_source = _release(runtime, candidate_revision)
    runtime._activate(prior_revision)
    runtime._activate(current_revision)
    atomic_link = runtime._atomic_link

    def fail_candidate_current(link: Path, target: Path) -> None:
        if link == runtime.paths.current_link and target == candidate_source.resolve():
            raise OSError("injected current-pointer failure")
        atomic_link(link, target)

    monkeypatch.setattr(runtime, "_atomic_link", fail_candidate_current)

    with pytest.raises(OSError, match="injected current-pointer failure"):
        runtime._activate(candidate_revision)

    assert runtime.active_release() == (current_revision, current_source.resolve())
    assert runtime.previous_release() == (prior_revision, prior_source.resolve())


def test_rollback_pointer_failure_restores_the_complete_pair(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    previous_revision = "a" * 40
    current_revision = "b" * 40
    previous_source = _release(runtime, previous_revision)
    current_source = _release(runtime, current_revision)
    runtime._activate(previous_revision)
    runtime._activate(current_revision)
    atomic_link = runtime._atomic_link

    def fail_previous_swap(link: Path, target: Path) -> None:
        if link == runtime.paths.previous_link and target == current_source.resolve():
            raise OSError("injected previous-pointer failure")
        atomic_link(link, target)

    monkeypatch.setattr(runtime, "_atomic_link", fail_previous_swap)

    with pytest.raises(OSError, match="injected previous-pointer failure"):
        runtime.rollback()

    assert runtime.active_release() == (current_revision, current_source.resolve())
    assert runtime.previous_release() == (previous_revision, previous_source.resolve())


def test_config_only_tracks_update_source_not_active_release(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._write_config(remote="https://github.com/RecursiveIntell/Ares.git", branch="main")

    payload = json.loads(runtime.paths.config_path.read_text(encoding="utf-8"))

    assert payload == {
        "branch": "main",
        "remote": "https://github.com/RecursiveIntell/Ares.git",
        "schema_version": 2,
        "upstream_branch": "main",
        "upstream_remote": "https://github.com/NousResearch/hermes-agent.git",
    }


def test_legacy_update_config_receives_safe_upstream_defaults(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._atomic_json(
        runtime.paths.config_path,
        {
            "schema_version": 1,
            "remote": "https://github.com/RecursiveIntell/Ares.git",
            "branch": "main",
        },
    )

    config = runtime._read_config()

    assert config["upstream_remote"] == "https://github.com/NousResearch/hermes-agent.git"
    assert config["upstream_branch"] == "main"


def test_upstream_candidate_applies_downstream_delta_in_staging(tmp_path: Path, monkeypatch) -> None:
    upstream = _repository(tmp_path / "upstream")
    (upstream / "hermes.txt").write_text("base\n", encoding="utf-8")
    _commit(upstream, "base")

    downstream = tmp_path / "downstream"
    subprocess.run(["git", "clone", str(upstream), str(downstream)], check=True)
    _git(downstream, "config", "user.name", "Ares Runtime Tests")
    _git(downstream, "config", "user.email", "ares-runtime-tests@example.invalid")
    (downstream / "ares.txt").write_text("downstream patch\n", encoding="utf-8")
    downstream_revision = _commit(downstream, "Ares patch")

    (upstream / "upstream.txt").write_text("new Hermes feature\n", encoding="utf-8")
    upstream_revision = _commit(upstream, "upstream change")

    runtime = _runtime(tmp_path)
    monkeypatch.setattr(runtime, "_build_runtime", lambda source, *, desktop: None)

    candidate_revision = runtime._materialize_upstream_candidate(
        downstream_remote=str(downstream),
        downstream_revision=downstream_revision,
        upstream_remote=str(upstream),
        upstream_branch="main",
        upstream_revision=upstream_revision,
        desktop=False,
    )

    candidate = runtime._release_source(candidate_revision)
    assert (candidate / "upstream.txt").read_text(encoding="utf-8") == "new Hermes feature\n"
    assert (candidate / "ares.txt").read_text(encoding="utf-8") == "downstream patch\n"
    assert _git(candidate, "status", "--porcelain") == ""
    metadata = runtime._release_metadata(candidate_revision)
    assert metadata["upstream_revision"] == upstream_revision
    assert metadata["downstream_revision"] == downstream_revision


def test_update_activates_only_the_verified_upstream_candidate(tmp_path: Path, monkeypatch) -> None:
    upstream = _repository(tmp_path / "upstream")
    (upstream / "hermes.txt").write_text("base\n", encoding="utf-8")
    _commit(upstream, "base")

    downstream = tmp_path / "downstream"
    subprocess.run(["git", "clone", str(upstream), str(downstream)], check=True)
    _git(downstream, "config", "user.name", "Ares Runtime Tests")
    _git(downstream, "config", "user.email", "ares-runtime-tests@example.invalid")
    (downstream / "ares.txt").write_text("Ares patch\n", encoding="utf-8")
    _commit(downstream, "Ares patch")

    (upstream / "upstream.txt").write_text("new Hermes feature\n", encoding="utf-8")
    _commit(upstream, "upstream change")

    runtime = _runtime(tmp_path)
    runtime._write_config(
        remote=str(downstream),
        branch="main",
        upstream_remote=str(upstream),
        upstream_branch="main",
    )
    monkeypatch.setattr(runtime, "_build_runtime", lambda source, *, desktop: None)

    candidate_revision, changed = runtime.update(desktop=False)

    assert changed is True
    assert runtime.active_release()[0] == candidate_revision
    assert runtime._release_metadata(candidate_revision)["upstream_revision"] == _git(
        upstream, "rev-parse", "HEAD"
    )
    assert runtime.update(desktop=False) == (candidate_revision, False)


def test_upstream_candidate_conflict_never_publishes_a_release(tmp_path: Path, monkeypatch) -> None:
    upstream = _repository(tmp_path / "upstream")
    (upstream / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit(upstream, "base")

    downstream = tmp_path / "downstream"
    subprocess.run(["git", "clone", str(upstream), str(downstream)], check=True)
    _git(downstream, "config", "user.name", "Ares Runtime Tests")
    _git(downstream, "config", "user.email", "ares-runtime-tests@example.invalid")
    (downstream / "shared.txt").write_text("Ares change\n", encoding="utf-8")
    downstream_revision = _commit(downstream, "Ares patch")

    (upstream / "shared.txt").write_text("Hermes change\n", encoding="utf-8")
    upstream_revision = _commit(upstream, "upstream change")

    runtime = _runtime(tmp_path)
    monkeypatch.setattr(runtime, "_build_runtime", lambda source, *, desktop: None)

    with pytest.raises(AresLocalRuntimeError):
        runtime._materialize_upstream_candidate(
            downstream_remote=str(downstream),
            downstream_revision=downstream_revision,
            upstream_remote=str(upstream),
            upstream_branch="main",
            upstream_revision=upstream_revision,
            desktop=False,
        )

    assert not runtime.paths.releases_dir.exists() or not any(runtime.paths.releases_dir.iterdir())


def test_desktop_launch_uses_xwayland_only_when_available() -> None:
    executable = Path("/tmp/Ares")

    assert _desktop_launch_arguments(executable, platform="linux", environment={"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"}) == [
        str(executable),
        "--ozone-platform=x11",
    ]
    assert _desktop_launch_arguments(executable, platform="linux", environment={"XDG_SESSION_TYPE": "wayland"}) == [str(executable)]
    assert _desktop_launch_arguments(executable, platform="linux", environment={"DISPLAY": ":0"}) == [str(executable)]
    assert _desktop_launch_arguments(executable, platform="darwin", environment={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}) == [str(executable)]


def test_desktop_gpu_policy_reads_the_ares_scoped_config(tmp_path: Path) -> None:
    home = tmp_path / "ares-home"
    home.mkdir()
    (home / "config.yaml").write_text("desktop:\n  disable_gpu: true\n", encoding="utf-8")

    assert _desktop_disable_gpu_policy(home) == "1"


def test_desktop_launch_bridges_configured_gpu_policy_without_overriding_environment(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    revision = "a" * 40
    source = _release(runtime, revision)
    runtime._activate(revision)
    runtime.paths.agent_home.mkdir()
    (runtime.paths.agent_home / "config.yaml").write_text("desktop:\n  disable_gpu: true\n", encoding="utf-8")
    executable = source / "Ares"
    executable.write_text("desktop", encoding="utf-8")
    captured: list[dict[str, str]] = []

    def fake_popen(_arguments, **kwargs):
        captured.append(kwargs["env"])
        return SimpleNamespace()

    monkeypatch.setattr(runtime, "_desktop_binary", lambda _source: executable)
    monkeypatch.setattr("ares_runtime.local_runtime.subprocess.Popen", fake_popen)

    runtime.desktop(rebuild=False)

    assert captured[-1]["HERMES_DESKTOP_DISABLE_GPU"] == "1"

    monkeypatch.setenv("HERMES_DESKTOP_DISABLE_GPU", "0")
    runtime.desktop(rebuild=False)

    assert captured[-1]["HERMES_DESKTOP_DISABLE_GPU"] == "0"


def test_launcher_resolves_the_selected_runtime_dynamically(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._install_launcher()

    launcher = runtime.paths.launcher_path.read_text(encoding="utf-8")

    assert str(runtime.paths.agent_home) in launcher
    assert 'runtime_root="$ARES_HOME/runtime/current"' in launcher
    assert 'if [[ -z "${ARES_HOME:-}" ]]' in launcher
    assert 'if [[ -z "${ARES_GATEWAY_UNIT_PATH:-}" ]]' in launcher
    assert "cd \"$runtime_root\"" in launcher
    assert "-m ares_runtime.local_runtime" in launcher
    assert "Coding" not in launcher


def test_default_paths_honors_an_explicit_gateway_unit_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARES_HOME", str(tmp_path / "ares-home"))
    monkeypatch.setenv("ARES_BIN_DIR", str(tmp_path / "bin"))
    unit_path = tmp_path / "isolated" / "isolated-gateway.service"
    monkeypatch.setenv("ARES_GATEWAY_UNIT_PATH", str(unit_path))

    from ares_runtime.local_runtime import _default_paths

    assert _default_paths().unit_path == unit_path


def test_custom_gateway_unit_path_never_probes_the_live_default_unit(tmp_path: Path, monkeypatch) -> None:
    runtime = AresLocalRuntime(
        AresLocalPaths(
            state_root=tmp_path / "state",
            data_root=tmp_path / "data",
            agent_home=tmp_path / "ares-home",
            launcher_path=tmp_path / "bin" / "ares",
            unit_path=tmp_path / "unit" / "isolated-gateway.service",
        )
    )
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr("ares_runtime.local_runtime.shutil.which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        "ares_runtime.local_runtime.subprocess.run",
        lambda command, **_kwargs: calls.append(tuple(command)) or SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    runtime._systemctl("is-active", "--quiet", "ares-gateway.service", required=False)

    assert calls == [("systemctl", "--user", "is-active", "--quiet", "isolated-gateway.service")]


def test_custom_gateway_unit_path_rejects_the_live_default_basename(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARES_HOME", str(tmp_path / "ares-home"))
    monkeypatch.setenv("ARES_GATEWAY_UNIT_PATH", str(tmp_path / "isolated" / "ares-gateway.service"))

    from ares_runtime.local_runtime import _default_paths

    with pytest.raises(AresLocalRuntimeError, match="distinct systemd unit name"):
        _default_paths()


def test_generated_launcher_is_shell_safe_for_unusual_paths(tmp_path: Path) -> None:
    runtime = AresLocalRuntime(
        AresLocalPaths(
            state_root=tmp_path / "state",
            data_root=tmp_path / "data",
            agent_home=tmp_path / "home with \"quotes\" $() `ticks` (x)",
            launcher_path=tmp_path / "bin with spaces" / "ares",
            unit_path=tmp_path / "unit with spaces" / "isolated-gateway.service",
        )
    )
    runtime._install_launcher()

    result = subprocess.run(["bash", "-n", str(runtime.paths.launcher_path)], check=False)

    assert result.returncode == 0


def test_isolated_setup_does_not_probe_legacy_live_gateway(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    revision = "c" * 40
    _release(runtime, revision)
    source = tmp_path / "candidate"
    source.mkdir()
    isolated_unit = tmp_path / "unit" / "isolated-gateway.service"
    runtime = AresLocalRuntime(
        AresLocalPaths(
            state_root=runtime.paths.state_root,
            data_root=runtime.paths.data_root,
            agent_home=runtime.paths.agent_home,
            launcher_path=runtime.paths.launcher_path,
            unit_path=isolated_unit,
        )
    )
    calls: list[tuple[str, ...]] = []
    handoff: list[bool] = []

    def git_output(_source: Path, *args: str) -> str:
        if args == ("rev-parse", "--is-inside-work-tree"):
            return "true"
        if args == ("rev-parse", "HEAD"):
            return revision
        if args == ("remote", "get-url", "origin"):
            return str(source)
        if args == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return "main"
        raise AssertionError(args)

    monkeypatch.setattr(runtime, "_git_output", git_output)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_seed_agent_home", lambda *_args: False)
    monkeypatch.setattr(runtime, "_provision_context_governor_key", lambda *_args: None)
    monkeypatch.setattr(runtime, "_write_config", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_install_launcher", lambda: None)
    monkeypatch.setattr(runtime, "_install_gateway_unit", lambda: None)
    monkeypatch.setattr(runtime, "_handoff_gateway", lambda *, legacy_active: handoff.append(legacy_active))
    monkeypatch.setattr(runtime, "_systemctl", lambda *args, **_kwargs: calls.append(args) or False)

    runtime.setup(source, desktop=False, gateway=True, seed_from=tmp_path / "seed")

    assert handoff == [False]
    assert all("hermes-gateway.service" not in call for call in calls)


def test_setup_handoff_failure_restores_pointer_and_launcher(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    prior_revision = "0" * 40
    old_revision = "a" * 40
    new_revision = "b" * 40
    prior_source = _release(runtime, prior_revision)
    old_source = _release(runtime, old_revision)
    _release(runtime, new_revision)
    runtime._activate(prior_revision)
    runtime._activate(old_revision)
    source = tmp_path / "candidate"
    source.mkdir()

    def git_output(_source: Path, *args: str) -> str:
        if args == ("rev-parse", "--is-inside-work-tree"):
            return "true"
        if args == ("rev-parse", "HEAD"):
            return new_revision
        if args == ("remote", "get-url", "origin"):
            return str(source)
        if args == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return "main"
        raise AssertionError(args)

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(runtime, "_git_output", git_output)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_seed_agent_home", lambda *_args: False)
    monkeypatch.setattr(runtime, "_provision_context_governor_key", lambda *_args: None)
    monkeypatch.setattr(runtime, "_write_config", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_install_gateway_unit", lambda: None)
    monkeypatch.setattr(runtime, "_handoff_gateway", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("handoff failed")))
    monkeypatch.setattr(runtime, "_systemctl", lambda *args, **_kwargs: calls.append(args) or True)

    with pytest.raises(RuntimeError, match="handoff failed"):
        runtime.setup(source, desktop=False, gateway=True, seed_from=tmp_path / "seed")

    assert runtime.active_release() == (old_revision, old_source.resolve())
    assert runtime.previous_release() == (prior_revision, prior_source.resolve())
    assert "cd \"$runtime_root\"" in runtime.paths.launcher_path.read_text(encoding="utf-8")
    assert ("enable", "ares-gateway.service") in calls
    assert ("restart", "ares-gateway.service") in calls


def test_update_failure_restores_complete_pointer_pair(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    prior_revision = "0" * 40
    old_revision = "a" * 40
    new_revision = "b" * 40
    prior_source = _release(runtime, prior_revision)
    old_source = _release(runtime, old_revision)
    _release(runtime, new_revision)
    runtime._activate(prior_revision)
    runtime._activate(old_revision)
    runtime.paths.unit_path.parent.mkdir(parents=True)
    runtime.paths.unit_path.write_text("unit", encoding="utf-8")

    monkeypatch.setattr(
        runtime,
        "_read_config",
        lambda: {
            "remote": "downstream",
            "branch": "main",
            "upstream_remote": "upstream",
            "upstream_branch": "main",
        },
    )
    monkeypatch.setattr(
        runtime,
        "_remote_revision",
        lambda remote, _branch: "d" * 40 if remote == "downstream" else "u" * 40,
    )
    monkeypatch.setattr(runtime, "_release_metadata", lambda _revision: {})
    monkeypatch.setattr(
        runtime,
        "_materialize_upstream_candidate",
        lambda **_kwargs: new_revision,
    )
    monkeypatch.setattr(runtime, "_install_gateway_unit", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_systemctl",
        lambda *args, **_kwargs: False if args[:2] == ("is-active", "--quiet") else True,
    )
    monkeypatch.setattr("ares_runtime.local_runtime.time.sleep", lambda _seconds: None)

    with pytest.raises(AresLocalRuntimeError, match="did not remain active"):
        runtime.update(desktop=False)

    assert runtime.active_release() == (old_revision, old_source.resolve())
    assert runtime.previous_release() == (prior_revision, prior_source.resolve())


def test_setup_pre_handoff_failure_restores_complete_pointer_pair(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    prior_revision = "0" * 40
    old_revision = "a" * 40
    new_revision = "b" * 40
    prior_source = _release(runtime, prior_revision)
    old_source = _release(runtime, old_revision)
    _release(runtime, new_revision)
    runtime._activate(prior_revision)
    runtime._activate(old_revision)
    source = tmp_path / "candidate"
    source.mkdir()

    def git_output(_source: Path, *args: str) -> str:
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): new_revision,
            ("remote", "get-url", "origin"): str(source),
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "main",
        }
        return values[args]

    monkeypatch.setattr(runtime, "_git_output", git_output)
    monkeypatch.setattr(runtime, "_materialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_seed_agent_home", lambda *_args: False)
    monkeypatch.setattr(runtime, "_provision_context_governor_key", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_write_config",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("config failed")),
    )
    monkeypatch.setattr(runtime, "_systemctl", lambda *args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="config failed"):
        runtime.setup(source, desktop=False, gateway=True, seed_from=tmp_path / "seed")

    assert runtime.active_release() == (old_revision, old_source.resolve())
    assert runtime.previous_release() == (prior_revision, prior_source.resolve())


def test_agent_environment_strips_ambient_python_import_controls(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
    ):
        monkeypatch.setenv(name, f"/hostile/{name.lower()}")

    environment = runtime._agent_environment()

    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
    ):
        assert name not in environment


def test_build_environment_strips_all_python_import_controls(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
    ):
        monkeypatch.setenv(name, f"/hostile/{name.lower()}")

    environment = runtime._build_environment(source)

    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
    ):
        assert name not in environment
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(source / ".venv")


def test_gateway_unit_uses_the_explicit_foreground_action(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._install_gateway_unit()

    unit = runtime.paths.unit_path.read_text(encoding="utf-8")

    assert f"ExecStart={runtime.paths.launcher_path} gateway foreground" in unit
    assert "TimeoutStopSec=210" in unit


def test_source_cleanliness_reports_dirty_and_clean_git_releases(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    source = _repository(tmp_path / "release")
    (source / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _commit(source, "initial")

    assert runtime._source_cleanliness(source) == (True, "clean")

    (source / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    clean, detail = runtime._source_cleanliness(source)
    assert clean is False
    assert detail == "dirty (1 path(s))"


def test_systemd_environment_preserves_an_existing_session_bus(monkeypatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/existing/runtime")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/existing/runtime/bus")

    environment = AresLocalRuntime._systemd_environment()

    assert environment["XDG_RUNTIME_DIR"] == "/existing/runtime"
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/existing/runtime/bus"
    assert environment["PATH"] == os.environ["PATH"]


def test_seed_adds_missing_auth_without_overwriting_an_ares_home(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    source_home = tmp_path / "hermes-home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"provider":"codex"}', encoding="utf-8")
    runtime.paths.agent_home.mkdir()
    (runtime.paths.agent_home / "config.yaml").write_text("provider: preserved\n", encoding="utf-8")
    runtime._atomic_json(
        runtime.paths.agent_home / "ares-migration.json",
        {"schema_version": 1, "source_home": str(source_home), "copied": [], "migrated_at": 0},
    )

    assert runtime._seed_agent_home(source_home) is True
    assert (runtime.paths.agent_home / "auth.json").read_text(encoding="utf-8") == '{"provider":"codex"}'
    assert (runtime.paths.agent_home / "config.yaml").read_text(encoding="utf-8") == "provider: preserved\n"


def test_context_governor_provisioning_uses_the_existing_governed_key_owner() -> None:
    source = Path(__file__).parents[2] / "ares_runtime" / "local_runtime.py"

    implementation = source.read_text(encoding="utf-8")

    assert "ContextGovernorKeyState" in implementation
    assert "initialize_first_install" in implementation
    assert "MissingGovernedKey" in implementation


def test_ares_runtime_is_included_in_the_noneditable_distribution() -> None:
    project = Path(__file__).parents[2] / "pyproject.toml"
    data = tomllib.loads(project.read_text(encoding="utf-8"))

    assert "ares_runtime" in data["tool"]["setuptools"]["packages"]["find"]["include"]


def test_runtime_builder_uses_editable_python_and_managed_node() -> None:
    source = Path(__file__).parents[2] / "ares_runtime" / "local_runtime.py"
    implementation = source.read_text(encoding="utf-8")

    assert "--no-editable" not in implementation
    assert "from hermes_cli.managed_uv import ensure_uv" in implementation
    assert "self._managed_npm()" in implementation
    assert '[npm, "ci", "--include=dev"]' in implementation


def test_chat_command_leaves_hermes_options_for_the_runtime() -> None:
    args, passthrough = _parser().parse_known_args(
        ["chat", "--oneshot", "Reply with exactly ARES_RUNTIME_OK"]
    )

    assert args.command == "chat"
    assert passthrough == ["--oneshot", "Reply with exactly ARES_RUNTIME_OK"]


def test_bare_ares_launches_the_tui_and_never_the_desktop(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    runtime = SimpleNamespace(
        tui=lambda arguments: calls.append(("tui", arguments)),
        desktop=lambda **kwargs: calls.append(("desktop", kwargs)),
    )
    monkeypatch.setattr("ares_runtime.local_runtime.AresLocalRuntime", lambda: runtime)

    main([])

    assert calls == [("tui", ())]


def test_doctor_uses_the_strict_context_governor_probe() -> None:
    source = Path(__file__).parents[2] / "ares_runtime" / "local_runtime.py"

    implementation = source.read_text(encoding="utf-8")

    assert "Context Governor strict probe" in implementation
    assert "ContextGovernorEngine().probe_activation()" in implementation


def test_doctor_reports_live_runtime_process_drift(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    revision = "a" * 40
    source = _release(runtime, revision)
    python = runtime._python_for(source)
    python.parent.mkdir(parents=True)
    python.write_text("python", encoding="utf-8")
    python.chmod(0o755)
    runtime._activate(revision)
    monkeypatch.setattr(runtime, "_source_cleanliness", lambda _source: (True, "clean"))
    monkeypatch.setattr(
        "hermes_cli.sqlite_runtime.probe_sqlite_runtime",
        lambda _python: SimpleNamespace(
            wal_reset_vulnerable=False,
            sqlite_version_string="3.53.2",
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='{"engine":"compressor","strict_probe":"not configured"}'
        ),
    )

    def subprocess_run(command, **_kwargs):
        program = command[2] if len(command) > 2 else ""
        stdout = '{"missing":[]}' if "probe_mcp_server_tools" in program else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("ares_runtime.local_runtime.subprocess.run", subprocess_run)
    monkeypatch.setattr(
        "ares_runtime.runtime_audit.audit_managed_runtime_processes",
        lambda **kwargs: SimpleNamespace(
            ok=False,
            summary=lambda: (
                f"managed=2 coherent=1 stale=1 active={kwargs['active_revision']}"
            ),
        ),
    )
    monkeypatch.setattr(runtime, "_systemctl", lambda *_args, **_kwargs: True)

    checks = {label: (passed, detail) for label, passed, detail in runtime.doctor()}

    assert checks["runtime process coherence"] == (
        False,
        f"managed=2 coherent=1 stale=1 active={revision}",
    )


def test_runtime_builder_refuses_an_installed_release_source(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    source = _release(runtime, "a" * 40)
    monkeypatch.setattr(
        "hermes_cli.managed_uv.ensure_uv",
        lambda: (_ for _ in ()).throw(AssertionError("managed build was entered")),
    )

    with pytest.raises(AresLocalRuntimeError, match="installed release"):
        runtime._build_runtime(source, desktop=False)


def test_materialize_reuses_a_complete_existing_release_without_rebuilding(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    revision = "a" * 40
    source = _release(runtime, revision)
    python = runtime._python_for(source)
    python.parent.mkdir(parents=True)
    python.write_text("python", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing release was rebuilt")
        ),
    )

    runtime._materialize("unused", revision, desktop=False)

    assert runtime._release_source(revision) == source


def test_materialize_quarantines_an_incomplete_nonactive_release_then_rebuilds(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    source_repository = _repository(tmp_path / "source")
    (source_repository / "canonical.txt").write_text("canonical\n", encoding="utf-8")
    revision = _commit(source_repository, "canonical source")
    incomplete_source = _release(runtime, revision)
    (incomplete_source / "preserved.txt").write_text("old incomplete release\n", encoding="utf-8")
    runtime._atomic_link(runtime.paths.previous_link, incomplete_source.resolve())

    def mark_ready(source: Path, *, desktop: bool) -> None:
        assert desktop is False
        assert not runtime._installed_release_source(source)
        python = runtime._python_for(source)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("python", encoding="utf-8")
        python.chmod(0o755)

    monkeypatch.setattr(runtime, "_build_runtime", mark_ready)

    runtime._materialize(str(source_repository), revision, desktop=False)

    rebuilt = runtime._release_source(revision)
    assert (rebuilt / "canonical.txt").read_text(encoding="utf-8") == "canonical\n"
    assert not (rebuilt / "preserved.txt").exists()
    quarantines = sorted((runtime.paths.data_root / "quarantine" / "incomplete-releases").glob(f"{revision}.*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "source" / "preserved.txt").read_text(encoding="utf-8") == "old incomplete release\n"
    assert runtime.previous_release() == (revision, rebuilt.resolve())


def test_materialize_restores_incomplete_release_when_recovery_build_fails(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    source_repository = _repository(tmp_path / "source")
    (source_repository / "canonical.txt").write_text("canonical\n", encoding="utf-8")
    revision = _commit(source_repository, "canonical source")
    incomplete_source = _release(runtime, revision)
    (incomplete_source / "preserved.txt").write_text("old incomplete release\n", encoding="utf-8")
    runtime._atomic_link(runtime.paths.previous_link, incomplete_source.resolve())
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected build failure")),
    )

    with pytest.raises(RuntimeError, match="injected build failure"):
        runtime._materialize(str(source_repository), revision, desktop=False)

    restored = runtime._release_source(revision)
    assert (restored / "preserved.txt").read_text(encoding="utf-8") == "old incomplete release\n"
    assert runtime.previous_release() == (revision, restored.resolve())
    assert not list((runtime.paths.data_root / "quarantine" / "incomplete-releases").glob(f"{revision}.*"))


def test_materialize_refuses_to_quarantine_an_incomplete_active_release(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    source_repository = _repository(tmp_path / "source")
    (source_repository / "canonical.txt").write_text("canonical\n", encoding="utf-8")
    revision = _commit(source_repository, "canonical source")
    incomplete_source = _release(runtime, revision)
    runtime._activate(revision)
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("build must not run")),
    )

    with pytest.raises(AresLocalRuntimeError, match="active Ares release"):
        runtime._materialize(str(source_repository), revision, desktop=False)

    assert runtime.active_release() == (revision, incomplete_source.resolve())
    assert not list((runtime.paths.data_root / "quarantine" / "incomplete-releases").glob(f"{revision}.*"))


def test_materialize_restores_incomplete_release_when_staging_cleanup_fails(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    source_repository = _repository(tmp_path / "source")
    (source_repository / "canonical.txt").write_text("canonical\n", encoding="utf-8")
    revision = _commit(source_repository, "canonical source")
    incomplete_source = _release(runtime, revision)
    (incomplete_source / "preserved.txt").write_text("old incomplete release\n", encoding="utf-8")
    runtime._atomic_link(runtime.paths.previous_link, incomplete_source.resolve())
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected build failure")),
    )
    monkeypatch.setattr(
        "ares_runtime.local_runtime.shutil.rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("injected cleanup failure")),
    )

    with pytest.raises(AresLocalRuntimeError, match="staging cleanup failed"):
        runtime._materialize(str(source_repository), revision, desktop=False)

    restored = runtime._release_source(revision)
    assert (restored / "preserved.txt").read_text(encoding="utf-8") == "old incomplete release\n"
    assert runtime.previous_release() == (revision, restored.resolve())


def test_upstream_candidate_reuse_never_rebuilds_the_installed_release(
    tmp_path: Path, monkeypatch
) -> None:
    upstream = _repository(tmp_path / "upstream-reuse")
    (upstream / "base.txt").write_text("base\n", encoding="utf-8")
    _commit(upstream, "base")

    downstream = tmp_path / "downstream-reuse"
    subprocess.run(["git", "clone", str(upstream), str(downstream)], check=True)
    _git(downstream, "config", "user.name", "Ares Runtime Tests")
    _git(downstream, "config", "user.email", "ares-runtime-tests@example.invalid")
    (downstream / "ares.txt").write_text("Ares\n", encoding="utf-8")
    downstream_revision = _commit(downstream, "Ares patch")

    (upstream / "upstream.txt").write_text("upstream\n", encoding="utf-8")
    upstream_revision = _commit(upstream, "upstream patch")

    runtime = _runtime(tmp_path)

    def mark_ready(source: Path, *, desktop: bool) -> None:
        assert desktop is False
        python = runtime._python_for(source)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("python", encoding="utf-8")
        python.chmod(0o755)

    monkeypatch.setattr(runtime, "_build_runtime", mark_ready)
    first = runtime._materialize_upstream_candidate(
        downstream_remote=str(downstream),
        downstream_revision=downstream_revision,
        upstream_remote=str(upstream),
        upstream_branch="main",
        upstream_revision=upstream_revision,
        desktop=False,
    )
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("installed candidate was rebuilt")
        ),
    )

    second = runtime._materialize_upstream_candidate(
        downstream_remote=str(downstream),
        downstream_revision=downstream_revision,
        upstream_remote=str(upstream),
        upstream_branch="main",
        upstream_revision=upstream_revision,
        desktop=False,
    )

    assert second == first


def test_desktop_rebuild_refuses_to_mutate_the_active_release(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path)
    revision = "a" * 40
    source = _release(runtime, revision)
    runtime._activate(revision)
    executable = source / "Ares"
    executable.write_text("desktop", encoding="utf-8")
    monkeypatch.setattr(runtime, "_desktop_binary", lambda _source: executable)
    monkeypatch.setattr(
        runtime,
        "_build_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("active release was rebuilt")
        ),
    )

    with pytest.raises(AresLocalRuntimeError, match="cannot mutate an installed release"):
        runtime.desktop(rebuild=True)
