from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tomllib

import pytest

from ares_runtime import local_runtime
from ares_runtime.local_runtime import AresLocalPaths, AresLocalRuntime, AresLocalRuntimeError, _parser, main


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
    upstream_revision = _commit(upstream, "upstream update")

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
    _commit(upstream, "upstream update")

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
    assert runtime.paths.launcher_path.is_file()
    assert runtime._release_metadata(candidate_revision)["upstream_revision"] == _git(
        upstream, "rev-parse", "HEAD"
    )
    assert runtime.update(desktop=False) == (candidate_revision, False)

    # A successful build remains a retryable candidate if its gateway handoff
    # failed.  The next attempt must promote it without rebuilding a second
    # candidate from the same revisions.
    runtime.paths.current_link.unlink()
    monkeypatch.setattr(
        runtime,
        "_materialize_upstream_candidate",
        lambda **_kwargs: pytest.fail("rebuilt an already verified candidate"),
    )
    assert runtime.update(desktop=False) == (candidate_revision, True)


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
    upstream_revision = _commit(upstream, "upstream update")

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


def test_upstream_change_is_not_masked_by_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _repository(tmp_path / "upstream")
    (upstream / "base.txt").write_text("base\n", encoding="utf-8")
    stale_revision = _commit(upstream, "base")
    (upstream / "new.txt").write_text("new upstream revision\n", encoding="utf-8")
    _commit(upstream, "upstream update")

    downstream = tmp_path / "downstream"
    subprocess.run(["git", "clone", str(upstream), str(downstream)], check=True)
    downstream_revision = _git(downstream, "rev-parse", "HEAD")
    runtime = _runtime(tmp_path)
    cleanup_options: list[bool] = []

    def _rmtree_that_can_only_ignore_errors(_path: Path, *, ignore_errors: bool = False) -> None:
        cleanup_options.append(ignore_errors)
        if not ignore_errors:
            raise OSError("simulated cleanup race")

    monkeypatch.setattr(local_runtime.shutil, "rmtree", _rmtree_that_can_only_ignore_errors)

    with pytest.raises(AresLocalRuntimeError, match="upstream changed"):
        runtime._materialize_upstream_candidate(
            downstream_remote=str(downstream),
            downstream_revision=downstream_revision,
            upstream_remote=str(upstream),
            upstream_branch="main",
            upstream_revision=stale_revision,
            desktop=False,
        )

    assert cleanup_options == [True]


def test_launcher_resolves_the_selected_runtime_dynamically(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._install_launcher()

    launcher = runtime.paths.launcher_path.read_text(encoding="utf-8")

    assert str(runtime.paths.current_link) in launcher
    assert "-m ares_runtime.local_runtime" in launcher
    assert 'cd "$runtime_root"' in launcher
    assert f"export HERMES_HOME={str(runtime.paths.agent_home)!r}" in launcher
    assert "export ARES_MANAGED_RUNTIME=1" in launcher
    assert "Coding" not in launcher


def test_build_environment_is_scoped_to_the_ares_agent_home(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    environment = runtime._build_environment(tmp_path / "candidate")

    assert environment["HERMES_HOME"] == str(runtime.paths.agent_home)
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "candidate" / ".venv")
    assert runtime._agent_environment()["ARES_MANAGED_RUNTIME"] == "1"


def test_runtime_build_uses_supported_editable_source_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    source = tmp_path / "candidate"
    python = source / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls: list[list[str]] = []

    import hermes_cli.managed_uv as managed_uv

    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: Path("/managed/uv"))
    monkeypatch.setattr(
        runtime,
        "_run",
        lambda command, **_kwargs: calls.append([str(value) for value in command]),
    )

    runtime._build_runtime(source, desktop=False)

    sync = calls[0]
    assert sync == ["/managed/uv", "sync", "--locked", "--extra", "all", "--no-dev"]


def test_gateway_unit_uses_the_explicit_foreground_action(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._install_gateway_unit()

    unit = runtime.paths.unit_path.read_text(encoding="utf-8")

    assert f"ExecStart={runtime.paths.launcher_path} gateway foreground" in unit
    assert "Environment=ARES_MANAGED_RUNTIME=1" in unit
    assert "TimeoutStopSec=210" in unit


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


def test_chat_command_leaves_hermes_options_for_the_runtime() -> None:
    args, passthrough = _parser().parse_known_args(
        ["chat", "--oneshot", "Reply with exactly ARES_RUNTIME_OK"]
    )

    assert args.command == "chat"
    assert passthrough == ["--oneshot", "Reply with exactly ARES_RUNTIME_OK"]


def test_role_gate_command_preserves_allowed_exit_and_consumer_note(tmp_path: Path, capsys) -> None:
    request = tmp_path / "allowed.json"
    request.write_text(
        json.dumps(
            {
                "role": "role.public_evidence_editor",
                "action": "publication_ready",
                "payload": {"claim_blockers": [], "evidence_blockers": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["role-gate", "--request", str(request)])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert '"allowed": true' in output
    assert "not connected to every Ares runtime or publication path" in output


def test_role_gate_command_preserves_rejected_exit(tmp_path: Path, capsys) -> None:
    request = tmp_path / "rejected.json"
    request.write_text(
        json.dumps(
            {
                "role": "role.public_evidence_editor",
                "action": "publication_ready",
                "payload": {"claim_blockers": ["missing source"], "evidence_blockers": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["role-gate", "--request", str(request)])

    assert exc_info.value.code == 1
    assert '"allowed": false' in capsys.readouterr().out


def test_role_gate_command_preserves_malformed_exit(tmp_path: Path, capsys) -> None:
    request = tmp_path / "malformed.json"
    request.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(["role-gate", "--request", str(request)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert '"code": "invalid_request"' in captured.out
    assert "role-gate consumer limitation" in captured.err


def test_doctor_uses_the_strict_context_governor_probe() -> None:
    source = Path(__file__).parents[2] / "ares_runtime" / "local_runtime.py"

    implementation = source.read_text(encoding="utf-8")

    assert "Context Governor strict probe" in implementation
    assert "ContextGovernorEngine().probe_activation()" in implementation
