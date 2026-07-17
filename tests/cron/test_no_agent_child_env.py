from __future__ import annotations

import os
import threading
from pathlib import Path
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest


class _ProtectedSource:
    name = "test-source"

    def protected_env_vars(self, cfg: dict) -> frozenset[str]:
        return frozenset({cfg.get("token_alias", "OP_SERVICE_ACCOUNT_TOKEN")})


def _patch_enabled_source(monkeypatch, *, alias: str = "CUSTOM_BOOTSTRAP") -> None:
    import agent.secret_sources.registry as registry
    import hermes_cli.env_loader as env_loader

    monkeypatch.setattr(
        env_loader,
        "_load_secrets_config",
        lambda _home: {
            "test-source": {"enabled": True, "token_alias": alias},
        },
    )
    monkeypatch.setattr(
        registry,
        "_ordered_enabled_sources",
        lambda _cfg: [_ProtectedSource()],
    )


def test_prepare_no_agent_child_env_is_immutable_and_strips_bootstrap_vars(
    tmp_path, monkeypatch
):
    import cron.scheduler as scheduler

    canonical_home = tmp_path / "profile"
    canonical_home.mkdir()
    home = canonical_home / ".." / "profile"
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    monkeypatch.setenv("PERSONAL_HUB_CRON_SECRET", "intended-job-secret")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "default-bootstrap")
    monkeypatch.setenv("CUSTOM_BOOTSTRAP", "configured-bootstrap")
    _patch_enabled_source(monkeypatch)

    with patch("hermes_cli.env_loader.reset_secret_source_cache"), patch(
        "hermes_cli.env_loader.load_hermes_dotenv"
    ):
        pinned_home, child_env = scheduler._prepare_no_agent_child_environment()

    assert pinned_home == home.resolve()
    assert isinstance(child_env, MappingProxyType)
    assert child_env["HERMES_HOME"] == str(home.resolve())
    assert child_env["PERSONAL_HUB_CRON_SECRET"] == "intended-job-secret"
    assert "OP_SERVICE_ACCOUNT_TOKEN" not in child_env
    assert "CUSTOM_BOOTSTRAP" not in child_env
    with pytest.raises(TypeError):
        child_env["MUTATION"] = "blocked"


def test_prepare_no_agent_child_env_resets_before_loading_exact_canonical_home(
    tmp_path, monkeypatch
):
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(scheduler, "_hermes_home", home / ".")
    calls: list[tuple[str, Path]] = []

    def reset(path: Path) -> None:
        calls.append(("reset", path))

    def load(*, hermes_home: Path) -> None:
        calls.append(("load", hermes_home))

    with patch("hermes_cli.env_loader.reset_secret_source_cache", side_effect=reset), patch(
        "hermes_cli.env_loader.load_hermes_dotenv", side_effect=load
    ):
        pinned_home, _child_env = scheduler._prepare_no_agent_child_environment()

    assert pinned_home == home.resolve()
    assert calls == [("reset", home.resolve()), ("load", home.resolve())]


def test_prepare_no_agent_child_env_rotates_across_consecutive_runs(tmp_path, monkeypatch):
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    values = iter(("first", "second"))

    def load(*, hermes_home: Path) -> None:
        assert hermes_home == home.resolve()
        os.environ["PERSONAL_HUB_CRON_SECRET"] = next(values)

    try:
        with patch("hermes_cli.env_loader.reset_secret_source_cache"), patch(
            "hermes_cli.env_loader.load_hermes_dotenv", side_effect=load
        ):
            _, first = scheduler._prepare_no_agent_child_environment()
            _, second = scheduler._prepare_no_agent_child_environment()
    finally:
        os.environ.pop("PERSONAL_HUB_CRON_SECRET", None)

    assert first["PERSONAL_HUB_CRON_SECRET"] == "first"
    assert second["PERSONAL_HUB_CRON_SECRET"] == "second"


def test_prepare_no_agent_child_env_two_homes_cannot_interleave(tmp_path, monkeypatch):
    import agent.secret_sources.registry as registry
    import cron.scheduler as scheduler
    import hermes_cli.env_loader as env_loader
    import tools.environments.local as local_env
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = {name: (tmp_path / name).resolve() for name in ("alpha", "beta")}
    for home in homes.values():
        home.mkdir()

    entered_snapshot = threading.Event()
    release_snapshot = threading.Event()
    beta_loaded = threading.Event()
    original_sanitize = local_env._sanitize_subprocess_env

    def load(*, hermes_home: Path) -> None:
        name = hermes_home.name
        os.environ["PROFILE_SCOPED_SECRET"] = name
        os.environ["HERMES_HOME"] = f"contaminated-by-{name}"
        if name == "beta":
            beta_loaded.set()

    def sanitize(env: dict[str, str]) -> dict[str, str]:
        if env.get("PROFILE_SCOPED_SECRET") == "alpha":
            entered_snapshot.set()
            assert release_snapshot.wait(timeout=5)
        return original_sanitize(env)

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(env_loader, "reset_secret_source_cache", lambda _home: None)
    monkeypatch.setattr(env_loader, "load_hermes_dotenv", load)
    monkeypatch.setattr(env_loader, "_load_secrets_config", lambda _home: {})
    monkeypatch.setattr(registry, "_ordered_enabled_sources", lambda _cfg: [])
    monkeypatch.setattr(local_env, "_sanitize_subprocess_env", sanitize)

    snapshots: dict[str, tuple[Path, MappingProxyType]] = {}

    def prepare(name: str) -> None:
        token = set_hermes_home_override(homes[name])
        try:
            snapshots[name] = scheduler._prepare_no_agent_child_environment()
        finally:
            reset_hermes_home_override(token)

    alpha = threading.Thread(target=prepare, args=("alpha",))
    beta = threading.Thread(target=prepare, args=("beta",))
    try:
        alpha.start()
        assert entered_snapshot.wait(timeout=5)
        beta.start()
        assert not beta_loaded.wait(timeout=0.1)
        release_snapshot.set()
        alpha.join(timeout=5)
        beta.join(timeout=5)
    finally:
        release_snapshot.set()
        alpha.join(timeout=5)
        if beta.ident is not None:
            beta.join(timeout=5)
        os.environ.pop("PROFILE_SCOPED_SECRET", None)

    assert not alpha.is_alive() and not beta.is_alive()
    for name, home in homes.items():
        pinned_home, child_env = snapshots[name]
        assert pinned_home == home
        assert child_env["HERMES_HOME"] == str(home)
        assert child_env["PROFILE_SCOPED_SECRET"] == name


def test_refresh_failure_fails_run_safely_without_starting_child(tmp_path, monkeypatch):
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(scheduler, "_hermes_home", home)

    job = {
        "id": "refresh-failure",
        "name": "refresh-failure",
        "no_agent": True,
        "script": "never-runs.py",
    }
    with patch(
        "hermes_cli.env_loader.reset_secret_source_cache",
        side_effect=RuntimeError("refresh failed"),
    ), patch("cron.scheduler.subprocess.run") as run:
        success, doc, response, error = scheduler.run_job(job)

    assert success is False
    assert doc == "" and response == ""
    assert error == "No-agent secret refresh failed: RuntimeError: refresh failed"
    run.assert_not_called()


def test_claim_heartbeat_passes_exact_prepared_snapshot_and_home(tmp_path, monkeypatch):
    import cron.scheduler as scheduler

    home = (tmp_path / "profile").resolve()
    snapshot = MappingProxyType({"HERMES_HOME": str(home), "PATH": "/snapshot/bin"})
    captured = {}

    def run_script(script_path: str, **kwargs):
        captured.update(script_path=script_path, **kwargs)
        return True, "done"

    monkeypatch.setattr(scheduler, "_run_job_script", run_script)
    result = scheduler._run_job_script_with_claim_heartbeat(
        {"id": "snapshot"},
        "script.py",
        child_env=snapshot,
        hermes_home=home,
    )

    assert result == (True, "done")
    assert captured["child_env"] is snapshot
    assert captured["hermes_home"] is home


def test_prepared_snapshot_path_controls_bash_lookup(tmp_path, monkeypatch):
    import cron.scheduler as scheduler

    home = (tmp_path / "profile").resolve()
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "watch.sh").write_text("echo ignored\n")
    snapshot = MappingProxyType(
        {"HERMES_HOME": str(home), "PATH": "/snapshot-only/bin"}
    )
    completed = MagicMock(returncode=0, stdout="ok\n", stderr="")

    with patch("cron.scheduler.shutil.which", return_value="/snapshot/bash") as which, patch(
        "cron.scheduler.subprocess.run", return_value=completed
    ) as run:
        result = scheduler._run_job_script(
            "watch.sh", child_env=snapshot, hermes_home=home
        )

    assert result == (True, "ok")
    which.assert_called_once_with("bash", path="/snapshot-only/bin")
    assert run.call_args.kwargs["env"] is snapshot


def test_no_agent_workdir_keeps_relative_script_resolution(tmp_path, monkeypatch):
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    scripts = home / "scripts"
    workdir = tmp_path / "job-workdir"
    scripts.mkdir(parents=True)
    workdir.mkdir()
    (workdir / "payload.txt").write_text("from-workdir\n")
    (scripts / "relative.py").write_text(
        "from pathlib import Path\nprint(Path('payload.txt').read_text().strip())\n"
    )
    monkeypatch.setattr(scheduler, "_hermes_home", home)

    job = {
        "id": "relative-workdir",
        "name": "relative-workdir",
        "no_agent": True,
        "script": "relative.py",
        "workdir": str(workdir),
    }
    with patch("hermes_cli.env_loader.reset_secret_source_cache"), patch(
        "hermes_cli.env_loader.load_hermes_dotenv"
    ):
        success, _doc, response, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert response == "from-workdir"
