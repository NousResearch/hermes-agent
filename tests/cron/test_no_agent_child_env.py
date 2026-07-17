from __future__ import annotations

import os
import threading
from pathlib import Path
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest

from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource


class _ProtectedSource:
    name = "test-source"

    def protected_env_vars(self, cfg: dict) -> frozenset[str]:
        return frozenset({cfg.get("token_alias", "OP_SERVICE_ACCOUNT_TOKEN")})


class _StrictTestSource(SecretSource):
    name = "stricttest"
    label = "Strict Test"
    shape = "mapped"

    def __init__(self, result: FetchResult):
        self.result = result
        self.invalidated: list[Path] = []
        self.fetch_configs: list[dict] = []

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        self.fetch_configs.append(dict(cfg))
        return self.result

    def invalidate_cache(self, home_path: Path) -> None:
        self.invalidated.append(home_path)


def _install_strict_source(monkeypatch, source: SecretSource) -> None:
    import agent.secret_sources.registry as registry

    monkeypatch.setattr(registry, "_BUILTINS_LOADED", True)
    monkeypatch.setattr(registry, "_SOURCES", {source.name: source})


def _patch_enabled_source(monkeypatch, *, alias: str = "CUSTOM_BOOTSTRAP") -> None:
    import agent.secret_sources.registry as registry
    import hermes_cli.env_loader as env_loader

    config = {
        "test-source": {"enabled": True, "token_alias": alias},
    }
    monkeypatch.setattr(
        env_loader,
        "refresh_hermes_dotenv_strict",
        lambda **_kwargs: env_loader.StrictDotenvRefresh((), config),
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

    pinned_home, child_env = scheduler._prepare_no_agent_child_environment()

    assert pinned_home == home.resolve()
    assert isinstance(child_env, MappingProxyType)
    assert child_env["HERMES_HOME"] == str(home.resolve())
    assert child_env["PERSONAL_HUB_CRON_SECRET"] == "intended-job-secret"
    assert "OP_SERVICE_ACCOUNT_TOKEN" not in child_env
    assert "CUSTOM_BOOTSTRAP" not in child_env
    with pytest.raises(TypeError):
        child_env["MUTATION"] = "blocked"


def test_prepare_strips_environment_expanded_bootstrap_alias(tmp_path, monkeypatch):
    import agent.secret_sources.registry as registry
    import cron.scheduler as scheduler
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  test-source:\n"
        "    enabled: true\n"
        "    token_alias: ${BOOTSTRAP_ALIAS_NAME}\n"
    )
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    monkeypatch.setenv("BOOTSTRAP_ALIAS_NAME", "EXPANDED_BOOTSTRAP_TOKEN")
    monkeypatch.setenv("EXPANDED_BOOTSTRAP_TOKEN", "must-not-reach-child")
    monkeypatch.setattr(
        registry, "_ordered_enabled_sources", lambda _cfg: [_ProtectedSource()]
    )

    exact_refresh_config = {
        "test-source": {
            "enabled": True,
            "token_alias": "EXPANDED_BOOTSTRAP_TOKEN",
        }
    }

    def refresh_then_replace_config(**_kwargs):
        config_path.write_text(
            "secrets:\n"
            "  test-source:\n"
            "    enabled: true\n"
            "    token_alias: DIFFERENT_BOOTSTRAP_TOKEN\n"
        )
        return env_loader.StrictDotenvRefresh((), exact_refresh_config)

    with patch(
        "hermes_cli.env_loader.refresh_hermes_dotenv_strict",
        side_effect=refresh_then_replace_config,
    ):
        _pinned_home, child_env = scheduler._prepare_no_agent_child_environment()

    assert "EXPANDED_BOOTSTRAP_TOKEN" not in child_env
    assert "${BOOTSTRAP_ALIAS_NAME}" not in child_env


def test_prepare_strips_onepassword_interactive_session_credentials(
    tmp_path, monkeypatch
):
    import agent.secret_sources.registry as registry
    import cron.scheduler as scheduler
    import hermes_cli.env_loader as env_loader
    from agent.secret_sources.onepassword import OnePasswordSource

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    monkeypatch.setenv("OP_SESSION_work", "interactive-session-secret")
    monkeypatch.setenv("OP_SESSION_personal", "another-session-secret")
    monkeypatch.setenv("NON_AUTH_VALUE", "keep-me")
    config = {"onepassword": {"enabled": True}}
    monkeypatch.setattr(
        env_loader,
        "refresh_hermes_dotenv_strict",
        lambda **_kwargs: env_loader.StrictDotenvRefresh((), config),
    )
    monkeypatch.setattr(
        registry,
        "_ordered_enabled_sources",
        lambda _cfg: [OnePasswordSource()],
    )

    _pinned_home, child_env = scheduler._prepare_no_agent_child_environment()

    assert "OP_SESSION_work" not in child_env
    assert "OP_SESSION_personal" not in child_env
    assert child_env["NON_AUTH_VALUE"] == "keep-me"


def test_prepare_no_agent_child_env_loads_exact_canonical_home(
    tmp_path, monkeypatch
):
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(scheduler, "_hermes_home", home / ".")
    calls: list[tuple[str, Path]] = []

    def reset(path: Path, **_kwargs) -> None:
        calls.append(("reset", path))

    def load(*, hermes_home: Path, **_kwargs) -> list[Path]:
        calls.append(("load", hermes_home))
        return []

    with patch("hermes_cli.env_loader.reset_secret_source_cache", side_effect=reset), patch(
        "hermes_cli.env_loader.load_hermes_dotenv", side_effect=load
    ):
        pinned_home, _child_env = scheduler._prepare_no_agent_child_environment()

    assert pinned_home == home.resolve()
    assert calls == [("load", home.resolve())]


def test_prepare_no_agent_child_env_rotates_across_consecutive_runs(tmp_path, monkeypatch):
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    values = iter(("first", "second"))

    def load(*, hermes_home: Path, **_kwargs) -> list[Path]:
        assert hermes_home == home.resolve()
        os.environ["PERSONAL_HUB_CRON_SECRET"] = next(values)
        return []

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

    def load(*, hermes_home: Path, **_kwargs) -> list[Path]:
        name = hermes_home.name
        os.environ["PROFILE_SCOPED_SECRET"] = name
        os.environ["HERMES_HOME"] = f"contaminated-by-{name}"
        if name == "beta":
            beta_loaded.set()
        return []

    def sanitize(env: dict[str, str]) -> dict[str, str]:
        if env.get("PROFILE_SCOPED_SECRET") == "alpha":
            entered_snapshot.set()
            assert release_snapshot.wait(timeout=5)
        return original_sanitize(env)

    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr(
        env_loader, "reset_secret_source_cache", lambda _home, **_kwargs: None
    )
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
        "hermes_cli.env_loader.refresh_hermes_dotenv_strict",
        side_effect=RuntimeError("refresh failed"),
    ), patch("cron.scheduler.subprocess.run") as run:
        success, doc, response, error = scheduler.run_job(job)

    assert success is False
    assert doc == "" and response == ""
    assert error == "No-agent secret refresh failed: secret refresh failed (type=RuntimeError)"
    run.assert_not_called()


def test_fetch_result_error_removes_stale_secret_and_never_starts_child(
    tmp_path, monkeypatch, caplog
):
    import cron.scheduler as scheduler
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  stricttest:\n    enabled: true\n    override_existing: true\n"
    )
    monkeypatch.setattr(scheduler, "_hermes_home", home)
    raw_secret = "backend token=do-not-log"
    _install_strict_source(
        monkeypatch,
        _StrictTestSource(
            FetchResult(error=raw_secret, error_kind=ErrorKind.NETWORK)
        ),
    )
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "ROTATING_SECRET", "stricttest")
    monkeypatch.setenv("ROTATING_SECRET", "stale-value")
    job = {
        "id": "strict-fetch-failure",
        "name": "strict-fetch-failure",
        "no_agent": True,
        "script": "never-runs.py",
    }

    with caplog.at_level("ERROR"), patch("cron.scheduler.subprocess.run") as run:
        success, doc, response, error = scheduler.run_job(job)

    assert success is False and doc == "" and response == ""
    assert error is not None
    assert "source=stricttest" in error and "stage=fetch" in error
    assert raw_secret not in error and raw_secret not in caplog.text
    assert "ROTATING_SECRET" not in os.environ
    run.assert_not_called()


def test_strict_apply_exception_is_sanitized(tmp_path, monkeypatch):
    import agent.secret_sources.registry as registry
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  stricttest:\n    enabled: true\n"
    )
    source = _StrictTestSource(FetchResult(secrets={"VALUE": "unused"}))
    _install_strict_source(monkeypatch, source)
    raw_secret = "exception contains SECRET-VALUE"
    monkeypatch.setattr(registry, "apply_all", lambda *_args: (_ for _ in ()).throw(RuntimeError(raw_secret)))

    with pytest.raises(env_loader.SecretSourceRefreshError) as raised:
        env_loader.refresh_hermes_dotenv_strict(hermes_home=home)
    assert "source=stricttest" in str(raised.value)
    assert "stage=apply" in str(raised.value)
    assert raw_secret not in str(raised.value)


def test_strict_refresh_expands_environment_backed_source_config(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  stricttest:\n"
        "    enabled: true\n"
        "    token_alias: ${STRICT_TOKEN_ALIAS}\n"
    )
    monkeypatch.setenv("STRICT_TOKEN_ALIAS", "EXPANDED_BOOTSTRAP_NAME")
    source = _StrictTestSource(FetchResult(secrets={"ROTATING_SECRET": "fresh"}))
    _install_strict_source(monkeypatch, source)

    env_loader.refresh_hermes_dotenv_strict(hermes_home=home)

    assert source.fetch_configs == [
        {"enabled": True, "token_alias": "EXPANDED_BOOTSTRAP_NAME"}
    ]


@pytest.mark.parametrize(
    "body",
    [
        (
            "secrets:\n  stricttest:\n    enabled: false\n"
            "secrets:\n  stricttest:\n    enabled: true\n"
        ),
        (
            "secrets:\n  stricttest:\n    enabled: false\n"
            "  stricttest:\n    enabled: true\n"
        ),
        (
            "secrets:\n  stricttest:\n    enabled: true\n"
            "    enabled: false\n"
        ),
    ],
)
def test_strict_refresh_rejects_duplicate_yaml_keys(tmp_path, monkeypatch, body):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(body)
    _install_strict_source(monkeypatch, _StrictTestSource(FetchResult()))

    with pytest.raises(env_loader.SecretSourceRefreshError) as raised:
        env_loader.refresh_hermes_dotenv_strict(hermes_home=home)
    assert raised.value.stage == "config"
    assert "duplicate" not in str(raised.value)


def test_strict_refresh_rejects_unresolved_environment_reference(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  stricttest:\n"
        "    enabled: true\n"
        "    token_alias: ${UNSET_STRICT_TOKEN_ALIAS}\n"
    )
    monkeypatch.delenv("UNSET_STRICT_TOKEN_ALIAS", raising=False)
    _install_strict_source(monkeypatch, _StrictTestSource(FetchResult()))

    with pytest.raises(env_loader.SecretSourceRefreshError) as raised:
        env_loader.refresh_hermes_dotenv_strict(hermes_home=home)
    assert raised.value.stage == "config"
    assert "UNSET_STRICT_TOKEN_ALIAS" not in str(raised.value)


def test_strict_refresh_deep_merges_managed_secret_config_over_user(
    tmp_path, monkeypatch
):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  stricttest:\n"
        "    enabled: true\n"
        "    token_alias: USER_BOOTSTRAP\n"
        "    user_setting: keep\n"
    )
    (managed / "config.yaml").write_text(
        "secrets:\n"
        "  stricttest:\n"
        "    token_alias: ${MANAGED_ALIAS_NAME}\n"
        "    managed_setting: authoritative\n"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("MANAGED_ALIAS_NAME", "MANAGED_BOOTSTRAP")
    source = _StrictTestSource(FetchResult())
    _install_strict_source(monkeypatch, source)

    result = env_loader.refresh_hermes_dotenv_strict(hermes_home=home)

    expected = {
        "enabled": True,
        "token_alias": "MANAGED_BOOTSTRAP",
        "user_setting": "keep",
        "managed_setting": "authoritative",
    }
    assert source.fetch_configs == [expected]
    assert result.secrets_config == {"stricttest": expected}


def test_strict_refresh_rejects_duplicate_keys_in_managed_config(
    tmp_path, monkeypatch
):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "secrets:\n"
        "  stricttest:\n"
        "    enabled: true\n"
        "    token_alias: FIRST\n"
        "    token_alias: SECOND\n"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _install_strict_source(monkeypatch, _StrictTestSource(FetchResult()))

    with pytest.raises(env_loader.SecretSourceRefreshError) as raised:
        env_loader.refresh_hermes_dotenv_strict(hermes_home=home)

    assert raised.value.stage == "config"
    assert "FIRST" not in str(raised.value)
    assert "SECOND" not in str(raised.value)


def test_strict_refresh_reapplies_managed_env_after_external_sources(
    tmp_path, monkeypatch
):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("LOCKED_API_KEY=managed-value\n")
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  stricttest:\n"
        "    enabled: true\n"
        "    override_existing: true\n"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    source = _StrictTestSource(
        FetchResult(secrets={"LOCKED_API_KEY": "external-value"})
    )
    _install_strict_source(monkeypatch, source)

    env_loader.refresh_hermes_dotenv_strict(hermes_home=home)

    assert os.environ["LOCKED_API_KEY"] == "managed-value"


def test_strict_refresh_success_replaces_stale_value(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  stricttest:\n    enabled: true\n    override_existing: true\n"
    )
    source = _StrictTestSource(FetchResult(secrets={"ROTATING_SECRET": "fresh"}))
    _install_strict_source(monkeypatch, source)
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "ROTATING_SECRET", "stricttest")
    monkeypatch.setenv("ROTATING_SECRET", "stale")

    env_loader.refresh_hermes_dotenv_strict(hermes_home=home)

    assert os.environ["ROTATING_SECRET"] == "fresh"
    assert env_loader.get_secret_source("ROTATING_SECRET") == "stricttest"
    assert source.invalidated == [home.resolve()]


def test_strict_refresh_with_no_enabled_sources_is_successful(tmp_path, monkeypatch):
    import hermes_cli.env_loader as env_loader

    home = tmp_path / "profile"
    home.mkdir()
    _install_strict_source(monkeypatch, _StrictTestSource(FetchResult()))
    result = env_loader.refresh_hermes_dotenv_strict(hermes_home=home)
    assert result.loaded_files == ()
    assert result.secrets_config == {}


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
