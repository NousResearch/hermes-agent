"""Regression coverage for idempotent long-lived dotenv reloads (#109902)."""

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import env_loader


def test_self_referential_path_does_not_grow_across_reloads(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("PATH=/opt/data/bin:${PATH}\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    for _ in range(5):
        env_loader.load_hermes_dotenv(
            hermes_home=tmp_path,
            load_external_secrets=False,
        )

    assert os.environ["PATH"] == "/opt/data/bin:/usr/bin:/bin"


def test_missing_explicit_project_path_shares_reload_scope(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("PATH=/opt/data/bin:${PATH}\n", encoding="utf-8")
    missing_project_env = tmp_path / "missing-project.env"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(env_loader, "_DOTENV_RELOAD_STATES", {})

    for index in range(6):
        env_loader.load_hermes_dotenv(
            hermes_home=tmp_path,
            project_env=missing_project_env if index % 2 else None,
            load_external_secrets=False,
        )

    assert os.environ["PATH"] == "/opt/data/bin:/usr/bin:/bin"


def test_non_self_reference_still_reads_current_environment(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("CHILD=${PARENT}\n", encoding="utf-8")
    monkeypatch.setenv("PARENT", "first")
    monkeypatch.delenv("CHILD", raising=False)

    env_loader._load_dotenv_with_fallback(env_path, override=True)
    assert os.environ["CHILD"] == "first"

    monkeypatch.setenv("PARENT", "second")
    env_loader._load_dotenv_with_fallback(env_path, override=True)
    assert os.environ["CHILD"] == "second"


def test_sequential_references_keep_python_dotenv_override_semantics(
    tmp_path, monkeypatch
):
    env_path = tmp_path / ".env"
    env_path.write_text("PARENT=file\nCHILD=${PARENT}/child\n", encoding="utf-8")
    monkeypatch.setenv("PARENT", "shell")
    monkeypatch.delenv("CHILD", raising=False)

    env_loader._load_dotenv_with_fallback(env_path, override=False)
    assert os.environ["PARENT"] == "shell"
    assert os.environ["CHILD"] == "shell/child"

    monkeypatch.delenv("CHILD")
    env_loader._load_dotenv_with_fallback(env_path, override=True)
    assert os.environ["PARENT"] == "file"
    assert os.environ["CHILD"] == "file/child"


def test_self_reference_default_is_stable_when_initial_value_is_missing(
    tmp_path, monkeypatch
):
    env_path = tmp_path / ".env"
    env_path.write_text("CACHE=${CACHE:-cold}\n", encoding="utf-8")
    monkeypatch.delenv("CACHE", raising=False)

    env_loader._load_dotenv_with_fallback(env_path, override=True)
    env_loader._load_dotenv_with_fallback(env_path, override=True)

    assert os.environ["CACHE"] == "cold"


def test_indirect_self_reference_does_not_grow_across_reloads(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "PREVIOUS_PATH=${PATH}\nPATH=/opt/data/bin:${PREVIOUS_PATH}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("PREVIOUS_PATH", raising=False)

    for _ in range(5):
        env_loader._load_dotenv_with_fallback(env_path, override=True)

    assert os.environ["PREVIOUS_PATH"] == "/usr/bin:/bin"
    assert os.environ["PATH"] == "/opt/data/bin:/usr/bin:/bin"


def test_managed_self_reference_follows_changed_user_layer(tmp_path, monkeypatch):
    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    user_env = home / ".env"
    user_env.write_text("PATH=/user-v1:${PATH}\n", encoding="utf-8")
    (managed / ".env").write_text("PATH=/managed:${PATH}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["PATH"] == "/managed:/user-v1:/usr/bin:/bin"

    user_env.write_text("PATH=/user-v2:${PATH}\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)

    assert os.environ["PATH"] == "/managed:/user-v2:/usr/bin:/bin"


def test_external_override_is_reapplied_after_dotenv_reload(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("SERVICE_TOKEN=dotenv-value\n", encoding="utf-8")
    monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(env_loader, "_DOTENV_RELOAD_STATES", {})
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    monkeypatch.setattr(env_loader, "_SECRET_SOURCE_VALUES_BY_HOME", {})
    monkeypatch.setattr(env_loader, "_SECRET_SOURCE_APPLIED_VALUES_BY_HOME", {})
    monkeypatch.setattr(
        env_loader,
        "_load_secrets_config",
        lambda _home: {"fake": {"enabled": True, "override_existing": True}},
    )

    from agent.secret_sources import registry

    calls = 0

    def fake_apply_all(_cfg, _home):
        nonlocal calls
        calls += 1
        os.environ["SERVICE_TOKEN"] = "vault-value"
        source = SimpleNamespace(
            name="fake",
            label="Fake",
            result=SimpleNamespace(error=None, error_kind=None, warnings=[]),
            applied=["SERVICE_TOKEN"],
            skipped_existing=[],
        )
        return SimpleNamespace(
            sources=[source],
            provenance={
                "SERVICE_TOKEN": SimpleNamespace(source="fake")
            },
            conflicts=[],
            applied_any=True,
        )

    monkeypatch.setattr(registry, "apply_all", fake_apply_all)

    env_loader.load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=True)
    assert os.environ["SERVICE_TOKEN"] == "vault-value"

    env_loader.load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=True)

    assert calls == 1
    assert os.environ["SERVICE_TOKEN"] == "vault-value"


def test_windows_interpolation_uses_case_insensitive_environment_keys():
    resolved = env_loader._resolve_dotenv_assignments(
        [("Path", "/opt/data/bin:${PATH}")],
        environ={"PATH": "C:\\Windows"},
        override=True,
        case_insensitive=True,
    )

    assert resolved == {"Path": "/opt/data/bin:C:\\Windows"}


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows environment")
def test_windows_reload_treats_path_key_case_insensitively(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("Path=C:\\tools;${PATH}\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "C:\\Windows")

    for _ in range(5):
        env_loader._load_dotenv_with_fallback(env_path, override=True)

    assert os.environ["PATH"] == "C:\\tools;C:\\Windows"


def test_dotenv_load_applies_the_single_snapshot_it_read(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("VALUE=first\n", encoding="utf-8")
    monkeypatch.delenv("VALUE", raising=False)
    original_read_bytes = Path.read_bytes
    reads = 0

    def replace_after_read(path):
        nonlocal reads
        data = original_read_bytes(path)
        if path == env_path:
            reads += 1
            env_path.write_text("VALUE=second\n", encoding="utf-8")
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    env_loader._load_dotenv_with_fallback(env_path, override=True)

    assert reads == 1
    assert os.environ["VALUE"] == "first"


def test_reload_never_publishes_the_temporary_baseline(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("PATH=/opt/data/bin:${PATH}\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env_loader._load_dotenv_with_fallback(env_path, override=True)

    resolving = threading.Event()
    release = threading.Event()
    errors = []
    original_resolve = env_loader._resolve_dotenv_assignments

    def pause_during_private_resolution(*args, **kwargs):
        resolving.set()
        release.wait(timeout=5)
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(
        env_loader, "_resolve_dotenv_assignments", pause_during_private_resolution
    )

    def reload_dotenv():
        try:
            env_loader._load_dotenv_with_fallback(env_path, override=True)
        except Exception as exc:  # pragma: no cover - surfaced by the main thread
            errors.append(exc)

    thread = threading.Thread(target=reload_dotenv)
    thread.start()
    assert resolving.wait(timeout=5)
    try:
        assert os.environ["PATH"] == "/opt/data/bin:/usr/bin:/bin"
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert os.environ["PATH"] == "/opt/data/bin:/usr/bin:/bin"
