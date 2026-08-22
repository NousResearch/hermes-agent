from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def config_module(monkeypatch, tmp_path):
    from hermes_cli import config
    from hermes_cli import managed_scope

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "missing-managed"))
    config._LOAD_CONFIG_CACHE.clear()
    config._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config._CONFIG_RESOLUTION_FAILURE_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    yield config
    config._LOAD_CONFIG_CACHE.clear()
    config._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config._CONFIG_RESOLUTION_FAILURE_CACHE.clear()
    managed_scope.invalidate_managed_cache()


def _rewrite_with_new_signature(path: Path, text: str) -> None:
    previous = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text, encoding="utf-8")
    current = path.stat()
    if current.st_mtime_ns == previous:
        os.utime(path, ns=(current.st_atime_ns, previous + 1_000_000))


def test_explicit_effective_config_value_is_cached_and_path_scoped(
    config_module, tmp_path, monkeypatch
):
    profile = tmp_path / "profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(
        "sessions:\n  trigram_fts: ${TRIGRAM_SWITCH}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRIGRAM_SWITCH", "false")

    calls = 0
    original = config_module.fast_safe_load

    def _counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(config_module, "fast_safe_load", _counting_load)

    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=True
    ) == "false"
    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=True
    ) == "false"
    assert calls == 1

    _rewrite_with_new_signature(
        config_path,
        "sessions:\n  trigram_fts: true\n# cache invalidation\n",
    )
    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=False
    ) is True
    assert calls == 2


def test_explicit_effective_config_failure_is_memoized_until_file_changes(
    config_module, tmp_path, monkeypatch
):
    profile = tmp_path / "profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )

    calls = 0
    original = config_module.fast_safe_load

    def _counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(config_module, "fast_safe_load", _counting_load)

    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=True
    ) is False
    assert calls == 1

    # A broken current file must raise even though a last-known-good value is
    # available. The failure signature then prevents repeated reparsing until
    # the file changes.
    _rewrite_with_new_signature(config_path, "sessions: [\n")
    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path, "sessions", "trigram_fts", default=True
        )
    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path, "sessions", "trigram_fts", default=True
        )
    assert calls == 2

    _rewrite_with_new_signature(
        config_path,
        "sessions:\n  trigram_fts: false\n# repaired\n",
    )
    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=True
    ) is False
    assert calls == 3


def test_missing_explicit_config_caches_resolved_defaults(
    config_module, tmp_path, monkeypatch
):
    config_path = tmp_path / "profile-without-config" / "config.yaml"
    calls = 0
    original_normalizer = config_module._normalize_root_model_keys

    def _counting_normalizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_normalizer(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "_normalize_root_model_keys",
        _counting_normalizer,
    )

    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=True
    ) is True
    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=True
    ) is True
    assert calls == 1


def test_missing_global_config_caches_resolved_defaults(
    config_module, monkeypatch
):
    calls = 0
    original_normalizer = config_module._normalize_root_model_keys

    def _counting_normalizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_normalizer(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "_normalize_root_model_keys",
        _counting_normalizer,
    )

    first = config_module.load_config_readonly()
    second = config_module.load_config_readonly()
    assert first is second
    assert calls == 1


def test_unresolved_managed_config_fails_strict_reader_until_repaired(
    config_module, tmp_path, monkeypatch
):
    from hermes_cli import managed_scope

    profile = tmp_path / "profile"
    managed = tmp_path / "managed"
    profile.mkdir()
    managed.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    managed_path = managed / "config.yaml"
    managed_path.write_text("sessions: [\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    managed_scope.invalidate_managed_cache()

    calls = 0
    original = managed_scope.yaml.safe_load

    def _counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(managed_scope.yaml, "safe_load", _counting_load)

    # General config reads intentionally fail open. A later strict explicit
    # read must not consume that shared user-only cache entry.
    assert config_module.load_config_readonly()["sessions"]["trigram_fts"] is False

    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path, "sessions", "trigram_fts", default=True
        )
    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path, "sessions", "trigram_fts", default=True
        )
    assert calls == 1

    _rewrite_with_new_signature(
        managed_path,
        "sessions:\n  trigram_fts: true\n# repaired managed policy\n",
    )
    assert config_module.resolve_effective_config_value(
        config_path, "sessions", "trigram_fts", default=False
    ) is True
    assert calls == 2


@pytest.mark.linux_only
def test_unreadable_config_failure_cache_invalidates_after_chmod_repair(
    config_module, tmp_path, monkeypatch
):
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() == 0:
        pytest.skip("permission denial cannot be exercised on this platform/user")

    profile = tmp_path / "permission-profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    before = config_path.stat()
    monkeypatch.setenv("HERMES_HOME", str(profile))
    config_path.chmod(0)
    try:
        # Prewarm the shared cache through the general fail-open reader. Strict
        # resolution must still reject it, then recover after chmod repair.
        assert config_module.load_config_readonly()["sessions"]["trigram_fts"] is True
        with pytest.raises(config_module.ConfigResolutionError):
            config_module.resolve_effective_config_value(
                config_path, "sessions", "trigram_fts", default=True
            )
        with pytest.raises(config_module.ConfigResolutionError):
            config_module.resolve_effective_config_value(
                config_path, "sessions", "trigram_fts", default=True
            )

        config_path.chmod(0o600)
        repaired = config_path.stat()
        assert (repaired.st_mtime_ns, repaired.st_size) == (
            before.st_mtime_ns,
            before.st_size,
        )
        assert repaired.st_ctime_ns != before.st_ctime_ns or repaired.st_mode != before.st_mode
        assert config_module.resolve_effective_config_value(
            config_path, "sessions", "trigram_fts", default=True
        ) is False
    finally:
        config_path.chmod(0o600)


@pytest.mark.parametrize(
    "invalid_yaml",
    (
        "false\n",
        "[]\n",
        "sessions: false\n",
        "sessions: []\n",
        "sessions: null\n",
    ),
)
def test_strict_reader_rejects_wrong_shape_config(
    config_module,
    tmp_path,
    invalid_yaml,
):
    profile = tmp_path / "wrong-shape-profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(invalid_yaml, encoding="utf-8")

    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path,
            "sessions",
            "trigram_fts",
            default=True,
        )


@pytest.mark.linux_only
def test_strict_reader_rejects_dangling_user_symlink_until_target_appears(
    config_module,
    tmp_path,
):
    profile = tmp_path / "dangling-user-profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    target = profile / "deployed-config.yaml"
    config_path.symlink_to(target)

    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path,
            "sessions",
            "trigram_fts",
            default=True,
        )

    target.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    assert config_module.resolve_effective_config_value(
        config_path,
        "sessions",
        "trigram_fts",
        default=True,
    ) is False


@pytest.mark.linux_only
def test_dangling_user_symlink_retains_last_known_good_for_general_readers(
    config_module,
    tmp_path,
    monkeypatch,
):
    profile = tmp_path / "dangling-lkg-profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(
        "approvals:\n  deny:\n    - dangerous-command\n"
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))
    loaded = config_module.load_config_readonly()
    assert loaded["approvals"]["deny"] == ["dangerous-command"]

    target = profile / "deployed-config.yaml"
    config_path.unlink()
    config_path.symlink_to(target)

    fallback = config_module.load_config_readonly()
    assert fallback["approvals"]["deny"] == ["dangerous-command"]
    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path,
            "sessions",
            "trigram_fts",
            default=True,
        )


@pytest.mark.linux_only
def test_strict_reader_rejects_dangling_managed_symlink_until_target_appears(
    config_module,
    tmp_path,
    monkeypatch,
):
    from hermes_cli import managed_scope

    profile = tmp_path / "profile"
    managed = tmp_path / "managed"
    profile.mkdir()
    managed.mkdir()
    config_path = profile / "config.yaml"
    config_path.write_text(
        "sessions:\n  trigram_fts: true\n",
        encoding="utf-8",
    )
    managed_path = managed / "config.yaml"
    target = managed / "deployed-config.yaml"
    managed_path.symlink_to(target)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    managed_scope.invalidate_managed_cache()
    config_module.load_config_readonly()

    with pytest.raises(config_module.ConfigResolutionError):
        config_module.resolve_effective_config_value(
            config_path,
            "sessions",
            "trigram_fts",
            default=True,
        )

    target.write_text(
        "sessions:\n  trigram_fts: false\n",
        encoding="utf-8",
    )
    assert config_module.resolve_effective_config_value(
        config_path,
        "sessions",
        "trigram_fts",
        default=True,
    ) is False