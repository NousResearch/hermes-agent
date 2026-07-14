"""Generic effective-config pin contracts outside the canonical gateway."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from hermes_cli import config as config_module
from hermes_cli import managed_scope


def test_nonisolated_load_config_keeps_defaults_and_env_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit process pin, the historical pipeline is unchanged."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"default": "user-model"},
                "ordinary_env_value": "${PIN_TEST_VALUE}",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(config_module, "ensure_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    monkeypatch.setenv("PIN_TEST_VALUE", "expanded-as-before")
    monkeypatch.setitem(
        config_module.DEFAULT_CONFIG,
        "future_default_sentinel",
        {"present_outside_isolation": True},
    )

    effective = config_module.load_config()
    raw = config_module.read_raw_config()

    assert effective["future_default_sentinel"] == {
        "present_outside_isolation": True
    }
    assert effective["ordinary_env_value"] == "expanded-as-before"
    assert effective["model"]["default"] == "user-model"
    assert "future_default_sentinel" not in raw
    assert raw["ordinary_env_value"] == "${PIN_TEST_VALUE}"
    assert config_module.effective_config_projection_is_pinned() is False


def test_pinned_projection_never_merges_defaults_or_expands_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit pin is the complete mapping, including literal strings."""
    from gateway import run

    config_path = tmp_path / "config.yaml"
    exact = {
        "model": {"default": "sealed-model"},
        "literal_env_reference": "${PIN_TEST_VALUE}",
    }
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    monkeypatch.setenv("PIN_TEST_VALUE", "must-not-expand")
    monkeypatch.setitem(
        config_module.DEFAULT_CONFIG,
        "future_default_sentinel",
        {"must_not_merge": True},
    )

    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        exact_mapping=exact,
    )

    assert config_module.load_config() == exact
    assert config_module.load_config_readonly() == exact
    assert config_module.read_raw_config() == exact
    assert run._load_gateway_config() == exact
    assert run._load_gateway_runtime_config() == exact
    assert "future_default_sentinel" not in config_module.load_config()
    assert (
        config_module.load_config()["literal_env_reference"]
        == "${PIN_TEST_VALUE}"
    )


def test_pin_rejects_symlinked_hermes_home_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    try:
        linked_home.symlink_to(real_home, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    exact = {"model": {"default": "sealed-model"}}
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    (real_home / "config.yaml").write_bytes(raw)
    config_path = linked_home / "config.yaml"
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="contains a symbolic link",
    ):
        config_module.pin_effective_config_projection(
            config_path=config_path,
            raw_bytes=raw,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            exact_mapping=exact,
        )


def test_post_pin_ancestor_identity_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "sealed-home"
    home.mkdir()
    config_path = home / "config.yaml"
    exact = {"model": {"default": "sealed-model"}}
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        exact_mapping=exact,
    )

    home.rename(tmp_path / "replaced-home")
    home.mkdir()
    config_path.write_bytes(raw)

    # Semantic consumers remain on the approved snapshot; only an explicit
    # authority boundary touches and rejects the replaced path chain.
    assert config_module.load_config() == exact
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="path identity drifted",
    ):
        config_module.attest_pinned_effective_config_projection()


def test_same_pin_replay_is_idempotent_and_conflict_preserves_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    exact = {
        "model": {"default": "sealed-model"},
        "agent": {"max_turns": 90},
    }
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    first = config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=digest,
        exact_mapping=exact,
    )
    replay = config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=digest,
        exact_mapping=exact,
    )
    assert first == replay == digest

    conflict = {
        "model": {"default": "conflicting-model"},
        "agent": {"max_turns": 1},
    }
    conflicting_raw = yaml.safe_dump(conflict, sort_keys=True).encode("utf-8")
    config_path.write_bytes(conflicting_raw)
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="already pinned",
    ):
        config_module.pin_effective_config_projection(
            config_path=config_path,
            raw_bytes=conflicting_raw,
            raw_sha256=hashlib.sha256(conflicting_raw).hexdigest(),
            exact_mapping=conflict,
        )

    # The rejected candidate never becomes semantic authority.
    assert config_module.load_config() == exact
    config_path.write_bytes(raw)
    assert config_module.attest_pinned_effective_config_projection() == digest


def test_pinned_readonly_and_raw_nested_mutation_cannot_mutate_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    exact = {
        "model": {"default": "sealed-model"},
        "nested": {
            "items": [{"name": "approved"}],
            "policy": {"enabled": True},
        },
    }
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        exact_mapping=exact,
    )

    readonly = config_module.load_config_readonly()
    readonly["nested"]["items"][0]["name"] = "mutated-readonly"
    readonly["nested"]["items"].append({"name": "injected"})

    raw_view = config_module.read_raw_config()
    raw_view["nested"]["policy"]["enabled"] = False
    raw_view["model"]["default"] = "mutated-raw"

    assert config_module.load_config() == exact
    assert config_module.load_config_readonly() == exact
    assert config_module.read_raw_config() == exact


@pytest.mark.skipif(os.name == "nt", reason="requires stable POSIX inode identity")
def test_same_byte_final_config_inode_replacement_fails_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    exact = {"model": {"default": "sealed-model"}}
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        exact_mapping=exact,
    )

    original_identity = (config_path.stat().st_dev, config_path.stat().st_ino)
    replacement = tmp_path / "replacement.yaml"
    replacement.write_bytes(raw)
    os.replace(replacement, config_path)
    replacement_identity = (config_path.stat().st_dev, config_path.stat().st_ino)
    if replacement_identity == original_identity:
        pytest.skip("filesystem reused the original inode")

    assert config_path.read_bytes() == raw
    assert config_module.load_config() == exact
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="path identity drifted",
    ):
        config_module.attest_pinned_effective_config_projection()


@pytest.mark.skipif(os.name == "nt", reason="requires stable POSIX inode identity")
def test_substituted_open_fd_inode_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    substitute_path = tmp_path / "substitute.yaml"
    exact = {"model": {"default": "sealed-model"}}
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    config_path.write_bytes(raw)
    substitute_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        exact_mapping=exact,
    )

    config_identity = (config_path.stat().st_dev, config_path.stat().st_ino)
    substitute_identity = (
        substitute_path.stat().st_dev,
        substitute_path.stat().st_ino,
    )
    if substitute_identity == config_identity:
        pytest.skip("filesystem did not provide distinct inode identities")

    real_open = config_module.os.open

    def open_substitute(_path: Path, flags: int) -> int:
        return real_open(substitute_path, flags)

    monkeypatch.setattr(config_module.os, "open", open_substitute)
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="path changed before open",
    ):
        config_module.attest_pinned_effective_config_projection()
