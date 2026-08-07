from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.main as main


def test_early_release_marker_guard_precedes_recovery_and_config_imports():
    source = Path(main.__file__).read_text(encoding="utf-8")

    guard = source.index("if _early_release_managed_update_requested():")
    recovery = source.index("from hermes_cli import _early_recovery")
    config = source.index("from hermes_cli.config import get_hermes_home")

    assert guard < recovery < config


def test_early_release_marker_detection_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_bootstrap_root", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", ["hermes", "update", "--check"])

    marker = tmp_path / ".hermes-release-manifest.json"
    marker.symlink_to(tmp_path / "missing-manifest")

    assert main._early_release_managed_update_requested() is True


@pytest.mark.parametrize(
    "argv",
    [
        ["hermes", "plugins", "update", "web"],
        ["hermes", "profile", "update", "work"],
        ["hermes", "-z", "update"],
    ],
)
def test_early_release_marker_does_not_capture_nested_or_prompt_values(
    tmp_path, monkeypatch, argv
):
    monkeypatch.setattr(main, "_bootstrap_root", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", argv)
    (tmp_path / ".hermes-release-manifest.json").write_text("{}")

    assert main._early_release_managed_update_requested() is False


@pytest.mark.parametrize(
    "argv",
    [
        ["hermes", "update"],
        ["hermes", "-p", "work", "update"],
        ["hermes", "--profile", "work", "update"],
        ["hermes", "--profile=work", "update"],
        ["hermes", "--ignore-rules", "update"],
        ["hermes", "-p", "work", "--safe-mode", "update"],
    ],
)
def test_early_release_marker_captures_profile_scoped_top_level_update(
    tmp_path, monkeypatch, argv
):
    monkeypatch.setattr(main, "_bootstrap_root", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", argv)
    (tmp_path / ".hermes-release-manifest.json").write_text("{}")

    assert main._early_release_managed_update_requested() is True


@pytest.mark.parametrize("check", [False, True])
def test_release_managed_checkout_refuses_before_update_side_effects(
    tmp_path, monkeypatch, check
):
    marker = tmp_path / ".hermes-release-manifest.json"
    marker.write_text("{}")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    touched = []
    monkeypatch.setattr(
        main,
        "_install_hangup_protection",
        lambda **kwargs: touched.append("logging"),
    )
    monkeypatch.setattr(
        main,
        "_cmd_update_check",
        lambda **kwargs: touched.append("check"),
    )
    monkeypatch.setattr(
        main,
        "_cmd_update_impl",
        lambda *args, **kwargs: touched.append("apply"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main.cmd_update(SimpleNamespace(check=check, branch=None, gateway=False))

    assert exc_info.value.code == 1
    assert touched == []


def test_release_managed_checkout_refuses_dangling_symlink_marker(
    tmp_path, monkeypatch
):
    (tmp_path / ".hermes-release-manifest.json").symlink_to(
        tmp_path / "missing-manifest"
    )
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    touched = []
    monkeypatch.setattr(
        main,
        "_install_hangup_protection",
        lambda **kwargs: touched.append("logging"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main.cmd_update(SimpleNamespace(check=False, branch=None, gateway=False))

    assert exc_info.value.code == 1
    assert touched == []
