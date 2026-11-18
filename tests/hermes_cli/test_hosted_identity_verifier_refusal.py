"""A bundled identity verifier whose directory resolves outside the image is refused by the shared
named-provider seam; that refusal is an ordinary hosted boot refusal (gateway and standalone exit 78),
and the planted code is never imported."""
from pathlib import Path
import pytest


def test_out_of_image_bundled_verifier_is_a_boot_refusal(monkeypatch, tmp_path):
    from hermes_cli import hosted_mode
    from plugins import identity_verifier

    bundled = tmp_path / "image" / "identity_verifier"
    bundled.mkdir(parents=True)
    target = tmp_path / "tenant-home" / "planted"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("raise AssertionError('must not import')\n")
    (bundled / "planted").symlink_to(target, target_is_directory=True)
    config = {"hosted": {"enabled": True, "identity_verifier": "planted"}}
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": True, "config": config})
    monkeypatch.setattr(identity_verifier, "_PLUGINS_DIR", bundled)
    with pytest.raises(hosted_mode.HostedBootRefused, match="identity verifier"):
        hosted_mode._require_hosted_backends(config)


def test_standalone_out_of_image_verifier_exits_78(monkeypatch, tmp_path):
    from hermes_cli import hosted_mode, web_server
    from plugins import identity_verifier

    bundled = tmp_path / "image" / "identity_verifier"
    bundled.mkdir(parents=True)
    target = tmp_path / "tenant-home" / "planted"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("raise AssertionError('must not import')\n")
    (bundled / "planted").symlink_to(target, target_is_directory=True)
    config = {"hosted": {"enabled": True, "identity_verifier": "planted"}}
    monkeypatch.setattr(hosted_mode, "_VERDICT", {"hosted": True, "config": config})
    monkeypatch.setattr(identity_verifier, "_PLUGINS_DIR", bundled)
    monkeypatch.setattr(hosted_mode, "hosted_boot_preflight", lambda **kwargs: None)
    monkeypatch.setattr(hosted_mode, "_refuse_direct_platform_adapters", lambda config: None)
    monkeypatch.setattr(hosted_mode, "_refuse_non_sandbox_terminal", lambda config: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    with pytest.raises(SystemExit) as exc:
        web_server._admit_hosted_serving()
    assert exc.value.code == 78

