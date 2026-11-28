"""A Docker's standalone TypeScript copy is not an installed pnpm workspace."""

import json
from unittest.mock import Mock

import pytest

from hermes_cli import doctor, doctor_tools
from pm.paths import install_stamp_path


@pytest.fixture
def bundled_root(tmp_path, monkeypatch):
    root = tmp_path / "image"
    modules = root / "node_modules"
    (modules / "typescript").mkdir(parents=True)
    (modules / ".bin").mkdir()
    (modules / "typescript" / "package.json").write_text(json.dumps({
        "name": "typescript", "version": "6.0.3",
    }))
    stamp = install_stamp_path(root)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"distribution": "docker"}))
    monkeypatch.setattr(doctor, "PROJECT_ROOT", root)
    monkeypatch.setattr(doctor_tools, "_pm_tool_path", lambda name: f"/tools/{name}" if name == "pnpm" else None)
    from gateway.platforms import whatsapp_common
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: root / "no-whatsapp")
    return root


def test_bundled_typescript_does_not_audit_uninstalled_workspace(bundled_root, monkeypatch, capsys):
    audit = Mock()
    monkeypatch.setattr(doctor_tools, "_audit_one", audit)

    finding = doctor_tools._check_npm_audit(False)

    audit.assert_not_called()
    output = capsys.readouterr().out
    assert "bundled TypeScript" in output
    assert "not installed" in output
    assert "no known vulnerabilities" not in output
    assert not finding.issues and not finding.manual_issues


@pytest.mark.parametrize("extra", ["checkout", "workspace", "unknown", "typescript-dependency", "optional", "invalid"])
def test_bundled_runtime_exception_never_hides_other_installs(bundled_root, monkeypatch, extra):
    modules = bundled_root / "node_modules"
    if extra == "checkout":
        (bundled_root / ".git").mkdir()
    elif extra == "workspace":
        (modules / ".pnpm").mkdir()
        (modules / ".pnpm" / "lock.yaml").write_text("lockfileVersion: '9.0'\n")
    elif extra == "unknown":
        (modules / "another-package").mkdir()
    elif extra == "invalid":
        (modules / "typescript" / "package.json").write_text("{invalid json")
    else:
        (modules / "typescript" / "package.json").write_text(json.dumps({
            "name": "typescript", "version": "6.0.3",
            "optionalDependencies" if extra == "optional" else "dependencies": {"minimatch": "10.2.6"},
        }))
    audit = Mock()
    monkeypatch.setattr(doctor_tools, "_audit_one", audit)

    doctor_tools._check_npm_audit(False)

    audit.assert_called_once()
    assert audit.call_args.args[2] == "Root npm package"


def test_bundled_root_does_not_skip_installed_workspace_audit(bundled_root, monkeypatch):
    (bundled_root / "web" / "node_modules").mkdir(parents=True)
    audit = Mock()
    monkeypatch.setattr(doctor_tools, "_audit_one", audit)

    doctor_tools._check_npm_audit(False)

    audit.assert_called_once()
    assert audit.call_args.args[2] == "web workspace"
