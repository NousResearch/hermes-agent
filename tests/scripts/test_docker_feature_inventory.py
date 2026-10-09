"""Docker assembly must keep its prepared dependency selection (#135329)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import sysconfig

import pytest

from docker.build_agent import assemble_image
from scripts.build.inputs import RESOURCE_ENV


def _image_layout(root: Path) -> None:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="image-fixture"\nversion="1"\n[project.scripts]\n', encoding="utf-8"
    )
    environment = root / ".venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").symlink_to(sys.executable)
    site = Path(sysconfig.get_path("purelib", vars={"base": str(environment), "platbase": str(environment)}))
    site.mkdir(parents=True)
    for name in (*RESOURCE_ENV, "tools", "pm-runtime", "ui-tui/dist", "hermes_cli/web_dist"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "pm-runtime/deps").mkdir()
    (root / "pm-runtime/pm-runtime.json").write_text(
        json.dumps({"python": "../.venv/bin/python", "sitePackages": "deps"}), encoding="utf-8"
    )
    (root / "ui-tui/dist/entry.js").write_text("export {}", encoding="utf-8")
    (root / "ui-tui/package.json").write_text('{"type":"module"}', encoding="utf-8")
    (root / "hermes_cli/web_dist/index.html").write_text("built", encoding="utf-8")


@pytest.mark.platforms("linux")
def test_image_assembly_refuses_missing_feature_inventory(tmp_path):
    root = tmp_path / "image"
    _image_layout(root)
    with pytest.raises(FileNotFoundError, match="enabled-features.json"):
        assemble_image(root)
    assert not (root / "manifest.json").exists()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("build_succeeds", [False, True])
def test_prepared_image_retains_features_only_after_success(tmp_path, monkeypatch, build_succeeds):
    from types import SimpleNamespace

    from docker import build_dependencies
    from pm.features import read_features
    from pm.install import _feature_policy, _target_selection
    from pm.package import InstallError

    root = tmp_path / "image"
    _image_layout(root)
    declared = {"all", "messaging", "discord", "telegram", "mcp", "edge-tts"}
    with (root / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write("[project.optional-dependencies]\n")
        stream.writelines(f'"{extra}" = []\n' for extra in sorted(declared))

    def prepare(**kwargs):
        assert kwargs["source"] == root
        assert kwargs["out"] == root / ".venv"
        assert kwargs["python"] == Path(sys.executable)
        assert kwargs["no_install_project"] and kwargs["frozen"]
        assert kwargs["sealed"] and kwargs["explicit"]
        assert not (root / "enabled-features.json").exists()
        assert kwargs["extras"] == list(build_dependencies.IMAGE_EXTRAS)
        if not build_succeeds:
            raise RuntimeError("dependency preparation failed")
        site = Path(sysconfig.get_path(
            "purelib", vars={"base": str(root / ".venv"), "platbase": str(root / ".venv")},
        ))
        for module in ("telegram", "discord", "mcp"):
            (site / f"{module}.py").write_text("", encoding="utf-8")
        return root / ".venv/bin/python"

    monkeypatch.setattr(build_dependencies, "build_environment", prepare)
    if not build_succeeds:
        with pytest.raises(RuntimeError, match="dependency preparation failed"):
            build_dependencies.build_image_dependencies(root, Path(sys.executable))
        assert not (root / "enabled-features.json").exists()
        with pytest.raises(FileNotFoundError, match="enabled-features.json"):
            assemble_image(root)
        return

    build_dependencies.build_image_dependencies(root, Path(sys.executable))
    inventory = (root / "enabled-features.json").read_bytes()
    carried = ["discord", "mcp", "messaging", "telegram"]
    assert json.loads(inventory)["extras"] == carried
    assemble_image(root)
    assert (root / "enabled-features.json").read_bytes() == inventory
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(root / "tools"))
    assert read_features() == carried
    package = SimpleNamespace(
        project_root=lambda: root,
        expected_stamp=lambda extras, **kwargs: ",".join(extras),
    )
    for lazy in (True, False):
        monkeypatch.setattr("pm.install.lazy_installs_allowed", lambda: lazy)
        shipped, frozen = _feature_policy(["discord", "telegram", "mcp"], repair=False)
        first, _, _ = _target_selection(
            package, {}, extras=["discord"], inputs={}, repair=False,
            shipped=shipped, frozen=frozen,
        )
        assert first == carried
    with pytest.raises(InstallError, match="outside this bundle's frozen feature set"):
        _feature_policy(["edge-tts"], repair=False)
    subsequent, _, _ = _target_selection(
        package, {"extras": ["discord"]}, extras=[], inputs={}, repair=False,
        shipped=read_features(), frozen=None,
    )
    assert subsequent == ["discord"]


@pytest.mark.platforms("linux")
def test_image_dependency_inventory_failure_never_publishes_features(tmp_path, monkeypatch):
    from docker import build_dependencies
    from pm.features import FeatureProbeError

    root = tmp_path / "image"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="empty-image"\nversion="1"\n[project.optional-dependencies]\ndiscord=[]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_dependencies, "build_environment", lambda **kwargs: root / ".venv/bin/python")
    # A nominally successful build without its dependency tree is not an empty inventory.
    with pytest.raises(FeatureProbeError, match="target dependency tree has no site-packages"):
        build_dependencies.build_image_dependencies(root, Path(sys.executable))
    assert not (root / "enabled-features.json").exists()
