"""Feature inventory reads the selected dependency tree, not the builder."""
import json
import os
from pathlib import Path
import subprocess
import sys

import packaging
import pytest

import pm.extras as extras
import pm.features as features
from hermes_cli.runtime_paths import site_packages


def test_inventory_uses_the_target_and_requires_every_anchor(tmp_path, monkeypatch):
    target_root = tmp_path / "target"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(target_root)],
        capture_output=True, check=True, timeout=60,
    )
    target = target_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    dependencies = tmp_path / "dependencies"
    site = site_packages(dependencies)
    site.mkdir(parents=True)
    witness = tmp_path / "child.json"
    package = site / "inventory_anchor"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(witness)!r}).write_text(json.dumps(sys.executable), encoding='utf-8')\n",
        encoding="utf-8",
    )
    (package / "present.py").write_text("VALUE = 'target'\n", encoding="utf-8")
    (site / "one_part.py").write_text("", encoding="utf-8")
    editable = tmp_path / "editable"
    editable.mkdir()
    (editable / "editable_anchor.py").write_text("", encoding="utf-8")
    (site / "editable.pth").write_text(str(editable) + "\n", encoding="utf-8")
    launches = tmp_path / "launches.jsonl"
    (site / "launches.pth").write_text(
        f"import json, pathlib, sys; p = pathlib.Path({str(launches)!r}); "
        "text = p.read_text(encoding='utf-8') if p.exists() else ''; "
        "p.write_text(text + json.dumps(sys.executable) + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    mapping = {
        "complete": "inventory_anchor.present",
        "partial": ("one_part", "absent_part"),
        "host-only": "packaging",
        "editable": "editable_anchor",
    }
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project.optional-dependencies]\n" + "".join(f'"{name}"=[]\n' for name in mapping),
        encoding="utf-8",
    )
    monkeypatch.setattr(extras, "ANCHORS", mapping)
    before_path = list(sys.path)
    before_env = dict(os.environ)
    assert packaging.__file__ and "packaging" in sys.modules
    result = features.installed_extras(repo, dependencies, python_exe=target)
    assert result == ["complete", "editable"]
    assert Path(json.loads(witness.read_text(encoding="utf-8"))) == target
    assert [Path(json.loads(line)) for line in launches.read_text(encoding="utf-8").splitlines()] == [target]
    assert "inventory_anchor" not in sys.modules
    assert sys.path == before_path
    assert dict(os.environ) == before_env

    (site / "absent_part.py").write_text("", encoding="utf-8")
    assert features.installed_extras(repo, dependencies, python_exe=target) == ["complete", "editable", "partial"]
    with pytest.raises(features.FeatureProbeError, match="feature inventory failed"):
        features.installed_extras(repo, dependencies, python_exe=tmp_path / "missing-python")
