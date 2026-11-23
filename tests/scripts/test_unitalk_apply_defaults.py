"""Boot-time Unitalk defaults reach homes created before the default existed.

The template only seeds a volume's first boot; ``scripts/unitalk_apply_defaults.py`` runs on every
boot and fills the image default in wherever the home has not chosen an image provider itself.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import hermes_yaml as yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "unitalk_apply_defaults.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("unitalk_apply_defaults_test_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(hermes_home: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"HERMES_HOME": str(hermes_home), "HERMES_SKIP_CHMOD": "1", "PYTHONPATH": str(REPO_ROOT)})
    return subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, check=False,
    )


def _image_gen(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("image_gen")


def test_existing_home_without_image_settings_gets_the_unitalk_default(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("# operator note\nagent:\n  max_turns: 42\n", encoding="utf-8")

    proc = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _image_gen(config) == _load_script_module().IMAGE_GEN_DEFAULT
    text = config.read_text(encoding="utf-8")
    assert "# operator note" in text
    assert yaml.safe_load(text)["agent"]["max_turns"] == 42


def test_home_that_chose_an_image_provider_is_left_alone(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    chosen = {"provider": "fal", "model": "fal-ai/flux-2-pro"}
    config.write_text(yaml.safe_dump({"image_gen": chosen}), encoding="utf-8")
    before = config.read_bytes()

    proc = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert config.read_bytes() == before


def test_named_profiles_receive_the_default_too(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("agent:\n  max_turns: 90\n", encoding="utf-8")
    profile_config = tmp_path / "profiles" / "designer" / "config.yaml"
    profile_config.parent.mkdir(parents=True)
    profile_config.write_text("agent:\n  max_turns: 90\n", encoding="utf-8")

    proc = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _image_gen(profile_config)["provider"] == "openai"


def test_second_boot_changes_nothing(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("agent:\n  max_turns: 90\n", encoding="utf-8")
    assert _run(tmp_path).returncode == 0
    after_first = config.read_bytes()

    proc = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert config.read_bytes() == after_first
    assert "added the Unitalk image_gen default" not in proc.stdout


def test_boot_default_matches_the_template_new_homes_are_seeded_from() -> None:
    template = yaml.safe_load((REPO_ROOT / "cli-config.yaml.example").read_text(encoding="utf-8"))

    assert template["image_gen"] == _load_script_module().IMAGE_GEN_DEFAULT
