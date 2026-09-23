"""Executable, read-only guided-routing examples shipped with the operator docs."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_public_examples_validate_and_explain_without_state(tmp_path):
    root = Path(__file__).resolve().parents[2]
    examples = root / "website/static/examples/guided-routing"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(tmp_path))
    policy = examples / "policy.json"
    requirements = examples / "requirements.json"
    for args in [("validate", str(policy)), ("explain", str(policy), str(requirements))]:
        proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", "routing", *args, "--json"],
                              cwd=root, env=env, text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        result = json.loads(proc.stdout)
        if args[0] == "validate":
            assert result["valid"]
        else:
            assert result["selected"]["route_id"] in json.loads(policy.read_text())["rankings"]["builder"]["deep"]
            assert result["requirements"]["quality"] == "deep", "a model-proposed task_class is not shallow authority"
    assert not (home / "model_routing.db").exists()
    assert not list(home.rglob("kanban.db"))


def test_routing_help_points_to_checkout_operator_guide(tmp_path):
    import re

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "routing", "--help"],
        cwd=root, env=dict(os.environ, HERMES_HOME=str(tmp_path / "home"), HOME=str(tmp_path)),
        text=True, capture_output=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    help_text = re.sub(r"-\n\s*", "-", proc.stdout)
    targets = re.findall(r"website/docs/[^\s)]+\.md", help_text)
    assert targets, "routing help must point to a guide shipped in the checkout"
    for target in targets:
        assert (root / target).is_file()


def test_public_adapter_examples_survive_validation():
    from hermes_cli.kanban_model_routing import validate_routing_requirements
    from hermes_cli.moa_config import normalize_moa_config

    examples = Path(__file__).resolve().parents[2] / "website/static/examples/guided-routing"
    intake = json.loads((examples / "intake.json").read_text())
    assert validate_routing_requirements(intake) == intake
    config = json.loads((examples / "moa.json").read_text())
    normalized = normalize_moa_config(config)
    preset = normalized["presets"]["guided-example"]
    assert len(preset["reference_models"]) == len(config["presets"]["guided-example"]["reference_models"])
    for slot in [*preset["reference_models"], preset["aggregator"]]:
        assert slot["routing_role"]
        assert validate_routing_requirements(slot["routing_requirements"]) == intake
