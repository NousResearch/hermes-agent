"""The external action must never install Hermes into the plugin checkout."""
import os
from pathlib import Path
import subprocess

import pytest
import hermes_yaml as yaml


@pytest.mark.platforms("posix")
def test_external_validator_checkout_uses_requested_ref_and_preserves_caller(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    action = yaml.safe_load((repo / ".github/actions/plugin-validate/action.yml").read_text())
    source_step = next(step for step in action["runs"]["steps"] if step.get("id") == "source")
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    caller = tmp_path / "plugin"
    caller.mkdir()
    manifest = caller / "pyproject.toml"
    manifest.write_text('[project]\nname = "caller-plugin"\nversion = "1.0.0"\n')
    before = manifest.read_bytes()
    runner = tmp_path / "runner"
    runner.mkdir()
    output = tmp_path / "output"
    env_file = tmp_path / "env"
    env = {**os.environ, "RUNNER_TEMP": str(runner), "RUNNER_OS": "Linux",
           "GITHUB_OUTPUT": str(output), "GITHUB_ENV": str(env_file), "_HERMES_REF": ref,
           "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": f"url.{repo.as_uri()}.insteadOf",
           "GIT_CONFIG_VALUE_0": "https://github.com/NousResearch/hermes-agent.git"}
    subprocess.run(["bash", "-c", source_step["run"]], cwd=caller, env=env, check=True,
                   capture_output=True, text=True, timeout=60)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    source = Path(outputs["source"])
    assert source.is_relative_to(runner)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == ref
    assert manifest.read_bytes() == before
    assert sorted(path.name for path in caller.iterdir()) == ["pyproject.toml"]
    assert "python-version" in outputs  # the real setup-pm pin reader ran
    assert not (source / ".build/validator").exists()  # preparation is not installation
