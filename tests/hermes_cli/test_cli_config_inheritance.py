"""The CLI's own config loader must honour profile inheritance.

``load_cli_config`` reads ``$HERMES_HOME/config.yaml`` directly rather than
going through ``hermes_cli.config.load_config``. An inheriting profile keeps
its model in the root config, so a loader that skips inheritance hands the
agent an empty model and it silently falls back to the provider's first
catalog entry — a different model than the one configured, with no error.
"""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _model_seen_by_the_agent(profile_home: Path) -> dict:
    """Read CLI_CONFIG's model block from a fresh interpreter.

    ``cli`` resolves HERMES_HOME at import time, so the value has to be set
    before the module is imported — a subprocess is the honest way to do that
    without leaking a half-configured module into the rest of the suite.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, cli; print(json.dumps(cli.CLI_CONFIG.get('model')))",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={"HERMES_HOME": str(profile_home), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


def _build_profile(tmp_path: Path, *, profile_config: dict, root_config: dict) -> Path:
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "heir"
    profile_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(root_config), encoding="utf-8")
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(profile_config), encoding="utf-8"
    )
    return profile_home


def test_inheriting_profile_gets_the_root_model(tmp_path):
    """The model the agent runs must be the one the root config names."""
    profile_home = _build_profile(
        tmp_path,
        profile_config={"inherit": True, "agent": {"max_turns": 7}},
        root_config={
            "model": {
                "provider": "anthropic",
                "default": "claude-opus-5",
                "base_url": "https://api.anthropic.com",
            }
        },
    )

    model = _model_seen_by_the_agent(profile_home)

    assert model.get("default") == "claude-opus-5"
    assert model.get("provider") == "anthropic"


def test_profile_model_still_wins_over_the_inherited_one(tmp_path):
    profile_home = _build_profile(
        tmp_path,
        profile_config={"inherit": True, "model": {"default": "claude-sonnet-5"}},
        root_config={"model": {"provider": "anthropic", "default": "claude-opus-5"}},
    )

    model = _model_seen_by_the_agent(profile_home)

    assert model.get("default") == "claude-sonnet-5"


def test_profile_without_inherit_is_unaffected(tmp_path):
    """Isolation stays the default: no opt-in, no inherited model."""
    profile_home = _build_profile(
        tmp_path,
        profile_config={"agent": {"max_turns": 7}},
        root_config={"model": {"provider": "anthropic", "default": "claude-opus-5"}},
    )

    model = _model_seen_by_the_agent(profile_home)

    assert model.get("default") != "claude-opus-5"
