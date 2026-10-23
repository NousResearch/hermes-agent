from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "skills"
    / "autonomous-ai-agents"
    / "remote-access-setup"
    / "scripts"
    / "setup_remote.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("remote_access_setup", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replace_env_preserves_unrelated_values_and_replaces_auth(tmp_path: Path) -> None:
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_ME=yes\nHERMES_DASHBOARD_BASIC_AUTH_USERNAME=old\n")

    module.replace_env(
        env_file,
        {
            "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "admin",
            "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH": "scrypt$redacted",
        },
    )

    assert env_file.read_text().splitlines() == [
        "KEEP_ME=yes",
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=scrypt$redacted",
    ]


def test_existing_env_ignores_comments(tmp_path: Path) -> None:
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("# SECRET=ignored\nSECRET=preserved\n")

    assert module.existing_env(env_file) == {"SECRET": "preserved"}
