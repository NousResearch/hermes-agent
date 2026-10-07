"""Regression test for config.yaml `security.redact_secrets: false` toggle.

Bug: `agent/redact.py` snapshots `_REDACT_ENABLED` from the env var
`RABBIT_REDACT_SECRETS` at module-import time. `rabbit_cli/main.py` at
line ~174 calls `setup_logging(mode="cli")` which transitively imports
`agent.redact` — BEFORE any config bridge ran. So if a user set
`security.redact_secrets: false` in config.yaml (instead of as an env var
in .env), the toggle was silently ignored in both `rabbit chat` and
`rabbit gateway run`.

Fix: bridge `security.redact_secrets` from config.yaml → `RABBIT_REDACT_SECRETS`
env var in `rabbit_cli/main.py` BEFORE the `setup_logging()` call.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_redact_secrets_false_in_config_yaml_is_honored(tmp_path):
    """Setting `security.redact_secrets: false` in config.yaml must disable
    redaction — even though it's set in YAML, not as an env var."""
    rabbit_home = tmp_path / ".rabbit"
    rabbit_home.mkdir()

    # Write a config.yaml with redact_secrets: false
    (rabbit_home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            security:
              redact_secrets: false
            """
        )
    )
    # Empty .env so nothing else sets the env var
    (rabbit_home / ".env").write_text("")

    # Spawn a fresh Python process that imports rabbit_cli.main and checks
    # _REDACT_ENABLED. Must be a subprocess — we need a clean module state.
    probe = textwrap.dedent(
        """\
        import sys, os
        # Make absolutely sure the env var is not pre-set
        os.environ.pop("RABBIT_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import rabbit_cli.main  # triggers the bridge + setup_logging
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('RABBIT_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["RABBIT_HOME"] = str(rabbit_home)
    env.pop("RABBIT_REDACT_SECRETS", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_ENABLED=False" in result.stdout, (
        f"Config toggle not honored.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "ENV_VAR=false" in result.stdout


def test_redact_secrets_default_true_when_unset(tmp_path):
    """Without the config key or env var, redaction is ON by default (#17691).

    Secret redaction is a secure default — users who need raw credential
    values in tool output (e.g. working on the redactor itself) must set
    `security.redact_secrets: false` explicitly (or
    `RABBIT_REDACT_SECRETS=false`).
    """
    rabbit_home = tmp_path / ".rabbit"
    rabbit_home.mkdir()
    (rabbit_home / "config.yaml").write_text("{}\n")  # empty config
    (rabbit_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import sys, os
        os.environ.pop("RABBIT_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import rabbit_cli.main
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["RABBIT_HOME"] = str(rabbit_home)
    env.pop("RABBIT_REDACT_SECRETS", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_ENABLED=True" in result.stdout




def test_dotenv_redact_secrets_beats_config_yaml(tmp_path):
    """.env RABBIT_REDACT_SECRETS takes precedence over config.yaml."""
    rabbit_home = tmp_path / ".rabbit"
    rabbit_home.mkdir()
    (rabbit_home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            security:
              redact_secrets: false
            """
        )
    )
    # .env force-enables redaction
    (rabbit_home / ".env").write_text("RABBIT_REDACT_SECRETS=true\n")

    probe = textwrap.dedent(
        """\
        import sys, os
        os.environ.pop("RABBIT_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import rabbit_cli.main
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('RABBIT_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["RABBIT_HOME"] = str(rabbit_home)
    env.pop("RABBIT_REDACT_SECRETS", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"probe failed: {result.stderr}"
    # .env value wins
    assert "REDACT_ENABLED=True" in result.stdout
    assert "ENV_VAR=true" in result.stdout
