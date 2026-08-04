"""Regression test for config.yaml `security.redact_secrets: false` toggle.

Bug: `agent/redact.py` snapshots `_REDACT_ENABLED` from the env var
`HERMES_REDACT_SECRETS` at module-import time. `hermes_cli/main.py` at
line ~174 calls `setup_logging(mode="cli")` which transitively imports
`agent.redact` — BEFORE any config bridge ran. So if a user set
`security.redact_secrets: false` in config.yaml (instead of as an env var
in .env), the toggle was silently ignored in both `hermes chat` and
`hermes gateway run`.

Fix: bridge `security.redact_secrets` from config.yaml → `HERMES_REDACT_SECRETS`
env var in `hermes_cli/main.py` BEFORE the `setup_logging()` call.
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
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    # Write a config.yaml with redact_secrets: false
    (hermes_home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            security:
              redact_secrets: false
            """
        )
    )
    # Empty .env so nothing else sets the env var
    (hermes_home / ".env").write_text("")

    # Spawn a fresh Python process that imports hermes_cli.main and checks
    # _REDACT_ENABLED. Must be a subprocess — we need a clean module state.
    probe = textwrap.dedent(
        """\
        import sys, os
        # Make absolutely sure the env var is not pre-set
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import hermes_cli.main  # triggers the bridge + setup_logging
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)

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
    `HERMES_REDACT_SECRETS=false`).
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("{}\n")  # empty config
    (hermes_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import sys, os
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import hermes_cli.main
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)

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
    """.env HERMES_REDACT_SECRETS takes precedence over config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            security:
              redact_secrets: false
            """
        )
    )
    # .env force-enables redaction
    (hermes_home / ".env").write_text("HERMES_REDACT_SECRETS=true\n")

    probe = textwrap.dedent(
        """\
        import sys, os
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import hermes_cli.main
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)

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


def test_gateway_config_redact_secrets_false_is_snapshotted_before_redactor_import(
    tmp_path,
):
    """Gateway startup bridges config before its session imports reach redactor."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_secrets: false\n")
    (hermes_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import gateway.run
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_ENABLED=False" in result.stdout
    assert "ENV_VAR=false" in result.stdout


def test_gateway_dotenv_redact_secrets_beats_config_yaml_in_fresh_process(tmp_path):
    """Gateway's later config bridge preserves the dotenv redaction override."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_secrets: false\n")
    (hermes_home / ".env").write_text("HERMES_REDACT_SECRETS=true\n")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import gateway.run
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)
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
    assert "ENV_VAR=true" in result.stdout


def test_redact_level_config_is_snapshotted_before_redactor_import(tmp_path):
    """Config bridges redact_level before agent.redact snapshots its environment."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_level: strict\n")
    (hermes_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_LEVEL", None)
        sys.path.insert(0, %r)
        import hermes_cli.main
        import agent.redact
        print(f"REDACT_LEVEL={agent.redact._REDACT_LEVEL}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_LEVEL', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_LEVEL=strict" in result.stdout
    assert "ENV_VAR=strict" in result.stdout


def test_dotenv_redact_level_beats_config_yaml_in_fresh_process(tmp_path):
    """An explicit dotenv redaction level remains the startup override."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_level: strict\n")
    (hermes_home / ".env").write_text("HERMES_REDACT_LEVEL=standard\n")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_LEVEL", None)
        sys.path.insert(0, %r)
        import hermes_cli.main
        import agent.redact
        print(f"REDACT_LEVEL={agent.redact._REDACT_LEVEL}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_LEVEL', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_LEVEL=standard" in result.stdout
    assert "ENV_VAR=standard" in result.stdout


def test_gateway_config_redact_level_is_snapshotted_before_redactor_import(tmp_path):
    """Gateway startup bridges config before its compression import reaches redactor."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_level: strict\n")
    (hermes_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_LEVEL", None)
        sys.path.insert(0, %r)
        import gateway.run
        import agent.redact
        print(f"REDACT_LEVEL={agent.redact._REDACT_LEVEL}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_LEVEL', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_LEVEL=strict" in result.stdout
    assert "ENV_VAR=strict" in result.stdout


def test_gateway_dotenv_redact_level_beats_config_yaml_in_fresh_process(tmp_path):
    """Gateway dotenv values remain the override over config.yaml fallbacks."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_level: strict\n")
    (hermes_home / ".env").write_text("HERMES_REDACT_LEVEL=standard\n")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ["HERMES_REDACT_LEVEL"] = "off"
        sys.path.insert(0, %r)
        import gateway.run
        import agent.redact
        print(f"REDACT_LEVEL={agent.redact._REDACT_LEVEL}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_LEVEL', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env["HERMES_REDACT_LEVEL"] = "off"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_LEVEL=standard" in result.stdout
    assert "ENV_VAR=standard" in result.stdout


def test_legacy_cli_dotenv_redact_level_beats_config_yaml(tmp_path):
    """The legacy cli.py bridge preserves the dotenv redaction-level override."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_level: strict\n")
    (hermes_home / ".env").write_text("HERMES_REDACT_LEVEL=standard\n")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_LEVEL", None)
        sys.path.insert(0, %r)
        import cli
        import agent.redact
        print(f"REDACT_LEVEL={agent.redact._REDACT_LEVEL}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_LEVEL', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_LEVEL=standard" in result.stdout
    assert "ENV_VAR=standard" in result.stdout


def test_legacy_cli_config_redact_level_is_snapshotted_before_redactor_import(tmp_path):
    """The legacy cli.py bridge reaches agent.redact before CLI mixin imports."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_level: strict\n")
    (hermes_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_LEVEL", None)
        sys.path.insert(0, %r)
        import cli
        import agent.redact
        print(f"REDACT_LEVEL={agent.redact._REDACT_LEVEL}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_LEVEL', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_LEVEL", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_LEVEL=strict" in result.stdout
    assert "ENV_VAR=strict" in result.stdout


def test_legacy_cli_config_redact_secrets_false_is_snapshotted_before_redactor_import(
    tmp_path,
):
    """The legacy CLI bridges YAML false before its mixins import the redactor."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_secrets: false\n")
    (hermes_home / ".env").write_text("")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import cli
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )

    assert result.returncode == 0, f"probe failed: {result.stderr}"
    assert "REDACT_ENABLED=False" in result.stdout
    assert "ENV_VAR=false" in result.stdout


def test_legacy_cli_dotenv_redact_secrets_beats_config_yaml(tmp_path):
    """The legacy CLI preserves a dotenv redaction override over YAML."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("security:\n  redact_secrets: false\n")
    (hermes_home / ".env").write_text("HERMES_REDACT_SECRETS=true\n")

    probe = textwrap.dedent(
        """\
        import os, sys
        os.environ.pop("HERMES_REDACT_SECRETS", None)
        sys.path.insert(0, %r)
        import cli
        import agent.redact
        print(f"REDACT_ENABLED={agent.redact._REDACT_ENABLED}")
        print(f"ENV_VAR={os.environ.get('HERMES_REDACT_SECRETS', '<unset>')}")
        """
    ) % str(REPO_ROOT)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_REDACT_SECRETS", None)
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
    assert "ENV_VAR=true" in result.stdout
