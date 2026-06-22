"""Regression tests for install.sh messaging-platform SDK recovery.

When the POSIX installer falls back from ``.[all]`` to a narrower dependency
tier, platform SDKs like ``slack-bolt`` can be missing even though the setup
wizard just collected matching gateway tokens. The installer must verify the
imports for configured tokens and repair the venv before offering to start the
gateway. See #3944.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_verifies_platform_sdks_after_setup() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert "install_platform_sdks()" in text
    assert 'log_info "Verifying platform SDKs for tokens found in $env_path ..."' in text
    assert 'run_setup_wizard\n    install_platform_sdks\n    maybe_start_gateway' in text


def test_install_script_repairs_slack_sdk_when_tokens_are_present() -> None:
    text = INSTALL_SH.read_text(encoding="utf-8")

    assert "SLACK_BOT_TOKEN|slack_sdk|slack-sdk>=3.40.1,<4" in text
    assert "SLACK_APP_TOKEN|slack_bolt|slack-bolt>=1.27.0,<2" in text
    assert 'if [ "$var" = "WHATSAPP_ENABLED" ] && [ "$raw" != "true" ]; then' in text
    assert 'VIRTUAL_ENV="$INSTALL_DIR/venv" $UV_CMD pip install "$spec"' in text
