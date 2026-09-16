"""CLI startup must resolve Bedrock's wire for the SESSION model, not the config default.

Regression for #50292.  ``hermes chat -m <non-Claude Bedrock id> --provider bedrock`` on a
profile whose ``model.default`` is a Claude id started on the AnthropicBedrock wire
(``anthropic_messages``); Bedrock rejected the Anthropic-format tool schema with
``HTTP 400 ... Invalid 'tools': missing field 'type'`` and the fallback chain replaced the
requested model.

The resolver half already honours ``target_model`` (``current_model = target_model or
default``); the defect is the startup caller, ``_ensure_runtime_credentials``, which never
passed it.  These tests drive that caller with the real ``resolve_runtime_provider`` and a
Claude default in config, and assert on the ``api_mode`` the CLI ends up with -- the one
thing that decides which wire the first request uses.
"""

from __future__ import annotations

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


CLAUDE_DEFAULT = "us.anthropic.claude-opus-5"


class _RuntimeCLI(CLIAgentSetupMixin):
    """The minimal HermesCLI shell ``_ensure_runtime_credentials`` reads and writes."""

    def __init__(self, *, model: str, provider: str):
        self.model = model
        self.requested_provider = provider
        self.provider = provider
        self.api_key = None
        self.base_url = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._fallback_model = []
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None
        self._provider_source = None
        self._active_agent_route_signature = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


@pytest.fixture
def bedrock_claude_default(monkeypatch):
    """A profile whose default is Claude on Bedrock, with credentials the resolver accepts.

    ``requested_provider="bedrock"`` is an *explicit* selection, so the resolver trusts
    boto3's credential chain instead of probing for keys; the env vars below only keep
    boto3 itself from wandering into IMDS/SSO lookups inside the test.  ``HERMES_HOME``
    is already a per-test tempdir (autouse ``_hermetic_environment``), so the config
    written here is the only one the mtime-cached loader can see.
    """
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        f"model:\n  default: {CLAUDE_DEFAULT}\n  provider: bedrock\nbedrock:\n  region: us-west-2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTNOTREAL0000000")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-real")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


def _startup(model: str) -> _RuntimeCLI:
    cli = _RuntimeCLI(model=model, provider="bedrock")
    assert cli._ensure_runtime_credentials() is True
    return cli


def test_non_claude_session_model_uses_converse_despite_claude_default(bedrock_claude_default):
    """The invariant: ``-m`` names a Converse-only model, the profile default is Claude."""
    cli = _startup("zai.glm-4.7-flash")
    assert cli.api_mode == "bedrock_converse"
    assert cli.provider == "bedrock"
    # The model the session runs is the one that was asked for, not the default.
    assert cli.model == "zai.glm-4.7-flash"


@pytest.mark.parametrize(
    "model",
    [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "global.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
def test_claude_session_model_keeps_anthropic_wire(bedrock_claude_default, model):
    """Control: passing ``target_model`` must not regress Claude ids -- on any inference-profile
    prefix ``is_anthropic_bedrock_model`` normalises -- to Converse."""
    cli = _startup(model)
    assert cli.api_mode == "anthropic_messages", model
    assert cli.provider == "bedrock"
