"""Tests for /slack-ingest — natural-language Slack history ingest entrypoint."""

import pytest

from agent.slack_ingest_prompt import build_slack_ingest_prompt, _EXECUTION_RULES


class TestBuildSlackIngestPrompt:
    def test_embeds_user_request_verbatim(self):
        req = "2026-06-24 기준으로 <#C0B7QVCLQF9> 채널 업무만 인제스트해줘"
        prompt = build_slack_ingest_prompt(req)
        assert req in prompt

    def test_always_includes_execution_rules(self):
        req = "6월 24일 3개 채널 인제스트"
        assert _EXECUTION_RULES in build_slack_ingest_prompt(req)

    def test_mentions_slack_history_and_wiki_update_requirements(self):
        prompt = build_slack_ingest_prompt("인제스트 실행")
        for needle in (
            "Slack channel history",
            "slack-channel-history-to-wiki-ingest",
            "wiki/index.md",
            "wiki/log.md",
        ):
            assert needle in prompt

    def test_blank_request_is_rejected(self):
        with pytest.raises(ValueError):
            build_slack_ingest_prompt("   \n ")


class TestSlackIngestRegistryWiring:
    def test_command_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("slack-ingest")
        assert cmd is not None
        assert cmd.name == "slack-ingest"

    def test_command_is_gateway_known(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "slack-ingest" in GATEWAY_KNOWN_COMMANDS

    def test_command_is_gateway_only(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("slack-ingest").gateway_only
