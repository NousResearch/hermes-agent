"""Incoming-webhook URLs (Slack, Discord, Teams): the URL itself is the credential (#118022)."""

import logging
import time

import pytest

from agent.redact import RedactingFormatter, redact_sensitive_text


class TestIncomingWebhookUrls:
    """An incoming-webhook URL is itself the credential: whoever holds it can post into the
    workspace as the app. Hermes keeps one as ``incoming_webhook_url`` for Teams delivery and
    ships Slack and Discord adapters, so they reach config dumps, tool output and logs."""

    # Built at runtime: a contiguous token-shaped webhook literal trips secret scanners.
    SLACK = "https://hooks." + "slack.com/services/" + "T" + "0" * 8 + "/B" + "0" * 8 + "/" + "X" * 24
    DISCORD = "https://discord.com/api/webhooks/" + "1234567890123456789/" + "AbCdEf-GhIjKlMnOpQrStUvWxYz0123456789"
    TEAMS = "https://acme.webhook.office.com/webhookb2/abc-def@ghi/IncomingWebhook/0123456789abcdef/aaaa-bbbb"

    def test_slack_webhook_secret_is_masked(self):
        result = redact_sensitive_text(f"posting to {self.SLACK}")
        assert self.SLACK not in result
        # The host and ids stay readable so the log still says which integration.
        assert "hooks.slack.com/services/T00000000/B00000000/" in result

    def test_discord_webhook_secret_is_masked(self):
        result = redact_sensitive_text(f"posting to {self.DISCORD}")
        assert self.DISCORD not in result
        assert "discord.com/api/webhooks/1234567890123456789/" in result

    def test_teams_webhook_secret_is_masked(self):
        result = redact_sensitive_text(f"posting to {self.TEAMS}")
        assert self.TEAMS not in result

    def test_file_read_uses_the_non_reusable_sentinel(self):
        """Same contract as the other credential shapes: a secret read out of a file is masked
        with a sentinel that cannot be written back as a truncated-looking credential (#35519)."""
        result = redact_sensitive_text(f"webhook: {self.SLACK}", file_read=True)

        assert self.SLACK not in result
        assert "redacted-secret" in result
        assert "XXXX" not in result  # no head/tail of the secret survives

    def test_ordinary_urls_still_pass_through(self):
        """Web URLs are passed through on purpose; only the vendor webhook shapes match."""
        for benign in (
            "see https://api.slack.com/messaging/webhooks for setup",
            "docs at https://example.com/api/webhooks/overview",
            "GET https://discord.com/api/v10/users/@me",
        ):
            assert redact_sensitive_text(benign) == benign

    SIG = "Xq9vT2mK8pL4rN6sW0yZ3aB5cD7eF1gH9jJ2kL4mN6o"
    SLACK_TRIGGER_SECRET = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    # Teams "Workflows" (Power Automate) webhooks: the ``sig`` query parameter is the secret.
    TEAMS_WORKFLOW = (
        "https://prod-27.westus.logic.azure.com:443/workflows/0a1b2c3d4e5f67890a1b2c3d4e5f6789"
        "/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun"
        f"&sv=1.0&sig={SIG}")
    TEAMS_POWER_PLATFORM = (
        "https://default0a1b2c3d4e5f.67.environment.api.powerplatform.com:443/powerautomate"
        "/automations/direct/workflows/0a1b2c3d4e5f67890a1b2c3d4e5f6789/triggers/manual/paths"
        f"/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig={SIG}")
    SLACK_TRIGGER = f"https://hooks.slack.com/triggers/T0123ABCD/7412589630125/{SLACK_TRIGGER_SECRET}"

    @pytest.mark.parametrize("url_attr,secret_attr", [
        ("TEAMS_WORKFLOW", "SIG"),
        ("TEAMS_POWER_PLATFORM", "SIG"),
        ("SLACK_TRIGGER", "SLACK_TRIGGER_SECRET"),
    ])
    def test_workflow_and_trigger_urls_masked_on_production_paths(self, url_attr, secret_attr):
        """terminal output (.env dump and plain), read_file of config.yaml and log lines."""
        from agent.redact import redact_terminal_output

        url, secret = getattr(self, url_attr), getattr(self, secret_attr)
        env_dump = redact_terminal_output(f"TEAMS_WEBHOOK_URL={url}\n", "cat .env")
        plain = redact_terminal_output(f'curl -X POST "{url}"\n', "history")
        read = redact_sensitive_text(
            f'platforms:\n  teams:\n    incoming_webhook_url: "{url}"\n', file_read=True)
        record = logging.LogRecord("gateway", logging.INFO, "", 0, "POST %s -> 202", (url,), None)
        log = RedactingFormatter("%(message)s").format(record)

        for out in (env_dump, plain, read, log):
            assert secret not in out
            assert secret[6:-4] not in out
        # Only the secret is masked: host, ids and the closing quote stay put.
        prefix = url[: url.index(secret)]
        assert f'"{prefix}' in plain and plain.rstrip().endswith('"')
        assert "redacted-secret" in read

    def test_teams_connector_mask_keeps_the_closing_quote(self):
        result = redact_sensitive_text(f'url: "{self.TEAMS}"', file_read=True)
        assert self.TEAMS not in result
        assert result.endswith('"')

    def test_ordinary_logic_app_and_slack_urls_pass_through(self):
        from agent.redact import redact_terminal_output

        for benign in (
            "https://prod-27.westus.logic.azure.com:443/workflows/abc/runs?api-version=2016-06-01",
            "https://management.azure.com/subscriptions/x/workflows/y?api-version=2019-05-01",
            "https://app.slack.com/client/T0123ABCD/C0123ABCD",
            "https://slack.com/help/articles/360041352714",
            "https://api.slack.com/automation/triggers/webhook",
        ):
            assert redact_terminal_output(benign, "echo") == benign
            assert redact_sensitive_text(benign, file_read=True) == benign

    @pytest.mark.parametrize("template", [
        "TEAMS_INCOMING_WEBHOOK_URL={url}",
        '{{"incoming_webhook_url": "{url}"}}',
        "incoming_webhook_url: {url}",
        "curl -X POST {url}",
    ])
    @pytest.mark.parametrize("url_attr", ["SLACK", "DISCORD", "TEAMS"])
    @pytest.mark.parametrize("scheme", ["https://", "http://"])
    def test_issue_shapes_mask_the_secret(self, template, url_attr, scheme):
        """#118022: env assignment, JSON, YAML and a command line all leaked verbatim, since the
        assignment pass skips ``://`` text; a plain-http scheme carries the same secret."""
        url = scheme + getattr(self, url_attr).removeprefix("https://")
        leak = {"SLACK": "X" * 12, "DISCORD": "GhIjKlMnOp", "TEAMS": "0123456789abcdef"}[url_attr]
        for out in (redact_sensitive_text(template.format(url=url)),
                    redact_sensitive_text(template.format(url=url), file_read=True)):
            assert url not in out
            assert leak not in out


class TestWebhookUrlShapes:
    """Round-2 shapes: Workflow Builder's four-segment Slack URL and equivalent host spellings."""

    SECRET = "AbCdEfGhIjKlMnOpQrStUvWx"

    def test_slack_workflow_builder_masks_the_last_segment(self):
        """workflows/{T}/{A}/{numeric id}/{secret}: the numeric id is public, the last segment is
        the secret. A three-segment rule masked the id and left the secret verbatim."""
        url = "https://hooks.slack.com/workflows/" + "T01ABCDEF/A02BCDEFG/412345678901234567/" + self.SECRET
        for out in (redact_sensitive_text(url), redact_sensitive_text(url, file_read=True)):
            assert self.SECRET[6:-4] not in out
            assert "/A02BCDEFG/412345678901234567/" in out

    @pytest.mark.parametrize("prefix", [
        "https://hooks.slack.com.:443/services/T0123ABCD/B0123ABCD/",
        "https://hooks.slack-gov.com/services/T0123ABCD/B0123ABCD/",
        "https://www.discord.com/api/webhooks/1234567890123456789/",
    ])
    def test_equivalent_hosts_mask_the_secret(self, prefix):
        out = redact_sensitive_text(prefix + self.SECRET)
        assert self.SECRET[6:-4] not in out
        assert out.startswith(prefix)

    @pytest.mark.parametrize("template", [
        "[hook]({url}) next", "use `{url}` here", "url={url},status=200", "Posted to {url}.",
    ])
    def test_teams_connector_mask_stops_at_the_url_end(self, template):
        """The connector path is GUIDs, ``@``, ``/``, hex and a base64url ``V2`` tail: punctuation
        after the URL is not secret and survives, as it does after a Slack or Discord URL."""
        url = ("https://acme.webhook.office.com/webhookb2/a1b2c3d4-e5f6@a7b8c9d0-e1f2/IncomingWebhook/"
               + "0123456789abcdef" * 2 + "/V2" + self.SECRET)
        text = template.format(url=url)
        suffix = text[text.index(url) + len(url):]
        for out in (redact_sensitive_text(text), redact_sensitive_text(text, file_read=True)):
            assert self.SECRET[6:-4] not in out
            assert out.endswith(suffix)
            assert redact_sensitive_text(out) == out  # a second pass leaves the mask alone

    @pytest.mark.parametrize("text", [
        "https://a.logic.azure.com/workflows/" * 1400,
        "https://a.logic.azure.com/" + "workflows/" * 8000,
        ("https://a.logic.azure.com/" + "b/" * 8 + "workflows/" + "c" * 500 + "?" + "a&" * 300) * 60,
    ], ids=["repeated-url", "repeated-path", "repeated-capped-url"])
    def test_teams_workflows_rule_is_linear_on_repeated_urls(self, text):
        """Attacker-shaped tool output must not stall redaction (the uncapped arm took ~48 s on
        the first input). The bound is generous; the capped rule runs in tens of ms."""
        t0 = time.perf_counter()
        redact_sensitive_text(text)
        assert time.perf_counter() - t0 < 2.0
