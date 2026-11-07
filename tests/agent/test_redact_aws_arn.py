"""AWS ARNs are not Telegram bot tokens.

``_TELEGRAM_RE`` matches ``<8+ digits>:<30+ token chars>``. An ARN's
``<12-digit account>:<resource>`` has that shape whenever the resource name is
30+ characters, so a plain SNS topic or IAM role ARN came back as
``...:123456789012:***``. Callers that compare ``redact_sensitive_text(x) != x``
as a credential preflight then refuse text that contains no secret at all.

All values are synthetic and assembled at runtime; 123456789012 is the AWS
documentation account ID.
"""

import pytest

from agent.redact import redact_sensitive_text

ACCOUNT = "123456789012"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _token_body(n: int = 35) -> str:
    # Telegram bot secrets are 35 chars of [-A-Za-z0-9_], starting "AA".
    return ("AAH" + "k3-Zq_9xW" * 5)[:n]


class TestArnsPassThrough:
    @pytest.mark.parametrize("topic_len", [30, 31, 64])
    def test_sns_topic_arn_unchanged(self, topic_len):
        arn = f"arn:aws:sns:us-east-1:{ACCOUNT}:" + ("Default_CloudWatch_Alarms_Topic" * 3)[:topic_len]
        text = f"Subscribe the alarm to {arn} and confirm."
        assert redact_sensitive_text(text, force=True) == text

    def test_iam_role_arn_unchanged(self):
        # IAM ARNs have an empty region field.
        arn = f"arn:aws:iam::{ACCOUNT}:role_service_role_for_application_deployments"
        assert redact_sensitive_text(arn, force=True) == arn

    def test_long_resource_arn_unchanged(self):
        resource = "a" * 20 + "-" + "B" * 20 + "_" + "0" * 10
        arn = f"arn:aws-us-gov:lambda:us-gov-west-1:{ACCOUNT}:{resource}"
        assert len(resource) > 40
        assert redact_sensitive_text(arn, force=True) == arn

    def test_arn_in_json_and_task_text_unchanged(self):
        arn = f"arn:aws:sns:eu-west-2:{ACCOUNT}:" + "x" * 31
        text = '{"TopicArn": "' + arn + '"}\nDeploy with --topic=' + arn
        assert redact_sensitive_text(text, force=True) == text

    def test_29_30_boundary_after_account(self):
        # 29 chars never matched; 30 used to be masked. Both pass through now.
        for n in (29, 30):
            arn = f"arn:aws:sns:us-east-1:{ACCOUNT}:" + "t" * n
            assert redact_sensitive_text(arn, force=True) == arn, n


class TestTelegramTokensStillRedacted:
    def test_bare_token_redacted(self):
        body = _token_body()
        result = redact_sensitive_text(f"token is 123456789:{body}", force=True)
        assert body not in result
        assert "123456789:***" in result

    def test_bot_prefixed_token_redacted(self):
        body = _token_body()
        result = redact_sensitive_text(f"bot123456789:{body}", force=True)
        assert body not in result
        assert "bot123456789:***" in result

    def test_token_in_api_url_redacted(self):
        body = _token_body()
        url = f"https://api.telegram.org/bot123456789:{body}/getMe"
        result = redact_sensitive_text(f"curl {url}", force=True)
        assert body not in result
        assert "https://api.telegram.org/bot123456789:***" in result

    def test_12_digit_id_without_arn_prefix_redacted(self):
        # Only the ARN prefix exempts a 12-digit number, not the digit count.
        body = _token_body()
        result = redact_sensitive_text(f"{ACCOUNT}:{body}", force=True)
        assert body not in result

    def test_token_next_to_arn_redacted(self):
        body = _token_body()
        arn = f"arn:aws:sns:us-east-1:{ACCOUNT}:" + "x" * 31
        result = redact_sensitive_text(f"{arn} bot123456789:{body}", force=True)
        assert arn in result
        assert body not in result

    def test_29_30_boundary_for_tokens(self):
        assert redact_sensitive_text("123456789:" + "a" * 29, force=True) == "123456789:" + "a" * 29
        assert redact_sensitive_text("123456789:" + "a" * 30, force=True) == "123456789:***"
