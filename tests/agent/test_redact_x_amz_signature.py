import pytest

from agent.redact import redact_sensitive_text

# 64-char hex, matching no vendor prefix pattern, so only the query-param rule can mask it.
_SIG = "0123456789abcdef" * 4


@pytest.mark.parametrize("param", ["X-Amz-Signature", "x-amz-signature", "X-AMZ-SIGNATURE"])
def test_strict_redactor_masks_aws_sigv4_signature(param: str) -> None:
    url = f"https://bucket.s3.amazonaws.com/key?{param}={_SIG}&view=public"

    result = redact_sensitive_text(url, redact_url_credentials=True)

    assert _SIG not in result
    assert f"{param}=***" in result
    assert "view=public" in result


def test_strict_url_redaction_stays_opt_in() -> None:
    url = f"https://host.example/p?X-Amz-Signature={_SIG}"

    assert redact_sensitive_text(url) == url
