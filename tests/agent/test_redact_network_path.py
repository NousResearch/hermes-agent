"""Network-path credentials after punctuation boundaries."""

import pytest

from agent.redact import redact_for_egress, redact_sensitive_text


@pytest.fixture(params=["force", "egress"])
def redact(request):
    if request.param == "egress":
        return redact_for_egress
    return lambda text: redact_sensitive_text(text, force=True)


@pytest.mark.parametrize("password", ["Syn9thetic99", "SyntheticSecret9"])
@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [("address\uff1a", ""), ("<", ">"), ("", ""), ("mount '", "'")],
    ids=["fullwidth-colon", "angle-brackets", "bare", "quoted-mount"],
)
def test_network_path_password_is_masked(redact, password, prefix, suffix):
    text = f"{prefix}//user:{password}@host/share{suffix}"
    assert redact(text) == f"{prefix}//user:***@host/share{suffix}"
