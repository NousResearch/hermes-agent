"""Self-hosted OIDC sign-in admits one account, the machine owner's (the Webapp is single-user)."""
import pytest

from hermes_cli.dashboard_auth import InvalidCodeError
from plugins.dashboard_auth.self_hosted import owner_pin


@pytest.fixture
def pin_file(tmp_path, monkeypatch):
    path = tmp_path / "dashboard-auth" / "self-hosted-owner.json"
    monkeypatch.setattr(owner_pin, "_pin_path", lambda: path)
    return path


def test_the_configured_owner_is_the_only_account_that_signs_in(pin_file):
    owner_pin.require_owner("https://idp", "owner-sub", "owner-sub")
    with pytest.raises(InvalidCodeError):
        owner_pin.require_owner("https://idp", "colleague-sub", "owner-sub")
    assert not pin_file.exists()


def test_unconfigured_the_first_account_is_pinned_and_no_other_signs_in(pin_file):
    owner_pin.require_owner("https://idp", "first-sub", "")
    owner_pin.require_owner("https://idp", "first-sub", "")
    for issuer, sub in (("https://idp", "second-sub"), ("https://other-idp", "first-sub")):
        with pytest.raises(InvalidCodeError):
            owner_pin.require_owner(issuer, sub, "")
