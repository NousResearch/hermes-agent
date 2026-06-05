from types import SimpleNamespace

from gateway.session import Platform, SessionSource


def _make_bare_runner(dynamic_allowed=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    if dynamic_allowed is None:
        runner.adapters = {}
    else:
        runner.adapters = {
            Platform.WHATSAPP: SimpleNamespace(
                _dynamic_allowed_users=lambda: set(dynamic_allowed),
            )
        }
    return runner


def _make_whatsapp_source(user_id="251083989471255@lid"):
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=user_id,
        chat_type="dm",
        user_id=user_id,
        user_name="Customer",
    )


def test_whatsapp_dynamic_outbound_allowlist_authorizes_customer_with_owner_allowlist(monkeypatch):
    """Outbound-authorized WhatsApp contacts must not be dropped by Python auth.

    The Node bridge writes recipients to dynamic-allowed-users.json after an
    outbound send. WHATSAPP_ALLOWED_USERS remains the owner/operator allowlist,
    so the Python gateway must also honor the dynamic customer list when that
    owner allowlist is configured.
    """
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "3059274821,13059274821")

    runner = _make_bare_runner(dynamic_allowed={"251083989471255@lid"})

    assert runner._is_user_authorized(_make_whatsapp_source()) is True


def test_whatsapp_unknown_customer_still_rejected_with_owner_allowlist(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "3059274821,13059274821")

    runner = _make_bare_runner(dynamic_allowed={"111111111111111@lid"})

    assert runner._is_user_authorized(_make_whatsapp_source("251083989471255@lid")) is False


def test_whatsapp_open_atc_authorizes_any_customer_without_owner_privilege(monkeypatch):
    """WHATSAPP_ALLOWED_USERS=* opens DM ATC while customer-service keeps non-owners as Sophie."""
    from gateway.run import _source_matches_customer_service_owner, _source_should_use_customer_service

    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    cfg = {
        "customer_service": {
            "enabled": True,
            "channels": ["whatsapp"],
            "owner_users": {"whatsapp": ["13059274821", "120697238081658@lid"]},
        }
    }
    runner = _make_bare_runner(dynamic_allowed=set())
    customer = _make_whatsapp_source("999999999999999@lid")
    owner = _make_whatsapp_source("120697238081658@lid")

    assert runner._is_user_authorized(customer) is True
    assert _source_should_use_customer_service(customer, cfg) is True
    assert _source_matches_customer_service_owner(customer, cfg) is False

    assert runner._is_user_authorized(owner) is True
    assert _source_should_use_customer_service(owner, cfg) is False
    assert _source_matches_customer_service_owner(owner, cfg) is True
