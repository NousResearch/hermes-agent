from pathlib import Path

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.run import (
    _customer_service_prompt,
    _customer_service_toolsets,
    _source_matches_customer_service_owner,
    _source_should_use_customer_service,
)
from gateway.platforms.whatsapp import WhatsAppAdapter
from toolsets import resolve_toolset


def test_customer_service_routes_non_owner_whatsapp_to_sophie_toolset(monkeypatch):
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)
    cfg = {
        "customer_service": {
            "enabled": True,
            "channels": ["whatsapp", "email"],
            "owner_users": {"whatsapp": ["13059274821"]},
        }
    }
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="13059274824@s.whatsapp.net",
        user_id="13059274824@s.whatsapp.net",
    )

    assert _source_should_use_customer_service(source, cfg) is True
    assert _customer_service_toolsets(cfg) == ["customer_service"]
    prompt = _customer_service_prompt(cfg)
    assert "Sophie de SitioUno" in prompt
    assert "customer_intent_raise" in prompt
    assert "Perfecto, ya tomé nota de tu solicitud" in prompt


def test_customer_service_toolset_is_restricted_to_crm_and_intent_raise():
    resolved = set(resolve_toolset("customer_service"))

    assert "customer_intent_raise" in resolved
    assert "crm_contact_upsert" in resolved
    assert "crm_interaction_record" in resolved
    assert "sales_quote_create" not in resolved
    assert "sales_customer_workspace_create" not in resolved
    assert "calendar_create_event" not in resolved
    assert "calendar_update_event" not in resolved
    assert "crm_quote_create" not in resolved
    assert "crm_invoice_create" not in resolved
    assert "crm_product_upsert" not in resolved
    assert "crm_twenty_raw_request" not in resolved
    assert "crm_twenty_sync" not in resolved
    assert "terminal" not in resolved
    assert "cronjob" not in resolved
    assert "delegate_task" not in resolved


def test_customer_intents_supervisor_toolset_excludes_raise_only_surface():
    resolved = set(resolve_toolset("customer_intents"))

    assert "customer_intent_list" in resolved
    assert "customer_intent_update" in resolved
    assert "customer_intent_raise" not in resolved


def test_customer_service_keeps_owner_whatsapp_on_zeus_operator_route(monkeypatch):
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)
    cfg = {
        "customer_service": {
            "enabled": True,
            "channels": ["whatsapp", "email"],
            "owner_users": {"whatsapp": ["13059274821"]},
        }
    }
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="13059274821@s.whatsapp.net",
        user_id="13059274821@s.whatsapp.net",
    )

    assert _source_matches_customer_service_owner(source, cfg) is True
    assert _source_should_use_customer_service(source, cfg) is False


def test_customer_service_routes_non_owner_email_to_sophie():
    cfg = {
        "customer_service": {
            "enabled": True,
            "channels": ["whatsapp", "email"],
            "owner_users": {"email": ["jean@example.com"]},
        }
    }
    source = SessionSource(
        platform=Platform.EMAIL,
        chat_id="cliente@example.com",
        user_id="cliente@example.com",
    )

    assert _source_should_use_customer_service(source, cfg) is True


def test_whatsapp_dynamic_allowed_users_pass_python_gateway_allowlist(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "dynamic-allowed-users.json").write_text('["13059274824"]', encoding="utf-8")

    config = PlatformConfig(extra={"session_path": str(session_dir), "dm_policy": "allowlist"})
    adapter = WhatsAppAdapter(config)

    assert adapter._is_dm_allowed("13059274824@s.whatsapp.net") is True
    assert adapter._is_dm_allowed("13059274825@s.whatsapp.net") is False
