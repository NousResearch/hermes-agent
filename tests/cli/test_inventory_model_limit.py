import sys
import types

from hermes_cli.inventory import ConfigContext, build_models_payload


def test_build_models_payload_defaults_to_large_model_catalog(monkeypatch):
    observed = {}

    fake_model_switch = types.ModuleType("hermes_cli.model_switch")

    def list_authenticated_providers(**kwargs):
        observed.update(kwargs)
        return []

    setattr(fake_model_switch, "list_authenticated_providers", list_authenticated_providers)
    monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", fake_model_switch)

    ctx = ConfigContext(
        current_provider="nvidia",
        current_model="",
        current_base_url="",
        user_providers={},
        custom_providers=[],
    )

    build_models_payload(ctx)

    assert observed["max_models"] == 200
