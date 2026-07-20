import inspect

import hermes_cli.doctor as doctor


def test_doctor_accepts_runtime_model_provider_plugin():
    """Doctor must not reject providers resolved by the runtime plugin registry."""
    source = inspect.getsource(doctor.run_doctor)

    assert "from providers import list_providers as _list_plugin_providers" in source
    assert "known_providers.update(plugin_provider_ids)" in source
    assert 'provider not in ({"auto", "custom"} | plugin_provider_ids)' in source
