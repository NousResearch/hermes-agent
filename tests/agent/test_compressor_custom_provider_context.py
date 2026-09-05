"""Deferred compressor resolution must honor per-model provider overrides.

``ContextCompressor._resolve_context_length`` is the only resolution site
that did not thread the compatible custom-provider list into
``get_model_context_length`` (startup, ``/model`` switch and gateway
``/info`` all do — see #15779). As a result a
``providers.<name>.models.<id>.context_length`` entry was silently skipped
on the lazy path and ``/usage`` reported the catalog window instead.
"""

from agent.context_compressor import ContextCompressor


def _providers(models):
    return [
        {
            "name": "test-relay",
            "base_url": "https://relay.internal.example.com/v1",
            "models": models,
        }
    ]


def test_lazy_resolution_honors_per_model_override(monkeypatch):
    """The lazy path returns the override without any network probe."""
    from hermes_cli import config as _config_mod

    monkeypatch.setattr(
        _config_mod,
        "get_compatible_custom_providers",
        lambda config=None: _providers(
            {"zz-override-probe-model": {"context_length": 272000}}
        ),
    )
    cc = ContextCompressor(
        model="zz-override-probe-model",
        base_url="https://relay.internal.example.com/v1",
        provider="custom",
        quiet_mode=True,
    )
    assert cc.context_length == 272000


def test_explicit_config_context_length_still_wins(monkeypatch):
    """Precedence is unchanged: explicit pin beats the per-model entry."""
    from hermes_cli import config as _config_mod

    monkeypatch.setattr(
        _config_mod,
        "get_compatible_custom_providers",
        lambda config=None: _providers(
            {"zz-override-probe-model": {"context_length": 272000}}
        ),
    )
    cc = ContextCompressor(
        model="zz-override-probe-model",
        base_url="https://relay.internal.example.com/v1",
        provider="custom",
        config_context_length=100000,
        quiet_mode=True,
    )
    assert cc.context_length == 100000


def test_unlisted_model_ignores_override(monkeypatch):
    """A model with no entry must not inherit another model's override."""
    from hermes_cli import config as _config_mod

    monkeypatch.setattr(
        _config_mod,
        "get_compatible_custom_providers",
        lambda config=None: _providers(
            {"zz-override-probe-model": {"context_length": 272000}}
        ),
    )
    cc = ContextCompressor(
        model="zz-unlisted-model",
        base_url="https://relay.internal.example.com/v1",
        provider="custom",
        config_context_length=100000,
        quiet_mode=True,
    )
    assert cc.context_length == 100000
