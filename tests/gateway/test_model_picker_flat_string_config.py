"""Regression: the bare /model picker must honor a flat-string model: value.

_ModelSwitchContext.read_config read cfg[model] behind an isinstance(model_cfg, dict)
guard, so a scalar model: deepseek-v4-flash was dropped and the picker opened with
current_model="" — no current marker and the switch resolved against "". The
sibling test_model_command_flat_string_config.py covers persistence, not display.
"""
from gateway.slash_commands_model import _ModelSwitchContext


def _ctx(tmp_path, monkeypatch, cfg):
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda **kw: cfg)
    ctx = _ModelSwitchContext("test", None, tmp_path / "config.yaml", False)
    ctx.read_config()
    return ctx


def test_flat_string_model_reaches_picker(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, {"model": "deepseek-v4-flash", "providers": {}})
    assert ctx.current_model == "deepseek-v4-flash"


def test_nested_dict_model_unchanged(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, {"model": {"default": "m", "provider": "p", "base_url": "u"}, "providers": {}})
    assert ctx.current_model == "m"
    assert ctx.current_provider == "p"
    assert ctx.current_base_url == "u"
