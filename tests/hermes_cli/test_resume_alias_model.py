"""Repeating the creation `-m <alias>` on resume is the same frozen route, not an override."""
import argparse

import pytest


def test_resume_accepts_the_creation_alias_and_still_refuses_another_model(tmp_path, monkeypatch):
    from hermes_cli import gateway_chat, model_switch
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://127.0.0.1:1/v1\n  default: model-main\n"
        "model_aliases:\n  fast: {model: model-alias, provider: custom, base_url: http://127.0.0.1:3/v1}\n",
        encoding="utf-8")
    model_switch.DIRECT_ALIASES.clear()
    frozen = {"info": {"launch_request": {"source": "cli", "model": "model-alias", "provider": "custom",
                                          "base_url": "http://127.0.0.1:3/v1"}}}
    gateway_chat.check_resume_policy(argparse.Namespace(resume="s", model="fast", query="x"), frozen)
    gateway_chat.check_resume_policy(argparse.Namespace(resume="s", model="model-alias", query="x"), frozen)
    for model in ("model-main", "other"):
        with pytest.raises(gateway_chat.GatewayClientError):
            gateway_chat.check_resume_policy(argparse.Namespace(resume="s", model=model, query="x"), frozen)
