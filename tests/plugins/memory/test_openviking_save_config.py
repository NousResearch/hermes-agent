"""OpenViking configuration writes honor the callback's explicit profile home."""
from __future__ import annotations

import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.openviking import OpenVikingMemoryProvider


def test_save_config_targets_explicit_home_and_restores_outer_scope(tmp_path, monkeypatch):
    active_home = tmp_path / "active"
    outer_home = tmp_path / "outer"
    target_home = tmp_path / "target"
    for home in (active_home, outer_home, target_home):
        home.mkdir()

    active_before = "model:\n  default: active-model\nmemory:\n  provider: openviking\n"
    outer_before = "model:\n  default: outer-model\nmemory:\n  provider: openviking\n"
    target_before = "model:\n  default: target-model\nmemory:\n  provider: openviking\n"
    (active_home / "config.yaml").write_text(active_before, encoding="utf-8")
    (outer_home / "config.yaml").write_text(outer_before, encoding="utf-8")
    (target_home / "config.yaml").write_text(target_before, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(active_home))

    outer_token = set_hermes_home_override(outer_home)
    try:
        OpenVikingMemoryProvider().save_config(
            {"endpoint": "https://target.example/v1", "recall_policy": "always"},
            str(target_home),
        )

        # The explicit callback target is updated; neither ambient scope is touched.
        assert (active_home / "config.yaml").read_text(encoding="utf-8") == active_before
        assert (outer_home / "config.yaml").read_text(encoding="utf-8") == outer_before
        target = yaml.safe_load((target_home / "config.yaml").read_text(encoding="utf-8"))
        assert target["model"]["default"] == "target-model"
        assert target["memory"]["provider"] == "openviking"
        assert target["memory"]["openviking"] == {
            "endpoint": "https://target.example/v1",
            "recall_policy": "always",
        }

        # The nested override was restored, so a canonical write still targets outer_home.
        from hermes_cli.config import save_config

        save_config({"memory": {"provider": "outer-restored"}}, merge_existing=True)
        outer = yaml.safe_load((outer_home / "config.yaml").read_text(encoding="utf-8"))
        assert outer["memory"]["provider"] == "outer-restored"
        assert (active_home / "config.yaml").read_text(encoding="utf-8") == active_before
    finally:
        reset_hermes_home_override(outer_token)
