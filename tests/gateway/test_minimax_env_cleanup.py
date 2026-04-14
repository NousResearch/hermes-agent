import importlib
import os
import sys


def _reload_gateway_run(monkeypatch, tmp_path, provider: str):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  provider: {provider}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    monkeypatch.setenv("MINIMAX_CN_API_KEY", "cn-key")
    sys.modules.pop("gateway.run", None)
    return importlib.import_module("gateway.run")


def test_gateway_run_clears_global_minimax_key_for_minimax_cn(monkeypatch, tmp_path):
    _reload_gateway_run(monkeypatch, tmp_path, "minimax-cn")

    assert os.getenv("MINIMAX_API_KEY") is None
    assert os.getenv("MINIMAX_CN_API_KEY") == "cn-key"


def test_gateway_run_preserves_global_minimax_key_for_non_cn_provider(monkeypatch, tmp_path):
    _reload_gateway_run(monkeypatch, tmp_path, "minimax")

    assert os.getenv("MINIMAX_API_KEY") == "global-key"
