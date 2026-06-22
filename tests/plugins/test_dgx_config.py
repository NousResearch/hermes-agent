"""Tests for plugins/dgx/_dgx_config.py

Covers config load/save, endpoint application, and URL helpers.
All tests use an isolated HERMES_HOME (from conftest) so they never
touch the developer's real ~/.hermes/config.yaml.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(monkeypatch, initial: dict):
    """Patch load_config / save_config with a simple in-memory store."""
    store = dict(initial)

    def _load():
        return dict(store)

    def _save(cfg):
        store.clear()
        store.update(cfg)

    import plugins.dgx._dgx_config as dc
    monkeypatch.setattr("hermes_cli.config.load_config", _load)
    monkeypatch.setattr("hermes_cli.config.save_config", _save)
    monkeypatch.setattr(dc, "load_config", _load, raising=False)
    monkeypatch.setattr(dc, "save_config", _save, raising=False)
    return store


# ---------------------------------------------------------------------------
# load_dgx_config
# ---------------------------------------------------------------------------

class TestLoadDgxConfig:
    def test_returns_defaults_when_no_dgx_section(self, monkeypatch):
        from plugins.dgx._dgx_config import load_dgx_config
        _make_config(monkeypatch, {})
        cfg = load_dgx_config()
        # host defaults to None until configured via `hermes dgx setup`
        assert cfg["host"] is None
        assert cfg["ollama_port"] == 11434
        assert cfg["vllm_port"] == 30800
        assert cfg["active_endpoint"] == "ollama"

    def test_merges_user_values_over_defaults(self, monkeypatch):
        from plugins.dgx._dgx_config import load_dgx_config
        _make_config(monkeypatch, {"dgx": {"host": "10.0.0.5", "ssh_user": "admin"}})
        cfg = load_dgx_config()
        assert cfg["host"] == "10.0.0.5"
        assert cfg["ssh_user"] == "admin"
        assert cfg["ollama_port"] == 11434  # default preserved

    def test_partial_override_preserves_remaining_defaults(self, monkeypatch):
        from plugins.dgx._dgx_config import load_dgx_config
        _make_config(monkeypatch, {"dgx": {"vllm_port": 9000}})
        cfg = load_dgx_config()
        assert cfg["vllm_port"] == 9000
        assert cfg["ollama_port"] == 11434  # untouched


# ---------------------------------------------------------------------------
# save_dgx_config
# ---------------------------------------------------------------------------

class TestSaveDgxConfig:
    def test_writes_dgx_key(self, monkeypatch):
        from plugins.dgx._dgx_config import save_dgx_config
        store = _make_config(monkeypatch, {})
        save_dgx_config({"host": "10.0.0.1", "active_endpoint": "vllm"})
        assert store.get("dgx", {}).get("host") == "10.0.0.1"
        assert store.get("dgx", {}).get("active_endpoint") == "vllm"

    def test_preserves_existing_non_dgx_keys(self, monkeypatch):
        from plugins.dgx._dgx_config import save_dgx_config
        store = _make_config(monkeypatch, {"model": {"default": "some-model"}})
        save_dgx_config({"host": "10.0.0.1"})
        assert store.get("model", {}).get("default") == "some-model"


# ---------------------------------------------------------------------------
# apply_endpoint
# ---------------------------------------------------------------------------

class TestApplyEndpoint:
    # A representative configured DGX (host set) — defaults alone leave host=None,
    # which is the "unconfigured" state.
    _CONFIGURED = {
        "host": "10.0.0.1",
        "ssh_user": "dgx",
        "ollama_port": 11434,
        "vllm_port": 30800,
        "vllm_32b_port": 30881,
        "litellm_host": "10.0.0.2",
        "litellm_port": 4000,
    }

    def _run(self, monkeypatch, endpoint: str, dgx: dict | None = None):
        from plugins.dgx._dgx_config import DEFAULTS, apply_endpoint
        base_dgx = dict(DEFAULTS)
        base_dgx.update(self._CONFIGURED)
        if dgx:
            base_dgx.update(dgx)
        store = _make_config(monkeypatch, {})
        apply_endpoint(base_dgx, endpoint)
        return store

    def test_ollama_sets_correct_provider_and_url(self, monkeypatch):
        store = self._run(monkeypatch, "ollama")
        model = store.get("model", {})
        assert model["provider"] == "ollama"
        assert "10.0.0.1" in model["base_url"]
        assert "11434" in model["base_url"]

    def test_vllm_sets_correct_provider_and_url(self, monkeypatch):
        store = self._run(monkeypatch, "vllm")
        model = store.get("model", {})
        assert model["provider"] == "custom"
        assert "30800" in model["base_url"]

    def test_litellm_sets_correct_provider_and_url(self, monkeypatch):
        store = self._run(monkeypatch, "litellm")
        model = store.get("model", {})
        assert model["provider"] == "custom"
        assert "10.0.0.2" in model["base_url"]
        assert "4000" in model["base_url"]

    def test_litellm_without_host_raises(self, monkeypatch):
        from plugins.dgx._dgx_config import DEFAULTS, apply_endpoint
        monkeypatch.delenv("HERMES_DGX_LITELLM_HOST", raising=False)
        _make_config(monkeypatch, {})
        # configured DGX but no litellm_host
        d = dict(DEFAULTS)
        d.update({"host": "10.0.0.1", "litellm_host": None})
        with pytest.raises(ValueError, match="litellm endpoint requires"):
            apply_endpoint(d, "litellm")

    def test_updates_active_endpoint_in_dgx_block(self, monkeypatch):
        store = self._run(monkeypatch, "vllm")
        assert store.get("dgx", {}).get("active_endpoint") == "vllm"

    def test_unknown_endpoint_raises(self, monkeypatch):
        from plugins.dgx._dgx_config import DEFAULTS, apply_endpoint
        _make_config(monkeypatch, {})
        d = dict(DEFAULTS)
        d.update(self._CONFIGURED)
        with pytest.raises(ValueError, match="Unknown endpoint"):
            apply_endpoint(d, "bogus")

    def test_custom_host_reflected_in_url(self, monkeypatch):
        store = self._run(monkeypatch, "ollama", dgx={"host": "10.0.0.99"})
        assert "10.0.0.99" in store["model"]["base_url"]


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

class TestUrlHelpers:
    def _dgx(self, **overrides):
        from plugins.dgx._dgx_config import DEFAULTS
        d = dict(DEFAULTS)
        d.update({
            "host": "10.0.0.1",
            "ssh_user": "dgx",
            "ollama_port": 11434,
            "vllm_port": 30800,
            "litellm_host": "10.0.0.2",
            "litellm_port": 4000,
        })
        d.update(overrides)
        return d

    def test_ollama_base_default(self):
        from plugins.dgx._dgx_config import ollama_base
        assert ollama_base(self._dgx()) == "http://10.0.0.1:11434"

    def test_vllm_base_default(self):
        from plugins.dgx._dgx_config import vllm_base
        assert vllm_base(self._dgx()) == "http://10.0.0.1:30800"

    def test_litellm_base_default(self):
        from plugins.dgx._dgx_config import litellm_base
        assert litellm_base(self._dgx()) == "http://10.0.0.2:4000"

    def test_litellm_base_returns_none_when_unset(self, monkeypatch):
        from plugins.dgx._dgx_config import litellm_base
        monkeypatch.delenv("HERMES_DGX_LITELLM_HOST", raising=False)
        d = self._dgx(litellm_host=None)
        assert litellm_base(d) is None

    def test_ollama_base_raises_when_host_missing(self):
        from plugins.dgx._dgx_config import DEFAULTS, DGXNotConfigured, ollama_base
        d = dict(DEFAULTS)
        d["host"] = None
        d.pop("_active_node", None)
        with pytest.raises(DGXNotConfigured):
            ollama_base(d)

    def test_custom_host_and_port(self):
        from plugins.dgx._dgx_config import ollama_base
        assert ollama_base(self._dgx(host="10.0.0.5", ollama_port=9999)) == "http://10.0.0.5:9999"


# ---------------------------------------------------------------------------
# Config policy — no HERMES_* env vars for non-secret settings (AGENTS.md)
# ---------------------------------------------------------------------------

class TestNoHermesEnvVars:
    """AGENTS.md ("What we don't want"): non-secret behavioral config — host,
    ports, the LiteLLM host — must live in config.yaml via `hermes dgx setup`,
    NOT in new ``HERMES_*`` env vars (``.env`` is for secrets only).
    """

    def _dgx_source_files(self):
        from pathlib import Path
        pkg = Path(__file__).parents[2] / "plugins" / "dgx"
        return sorted(pkg.glob("*.py"))

    def test_no_hermes_dgx_env_vars_in_source(self):
        offenders = []
        for py in self._dgx_source_files():
            for i, line in enumerate(py.read_text().splitlines(), 1):
                if "HERMES_DGX" in line:
                    offenders.append(f"{py.name}:{i}: {line.strip()}")
        assert not offenders, (
            "HERMES_DGX_* env vars are banned for non-secret config "
            "(AGENTS.md — use config.yaml via `hermes dgx setup`):\n"
            + "\n".join(offenders)
        )

    def test_defaults_do_not_read_host_from_environment(self, monkeypatch):
        # Even with a HERMES_DGX_HOST exported, a fresh import must leave the
        # host unset — the value comes only from config.yaml.
        import importlib
        import plugins.dgx._dgx_config as dc
        monkeypatch.setenv("HERMES_DGX_HOST", "10.9.9.9")
        monkeypatch.setenv("HERMES_DGX_SSH_USER", "intruder")
        monkeypatch.setenv("HERMES_DGX_OLLAMA_PORT", "59999")
        try:
            importlib.reload(dc)
            assert dc.DEFAULTS["host"] is None
            assert dc.NODE_DEFAULTS["host"] is None
            assert dc.DEFAULTS["ssh_user"] != "intruder"
            assert dc.DEFAULTS["ollama_port"] == 11434
        finally:
            monkeypatch.undo()
            importlib.reload(dc)
