"""A leftover ``model.base_url`` must not redirect a named vendor's credential (#133179).

``model.base_url`` is the knob for the ``custom`` provider. When ``model.provider`` names a
vendor whose endpoint is authoritative, a stale URL from a previous ``custom`` setup must not
override that endpoint — the vendor's own credential would be sent to whatever host the stale
URL names. These tests pin the host-plausibility gate in
``runtime_provider._config_base_url_for_provider`` and the overrides that must survive it.
"""

from types import SimpleNamespace

from hermes_cli import runtime_provider as rp

_ZEN_TEST_KEY = "oc_sk_zen_" + "test"
_GEMINI_TEST_KEY = "AIzaSy_test_" + "key"
_POOLED_DS_KEY = "pool_deepseek_" + "test"

_STALE_URL = "https://api.apmix.ai/v1"


def _no_pool(_provider):
    return SimpleNamespace(has_credentials=lambda: False)


def test_stale_custom_base_url_cannot_hijack_opencode_zen_credential(monkeypatch):
    """model.provider switched to opencode-zen while model.base_url still names the previous
    custom host: the Zen bearer token must go to opencode.ai, not the stale third-party host."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-zen")
    monkeypatch.setattr(rp, "load_pool", _no_pool)
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", _ZEN_TEST_KEY)
    monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "opencode-zen",
            "default": "space-bunny-free",
            "base_url": _STALE_URL,
        },
    )

    resolved = rp.resolve_runtime_provider(requested="opencode-zen")

    assert "opencode.ai" in resolved["base_url"]
    assert "apmix" not in resolved["base_url"]
    assert resolved["api_key"] == _ZEN_TEST_KEY


def test_stale_custom_base_url_cannot_hijack_gemini_credential(monkeypatch):
    """Same shape for gemini: the Google API key must stay on the Google endpoint."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "gemini")
    monkeypatch.setattr(rp, "load_pool", _no_pool)
    monkeypatch.setenv("GEMINI_API_KEY", _GEMINI_TEST_KEY)
    for var in ("GEMINI_BASE_URL", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "gemini",
            "default": "gemini-2.5-flash",
            "base_url": _STALE_URL,
        },
    )

    resolved = rp.resolve_runtime_provider(requested="gemini")

    assert "googleapis.com" in resolved["base_url"]
    assert "apmix" not in resolved["base_url"]
    assert resolved["api_key"] == _GEMINI_TEST_KEY


def test_stale_custom_base_url_cannot_hijack_pooled_deepseek_key(monkeypatch):
    """The credential-pool tail must get the same defence: an env-seeded deepseek row keeps the
    registry host, and a stale model.base_url cannot borrow the pooled key for another host."""

    class _Entry:
        runtime_api_key = _POOLED_DS_KEY
        access_token = ""
        source = "env:DEEPSEEK_API_KEY"
        base_url = "https://api.deepseek.com/v1"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self, **_kwargs):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "deepseek")
    monkeypatch.setattr(rp, "load_pool", lambda _p: _Pool())
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "deepseek",
            "default": "deepseek-v4-pro",
            "base_url": _STALE_URL,
        },
    )

    resolved = rp.resolve_runtime_provider(requested="deepseek")

    assert "deepseek.com" in resolved["base_url"]
    assert "apmix" not in resolved["base_url"]
    assert resolved["api_key"] == _POOLED_DS_KEY


def test_explicit_base_url_still_overrides_vendor_endpoint(monkeypatch):
    """Control for #133179: an explicit --base-url is the user's own pairing and wins over the
    vendor's official endpoint — the gate only judges a config-file model.base_url."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-zen")
    monkeypatch.setattr(rp, "load_pool", _no_pool)
    monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "opencode-zen",
            "default": "space-bunny-free",
            "base_url": _STALE_URL,
        },
    )

    resolved = rp.resolve_runtime_provider(
        requested="opencode-zen",
        explicit_api_key=_ZEN_TEST_KEY,
        explicit_base_url=_STALE_URL,
    )

    assert resolved["base_url"] == _STALE_URL


def test_minimax_china_endpoint_still_honoured(monkeypatch):
    """A model.base_url on a family sibling's host (minimax → the minimax-cn China endpoint) is a
    legitimate override and must survive the host gate."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "load_pool", _no_pool)
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "minimax",
            "api_key": "mx-" + "test",
            "base_url": "",
            "source": "env",
        },
    )
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "minimax",
            "default": "MiniMax-M2",
            "base_url": "https://api.minimaxi.com/anthropic",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["base_url"] == "https://api.minimaxi.com/anthropic"


def test_zai_china_endpoint_still_honoured(monkeypatch):
    """zai's global/China endpoints are probed per account (auth_zai_kimi.ZAI_ENDPOINTS), so an
    off-registry model.base_url (open.bigmodel.cn) stays an intentional choice for it."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "zai")
    monkeypatch.setattr(rp, "load_pool", _no_pool)
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "zai",
            "api_key": "glm-" + "test",
            "base_url": "",
            "source": "env",
        },
    )
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "zai",
            "default": "glm-5",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="zai")

    assert resolved["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
