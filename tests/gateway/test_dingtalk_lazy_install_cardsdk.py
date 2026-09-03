"""Regression: DingTalk card SDK symbols must be rebound after lazy-install.

When the alibabacloud-dingtalk card SDK is NOT importable at module load time,
the top-level ``try/except`` in plugins/platforms/dingtalk/adapter.py stubs its
seven symbols to ``None`` and leaves ``CARD_SDK_AVAILABLE`` False.

``ensure_dingtalk_deps()`` lazy-installs the ``platform.dingtalk`` bundle, which
ships both dingtalk-stream AND alibabacloud-dingtalk (see
tools/lazy_deps.py). Its ``global`` declaration originally listed only the
dingtalk-stream/httpx names, so after a *successful* install the seven
card-SDK symbols stayed bound to the import-time ``None`` placeholder and
``CARD_SDK_AVAILABLE`` stayed False for the life of the process -- silently
disabling AI Cards / emotion reactions / media download even though the SDK
was now installed.

``ensure_dingtalk_deps()`` must add the card-SDK names to its ``global``
declaration, re-import them, rebind them, and flip ``CARD_SDK_AVAILABLE`` to
True so those features light up the moment the deps become importable.

Mirrors tests/gateway/test_discord_lazy_install_views.py and
tests/gateway/test_feishu_lazy_import.py: the lazy install itself is stubbed
(we assert the rebind wiring, not pip), and the SDK modules are injected into
``sys.modules`` so the re-import inside ensure_dingtalk_deps() resolves without
the real packages being present in the venv. The assertion is deliberately
widened over the whole card-SDK symbol set so a future dropped symbol fails and
names itself.

Fixes: #86211.
"""
import sys
import types
from unittest.mock import patch

import pytest

# Import the adapter BEFORE any fake SDK modules are injected into sys.modules,
# so the module body loads against the genuinely-absent deps
# (DINGTALK_STREAM_AVAILABLE=False -> class bodies fall back to ``object``),
# exactly as it would in a real deferred-install deployment.
from plugins.platforms.dingtalk import adapter


# The seven symbols the top-level ``except`` block stubs to None and that the
# lazy-install rebind must restore. Named explicitly so a regression that drops
# any one of them fails here with the offending symbol.
_CARD_SDK_NAMES = [
    "dingtalk_card_client",
    "dingtalk_card_models",
    "dingtalk_robot_client",
    "dingtalk_robot_models",
    "open_api_models",
    "tea_util_models",
]


def _fake_module(name):
    """A bare stand-in module usable as an importable SDK submodule."""
    return types.ModuleType(name)


@pytest.fixture
def fake_sdk_modules(monkeypatch):
    """Inject fake dingtalk-stream + card-SDK modules into ``sys.modules``.

    Lets the re-import statements inside ensure_dingtalk_deps() resolve to real
    (fake) module objects -- distinguishable from the import-time ``None``
    placeholder -- without the packages being installed in the test venv.
    """
    # Base dingtalk-stream + httpx (rebound before the card block).
    ds = _fake_module("dingtalk_stream")
    ds.ChatbotMessage = type("ChatbotMessage", (), {})
    ds_frames = _fake_module("dingtalk_stream.frames")
    ds_frames.CallbackMessage = type("CallbackMessage", (), {})
    ds_frames.AckMessage = type("AckMessage", (), {})
    httpx_mod = _fake_module("httpx")

    # Card SDK: alibabacloud_dingtalk.card_1_0 / robot_1_0 expose ``client`` and
    # ``models`` submodules; tea_openapi / tea_util expose ``models``.
    ali = _fake_module("alibabacloud_dingtalk")
    card = _fake_module("alibabacloud_dingtalk.card_1_0")
    card.client = _fake_module("alibabacloud_dingtalk.card_1_0.client")
    card.models = _fake_module("alibabacloud_dingtalk.card_1_0.models")
    robot = _fake_module("alibabacloud_dingtalk.robot_1_0")
    robot.client = _fake_module("alibabacloud_dingtalk.robot_1_0.client")
    robot.models = _fake_module("alibabacloud_dingtalk.robot_1_0.models")
    tea_openapi = _fake_module("alibabacloud_tea_openapi")
    tea_openapi.models = _fake_module("alibabacloud_tea_openapi.models")
    tea_util = _fake_module("alibabacloud_tea_util")
    tea_util.models = _fake_module("alibabacloud_tea_util.models")

    injected = {
        "dingtalk_stream": ds,
        "dingtalk_stream.frames": ds_frames,
        "httpx": httpx_mod,
        "alibabacloud_dingtalk": ali,
        "alibabacloud_dingtalk.card_1_0": card,
        "alibabacloud_dingtalk.card_1_0.client": card.client,
        "alibabacloud_dingtalk.card_1_0.models": card.models,
        "alibabacloud_dingtalk.robot_1_0": robot,
        "alibabacloud_dingtalk.robot_1_0.client": robot.client,
        "alibabacloud_dingtalk.robot_1_0.models": robot.models,
        "alibabacloud_tea_openapi": tea_openapi,
        "alibabacloud_tea_openapi.models": tea_openapi.models,
        "alibabacloud_tea_util": tea_util,
        "alibabacloud_tea_util.models": tea_util.models,
    }
    for mod_name, mod in injected.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)
    return injected


class TestEnsureDingtalkDepsRebindsCardSDK:
    """ensure_dingtalk_deps() must rebind the card SDK after a lazy install."""

    def test_card_sdk_symbols_rebound_after_lazy_install(
        self, monkeypatch, fake_sdk_modules
    ):
        """After a successful install none of the 7 card symbols stay stubbed."""
        # Simulate the state where the module was imported with BOTH the base
        # dingtalk-stream deps and the card SDK missing (the lazy-install
        # scenario): base + card availability flags False, card symbols stubbed
        # to the import-time None placeholder.
        monkeypatch.setattr(adapter, "DINGTALK_STREAM_AVAILABLE", False)
        monkeypatch.setattr(adapter, "HTTPX_AVAILABLE", False)
        monkeypatch.setattr(adapter, "CARD_SDK_AVAILABLE", False)
        for name in _CARD_SDK_NAMES:
            monkeypatch.setattr(adapter, name, None)

        # Pre-condition: every card symbol is the None placeholder.
        for name in _CARD_SDK_NAMES:
            assert getattr(adapter, name) is None, (
                f"{name} should be the None placeholder before the rebind"
            )

        # Stub the lazy install to a no-op success; the injected sys.modules
        # entries stand in for the freshly-installed packages.
        with patch("tools.lazy_deps.ensure", autospec=True) as ensure:
            assert adapter.ensure_dingtalk_deps() is True

        ensure.assert_called_once_with("platform.dingtalk", prompt=False)

        # Widened assertion: NONE of the seven card symbols may remain bound to
        # the import-time None placeholder. Report the whole offending set so a
        # future dropped symbol fails and names itself.
        still_stubbed = [n for n in _CARD_SDK_NAMES if getattr(adapter, n) is None]
        assert not still_stubbed, (
            "card SDK symbols left bound to the import-time None placeholder "
            f"after ensure_dingtalk_deps(): {still_stubbed}"
        )

        assert adapter.CARD_SDK_AVAILABLE is True, (
            "CARD_SDK_AVAILABLE must flip True once the card SDK is rebound"
        )
