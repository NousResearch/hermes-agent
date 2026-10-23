"""Regression tests for issue #135811.

``_PROVIDER_VISION_MODELS`` pinned xiaomi auxiliary vision to ``mimo-v2.5``,
which Xiaomi retires on 2026-10-21 — after that date every ``provider: xiaomi``
profile with no explicit ``auxiliary.vision.model`` resolves a dead model id
(HTTP 404 / model_not_found) for vision_analyze, screenshot analysis and the
per-turn tool-registry vision gate.

The pin moves to the image-capable successor ``mimo-v2.6-flash`` (same repair
as the zai glm-5v-turbo → glm-5.3-flash pin, #111429). The pin must stay a pin:
the xiaomi ProviderProfile has no ``default_vision_model()``, so removing the
entry would route vision to the user's (possibly text-only) chat model.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def isolated_home(monkeypatch):
    """Temp HERMES_HOME with config + clean credential env vars."""
    test_home = tempfile.mkdtemp(prefix="hermes_test_135811_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    monkeypatch.setenv("HERMES_HOME", hermes_home)

    # Strip all credential-shaped env vars so each scenario starts hermetic.
    for k in list(os.environ.keys()):
        if k.endswith("_API_KEY") or k.endswith("_TOKEN"):
            monkeypatch.delenv(k, raising=False)

    yield hermes_home
    shutil.rmtree(test_home, ignore_errors=True)


def _write_config(home: str, text: str) -> None:
    """Write config.yaml inside the isolated home only (containment-checked)."""
    config_file = Path(home, "config.yaml").resolve()
    if not config_file.is_relative_to(Path(home).resolve()):
        raise ValueError("config path escapes the isolated home")
    config_file.write_text(text, encoding="utf-8")


_RELOAD_PREFIXES = ("agent.auxiliary_client", "agent.image_routing",
                    "tools.vision_tools", "hermes_cli.config")


def _drop_reload_targets():
    for mod in list(sys.modules.keys()):
        if mod.startswith(_RELOAD_PREFIXES):
            del sys.modules[mod]


@pytest.fixture(autouse=True)
def _module_isolation():
    """Save/restore sys.modules entries this file reloads (issue #61597)."""
    saved = {name: mod for name, mod in sys.modules.items()
             if name.startswith(_RELOAD_PREFIXES)}
    yield
    _drop_reload_targets()
    sys.modules.update(saved)


def _fresh_modules():
    _drop_reload_targets()


class TestXiaomiVisionDefaultPin:
    def test_pin_points_at_live_successor_model(self, isolated_home):
        """The xiaomi static pin must be the live mimo-v2.6-flash, not the retired mimo-v2.5."""
        _fresh_modules()

        from agent.auxiliary_client import _resolve_provider_vision_default
        assert _resolve_provider_vision_default("xiaomi") == "mimo-v2.6-flash"

    def test_auto_route_uses_pin_not_chat_model(self, isolated_home, monkeypatch):
        """End-to-end: a xiaomi main provider whose chat model differs from the pin
        resolves the successor id for auxiliary vision, not the retired mimo-v2.5
        and not the chat model."""
        _write_config(isolated_home, """
model:
  provider: xiaomi
  default: mimo-v2.6-pro
""")
        monkeypatch.setenv("XIAOMI_API_KEY", "sk-test")
        _fresh_modules()

        # Offline hermeticity: the catalog verdict for the successor id is asserted
        # (True) instead of fetched — the pin's routing, not the catalog, is under test.
        import agent.image_routing as image_routing
        monkeypatch.setattr(
            image_routing, "_lookup_supports_vision",
            lambda provider, model, cfg=None, **_: True,
        )

        from agent.auxiliary_client import resolve_vision_provider_client
        provider, client, model = resolve_vision_provider_client(provider="auto")
        assert provider == "xiaomi"
        assert client is not None, "xiaomi auto vision should produce a usable client"
        assert model == "mimo-v2.6-flash", (
            "auto vision must resolve the live successor id, not the retired "
            "mimo-v2.5 (dead after Xiaomi's 2026-10-21 cutoff)"
        )
