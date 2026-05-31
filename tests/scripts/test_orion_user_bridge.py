"""Regression tests for the Dan-user Telegram bridge to Orión."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BRIDGE_PATH = Path(__file__).resolve().parents[2].parent / "scripts" / "orion_user_bridge.py"


def load_bridge():
    spec = importlib.util.spec_from_file_location("orion_user_bridge", BRIDGE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_status_noise_guard_blocks_known_loop_messages():
    bridge = load_bridge()

    assert bridge._is_status_noise("[Orion] ⚠️ Gateway shutting down — Your current task will be interrupted.")
    assert bridge._is_status_noise("HECHO\nRESULTADO\nSin acción")
    assert bridge._is_status_noise("⏳ Still working...")


def test_status_noise_guard_allows_real_human_work_request():
    bridge = load_bridge()

    assert not bridge._is_status_noise("Revisa el demo de StudioLab y dime qué falta, sin publicar producción.")


def test_validate_outbound_adds_orion_mention_and_bounds_prompt():
    bridge = load_bridge()

    out = bridge._validate_outbound("haz eco", allow_status_noise=False, max_chars=100)

    assert out.startswith("@orion_dev_das_bot")
    assert "haz eco" in out


def test_controlled_echo_prompt_contains_request_id_and_no_side_effect_guardrails():
    bridge = load_bridge()

    prompt = bridge._controlled_echo_prompt("payload", "abc123")

    assert "ORION_BRIDGE_CONTROLLED_TEST request_id=abc123" in prompt
    assert "ORION_BRIDGE_OK request_id=abc123" in prompt
    assert "No ejecutes herramientas" in prompt
    assert "no cambies archivos" in prompt
