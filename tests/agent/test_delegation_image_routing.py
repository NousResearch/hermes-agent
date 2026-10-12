"""Delegated children keep their own image-routing identity.

Child precedence: ``delegation.image_input_mode`` → ``agent.image_input_mode``
→ capability, with only the ``auxiliary.vision`` pin exempted for children (it
serves the main conversation's vision tool). Capability overrides are
route-scoped: a top-level ``model.supports_vision`` declaration answers for the
MAIN route only, never for a differently-resolved child.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent import image_routing
from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER, delegated_child_context
from agent.image_routing import decide_image_input_mode
from hermes_constants import get_hermes_home


_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# The main conversation pins a describer (e.g. a text-only main on a vision-capable flash).
_AUX_PINNED = {
    "agent": {},
    "auxiliary": {
        "vision": {
            "provider": "zai",
            "model": "glm-5.3-flash",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
        }
    },
}


def _force_vision(monkeypatch, verdict: bool) -> None:
    """Deterministic capability lookup — no catalog, no network."""
    monkeypatch.setattr(image_routing, "_lookup_supports_vision", lambda *a, **k: verdict)


def test_pinned_aux_keeps_main_conversation_text_mode(monkeypatch):
    _force_vision(monkeypatch, True)
    assert decide_image_input_mode("zai", "glm-5.3", _AUX_PINNED) == "text"


def test_pinned_aux_does_not_demote_vision_capable_child(monkeypatch):
    _force_vision(monkeypatch, True)
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3-flash", _AUX_PINNED) == "native"


def test_child_on_text_only_model_stays_on_text_lane(monkeypatch):
    _force_vision(monkeypatch, False)
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3", _AUX_PINNED) == "text"


def test_delegation_mode_text_pins_vision_capable_child(monkeypatch):
    _force_vision(monkeypatch, True)
    cfg = {**_AUX_PINNED, "delegation": {"image_input_mode": "text"}}
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3-flash", cfg) == "text"


def test_delegation_mode_native_overrides_child_capability(monkeypatch):
    _force_vision(monkeypatch, False)
    cfg = {**_AUX_PINNED, "delegation": {"image_input_mode": "native"}}
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3", cfg) == "native"


def test_delegation_mode_does_not_leak_into_main_conversation(monkeypatch):
    _force_vision(monkeypatch, True)
    cfg = {**_AUX_PINNED, "delegation": {"image_input_mode": "native"}}
    assert decide_image_input_mode("zai", "glm-5.3", cfg) == "text"


def test_child_inherits_agent_image_input_mode_pin(monkeypatch):
    """Compatibility fallback: an explicit agent-level pin keeps protecting
    children when no child-specific policy is set (an image-rejecting backend
    must not start receiving pixels because a child can see)."""
    _force_vision(monkeypatch, True)
    cfg = {**_AUX_PINNED, "agent": {"image_input_mode": "text"}}
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3-flash", cfg) == "text"


def test_delegation_pin_overrides_inherited_agent_pin(monkeypatch):
    """An explicit child policy wins over the inherited agent-level pin."""
    _force_vision(monkeypatch, False)
    cfg = {**_AUX_PINNED, "agent": {"image_input_mode": "text"},
           "delegation": {"image_input_mode": "native"}}
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3", cfg) == "native"


def test_main_capability_declaration_does_not_answer_for_child():
    """A top-level ``model.supports_vision`` declares the MAIN model; a child on
    a different route must not inherit the verdict. Real override resolution
    (no patched lookup): the per-provider entry for the child's own model wins,
    in both directions."""
    cfg = {
        "model": {"default": "main-model", "supports_vision": True},
        "providers": {"zai": {"models": {"child-blind": {"supports_vision": False}}}},
    }
    with delegated_child_context("test-child"):
        # explicitly blind child stays on text even though the main declares vision
        assert decide_image_input_mode("zai", "child-blind", cfg) == "text"
    # the main route keeps its own declaration
    assert decide_image_input_mode("zai", "main-model", cfg) == "native"


def test_main_blind_declaration_does_not_suppress_vision_child():
    cfg = {
        "model": {"default": "main-model", "supports_vision": False},
        "providers": {"zai": {"models": {"child-seeing": {"supports_vision": True}}}},
    }
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "child-seeing", cfg) == "native"


def test_env_marker_enters_child_branch(monkeypatch):
    """The ``HERMES_DELEGATED_CHILD_CONTEXT`` marker (inherited by descendant
    processes) routes like the ContextVar: agent pin inherited, aux pin skipped."""
    _force_vision(monkeypatch, True)
    cfg = {**_AUX_PINNED, "agent": {"image_input_mode": "text"}}
    with patch.dict("os.environ", {DELEGATED_CHILD_ENV_MARKER: "/boards/demo"}):
        assert decide_image_input_mode("zai", "glm-5.3-flash", cfg) == "text"


def test_delegation_mode_read_through_real_loader(tmp_path, monkeypatch):
    """The new key round-trips: config.yaml → load_config_readonly → routing."""
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "delegation:\n"
        "  image_input_mode: text\n"
        "auxiliary:\n"
        "  vision:\n"
        "    provider: zai\n"
        "    model: glm-5.3-flash\n",
        encoding="utf-8",
    )
    _force_vision(monkeypatch, True)
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3-flash", cfg) == "text"


def test_spawn_site_builds_goal_inside_child_context(tmp_path):
    """The production boundary: ``_ChildRun.await_child`` must enter
    delegated-child context BEFORE the multimodal goal message is built, so the
    routing decision (recorder-wrapped) sees child scope without any test
    supplying it. Deleting the spawn-site wrap turns this red."""
    from tools.delegate_tool_child_run import _ChildRun

    seen = {}

    def _recorder(provider, model, cfg, **kw):
        from agent.delegation_context import is_delegated_child_process_context

        seen["child_scope"] = is_delegated_child_process_context()
        return "native"

    image = tmp_path / "img.png"
    image.write_bytes(_TINY_PNG)

    class _Child:
        provider = "zai"
        model = "glm-5.3-flash"
        requested_provider = ""
        base_url = ""
        api_key = ""
        api_mode = ""
        _delegate_images = [str(image)]

        def run_conversation(self, user_message, **kw):
            seen["user_message"] = user_message
            return {"final_response": "ok", "completed": True}

    class _Parent:
        _current_task_id = None

    run = _ChildRun(_Child(), _Parent(), 0, "describe the image", None, None)
    with patch("tools.delegate_tool_child_run._create_isolated_worktree", return_value=None), \
            patch("agent.image_routing.decide_image_input_mode", side_effect=_recorder):
        run.seed_workspace()
        result, failure, _ = run.await_child()
    assert failure is None and result["final_response"] == "ok"
    assert seen["child_scope"] is True, "spawn site must supply the child scope itself"
    content = seen["user_message"]
    assert isinstance(content, list) and any(p.get("type") == "image_url" for p in content)


def test_multimodal_goal_embeds_pixels_inside_child_context(tmp_path, monkeypatch):
    """The goal-message helper honors the child scope: inside
    ``delegated_child_context`` a vision-capable child (lookup patched
    deterministically — no catalog, no network) gets image parts even when the
    persisted config pins ``auxiliary.vision``; outside the child scope the
    same pinned config keeps the hint lane (main-conversation contract
    unchanged — the spawn site is what wraps the call in the child context)."""
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "auxiliary:\n"
        "  vision:\n"
        "    provider: zai\n"
        "    model: glm-5.3-flash\n",
        encoding="utf-8",
    )
    image = tmp_path / "img.png"
    image.write_bytes(_TINY_PNG)
    child = SimpleNamespace(provider="zai", model="child-model", requested_provider="",
                            base_url="", api_key="", api_mode="")
    monkeypatch.setattr(image_routing, "_lookup_supports_vision", lambda *a, **k: True)

    from tools.delegate_tool_child_run import _build_child_goal_message

    with delegated_child_context("test-child"):
        parts = _build_child_goal_message("describe the image", [str(image)], child)
    assert isinstance(parts, list), f"expected multimodal parts, got {parts!r}"
    assert any(p.get("type") == "image_url" for p in parts)

    hints = _build_child_goal_message("describe the image", [str(image)], child)
    assert isinstance(hints, str) and "vision_analyze" in hints
