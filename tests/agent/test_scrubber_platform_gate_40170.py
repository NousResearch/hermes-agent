"""Adjustment 2 (#40170): the outbound recall scrubber is gated to customer-
facing platforms. On local single-user surfaces (cli/desktop/tui) the recalled
<memory-context> block is the operator's own data, so scrubbing it is a false
positive — the scrubber passes the stream through untouched there. On gateway
platforms one instance may serve multiple users, so it stays on by default.

Default is by platform; both layers are overridable via config:
  memory.scrub_recall_output                       (global)
  platforms.<p>.memory.scrub_recall_output         (flattened per-platform)
  gateway.platforms.<p>.memory.scrub_recall_output (nested per-platform)
"""
from agent.agent_init import _resolve_memory_scrub_recall_output
from agent.memory_manager import build_memory_context_block, StreamingContextScrubber

SECRET = "OPERATOR_SECRET_PEER_CARD"


def _feed(scrubber: StreamingContextScrubber, text: str, size: int) -> str:
    out = [scrubber.feed(text[i:i + size]) for i in range(0, len(text), size)]
    out.append(scrubber.flush())
    return "".join(out)


def _real_block() -> str:
    return build_memory_context_block(f"peer card: {SECRET}")


# -- scrubber enabled flag ---------------------------------------------------

def test_disabled_scrubber_passes_block_through_verbatim():
    """Single-user surface: the agent may surface the operator's own memory."""
    block = _real_block()
    for size in (1, 4, 999):
        out = _feed(StreamingContextScrubber(enabled=False), block, size)
        assert out == block, f"mutated at size {size}"
        assert SECRET in out


def test_enabled_scrubber_still_scrubs():
    """Customer-facing surface: regression guard that gating didn't disable it."""
    text = "Sure, here is what I recalled: " + _real_block() + " — anything else?"
    for size in (1, 4, 999):
        assert SECRET not in _feed(StreamingContextScrubber(enabled=True), text, size)


def test_default_constructor_is_enabled():
    """Direct callers (and existing tests) keep always-on behavior."""
    assert SECRET not in _feed(StreamingContextScrubber(), _real_block(), 4)


# -- platform resolver -------------------------------------------------------

def test_local_platforms_default_off():
    for p in ("cli", "desktop", "tui"):
        assert _resolve_memory_scrub_recall_output({}, p) is False


def test_gateway_platforms_default_on():
    for p in ("whatsapp", "telegram", "discord", "signal"):
        assert _resolve_memory_scrub_recall_output({}, p) is True


def test_empty_platform_defaults_off():
    assert _resolve_memory_scrub_recall_output({}, "") is False
    assert _resolve_memory_scrub_recall_output({}, None) is False


def test_global_override_forces_on_for_local():
    cfg = {"memory": {"scrub_recall_output": True}}
    assert _resolve_memory_scrub_recall_output(cfg, "cli") is True


def test_global_override_forces_off_for_gateway():
    cfg = {"memory": {"scrub_recall_output": False}}
    assert _resolve_memory_scrub_recall_output(cfg, "whatsapp") is False


def test_flat_platform_override_wins():
    cfg = {"platforms": {"whatsapp": {"memory": {"scrub_recall_output": False}}}}
    assert _resolve_memory_scrub_recall_output(cfg, "whatsapp") is False


def test_nested_gateway_platform_override():
    cfg = {"gateway": {"platforms": {"telegram": {"memory": {"scrub_recall_output": False}}}}}
    assert _resolve_memory_scrub_recall_output(cfg, "telegram") is False


def test_flat_platform_override_precedence_over_nested():
    cfg = {
        "gateway": {"platforms": {"whatsapp": {"memory": {"scrub_recall_output": True}}}},
        "platforms": {"whatsapp": {"memory": {"scrub_recall_output": False}}},
    }
    assert _resolve_memory_scrub_recall_output(cfg, "whatsapp") is False
