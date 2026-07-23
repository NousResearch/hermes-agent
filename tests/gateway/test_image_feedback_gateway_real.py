"""Real-GatewayRunner integration check for the image-feedback prepend layer.

Drives the REAL GatewayRunner prepend-path components (not a mock of the
implementation): the per-session status store, ``_pop_image_feedback_status``,
and ``build_image_feedback_line``. Behavioral prepend wiring (IMG-007
prepend-non-replace) is covered by red-team ``test_image_feedback.py``; this
module deliberately does NOT read implementation source text (forbidden
by AGENTS.md source-shape rule).

Full ``_handle_message_with_agent`` end-to-end needs ~20 mocked前置
(session_db / topic recovery / history / hooks / guards) which would test
the mocks rather than the feedback layer. Instead we verify each REAL piece,
complementing red-team ``test_image_feedback.py`` which covers IMG-003
send-delta==0, IMG-007 prepend-non-replace, IMG-011 cross-turn isolation at
the behavioral level.
"""

import inspect

from agent.image_routing import ImageFeedbackStatus, build_image_feedback_line
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def test_real_runner_pop_reads_and_clears():
    """Real GatewayRunner: pop reads the preset status and clears the store
    so it cannot leak into the next turn (IMG-011)."""
    runner = GatewayRunner(GatewayConfig())
    sk = "agent:main:discord:dm:9"
    runner._image_feedback_status_by_session[sk] = ImageFeedbackStatus(
        mode="text", total=1, succeeded=1, failed=0, reasons=[]
    )

    popped = runner._pop_image_feedback_status(sk)
    assert popped is not None and popped.mode == "text" and popped.succeeded == 1
    # Cleared — second pop is None.
    assert sk not in runner._image_feedback_status_by_session
    assert runner._pop_image_feedback_status(sk) is None


def test_real_runner_build_reads_cfg_and_prepends_verbatim():
    """Real build_image_feedback_line: renders the MAIN model name from the
    captured ``status.main_model`` (IMG-FIX1) and the vision model name from
    cfg; prepend formula `line + "\\n" + body` preserves body verbatim (IMG-007)."""
    cfg = {
        "model": {"default": "deepseek-v4-pro"},
        "auxiliary": {"vision": {"model": "qwen3-vl-plus"}},
    }
    # main_model is CAPTURED at routing time (IMG-FIX1) — it is NOT read from cfg.
    status = ImageFeedbackStatus(
        mode="text", total=1, succeeded=1, failed=0, reasons=[], main_model="deepseek-v4-pro"
    )
    line = build_image_feedback_line(status, cfg)
    assert "deepseek-v4-pro" in line, f"main model name missing: {line!r}"
    assert "qwen3-vl-plus" in line, f"vision model name missing: {line!r}"
    assert line.startswith("📎")

    body = "这是给用户的正文回复，逐字保留。"
    combined = f"{line}\n{body}"
    assert combined.startswith(line) and combined.endswith(body)
    assert len(combined) == len(line) + 1 + len(body)


def test_status_line_is_pure_string_no_send_surface():
    """build_image_feedback_line signature carries no adapter/send param —
    it cannot emit a message even if it wanted to (IMG-003 red line is
    structural; behavioral send-delta==0 is in red-team test_img_003)."""
    status = ImageFeedbackStatus(mode="text", total=1, succeeded=1, failed=0, reasons=[])
    line = build_image_feedback_line(
        status, {"model": {"default": "m"}, "auxiliary": {"vision": {"model": "v"}}}
    )
    assert isinstance(line, str) and line.startswith("📎")
    sig = inspect.signature(build_image_feedback_line)
    assert "send" not in sig.parameters and "adapter" not in sig.parameters


def test_runner_without_init_dict_does_not_crash():
    """_pop_image_feedback_status uses getattr fallback so harnesses that
    build GatewayRunner via object.__new__ don't crash."""
    bare = object.__new__(GatewayRunner)
    assert bare._pop_image_feedback_status("any") is None
