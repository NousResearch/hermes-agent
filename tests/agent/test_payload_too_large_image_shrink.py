"""413 on a fresh oversized attachment must shrink the image, not end the turn.

Real case: three iPhone photos attached in one desktop turn were each a ~30 MB base64
part. The first request 413'd; compaction cannot shed the newest user image (by design)
and the last-resort strip only touches tool messages, so the turn ended "Request payload
too large (413). Cannot compress further." and every later turn replayed the same body.
The image_too_large path already re-encodes oversized parts once; 413 now does too,
before spending compression attempts.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.error_classifier import FailoverReason
from agent.turn_overflow import recover_from_overflow
from agent.turn_retry_state import TurnRetryState


def _oversized_jpeg_data_url() -> str:
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed; shrink is best-effort")
    # Noise keeps JPEG from compressing it below the 4 MB shrink target.
    img = Image.effect_noise((2600, 2600), 128).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    assert len(url) > 4 * 1024 * 1024, "fixture must exceed the shrink target"
    return url


def _agent():
    agent = MagicMock()
    agent.base_url = "https://api.anthropic.com"
    agent.log_prefix = ""
    return agent


def _recover(agent, api_messages, retry):
    history = [{"role": "user", "content": [dict(p) for p in api_messages[0]["content"]]}]
    return recover_from_overflow(
        agent, RuntimeError("HTTP 413: Request exceeds the maximum size"),
        SimpleNamespace(reason=FailoverReason.payload_too_large), retry,
        status_code=413, error_msg="request exceeds the maximum size", wrapped_output_cap_budget=None,
        messages=history, api_messages=api_messages, system_message="sys", active_system_prompt="sys",
        conversation_history=history, approx_tokens=1000, compression_attempts=0,
        max_compression_attempts=3, api_call_count=1, effective_task_id="t",
    ), history


def test_413_with_oversized_user_image_shrinks_and_retries_without_compressing():
    url = _oversized_jpeg_data_url()
    api_messages = [{"role": "user", "content": [
        {"type": "text", "text": "here are some screen grabs"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]
    agent = _agent()
    retry = TurnRetryState()

    verdict, history = _recover(agent, api_messages, retry)

    assert verdict.action == "continue"
    assert verdict.compression_attempts == 0, "the shrink must not spend a compression attempt"
    agent._compress_context.assert_not_called()
    sent = api_messages[0]["content"][1]["image_url"]["url"]
    assert sent.startswith("data:image/") and len(sent) < len(url)
    # Copy-on-write: the stored history keeps the original bytes.
    assert history[0]["content"][1]["image_url"]["url"] == url


def test_413_shrink_is_one_shot_then_compression_takes_over(monkeypatch):
    """A second 413 on the same call must not loop on the shrink; it falls to compression."""
    agent = _agent()
    retry = TurnRetryState()
    retry.image_shrink_retry_attempted = True
    shrink = MagicMock(return_value=True)
    monkeypatch.setattr("agent.conversation_compression.try_shrink_image_parts_in_messages", shrink)
    agent._compress_context.side_effect = lambda msgs, sysmsg, **kw: (msgs, sysmsg)
    api_messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    verdict, _ = _recover(agent, api_messages, retry)

    shrink.assert_not_called()
    assert verdict.compression_attempts == 1
