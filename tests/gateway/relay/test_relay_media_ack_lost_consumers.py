"""A relay attachment whose ack is lost reaches its consumers as "possibly delivered".

The connector applied ``send_media`` but its ack never came back (socket hiccup,
connector restart). The relay adapter reports that as success with
``raw_response["ambiguous"]``. Both consumers that re-send on failure must carry
the doubt instead of dropping it: the model's ``send_message`` tool says "do not
resend", and a cron run marks the target unverified (``last_delivery_unverified``)
rather than reporting a clean delivery.

Real ``RelayAdapter`` + ``WebSocketRelayTransport`` + ``RelayMediaClient``; only the
connector is fake, at its network boundary.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from hermes_agent.gateway.config import Platform
from tests.cron.test_cron_relay_media_upload_failure import CHAT, _gateway_config, live_relay


@pytest.fixture(params=["ack_timeout", "socket_drop"])
def ack_lost_relay(request):
    """The connected relay, with every ``send_media`` applied but its ack lost:
    either it never comes back (timeout) or the socket drops right after the
    connector received the frame (reader exit settles the pending request)."""
    connector, adapter, loop = request.getfixturevalue("live_relay")
    if request.param == "ack_timeout":
        connector.unacked_ops = {"send_media"}
    else:
        connector.dropping_ops = {"send_media"}
    adapter._transport._outbound_timeout_s = 0.5
    return connector, adapter, loop


def _no_fallback_notice(connector) -> bool:
    return not any("q3-report" in str(a.get("content", "")) for a in connector.ops("send"))


def test_send_message_tool_reports_an_unconfirmed_attachment(ack_lost_relay, tmp_path):
    from hermes_agent.tools.send_message_tool import _send_live_adapter_media

    connector, adapter, loop = ack_lost_relay
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.4 q3")

    result = asyncio.run_coroutine_threadsafe(
        _send_live_adapter_media(adapter, CHAT, "Q3 report", [(str(pdf), False)]), loop
    ).result(timeout=20)

    assert len(connector.ops("send_media")) == 1        # it went out once
    assert result["success"] is True                     # not a failure the model would retry
    assert "do not resend" in result["delivery_unconfirmed"]
    assert _no_fallback_notice(connector)


def test_cron_marks_an_unconfirmed_attachment_unverified(ack_lost_relay, tmp_path):
    from hermes_agent.cron.jobs import create_job, get_job, mark_job_run
    from hermes_agent.cron.scheduler import _deliver_result

    connector, adapter, loop = ack_lost_relay
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.4 q3")
    job = create_job("Post the Q3 report", "every 1d", name="Q3 report", deliver="origin",
                     origin={"platform": "slack", "chat_id": CHAT})
    with patch("hermes_agent.gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("hermes_agent.cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("hermes_agent.cron.scheduler_delivery._maybe_mirror_cron_delivery", lambda *a, **k: None):
        error = _deliver_result(job, f"Q3 numbers are in.\n\nMEDIA:{pdf}",
                                adapters={Platform.RELAY: adapter}, loop=loop)
    mark_job_run(job["id"], True, delivery_error=error)
    record = get_job(job["id"])
    assert record is not None

    assert len(connector.ops("send_media")) == 1
    assert error is None                                  # possibly delivered: not a failed run
    assert record["last_delivery_unverified"], record     # ...but not a confirmed one either
    assert any(CHAT in target for target in record["last_delivery_unverified"])
    assert _no_fallback_notice(connector)
