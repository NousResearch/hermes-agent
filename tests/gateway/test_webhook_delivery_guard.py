"""The webhook delivery guard: an error diagnostic is not an answer.

When a turn ends without producing an answer, ``agent/conversation_loop.py``
puts a short diagnostic in ``final_response`` ("Response truncated due to
output length limit", "Context length exceeded (174,833 tokens). Cannot
compress further.", ...). Interactive surfaces render that as an error. A
webhook route hands whatever it receives to its delivery target, so before
this guard the recipient got the diagnostic itself — a one-line string that
reads like a terse reply — with nothing in the logs saying the request had
produced no output.

`gateway/run.py:_sanitize_gateway_final_response` already does this for the
chat gateways and deliberately exempts `webhook` as a programmatic surface.
That holds for the route's own HTTP response and for `deliver: log`; it does
not hold for a route configured to deliver to a person, which is what the
guard here covers.

Covers:
- every guarded diagnostic is caught, fixed sentences and templates alike
- the delivered text says no answer was produced, and keeps the reason
- a long provider message is capped in the notice but not in the log
- a WARNING naming the session and route is logged
- the route is named even when the delivery entry predates the `route` key
- `deliver: log` keeps the raw diagnostic (programmatic surface, unguarded)
- ordinary responses pass through untouched
- the inventory has not drifted from conversation_loop.py, in EITHER
  direction: a rewording and a newly added diagnostic both fail the test
"""

import ast
import asyncio
import logging
import pathlib
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import (
    WebhookAdapter,
    _GUARD_REASON_MAX_CHARS,
    _TERMINAL_ERROR_PLACEHOLDERS,
    _is_terminal_error_placeholder,
    _route_from_chat_id,
)

CHAT_ID = "webhook:daily-report:1785970121225"

# A plausible webhook report: an actual answer, which must pass untouched.
ORDINARY_RESPONSE = (
    "Overnight batch finished at 04:12 UTC.\n"
    "- 1,204 rows ingested, 3 rejected (schema mismatch on `order_ref`)\n"
    "- Retry queue empty\n"
    "No action needed."
)

# Every string literal / f-string template `conversation_loop.py` can put in
# `final_response`, mapped to a rendering of it — or to None when it is
# deliberately not guarded, with the reason. `test_inventory_matches_source`
# below extracts the same set from the source and compares, so this dict is
# the classification of the loop's diagnostics, not a copy that can rot.
KNOWN_FINAL_RESPONSES = {
    # -- guarded: the turn ended with no answer -----------------------------
    "Response truncated due to output length limit":
        "Response truncated due to output length limit",
    "First response truncated due to output length limit":
        "First response truncated due to output length limit",
    "Stream repeatedly dropped mid tool-call (network); the tool was not executed":
        "Stream repeatedly dropped mid tool-call (network); the tool was not executed",
    "Incomplete REASONING_SCRATCHPAD after 2 retries":
        "Incomplete REASONING_SCRATCHPAD after 2 retries",
    "Codex response remained incomplete after 3 continuation attempts":
        "Codex response remained incomplete after 3 continuation attempts",
    "Request payload too large (413). Cannot compress further.":
        "Request payload too large (413). Cannot compress further.",
    "Context overflow and auto-compaction is disabled (compression.enabled: "
    "false). Run /compress to compact manually, /new to start fresh, or "
    "switch to a larger-context model.":
        "Context overflow and auto-compaction is disabled (compression.enabled: "
        "false). Run /compress to compact manually, /new to start fresh, or "
        "switch to a larger-context model.",
    "max_tokens exceeds the provider's output cap for this model. Lower "
    "model.max_tokens in config.yaml.":
        "max_tokens exceeds the provider's output cap for this model. Lower "
        "model.max_tokens in config.yaml.",
    "(empty)": "(empty)",
    "Invalid API response after {} retries: {}":
        "Invalid API response after 3 retries: no choices in response",
    "API call failed after {} retries: {}":
        "API call failed after 3 retries: Connection reset by peer",
    "Model generated invalid tool call: {}":
        'Model generated invalid tool call: {"name": "read_file", "argum',
    "Request payload too large: max compression attempts ({}) reached.":
        "Request payload too large: max compression attempts (5) reached.",
    "Context length exceeded: max compression attempts ({}) reached.":
        "Context length exceeded: max compression attempts (5) reached.",
    "Context length exceeded ({} tokens). Cannot compress further.":
        "Context length exceeded (174,833 tokens). Cannot compress further.",
    "⏳ {}\n\nNo fallback provider available. Try again after the reset, or "
    "add a fallback provider in config.yaml.":
        "⏳ Rate limit reached, resets at 04:00 UTC\n\nNo fallback provider "
        "available. Try again after the reset, or add a fallback provider in "
        "config.yaml.",
    "I apologize, but I encountered an error while processing the model "
    "response: {}":
        "I apologize, but I encountered an error while processing the model "
        "response: Expecting value: line 1 column 1",
    "I apologize, but I encountered repeated errors: {}":
        "I apologize, but I encountered repeated errors: 502 Bad Gateway",
    # -- not guarded --------------------------------------------------------
    # Nothing is delivered for an empty response, so there is nothing to
    # mistake for an answer.
    "": None,
    # The interrupt sentinel. Its fixed part lives in
    # `conversation_loop.INTERRUPT_WAITING_FOR_MODEL_PREFIX`, so only the tail
    # appears here; `_sanitize_gateway_final_response` already suppresses this
    # sentinel on chat surfaces (#7921) and the interruption is operator-
    # initiated, so it is not a silent failure.
    "{}{}s elapsed).": None,
}


def _final_response_templates_in_source() -> set:
    """Every literal `final_response` value in conversation_loop.py.

    Parsed rather than grepped so a diagnostic wrapped across source lines is
    seen as the one string it is. f-strings come back as templates with `{}`
    for each interpolation; values that are not literals (a variable, a call)
    are not classifiable from the source and are skipped.
    """
    source = (
        pathlib.Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
    ).read_text(encoding="utf-8")

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.JoinedStr):
            out = ""
            for part in node.values:
                out += (
                    part.value
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                    else "{}"
                )
            return [out]
        if isinstance(node, ast.IfExp):  # a ternary picks between two of them
            return literal(node.body) + literal(node.orelse)
        return []

    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id in ("final_response", "_final_response")
                for t in node.targets
            ):
                found.update(literal(node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "final_response":
                    found.update(literal(value))
    return found


def _make_adapter() -> WebhookAdapter:
    config = PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": {}}
    )
    return WebhookAdapter(config)


def _wire_target(adapter: WebhookAdapter):
    target = AsyncMock()
    target.send = AsyncMock(return_value=SendResult(success=True))
    runner = MagicMock()
    runner.adapters = {Platform("telegram"): target}
    runner.config.get_home_channel.return_value = None
    adapter.gateway_runner = runner
    return target


def _send(adapter: WebhookAdapter, content: str) -> SendResult:
    return asyncio.run(adapter.send(CHAT_ID, content))


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

def test_every_guarded_diagnostic_is_classified_as_terminal():
    guarded = [v for v in KNOWN_FINAL_RESPONSES.values() if v is not None]
    assert len(guarded) > len(_TERMINAL_ERROR_PLACEHOLDERS), (
        "the templated diagnostics are guarded by pattern, not by exact string"
    )
    for rendered in guarded:
        assert _is_terminal_error_placeholder(rendered), rendered
        # Surrounding whitespace must not smuggle one past the guard.
        assert _is_terminal_error_placeholder(f"\n  {rendered}  \n")


def test_ordinary_response_is_not_classified_as_terminal():
    assert not _is_terminal_error_placeholder(ORDINARY_RESPONSE)
    assert not _is_terminal_error_placeholder("")
    assert not _is_terminal_error_placeholder(None)
    # A response that merely *quotes* a diagnostic is a real answer.
    assert not _is_terminal_error_placeholder(
        "The run failed: Response truncated due to output length limit. Retrying."
    )
    # ... including one that quotes a templated diagnostic mid-sentence.
    assert not _is_terminal_error_placeholder(
        "Two runs died with 'Context length exceeded (174,833 tokens). Cannot "
        "compress further.' — raising the compression tier fixed it."
    )


def test_inventory_matches_source_in_both_directions():
    """The guard's inventory is exactly what conversation_loop.py can emit.

    Set membership alone only proves the guarded strings still exist upstream;
    it cannot see a *new* diagnostic, which would silently fall outside the
    guard while every test stayed green. Comparing both ways turns that into
    a failure that names the string to classify.
    """
    in_source = _final_response_templates_in_source()
    classified = set(KNOWN_FINAL_RESPONSES)
    assert not in_source - classified, (
        "conversation_loop.py emits final_response values this guard has never "
        f"been told about: {sorted(in_source - classified)!r} — add each one to "
        "KNOWN_FINAL_RESPONSES with a rendering (guarded) or None (with the "
        "reason it is not)"
    )
    assert not classified - in_source, (
        "no longer emitted by conversation_loop.py, so the guard is matching "
        f"strings that cannot occur: {sorted(classified - in_source)!r}"
    )


def test_route_is_recovered_from_the_session_id():
    assert _route_from_chat_id(CHAT_ID) == "daily-report"
    # Route names come from a URL path segment, so a colon is possible.
    assert _route_from_chat_id("webhook:team:daily:1785970121225") == "team:daily"
    assert _route_from_chat_id("telegram:-1001234567890") == ""
    assert _route_from_chat_id("") == ""


# ---------------------------------------------------------------------------
# The delivered message
# ---------------------------------------------------------------------------

def test_placeholder_is_not_delivered_as_the_answer():
    adapter = _make_adapter()
    target = _wire_target(adapter)
    adapter._delivery_info[CHAT_ID] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-1001234567890"},
        "route": "daily-report",
    }

    placeholder = "Response truncated due to output length limit"
    result = _send(adapter, placeholder)

    assert result.success
    delivered = target.send.await_args.args[1]
    assert delivered != placeholder
    assert "No answer was produced" in delivered
    # The reason still travels — the guard explains, it does not hide.
    assert placeholder in delivered
    assert "daily-report" in delivered


def test_templated_diagnostic_is_guarded_too():
    """The set of fixed sentences is not the whole surface (six of eighteen)."""
    adapter = _make_adapter()
    target = _wire_target(adapter)
    adapter._delivery_info[CHAT_ID] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-1001234567890"},
        "route": "daily-report",
    }

    assert _send(
        adapter, "Context length exceeded (174,833 tokens). Cannot compress further."
    ).success
    delivered = target.send.await_args.args[1]
    assert "No answer was produced" in delivered
    assert "174,833 tokens" in delivered


def test_long_provider_message_is_capped_in_the_notice_but_not_in_the_log(caplog):
    adapter = _make_adapter()
    target = _wire_target(adapter)
    adapter._delivery_info[CHAT_ID] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-1001234567890"},
        "route": "daily-report",
    }

    tail = "x" * 900
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.webhook"):
        assert _send(adapter, f"API call failed after 3 retries: {tail}").success

    delivered = target.send.await_args.args[1]
    assert "…" in delivered
    assert tail not in delivered
    assert len(delivered) < _GUARD_REASON_MAX_CHARS + 300
    # The operator still gets the whole thing where it belongs.
    assert tail in " ".join(r.getMessage() for r in caplog.records)


def test_ordinary_response_is_delivered_untouched():
    adapter = _make_adapter()
    target = _wire_target(adapter)
    adapter._delivery_info[CHAT_ID] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-1001234567890"},
        "route": "daily-report",
    }

    assert _send(adapter, ORDINARY_RESPONSE).success
    assert target.send.await_args.args[1] == ORDINARY_RESPONSE


def test_guard_logs_a_warning_naming_the_session_and_route(caplog):
    adapter = _make_adapter()
    _wire_target(adapter)
    adapter._delivery_info[CHAT_ID] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-1001234567890"},
        "route": "daily-report",
    }

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.webhook"):
        _send(adapter, "Request payload too large (413). Cannot compress further.")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "the guard must leave a WARNING behind"
    line = warnings[-1].getMessage()
    assert "delivery-guard" in line
    assert CHAT_ID in line
    assert "daily-report" in line


def test_route_is_named_even_without_the_delivery_key():
    """A delivery entry that predates the `route` key still names its route."""
    adapter = _make_adapter()
    target = _wire_target(adapter)
    adapter._delivery_info[CHAT_ID] = {
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "-1001234567890"},
    }

    assert _send(adapter, "Incomplete REASONING_SCRATCHPAD after 2 retries").success
    delivered = target.send.await_args.args[1]
    assert "No answer was produced" in delivered
    assert "daily-report" in delivered
    assert "?" not in delivered


def test_log_delivery_keeps_the_raw_diagnostic(caplog):
    """`deliver: log` is a programmatic surface: no notice, no rewrite."""
    adapter = _make_adapter()
    adapter._delivery_info[CHAT_ID] = {"deliver": "log", "route": "daily-report"}

    placeholder = "Response truncated due to output length limit"
    with caplog.at_level(logging.INFO, logger="gateway.platforms.webhook"):
        assert _send(adapter, placeholder).success

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert placeholder in logged
    assert "delivery-guard" not in logged
