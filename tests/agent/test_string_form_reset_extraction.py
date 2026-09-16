"""Rate-limit reset extraction from string-form provider errors.

Cluster: ``extract_api_error_context`` falls back to ``str(error)[:500]`` when the SDK
exception carries no structured ``body``. Providers (and wrappers around the OpenAI SDK)
embed the full error payload in that string — ``Error code: 429 - {'error': {...}}`` —
and the extractor must parse it out, otherwise the credential pool persists
``last_error_reset_at=None`` for a quota wall that explicitly reports ``resets_at``.
The reset-aware restore gate then never engages and every turn re-fails the primary
against a proven-empty quota window.
"""

import json
import time


class TestEmbeddedPayloadExtraction:
    def test_string_form_429_extracts_reset_at(self):
        """A 429 whose payload rides in str(error) (body=None) still yields reset_at;
        Python-repr (single quotes) and JSON (double quotes) forms parse identically."""
        from agent.agent_runtime_helpers import extract_api_error_context
        from agent.credential_pool import _normalize_error_context

        reset_at = int(time.time()) + 212_366
        payload = {
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "plan_type": "pro",
                "resets_at": reset_at,
                "resets_in_seconds": 212_366,
            }
        }
        for rendered in (repr(payload), json.dumps(payload)):
            err = Exception(f"Error code: 429 - {rendered}")
            ctx = extract_api_error_context(err)
            assert ctx.get("reason") == "usage_limit_reached", ctx
            assert ctx.get("message") == "The usage limit has been reached", ctx
            assert ctx.get("reset_at") == reset_at, ctx
            assert _normalize_error_context(ctx).get("reset_at") == float(reset_at)

    def test_string_form_resets_in_seconds_becomes_absolute(self):
        """``resets_in_seconds`` (relative) in the embedded payload converts to an
        absolute reset_at like the structured-body path already does."""
        from agent.agent_runtime_helpers import extract_api_error_context

        payload = {"error": {"type": "usage_limit_reached", "resets_in_seconds": 90}}
        err = Exception(f"Error code: 429 - {payload!r}")
        ctx = extract_api_error_context(err)
        assert "reset_at" in ctx
        assert 85 <= ctx["reset_at"] - time.time() <= 95

    def test_embedded_parse_does_not_shadow_structured_body(self):
        """A real SDK error with a structured body still wins; the string fallback
        only runs when the body produced nothing."""
        import httpx
        from openai import RateLimitError

        body = {"error": {"type": "usage_limit_reached", "message": "gone",
                          "resets_at": 1_789_806_328}}
        resp = httpx.Response(429, headers={"content-type": "application/json"},
                              content=json.dumps(body).encode(),
                              request=httpx.Request("POST", "https://x/responses"))
        err = RateLimitError(f"Error code: 429 - {body!r}", response=resp, body=body)
        from agent.agent_runtime_helpers import extract_api_error_context
        ctx = extract_api_error_context(err)
        assert ctx["message"] == "gone"
        assert ctx["reset_at"] == 1_789_806_328

    def test_non_error_trailing_text_is_ignored(self):
        """A string that merely contains braces but no trailing error payload must not
        be force-parsed (parse failure is swallowed; behavior unchanged)."""
        from agent.agent_runtime_helpers import extract_api_error_context

        err = Exception("Error code: 429 - usage limit, see docs {} for details")
        ctx = extract_api_error_context(err)
        assert "reset_at" not in ctx
