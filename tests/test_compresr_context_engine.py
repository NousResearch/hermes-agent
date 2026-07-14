"""Contract test for the context_engine/compresr compaction engine.

The engine couples to the parent ``ContextCompressor``'s PRIVATE seam
(``_generate_summary`` + summary-prefix helpers). This test pins that contract
so an upstream refactor fails loudly in CI instead of silently degrading the
plugin to a no-op:

  * the override's signature matches the parent's,
  * a successful API call returns the prefixed compressed body,
  * an API failure returns None and trips the error counter + cooldown,
  * the external engine defaults to aborting failed compactions so context is
    preserved instead of replaced by a deterministic placeholder.
"""

import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.context_compressor import ContextCompressor  # noqa: E402
from plugins.context_engine.compresr import CompresrContextEngine  # noqa: E402

TURNS = [
    {"role": "user", "content": "Refactor the auth module and fix the login bug."},
    {"role": "assistant", "content": "Edited auth.py; tests pass. Token TTL set to 3600."},
]


def _engine():
    e = CompresrContextEngine()
    e.compresr_api_key = "cmp_test"  # avoid the unset-key warning path
    return e


def test_override_signature_matches_parent():
    parent = inspect.signature(ContextCompressor._generate_summary).parameters
    child = inspect.signature(CompresrContextEngine._generate_summary).parameters
    assert list(parent) == list(child), (
        "parent _generate_summary signature changed — update the override"
    )


def test_update_model_signature_matches_parent():
    """The update_model override must stay signature-compatible with the parent
    so a caller passing e.g. max_tokens never hits a TypeError or a silently
    dropped argument. Pins the parameter list against the authoritative parent."""
    parent = inspect.signature(ContextCompressor.update_model).parameters
    child = inspect.signature(CompresrContextEngine.update_model).parameters
    assert list(parent) == list(child), (
        "parent update_model signature changed — update the override"
    )


def test_update_model_forwards_max_tokens_to_parent():
    """max_tokens must be forwarded, not dropped."""
    e = _engine()
    seen = {}

    def _spy(self, model, context_length, base_url="", api_key="", provider="",
             api_mode="", max_tokens=None):
        seen["max_tokens"] = max_tokens

    orig = ContextCompressor.update_model
    try:
        ContextCompressor.update_model = _spy  # type: ignore[method-assign]
        e.update_model("m", 128000, max_tokens=4096)
    finally:
        ContextCompressor.update_model = orig  # type: ignore[method-assign]
    assert seen["max_tokens"] == 4096


def test_call_compresr_builds_request_and_parses_compressed_context(monkeypatch):
    """Pin the HTTP contract of the question-specific client: payload shape,
    X-API-Key header, and the response envelope key ``compressed_context``
    (which DIFFERS from the tool-output endpoint's ``compressed_output`` — a
    real drift footgun if the two are ever conflated)."""
    import io
    import urllib.error
    import urllib.request

    captured = {}

    class _Resp:
        def __init__(self, body):
            self._b = body.encode("utf-8")

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _Resp(json.dumps({"success": True, "data": {
            "compressed_context": "KEPT summary", "tokens_saved": 12,
        }}))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    e = _engine()
    e.compresr_model = "latte_v2"
    text, stats = e._call_compresr("some context", "the query")

    assert text == "KEPT summary"
    assert stats["tokens_saved"] == 12
    req = captured["req"]
    assert req.full_url.endswith("/compress/question-specific/")
    assert req.get_header("X-api-key") == "cmp_test"
    payload = json.loads(req.data)
    assert payload["context"] == "some context"
    assert payload["query"] == "the query"
    assert payload["compression_model_name"] == "latte_v2"
    assert "target_compression_ratio" in payload
    assert payload["source"] == "integration:hermes"

    # success:false → RuntimeError so the caller can fall back.
    def _fail(req, timeout=None):
        return _Resp(json.dumps({"success": False, "message": "nope"}))

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        e._call_compresr("c", "q")

    # HTTPError → RuntimeError carrying the status + server detail.
    def _http_err(req, timeout=None):
        raise urllib.error.HTTPError(
            "u", 422, "Unprocessable", None, io.BytesIO(b'{"detail":"bad"}')
        )

    monkeypatch.setattr(urllib.request, "urlopen", _http_err)
    with _pytest.raises(RuntimeError) as ei:
        e._call_compresr("c", "q")
    assert "422" in str(ei.value)


def test_success_returns_prefixed_body():
    e = _engine()
    e._call_compresr = lambda context, query: ("KEPT: login bug, TTL 3600", {"tokens_saved": 42})
    out = e._generate_summary(TURNS, focus_topic="login bug")
    assert out is not None
    # Carries the standard summary prefix so iterative re-compaction recognizes it.
    assert out == e._with_summary_prefix(e._strip_summary_prefix(out))
    assert "login bug" in out
    assert e.compresr_calls == 1
    assert e._previous_summary  # continuity state updated


def test_failure_falls_back_to_none():
    e = _engine()

    def _boom(context, query):
        raise RuntimeError("HTTP 500")

    e._call_compresr = _boom
    out = e._generate_summary(TURNS, focus_topic="anything")
    assert out is None  # → inherited compress() uses its deterministic handoff
    assert e.compresr_errors == 1
    assert e._summary_failure_cooldown_until > 0  # cooldown tripped
    assert e.abort_on_summary_failure is True


def test_empty_compression_is_treated_as_failure():
    e = _engine()
    e._call_compresr = lambda context, query: ("", {})
    assert e._generate_summary(TURNS, focus_topic="x") is None
    assert e.compresr_errors == 1


def test_failure_cooldown_persists_to_session_db():
    """Fix #4: the failure cooldown must go through the parent's
    _record_compression_failure_cooldown so it persists to the session DB and
    survives across processes — not just an in-memory field assignment."""
    e = _engine()

    class _FakeDB:
        def __init__(self):
            self.calls = []

        def record_compression_failure_cooldown(self, session_id, cooldown_until, error):
            self.calls.append((session_id, cooldown_until, error))

    db = _FakeDB()
    e._session_db = db
    e._session_id = "sess-1"

    def _boom(context, query):
        raise RuntimeError("HTTP 500")

    e._call_compresr = _boom
    assert e._generate_summary(TURNS, focus_topic="x") is None
    assert e._summary_failure_cooldown_until > 0        # in-memory still set
    assert len(db.calls) == 1                            # AND persisted
    assert db.calls[0][0] == "sess-1"
    assert "HTTP 500" in (db.calls[0][2] or "")


def test_ratio_mapping_keep_to_nx():
    e = _engine()
    e.summary_target_ratio = 0.2  # keep ~20%
    assert e._target_compression_ratio() == 5.0  # → 5x
    e.compresr_ratio_override = 0.8
    assert e._target_compression_ratio() == 0.8  # explicit override wins


def test_api_key_is_env_only(monkeypatch, tmp_path):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compresr:\n  api_key: cmp_from_config\n  model: latte_v1\n",
        encoding="utf-8",
    )

    e = CompresrContextEngine()
    assert e.compresr_api_key == ""
    assert e.compresr_model == "latte_v1"
    assert not e.is_available()


def _parent_reference(context_length):
    """Build a bare parent ContextCompressor and drive its update_model so we
    can compare the child's derived budgets against the authoritative parent."""
    parent = ContextCompressor(model="parent-placeholder", config_context_length=200_000)
    parent.update_model("m", context_length)
    return parent


def test_update_model_threshold_matches_parent_and_can_fire_small_ctx():
    """update_model must not re-floor threshold_tokens above the window.

    For ctx <= 64K the parent's small-context carve-out keeps the trigger below
    the window; re-flooring to MINIMUM_CONTEXT_LENGTH regressed #14690 (threshold
    >= window → compaction never fires → provider 400)."""
    e = _engine()
    e.update_model("m", 64000)
    parent = _parent_reference(64000)
    assert e.threshold_tokens == parent.threshold_tokens
    # The whole point: compaction can still fire before the window fills.
    assert e.threshold_tokens < 64000
    # Derived budgets stay in parity with the parent too.
    assert e.tail_token_budget == parent.tail_token_budget
    assert e.max_summary_tokens == parent.max_summary_tokens


def test_update_model_parity_for_large_contexts():
    for ctx in (100000, 128000):
        e = _engine()
        e.update_model("m", ctx)
        parent = _parent_reference(ctx)
        assert e.threshold_tokens == parent.threshold_tokens
        assert e.threshold_tokens < ctx
        assert e.tail_token_budget == parent.tail_token_budget
        assert e.max_summary_tokens == parent.max_summary_tokens


def test_secure_base_url_rejects_cloud_metadata_ip():
    # Regression: a misconfigured base_url pointing at IMDS would leak the
    # API key to the metadata endpoint. Reject cloud-metadata hosts.
    from plugins.context_engine.compresr import _secure_base_url

    default = "https://api.compresr.ai"
    assert _secure_base_url("https://169.254.169.254/latest/", default) == default
    assert _secure_base_url("https://metadata.google.internal/", default) == default
    assert _secure_base_url("https://100.100.100.200/", default) == default


def test_secure_base_url_allows_valid_https():
    from plugins.context_engine.compresr import _secure_base_url

    default = "https://api.compresr.ai"
    assert _secure_base_url("https://compresr.internal", default) == "https://compresr.internal"
    assert _secure_base_url("http://localhost:8000", default) == "http://localhost:8000"


def test_url_error_routed_through_runtime_error(monkeypatch):
    # Regression: urllib.error.URLError (DNS failure, connect refused,
    # socket.timeout) is NOT a subclass of HTTPError; it slipped past the
    # except HTTPError and surfaced as an opaque exception. Must be caught.
    import urllib.error
    import urllib.request as ureq

    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test")
    e = CompresrContextEngine()

    def _boom(*_a, **_kw):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(ureq, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="connection error"):
        e._call_compresr("q", "ctx")


def test_compresr_config_metadata_registered():
    from hermes_cli.config import DEFAULT_CONFIG, OPTIONAL_ENV_VARS, validate_config_structure

    compresr = DEFAULT_CONFIG["compresr"]
    assert compresr["tool_output_enabled"] is True
    assert compresr["tool_output_max_cache_mb"] == 256

    env_info = OPTIONAL_ENV_VARS["COMPRESR_API_KEY"]
    assert env_info["password"] is True
    assert env_info["tools"] == ["context_engine", "tool_output_compresr"]
    assert validate_config_structure({"compresr": {"tool_output_enabled": True}}) == []


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
