"""Regression tests for three fixes to the Compresr integration.

Each test pins a specific bug so it cannot silently reappear:

* **F1** — the exact post-footer size gate used to ``unlink`` the content-addressed
  cache file when it failed. Because the cache key is a hash of the *content*, a
  concurrent (or prior) compression of identical content could already have handed
  that path to the model as a recovery pointer; deleting it dangled the pointer and
  broke the recover-verbatim guarantee. The gate must now fail open WITHOUT deleting.
* **F2** — ``_is_recovery_read`` matched a bare ``"cache/compresr"`` substring in any
  arg, so an ordinary grep/read that merely mentioned it (e.g. working on this repo)
  was misread as a recovery and skipped compression. It must now match the full
  ``cache/compresr/tool-output`` segment, while genuine recovery paths still skip.
* **F3** — plugin ``__init__`` coerced numeric config/env with bare ``int()``/
  ``float()``; a typo like ``COMPRESR_TIMEOUT=abc`` raised, which the loader swallowed
  into a silent whole-feature disable. Bad values must now fall back to defaults.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import plugins.tool_output_compresr as toc  # noqa: E402
from plugins.tool_output_compresr import ToolOutputCompressor, cache, compress  # noqa: E402
from plugins.tool_output_compresr.compress import compress_tool_output  # noqa: E402
from plugins.context_engine.compresr import CompresrContextEngine  # noqa: E402

# ~240 tokens — comfortably above the min_tokens gates used below.
ORIGINAL = "\n".join(f"line{i} payload" for i in range(400))
COMPRESSED = "line0 payload\n[many lines removed]"

# An agent-visible path long enough that the real footer (which embeds the path
# twice) costs more than the nominal FOOTER_TOKEN_BUDGET, so the exact gate can
# actually fire. store_original returns this; the host file that must survive is
# whatever cache_file_path resolves to — in reality these differ (agent-visible
# vs host path), which is exactly the long-remote-home case F1 rode on.
_LONG_VISIBLE_PATH = "/root/.hermes/cache/compresr/tool-output/" + ("a" * 560)


class _FakeClient:
    def __init__(self, out=COMPRESSED):
        self.out = out

    def compress(self, **kw):
        return self.out, {}


# --------------------------------------------------------------------------- #
# F1 — the gate-fail path must not delete a content-addressed cache file
# --------------------------------------------------------------------------- #
def test_f1_gate_fail_does_not_delete_cache_file(tmp_path, monkeypatch):
    real_file = tmp_path / "host-cache-file"
    real_file.write_text("VERBATIM ORIGINAL")
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: _LONG_VISIBLE_PATH)
    monkeypatch.setattr(cache, "cache_file_path", lambda cid: real_file)

    base = "x" * 4000  # ~1000 tokens
    body = "x" * 3400  # passes the cheap pre-filter, fails the exact gate via the long path
    out, info = compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="c",
        client=_FakeClient(out=body), task_id="t",
    )
    assert out == base                                   # failed open
    assert info["skipped_reason"] == "not smaller after footer"  # benign, not an error
    assert real_file.exists()                            # F1: NOT deleted
    assert real_file.read_text() == "VERBATIM ORIGINAL"  # verbatim original intact


def test_f1_second_call_does_not_dangle_first_calls_pointer(tmp_path, monkeypatch):
    """The original failure: two compressions of identical content share one
    content-addressed cache file. Call A wins and returns a footer pointing at it;
    call B over the same content fails the exact gate. B must not delete A's file."""
    real_file = tmp_path / "host-cache-file"

    def _store(cid, content, task_id="default", **_):
        real_file.write_text(content)   # host write, as the real store_original does
        return _LONG_VISIBLE_PATH        # agent-visible path handed to the model

    monkeypatch.setattr(cache, "store_original", _store)
    monkeypatch.setattr(cache, "cache_file_path", lambda cid: real_file)

    base = "x" * 4000
    # A: small body → net win → returns a footer that names the shared cache file.
    out_a, info_a = compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="shared",
        client=_FakeClient(out="y" * 2000), task_id="t",
    )
    assert info_a["shortened"] is True
    assert info_a["cache_path"] in out_a
    assert real_file.exists()

    # B: larger body over identical content → fails the exact gate.
    out_b, info_b = compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="shared",
        client=_FakeClient(out="z" * 3400), task_id="t",
    )
    assert out_b == base
    assert info_b["skipped_reason"] == "not smaller after footer"

    # F1: A's recovery pointer still resolves to the verbatim original.
    assert real_file.exists()
    assert real_file.read_text() == base


# --------------------------------------------------------------------------- #
# F2 — recovery guard matches the full segment, not a bare substring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("args", [
    {"pattern": "cache/compresr"},                       # grepping for the string
    {"query": "find TODOs under cache/compresr"},        # query mentioning it
    {"file_path": "/work/src/cache/compresr_notes.md"},  # a file whose path contains it
])
def test_f2_ordinary_call_mentioning_substring_is_still_compressed(tmp_path, monkeypatch, args):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: f"{tmp_path}/cache/compresr/tool-output/{cid}",
    )
    before = c.recoveries
    out = c.on_transform_tool_result(tool_name="search_files", args=args, result=ORIGINAL)
    assert out is not None and compress.FOOTER_MARKER in out  # compressed, not skipped
    assert c.recoveries == before                            # recoveries stat not inflated


def test_f2_genuine_recovery_path_still_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    called = {"n": 0}

    def _spy(**kw):
        called["n"] += 1
        return COMPRESSED, {}

    monkeypatch.setattr(c._client, "compress", _spy)
    before = c.recoveries
    # A container-translated recovery path carries the full cache/compresr/tool-output segment.
    out = c.on_transform_tool_result(
        tool_name="read_file",
        args={"file_path": "/root/.hermes/cache/compresr/tool-output/deadbeef"},
        result=ORIGINAL,
    )
    assert out is None                    # verbatim passthrough
    assert c.recoveries == before + 1     # correctly counted as a recovery
    assert called["n"] == 0               # never hit the API


# --------------------------------------------------------------------------- #
# F3 — malformed numeric config/env falls back to defaults instead of raising
# --------------------------------------------------------------------------- #
_TOOL_OUTPUT_DEFAULTS = {
    "COMPRESR_TOOL_OUTPUT_MIN_TOKENS": ("min_tokens", toc._DEFAULT_MIN_TOKENS),
    "COMPRESR_TOOL_OUTPUT_TIMEOUT": ("timeout", toc._DEFAULT_TIMEOUT),
    "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB": ("max_cache_mb", toc._DEFAULT_MAX_CACHE_MB),
    "COMPRESR_TOOL_OUTPUT_TARGET_RATIO": ("target_ratio", toc._DEFAULT_TARGET_RATIO),
}


@pytest.mark.parametrize("var", list(_TOOL_OUTPUT_DEFAULTS))
def test_f3_bad_tool_output_numeric_env_falls_back(monkeypatch, var):
    attr, default = _TOOL_OUTPUT_DEFAULTS[var]
    monkeypatch.setenv(var, "not-a-number")
    c = ToolOutputCompressor()  # must not raise
    assert getattr(c, attr) == default


def test_f3_bad_context_engine_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("COMPRESR_TIMEOUT", "abc")
    eng = CompresrContextEngine()  # must not raise
    assert eng.compresr_timeout == 60


def test_f3_bad_context_engine_ratio_override_becomes_none(monkeypatch):
    monkeypatch.setenv("COMPRESR_TARGET_RATIO", "abc")
    eng = CompresrContextEngine()  # must not raise
    assert eng.compresr_ratio_override is None


def test_f3_register_still_succeeds_with_bad_env(monkeypatch):
    """The end goal of F3: a typo'd tunable must not silently disable the feature.
    Both plugins must still register their hook / engine."""
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    monkeypatch.setenv("COMPRESR_TIMEOUT", "abc")
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_TIMEOUT", "abc")
    hooks, engines = [], []
    ctx = type("Ctx", (), {
        "register_hook": lambda self, name, cb: hooks.append(name),
        "register_context_engine": lambda self, e: engines.append(e),
    })()

    import plugins.context_engine.compresr as ce

    toc.register(ctx)                                 # must not raise
    ce.register(ctx)                                  # must not raise

    assert "transform_tool_result" in hooks
    assert len(engines) == 1


# F4 — CR/LF/NUL in COMPRESR_API_KEY: sanitize on read + swallow ValueError.
def test_f4_crlf_in_api_key_is_rejected_context_engine(monkeypatch, caplog):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_secret\r\ninjected")
    eng = CompresrContextEngine()
    assert eng.compresr_api_key == ""
    assert eng.is_available() is False
    assert "cmp_secret" not in caplog.text


def test_f4_crlf_in_api_key_is_rejected_tool_output(monkeypatch, caplog):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_secret\r\ninjected")
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_ENABLED", "1")
    comp = ToolOutputCompressor()
    assert comp.api_key == ""
    assert comp.active is False
    assert "cmp_secret" not in caplog.text


# F5 — metadata / private-net blocklist covers IPv6-mapped, decimal/hex shorthand,
# full 169.254/16, RFC1918, CGNAT, fd00::/8, trailing-dot FQDN. Log-safe URL.
@pytest.mark.parametrize("bad_url", [
    "https://169.254.169.254/",
    "https://169.254.170.2/",
    "https://169.254.169.253/",
    "https://[::ffff:169.254.169.254]/",
    "https://2852039166/",
    "https://0xa9fea9fe/",
    "https://10.0.0.1/",
    "https://192.168.1.1/",
    "https://100.64.0.1/",
    "https://[fd00:ec2::254]/",
    "https://metadata.google.internal/",
    "https://metadata.google.internal./",
])
def test_f5_context_engine_rejects_all_metadata_and_private_hosts(monkeypatch, bad_url):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    monkeypatch.setenv("COMPRESR_BASE_URL", bad_url)
    eng = CompresrContextEngine()
    assert eng.compresr_base_url == "https://api.compresr.ai/api", (
        f"base_url {bad_url!r} was not rejected — became {eng.compresr_base_url!r}"
    )


@pytest.mark.parametrize("bad_url", [
    "https://169.254.169.254/",
    "https://[::ffff:169.254.169.254]/",
    "https://2852039166/",
    "https://10.0.0.1/",
])
def test_f5_tool_output_plugin_rejects_all_metadata_and_private_hosts(monkeypatch, bad_url):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_ENABLED", "1")
    monkeypatch.setenv("COMPRESR_BASE_URL", bad_url)
    comp = ToolOutputCompressor()
    assert comp.base_url == "https://api.compresr.ai/api", (
        f"base_url {bad_url!r} was not rejected — became {comp.base_url!r}"
    )


def test_f5_localhost_and_valid_https_still_pass(monkeypatch):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    monkeypatch.setenv("COMPRESR_BASE_URL", "http://localhost:8000/api")
    eng = CompresrContextEngine()
    assert eng.compresr_base_url == "http://localhost:8000/api"

    monkeypatch.setenv("COMPRESR_BASE_URL", "https://api.compresr.ai/api")
    eng = CompresrContextEngine()
    assert eng.compresr_base_url == "https://api.compresr.ai/api"


def test_f5_rejected_url_log_never_contains_userinfo(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    monkeypatch.setenv(
        "COMPRESR_BASE_URL", "https://user:cmp_secret_in_url@169.254.169.254/"
    )
    eng = CompresrContextEngine()
    assert eng.compresr_base_url == "https://api.compresr.ai/api"
    assert "cmp_secret_in_url" not in caplog.text
    assert "user:cmp_secret_in_url" not in caplog.text


# F6 — register() refuses when is_available()==False.
def test_f6_register_refuses_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    engines = []
    ctx = type("Ctx", (), {
        "register_context_engine": lambda self, e: engines.append(e),
    })()
    import plugins.context_engine.compresr as ce
    ce.register(ctx)
    assert engines == []


def test_f6_register_succeeds_when_api_key_present(monkeypatch):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    engines = []
    ctx = type("Ctx", (), {
        "register_context_engine": lambda self, e: engines.append(e),
    })()
    import plugins.context_engine.compresr as ce
    ce.register(ctx)
    assert len(engines) == 1
    assert engines[0].is_available() is True


def test_f6_loader_does_not_resurrect_declined_engine(monkeypatch):
    """The real load path must honor register()'s decline, not fall through to
    the subclass scan (which would instantiate CompresrContextEngine anyway)."""
    from plugins.context_engine import load_context_engine

    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    assert load_context_engine("compresr") is None


def test_f6_loader_returns_engine_when_available(monkeypatch):
    from plugins.context_engine import load_context_engine

    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    engine = load_context_engine("compresr")
    assert engine is not None
    assert engine.is_available() is True


# F7 — redact tool args (query) + outbound + cached content.
def test_f7_tool_output_redacts_query_and_content(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_ENABLED", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_MIN_TOKENS", "1")

    comp = ToolOutputCompressor()
    seen = {}

    def fake_compress(query, content, tool_name, cache_id, client, task_id,
                      max_cache_mb, target_ratio, cache_content=None):
        seen["query"] = query
        seen["content"] = content
        seen["cache_content"] = cache_content
        return "compressed body\n\n[compresr:recover] ...", {
            "called_api": True, "shortened": True, "cache_path": "/tmp/x",
            "base_tokens": 1000, "out_tokens": 100, "saved": 900,
        }

    monkeypatch.setattr(
        "plugins.tool_output_compresr.compress_tool_output", fake_compress
    )

    secret_args = {"command": "curl -H 'Authorization: Bearer sk-live-SECRETVALUE' https://api.example.com"}
    # Short lines only, to avoid _has_unrecoverable_long_line's fail-open gate.
    secret_output = (
        "AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE\n"
        + "\n".join(f"line {i} of output body xxxxxxxxxxxx" for i in range(500))
    )

    comp.on_transform_tool_result(
        tool_name="terminal", args=secret_args, result=secret_output, status="ok",
    )

    from agent.redact import redact_sensitive_text
    assert seen["query"] == redact_sensitive_text(f"terminal: {secret_args['command']}"[:600])
    assert seen["content"] == redact_sensitive_text(secret_output)


# F8 — success clears prior failure back-off state.
class _FakeSessionDB:
    """Minimal session-DB double exposing the cooldown persist/clear hooks."""

    def __init__(self):
        self.rows = {}

    def record_compression_failure_cooldown(self, session_id, cooldown_until, error):
        self.rows[session_id] = (cooldown_until, error)

    def clear_compression_failure_cooldown(self, session_id):
        self.rows.pop(session_id, None)


def test_f8_success_clears_prior_failure_cooldown(monkeypatch):
    monkeypatch.setenv("COMPRESR_API_KEY", "cmp_test_key")
    eng = CompresrContextEngine()

    # Bind a session DB and record a REAL prior failure so the persisted row
    # exists (in-memory-only assertions were the false positive this exercises).
    db = _FakeSessionDB()
    eng._session_db = db
    eng._session_id = "sess-1"
    eng._record_compression_failure_cooldown(3600.0, "compresr: prior transient error")
    assert "sess-1" in db.rows
    # Simulate the in-memory cooldown having elapsed (or a fresh process where
    # only the persisted DB row was reloaded) so _generate_summary may run.
    eng._summary_failure_cooldown_until = 1e-9

    monkeypatch.setattr(
        eng, "_call_compresr",
        lambda ctx, q: ("compressed body", {"original_tokens": 1000, "tokens_saved": 800}),
    )
    monkeypatch.setattr(eng, "_serialize_for_summary", lambda turns: "some turns")
    out = eng._generate_summary([{"role": "user", "content": "x"}], focus_topic="topic")
    assert out is not None
    assert eng._summary_failure_cooldown_until == 0.0
    assert eng._last_summary_error is None
    # The persisted session-DB row must be cleared too, not just the in-memory
    # fields — otherwise the cooldown reloads on resume and suppresses the engine.
    assert "sess-1" not in db.rows


# F9 — atomic cache write via O_NOFOLLOW + os.replace; concurrent-write safe.
def test_f9_store_original_uses_atomic_replace(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: None)

    replaced = []
    real_replace = cache.os.replace

    def _spy_replace(src, dst):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", _spy_replace)
    content = "atomic body\n" * 200
    cache.store_original("atomic1", content, task_id="t", max_cache_mb=0)

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert ".atomic1." in os.path.basename(src) and src.endswith(".tmp")
    assert dst.endswith("/atomic1")

    final_path = tmp_path / "cache" / "compresr" / "tool-output" / "atomic1"
    assert final_path.read_text() == content


def test_f9_concurrent_writes_never_produce_partial_content(monkeypatch, tmp_path):
    import threading
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: None)

    content_a = "AAAA" * 5000
    content_b = "BBBB" * 5000

    def writer(cid, content, n):
        for _ in range(n):
            cache.store_original(cid, content, task_id="t", max_cache_mb=0)

    ta = threading.Thread(target=writer, args=("cw", content_a, 30))
    tb = threading.Thread(target=writer, args=("cw", content_b, 30))
    ta.start(); tb.start(); ta.join(); tb.join()

    final = (tmp_path / "cache" / "compresr" / "tool-output" / "cw").read_text()
    assert final == content_a or final == content_b
    assert len(final) == len(content_a)
