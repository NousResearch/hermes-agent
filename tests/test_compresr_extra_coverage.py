"""Extra unit coverage for the Compresr integration (audit gap-fill).

These tests target branches the existing suite leaves uncovered:

  * JSON-unwrap edge cases (`_try_unwrap_json_tool_result`) — non-brace prefix,
    malformed JSON, non-dict top level, missing/blank inner key.
  * De-numbering helpers (`_strip_line_gutter`, `_is_fully_guttered`) edge cases.
  * The recovery-read guard (`_is_recovery_read`) — non-dict args, resolved
    host-path match, exception fallback.
  * Config/env precedence in `_opt` (env wins over config.yaml block).
  * Query derivation (`_derive_query`) path/tool-name/fallback branches.
  * Hook gating: error-status results, monotonic cooldown, unwrapped-inner too
    small, and the cooldown that trips on an API error (`info["error"]`).
  * compress.py exact-gate cleanup and target_ratio plumbing.
  * cache.py: LocalEnvironment host-path + no-op sync, no-active-env docker path,
    write failure, and prune stat/total edge cases.
  * context_engine: cooldown skip, empty-context skip, prior-summary fold,
    ratio clamping bounds, latte_v1 coarse payload, disable_placeholders,
    missing-key raise, and get_status extension.

The Compresr HTTP client is always mocked — no network calls.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import plugins.tool_output_compresr as toc  # noqa: E402
from plugins.tool_output_compresr import ToolOutputCompressor, cache  # noqa: E402
from plugins.tool_output_compresr.compress import compress_tool_output  # noqa: E402
from plugins.context_engine.compresr import CompresrContextEngine  # noqa: E402

# Reuse the big fixture shape from the primary test module.
ORIGINAL = "\n".join(
    f"line{i} {w}"
    for i, w in enumerate(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"] * 40)
)
COMPRESSED = "line0 alpha\nline1 bravo\n[4 lines removed]"


class _FakeClient:
    def __init__(self, out=COMPRESSED, stats=None, boom=False):
        self.out, self.stats, self.boom = out, stats or {}, boom
        self.seen = {}

    def compress(self, **kw):
        self.seen = kw
        if self.boom:
            raise RuntimeError("HTTP 500")
        return self.out, self.stats


def _fake_env(name, **attrs):
    env = type(name, (), {})()
    for k, v in attrs.items():
        setattr(env, k, v)
    return env


def _active(min_tokens=5):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", min_tokens
    return c


# --------------------------------------------------------------------------- #
# _try_unwrap_json_tool_result — every early-return branch
# --------------------------------------------------------------------------- #
def test_unwrap_unknown_tool_returns_none():
    assert toc._try_unwrap_json_tool_result("web_search", "{...}") == (None, None)


def test_unwrap_non_brace_prefix_returns_none():
    # read_file IS unwrappable, but a non-JSON payload must not be parsed.
    assert toc._try_unwrap_json_tool_result("read_file", "just plain text") == (None, None)


def test_unwrap_malformed_json_returns_none():
    assert toc._try_unwrap_json_tool_result("read_file", '{"content": ') == (None, None)


def test_unwrap_non_dict_top_level_returns_none():
    assert toc._try_unwrap_json_tool_result("read_file", "[1,2,3]") == (None, None)


def test_unwrap_missing_or_blank_inner_returns_none():
    assert toc._try_unwrap_json_tool_result("read_file", '{"path": "/x"}') == (None, None)
    assert toc._try_unwrap_json_tool_result("read_file", '{"content": "   "}') == (None, None)
    assert toc._try_unwrap_json_tool_result("read_file", '{"content": 5}') == (None, None)


def test_unwrap_success_and_splice_roundtrip():
    inner, splice = toc._try_unwrap_json_tool_result(
        "terminal", json.dumps({"output": "hello", "rc": 0})
    )
    assert inner == "hello"
    spliced = json.loads(splice("BYE"))
    assert spliced == {"output": "BYE", "rc": 0}


# --------------------------------------------------------------------------- #
# de-numbering helpers — extra edges
# --------------------------------------------------------------------------- #
def test_is_fully_guttered_empty_and_blank_lines():
    assert toc._is_fully_guttered("") is False          # no non-blank lines
    assert toc._is_fully_guttered("   \n\t") is False
    # blank interior lines are ignored, gutters on the rest still qualify
    assert toc._is_fully_guttered("1|a\n\n2|b") is True
    # one un-guttered non-blank line disqualifies the whole payload
    assert toc._is_fully_guttered("1|a\nplain\n2|b") is False


def test_strip_line_gutter_multidigit_and_no_gutter():
    assert toc._strip_line_gutter("123|x\n4567|y") == "x\ny"
    assert toc._strip_line_gutter("no gutter here") == "no gutter here"


# --------------------------------------------------------------------------- #
# _is_recovery_read — non-dict, resolved-path match, cache_root exception
# --------------------------------------------------------------------------- #
def test_recovery_read_non_dict_args():
    assert ToolOutputCompressor._is_recovery_read(None) is False
    assert ToolOutputCompressor._is_recovery_read("string") is False
    assert ToolOutputCompressor._is_recovery_read({"file_path": 5}) is False
    assert ToolOutputCompressor._is_recovery_read({"file_path": ""}) is False


def test_recovery_read_resolved_host_path_match(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    root = cache.ensure_cache_root()
    target = root / "abc"
    target.write_text("x", encoding="utf-8")
    # No "cache/compresr" substring shortcut here would be exercised on a fully
    # resolved absolute path that starts with the resolved cache root.
    assert ToolOutputCompressor._is_recovery_read({"file_path": str(target)}) is True


def test_recovery_read_cache_root_exception_falls_back(monkeypatch):
    # If get_cache_root().resolve() blows up, the substring shortcut must still
    # catch a container-translated path.
    monkeypatch.setattr(cache, "get_cache_root", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ToolOutputCompressor._is_recovery_read(
        {"file_path": "/root/.hermes/cache/compresr/tool-output/x"}
    ) is True
    # And a non-cache path returns False even with the broken cache_root.
    assert ToolOutputCompressor._is_recovery_read({"file_path": "/etc/passwd"}) is False


def test_recovery_read_resolve_oserror_is_swallowed(monkeypatch, tmp_path):
    """If Path(v).resolve() raises OSError/ValueError for a candidate value, the
    guard must swallow it and keep checking (not crash the hook)."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    real_resolve = cache.Path.resolve

    def _maybe_boom(self, *a, **k):
        if "boomtarget" in str(self):
            raise OSError("bad path")
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(cache.Path, "resolve", _maybe_boom)
    # value has no cache/compresr substring, so it reaches the resolve() try/except.
    assert ToolOutputCompressor._is_recovery_read({"file_path": "/some/boomtarget/x"}) is False


# --------------------------------------------------------------------------- #
# register() — plugin entry point wires the hook and stays quiet when inactive
# --------------------------------------------------------------------------- #
def test_register_wires_hook(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    hooks = {}

    class _Ctx:
        def register_hook(self, name, fn):
            hooks[name] = fn

    toc.register(_Ctx())
    assert "transform_tool_result" in hooks
    assert callable(hooks["transform_tool_result"])


def test_register_survives_cache_init_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "ensure_cache_root", lambda: (_ for _ in ()).throw(OSError("no perms")))
    hooks = {}

    class _Ctx:
        def register_hook(self, name, fn):
            hooks[name] = fn

    toc.register(_Ctx())  # must not raise despite ensure_cache_root blowing up
    assert "transform_tool_result" in hooks


# --------------------------------------------------------------------------- #
# _derive_query — path/tool-name/fallback branches
# --------------------------------------------------------------------------- #
def test_derive_query_branches():
    d = ToolOutputCompressor._derive_query
    assert d("grep", {"pattern": "foo"}) == "grep: foo"
    assert d("read_file", {"file_path": "/a/b.py"}).startswith("Relevant content of /a/b.py")
    assert d("read_file", {}) == "Relevant output of the read_file call for the current task"
    assert "Preserve the facts" in d("", None)  # empty tool + no args → fallback query
    # non-str arg values are skipped, falls through to tool-name form
    assert d("grep", {"pattern": 123}) == "Relevant output of the grep call for the current task"


def test_derive_query_truncates_to_600():
    long = "x" * 5000
    out = ToolOutputCompressor._derive_query("grep", {"query": long})
    assert len(out) == 600


# --------------------------------------------------------------------------- #
# _opt config/env precedence
# --------------------------------------------------------------------------- #
def test_opt_env_wins_over_config(monkeypatch, tmp_path):
    hermes_home = tmp_path
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        "compresr:\n"
        "  tool_output_model: model_from_cfg\n"
        "  tool_output_min_tokens: 111\n"
        "  base_url: https://cfg.example/api/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_MODEL", "model_from_env")
    monkeypatch.delenv("COMPRESR_TOOL_OUTPUT_MIN_TOKENS", raising=False)
    c = ToolOutputCompressor()
    assert c.model == "model_from_env"     # env wins
    assert c.min_tokens == 111             # config fills the gap
    assert c.base_url == "https://cfg.example/api"  # trailing slash stripped


def test_opt_defaults_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml
    for k in ("COMPRESR_TOOL_OUTPUT_MODEL", "COMPRESR_TOOL_OUTPUT_MIN_TOKENS",
              "COMPRESR_TOOL_OUTPUT_TARGET_RATIO", "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB"):
        monkeypatch.delenv(k, raising=False)
    c = ToolOutputCompressor()
    assert c.min_tokens == 1500
    assert c.max_cache_mb == 256
    assert c.target_ratio == 2.0


# --------------------------------------------------------------------------- #
# hook gating branches
# --------------------------------------------------------------------------- #
def test_hook_inactive_when_no_key():
    c = ToolOutputCompressor()
    c.enabled, c.api_key = True, ""
    assert c.active is False
    assert c.on_transform_tool_result(tool_name="grep", args={}, result=ORIGINAL) is None


def test_hook_non_string_result_is_ignored():
    c = _active()
    assert c.on_transform_tool_result(tool_name="grep", args={}, result={"a": 1}) is None
    assert c.on_transform_tool_result(tool_name="grep", args={}, result=None) is None


def test_hook_skips_error_status(monkeypatch):
    c = _active()
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: "/root/.hermes/cache/compresr/tool-output/x")
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, status="error",
    )
    assert out is None  # error-status results are never mangled


def test_hook_respects_cooldown(monkeypatch):
    c = _active()
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: "/root/.hermes/cache/compresr/tool-output/x")
    c._cooldown_until = time.monotonic() + 100.0  # in cooldown
    out = c.on_transform_tool_result(tool_name="grep", args={"pattern": "x"}, result=ORIGINAL)
    assert out is None


def test_hook_api_error_trips_cooldown(monkeypatch):
    """An API error inside compress_tool_output must bump errors + set a 30s
    cooldown so a failing endpoint isn't re-hit every turn."""
    c = _active()

    def _boom(**kw):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(c._client, "compress", _boom)
    before = time.monotonic()
    out = c.on_transform_tool_result(tool_name="grep", args={"pattern": "x"}, result=ORIGINAL)
    assert out is None
    assert c.errors == 1
    assert c._cooldown_until >= before + 29.0


def test_hook_successful_but_not_shorter_does_not_cooldown(monkeypatch):
    """Fix #2: a SUCCESSFUL API call whose output simply isn't a net win reports
    skipped_reason (NOT error), so it must NOT arm the 30s cooldown or bump the
    error counter — otherwise one incompressible output silently blocks
    compression of every later tool output for 30s."""
    c = _active()

    def _not_shorter(**kw):
        return ORIGINAL, {
            "called_api": True,
            "shortened": False,
            "skipped_reason": "not smaller after footer",
            "base_tokens": 500,
            "out_tokens": 500,
        }

    monkeypatch.setattr(toc, "compress_tool_output", _not_shorter)
    out = c.on_transform_tool_result(tool_name="grep", args={"pattern": "x"}, result=ORIGINAL)
    assert out is None
    assert c.errors == 0            # benign no-win — no error bump
    assert c._cooldown_until == 0.0  # cooldown NOT armed


def test_hook_cache_write_failure_does_cooldown(monkeypatch):
    """Fix #2 (refined): a GENUINE failure after a successful API call —
    store_original returning None (e.g. a reused Docker container that lost its
    cache mount, per Fix #1) — must still arm the cooldown and count as an error
    so the plugin doesn't make an un-throttled paid API call on every subsequent
    output and doesn't hide the infra failure from /usage."""
    c = _active()
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: None)  # cache write fails
    before = time.monotonic()
    out = c.on_transform_tool_result(tool_name="grep", args={"pattern": "x"}, result=ORIGINAL)
    assert out is None                          # fail open
    assert c.errors == 1                         # counted as a real error
    assert c._cooldown_until >= before + 29.0    # cooldown armed


def test_hook_long_line_fails_open_without_calling_api(monkeypatch):
    """Fix #3: a tool output whose cached original would contain a line longer
    than the recovery per-line cap (get_max_line_length, default 2000) can't be
    recovered byte-exact, so the hook fails open and never calls the API."""
    c = _active(min_tokens=10)
    long_line = "x" * 3000  # exceeds the 2000-char recovery clamp
    payload = "short head line\n" + long_line + "\nshort tail line"
    result = json.dumps({"output": payload})  # terminal-style envelope

    called = {"n": 0}

    def _spy(**kw):
        called["n"] += 1
        return COMPRESSED, {}

    monkeypatch.setattr(c._client, "compress", _spy)
    out = c.on_transform_tool_result(
        tool_name="terminal", args={"command": "cat big"}, result=result
    )
    assert out is None
    assert called["n"] == 0  # never hit the API — failed open before compressing


def test_hook_unwrapped_inner_below_min_tokens_skips(monkeypatch):
    """A JSON envelope whose *inner* payload is below min_tokens (even though the
    whole envelope exceeds it) must skip — the inner text is what gets compressed."""
    c = ToolOutputCompressor()
    c.enabled, c.api_key = True, "cmp_test"
    # min_tokens between inner size and envelope size.
    small_inner = "tiny content"
    result = json.dumps({"content": small_inner, "padding": "P" * 4000})
    c.min_tokens = (len(small_inner) + 3) // 4 + 5  # just above inner token count
    called = {"n": 0}

    def _spy(**kw):
        called["n"] += 1
        return COMPRESSED, {}

    monkeypatch.setattr(c._client, "compress", _spy)
    out = c.on_transform_tool_result(tool_name="read_file", args={"file_path": "/x"}, result=result)
    assert out is None
    assert called["n"] == 0  # never hit the API


def test_hook_defensive_except_on_store_raises(monkeypatch):
    """compress_tool_output is fail-open, but the hook wraps it in try/except as a
    belt-and-suspenders guard. Force a raise from within to hit that path."""
    c = _active()
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))

    def _raise(*a, **k):
        raise RuntimeError("unexpected cache explosion")

    monkeypatch.setattr(cache, "store_original", _raise)
    before = time.monotonic()
    out = c.on_transform_tool_result(tool_name="grep", args={"pattern": "x"}, result=ORIGINAL)
    assert out is None
    assert c.errors == 1
    assert c._cooldown_until >= before + 29.0


def test_hook_target_ratio_is_plumbed_to_client(monkeypatch):
    c = _active()
    c.target_ratio = 3.5
    fake = _FakeClient(out=COMPRESSED)
    c._client = fake
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: "/root/.hermes/cache/compresr/tool-output/x")
    c.on_transform_tool_result(tool_name="grep", args={"pattern": "x"}, result=ORIGINAL)
    assert fake.seen.get("target_ratio") == 3.5
    assert fake.seen.get("coarse") is True


# --------------------------------------------------------------------------- #
# compress.py — target_ratio plumbing
# --------------------------------------------------------------------------- #
def test_compress_forwards_target_ratio(monkeypatch):
    monkeypatch.setattr(cache, "store_original", lambda cid, content, task_id="default", **_: "/p/" + cid)
    fake = _FakeClient(out=COMPRESSED)
    compress_tool_output(
        query="q", content=ORIGINAL, tool_name="grep", cache_id="c",
        client=fake, task_id="t", target_ratio=7.0,
    )
    assert fake.seen.get("target_ratio") == 7.0


# --------------------------------------------------------------------------- #
# cache.py — LocalEnvironment + no-active-env docker + write failure + prune
# --------------------------------------------------------------------------- #
def test_store_original_local_env_returns_resolved_host_path(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: _fake_env("LocalEnvironment"))
    path = cache.store_original("local1", ORIGINAL, task_id="t", max_cache_mb=0)
    assert path == str((hermes_home / "cache" / "compresr" / "tool-output" / "local1").resolve())


def test_store_original_no_env_docker_backend(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: None)
    path = cache.store_original("dk", ORIGINAL, task_id="t", max_cache_mb=0)
    assert path == "/root/.hermes/cache/compresr/tool-output/dk"


def test_store_original_no_env_unknown_backend_fails_open(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "daytona")  # not local/docker/modal
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: None)
    path = cache.store_original("dt", ORIGINAL, task_id="t", max_cache_mb=0)
    assert path is None
    # Content-addressed: the host file is retained (not unlinked) when visibility
    # can't be proven, so a concurrent sibling's recovery pointer can't dangle;
    # the size-based pruner reclaims it later.
    assert (hermes_home / "cache" / "compresr" / "tool-output" / "dt").exists()


def test_store_original_write_failure_returns_none(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(cache, "_get_active_env", lambda task_id: None)

    real_open = cache.os.open

    def _boom_open(path, *a, **k):
        if "compresr" in str(path):
            raise OSError("disk full")
        return real_open(path, *a, **k)

    monkeypatch.setattr(cache.os, "open", _boom_open)
    assert cache.store_original("wf", ORIGINAL, task_id="t") is None


def test_prune_total_under_budget_is_noop(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir

    a = tmp_path / "a"
    a.write_text("x" * 5, encoding="utf-8")
    _prune_cache_dir(str(tmp_path), 1000, str(tmp_path / "keep"))
    assert a.exists()  # under budget → nothing removed


def test_prune_ignores_subdirectories(tmp_path):
    """Only files count toward the budget; a subdir under the cache root must be
    skipped by the ``is_file()`` filter and never unlinked."""
    from plugins.tool_output_compresr.cache import _prune_cache_dir

    subdir = tmp_path / "adir"
    subdir.mkdir()
    old = tmp_path / "old"
    current = tmp_path / "current"
    old.write_text("x" * 100, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")
    os.utime(old, (1, 1))

    _prune_cache_dir(str(tmp_path), 1, str(current))
    assert not old.exists()
    assert current.exists()
    assert subdir.is_dir()  # directory untouched


# --------------------------------------------------------------------------- #
# context_engine — cooldown, empty context, prior-summary fold, ratio bounds,
# latte_v1 coarse, disable_placeholders, missing key, get_status
# --------------------------------------------------------------------------- #
def _engine():
    e = CompresrContextEngine()
    e.compresr_api_key = "cmp_test"
    return e


TURNS = [
    {"role": "user", "content": "Fix the login bug in auth.py."},
    {"role": "assistant", "content": "Edited auth.py; token TTL 3600."},
]


def test_engine_cooldown_skips_generate(monkeypatch):
    e = _engine()
    e._summary_failure_cooldown_until = time.monotonic() + 100.0
    called = {"n": 0}
    e._call_compresr = lambda c, q: (called.__setitem__("n", called["n"] + 1), ("x", {}))[1]
    assert e._generate_summary(TURNS, focus_topic="x") is None
    assert called["n"] == 0


def test_engine_empty_context_skips(monkeypatch):
    e = _engine()
    monkeypatch.setattr(e, "_serialize_for_summary", lambda turns: "   \n\t")
    called = {"n": 0}
    e._call_compresr = lambda c, q: called.__setitem__("n", called["n"] + 1)
    assert e._generate_summary(TURNS) is None
    assert called["n"] == 0


def test_engine_folds_prior_summary_into_context():
    e = _engine()
    e._previous_summary = "EARLIER FACTS"
    seen = {}

    def _spy(context, query):
        seen["context"] = context
        return "KEPT body", {"tokens_saved": 5}

    e._call_compresr = _spy
    out = e._generate_summary(TURNS, focus_topic="topic")
    assert out is not None
    assert "[PRIOR CONTEXT SUMMARY]" in seen["context"]
    assert "EARLIER FACTS" in seen["context"]
    assert "[NEW CONVERSATION TURNS]" in seen["context"]


def test_engine_ratio_clamps_bounds():
    e = _engine()
    e.compresr_ratio_override = None
    # keep=0 is falsy so `keep or 0.2` substitutes the 0.2 default → 5x.
    e.summary_target_ratio = 0.0
    assert e._target_compression_ratio() == 5.0
    # A tiny but truthy keep clamps to the 0.01 floor → 100x (well under the 200 cap).
    e.summary_target_ratio = 0.001
    assert e._target_compression_ratio() == 100.0
    # keep near 1 clamps to the 0.95 ceiling → ~1.053x
    e.summary_target_ratio = 0.99
    assert e._target_compression_ratio() == round(1.0 / 0.95, 3)


def test_engine_latte_v1_sends_coarse(monkeypatch):
    e = _engine()
    e.compresr_model = "latte_v1"
    e.compresr_coarse = True
    e.compresr_disable_placeholders = True
    captured = {}

    class _Resp:
        def __init__(self, body):
            self._b = body.encode()

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request
    def _fake(req, timeout=None):
        captured["payload"] = json.loads(req.data)
        return _Resp(json.dumps({"success": True, "data": {"compressed_context": "x"}}))
    monkeypatch.setattr(urllib.request, "urlopen", _fake)

    e._call_compresr("ctx", "q")
    assert captured["payload"]["coarse"] is True
    assert captured["payload"]["disable_placeholders"] is True


def test_engine_missing_key_raises_in_call():
    e = CompresrContextEngine()
    e.compresr_api_key = ""
    with pytest.raises(RuntimeError):
        e._call_compresr("ctx", "q")


def test_engine_http_error_body_read_failure_still_raises(monkeypatch):
    import urllib.error
    import urllib.request

    class _BadReadErr(urllib.error.HTTPError):
        def read(self):
            raise RuntimeError("cannot read body")

    def _raise(req, timeout=None):
        raise _BadReadErr("u", 500, "Server Error", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    e = _engine()
    with pytest.raises(RuntimeError) as ei:
        e._call_compresr("ctx", "q")
    assert "500" in str(ei.value)


def test_engine_get_status_extends_parent():
    e = _engine()
    e.compresr_calls = 3
    e.compresr_tokens_saved = 99
    st = e.get_status()
    assert st["engine"] == "compresr"
    assert st["compresr_calls"] == 3
    assert st["compresr_tokens_saved"] == 99
    assert "compresr_last_duration_ms" in st


# --------------------------------------------------------------------------- #
# base_url must be https (except localhost) — no silent cleartext downgrade of
# the API key / egress via a stray config or env value.
# --------------------------------------------------------------------------- #
def test_secure_base_url_rejects_plaintext_remote():
    from plugins.tool_output_compresr import _secure_base_url as _sec_toc
    from plugins.context_engine.compresr import _secure_base_url as _sec_ce

    default = "https://api.compresr.ai/api"
    for _sec in (_sec_toc, _sec_ce):
        assert _sec("https://api.compresr.ai/api", default) == "https://api.compresr.ai/api"
        assert _sec("http://evil.example.com/api", default) == default  # remote http rejected
        assert _sec("http://localhost:8000/api", default) == "http://localhost:8000/api"
        assert _sec("http://127.0.0.1:8000", default) == "http://127.0.0.1:8000"
        assert _sec("not a url", default) == default  # garbage → default


def test_tool_output_compressor_ignores_insecure_base_url(monkeypatch):
    monkeypatch.setenv("COMPRESR_BASE_URL", "http://evil.example.com/api")
    assert ToolOutputCompressor().base_url == "https://api.compresr.ai/api"


def test_context_engine_ignores_insecure_base_url(monkeypatch):
    monkeypatch.setenv("COMPRESR_BASE_URL", "http://evil.example.com/api")
    assert CompresrContextEngine().compresr_base_url == "https://api.compresr.ai/api"


# --------------------------------------------------------------------------- #
# context.engine: compresr forwards the user's compression.* geometry instead
# of silently using ContextCompressor's hardcoded defaults.
# --------------------------------------------------------------------------- #
def test_context_engine_forwards_compression_config(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "compression:\n"
        "  protect_first_n: 5\n"
        "  protect_last_n: 40\n"
        "  target_ratio: 0.10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    eng = CompresrContextEngine()
    assert eng.protect_first_n == 5
    assert eng.protect_last_n == 40
    assert eng.summary_target_ratio == 0.10


def test_context_engine_uses_parent_defaults_without_compression_config(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    eng = CompresrContextEngine()
    assert eng.protect_last_n == 20      # ContextCompressor default, unchanged
    assert eng.summary_target_ratio == 0.20


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
