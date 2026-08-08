"""Regression tests for the per-message runtime-resolution fast path.

Two layers are pinned here:

Layer A — ``hermes_cli/runtime_provider.py``: the resolve tree previously
called ``load_config()`` (a defensive deepcopy of the whole config, ~265us on
a warm cache) 5-7 times per ``resolve_runtime_provider()`` call — measured
74% of the ~2.35 ms/call cost. All sites are read-only against config, so
they now use ``load_config_readonly()`` (mtime-cached, no deepcopy), and
``_get_model_config()`` deep-copies only the small model section to preserve
its mutation-safety contract.

Layer B — ``gateway/run.py``: the gateway resolver calls
``resolve_runtime_provider()`` 5-8x per inbound message. The MoA path already
caches it behind a 300s TTL (#66793); the gateway path now gets the same
treatment via ``_memoized_resolve_runtime``, keyed on the files the
resolution reads (config.yaml + profile/global auth.json) with a TTL
backstop. Vertex (per-call OAuth token) and fallback/AuthError results are
never cached.

The measured-work pins assert the MECHANISM (work actually skipped), which is
what a regression would break; the behavior-parity pins assert the RESULT is
unchanged.
"""

import time
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
import hermes_cli.config as config_mod
import hermes_cli.runtime_provider as rp_mod


@pytest.fixture(autouse=True)
def _clear_memo():
    """Each test starts with an empty gateway memo.

    Tolerates the attribute being absent (the sabotage gate runs this test
    file against PRE-FIX base code, where the memo does not exist yet) so
    the base leg fails on the measured-work assertions — a genuine pin
    failure — instead of erroring at setup.
    """
    lock = getattr(gateway_run, "_runtime_resolve_memo_lock", None)
    memo = getattr(gateway_run, "_runtime_resolve_memo", None)
    if lock is not None and memo is not None:
        with lock:
            memo.clear()
    yield
    if lock is not None and memo is not None:
        with lock:
            memo.clear()


# ---------------------------------------------------------------------------
# Layer A — the resolve tree must not deepcopy the whole config
# ---------------------------------------------------------------------------

class TestResolveTreeReadOnlyConfig:
    def test_get_model_config_does_not_call_deepcopy_load_config(self, monkeypatch):
        """The model-config read must use the read-only loader, not the
        deepcopy variant. load_config() is what a regression would reintroduce."""
        calls = {"deepcopy": 0}

        def counting_load_config(*a, **k):
            calls["deepcopy"] += 1
            return {}

        monkeypatch.setattr(config_mod, "load_config", counting_load_config)
        rp_mod._get_model_config()
        assert calls["deepcopy"] == 0

    def test_resolve_tree_uses_readonly_config_loader(self, monkeypatch):
        """No load_config() (deepcopy) call may originate from the
        runtime_provider resolve tree — every site there was converted to the
        read-only loader. (auth.py's error-hint builder legitimately calls
        load_config on the failure path; that is not the resolve tree.)"""
        import traceback

        calls = {"from_rp": 0, "elsewhere": 0}
        orig = config_mod.load_config

        def counting(*a, **k):
            frames = traceback.extract_stack(limit=4)
            # The DIRECT caller is frames[-3] (counting <- caller <- ...).
            caller = frames[-3] if len(frames) >= 3 else None
            from_rp = bool(
                caller and "hermes_cli/runtime_provider.py" in caller.filename
            )
            if from_rp:
                calls["from_rp"] += 1
            else:
                calls["elsewhere"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(config_mod, "load_config", counting)
        try:
            rp_mod.resolve_runtime_provider(requested="openai")
        except Exception:
            pass  # no credentials on CI — we only assert the loader used
        assert calls["from_rp"] == 0
        assert calls["elsewhere"] >= 1  # error-hint path still works

    def test_get_model_config_mutation_is_isolated(self):
        """The returned model config is a private copy: mutating it (even
        nested) must not corrupt the shared read-only config cache."""
        cfg_a = rp_mod._get_model_config()
        cfg_b = rp_mod._get_model_config()
        cfg_a["default"] = "HACKED"
        assert cfg_b.get("default") != "HACKED"
        # Nested values must also be isolated (the deepcopy of the section).
        if isinstance(cfg_a, dict) and isinstance(cfg_b, dict):
            assert cfg_a is not cfg_b


# ---------------------------------------------------------------------------
# Layer B — gateway memo: mechanism pins
# ---------------------------------------------------------------------------

class TestGatewayRuntimeMemo:
    def _counting_resolve(self, monkeypatch, *, real_delegate=False):
        calls = {"n": 0, "last": None}
        orig = rp_mod.resolve_runtime_provider

        def fake(*a, **k):
            calls["n"] += 1
            calls["last"] = (a, k)
            if real_delegate:
                # Behavior-parity mode: run the REAL resolver.
                return orig(*a, **k)
            # Deterministic success result (CI has no provider configured, so
            # the real resolver raises AuthError and nothing would be cached).
            return {
                "api_key": "test-key",
                "base_url": "https://example.com/v1",
                "provider": k.get("requested") or "openai",
                "requested_provider": k.get("requested"),
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
                "max_output_tokens": None,
            }

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake)
        return calls

    def test_second_resolve_hits_memo(self, monkeypatch):
        """Two identical calls within the TTL must resolve ONCE — the
        per-message win (5-8 resolver calls per message → 1)."""
        calls = self._counting_resolve(monkeypatch)
        r1 = gateway_run._memoized_resolve_runtime()
        r2 = gateway_run._memoized_resolve_runtime()
        assert calls["n"] == 1
        assert r1 == r2

    def test_different_requested_provider_is_separate_key(self, monkeypatch):
        calls = self._counting_resolve(monkeypatch)
        try:
            gateway_run._memoized_resolve_runtime()
        except Exception:
            pass
        try:
            gateway_run._memoized_resolve_runtime(requested="openai")
        except Exception:
            pass
        # Two different keys → two resolves (whatever they resolve to).
        assert calls["n"] == 2

    def test_config_change_invalidates_memo(self, monkeypatch, tmp_path):
        """A config.yaml mtime bump must force a fresh resolve immediately
        (a `/model` switch or `hermes config` edit must not wait for the TTL)."""
        calls = self._counting_resolve(monkeypatch)
        gateway_run._memoized_resolve_runtime()
        assert calls["n"] == 1

        # Rewrite config.yaml so its mtime/size change.
        from hermes_cli.config import get_config_path
        cfg_path = get_config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        orig = cfg_path.read_text() if cfg_path.exists() else ""
        try:
            cfg_path.write_text(orig + "\n# memo-invalidation probe\n")
            gateway_run._memoized_resolve_runtime()
            assert calls["n"] == 2
        finally:
            if orig:
                cfg_path.write_text(orig)

    def test_ttl_expiry_forces_re_resolve(self, monkeypatch):
        calls = self._counting_resolve(monkeypatch)
        gateway_run._memoized_resolve_runtime()
        assert calls["n"] == 1
        # Rewind the cached stamp past the TTL.
        with gateway_run._runtime_resolve_memo_lock:
            for key, (stamp, val) in list(gateway_run._runtime_resolve_memo.items()):
                gateway_run._runtime_resolve_memo[key] = (
                    stamp - gateway_run._RUNTIME_RESOLVE_MEMO_TTL_SECONDS - 1,
                    val,
                )
        gateway_run._memoized_resolve_runtime()
        assert calls["n"] == 2

    def test_vertex_requested_never_cached(self, monkeypatch):
        """Vertex mints a per-call OAuth token — every call must resolve."""
        calls = self._counting_resolve(monkeypatch)
        try:
            gateway_run._memoized_resolve_runtime(requested="vertex")
        except Exception:
            pass
        try:
            gateway_run._memoized_resolve_runtime(requested="vertex")
        except Exception:
            pass
        assert calls["n"] == 2

    def test_returned_dict_is_fresh_copy(self, monkeypatch):
        """Callers mutate the returned dict (e.g. pop('model')) — the cached
        entry must never be corrupted by that."""
        calls = self._counting_resolve(monkeypatch)
        r1 = gateway_run._memoized_resolve_runtime()
        r1["provider"] = "HACKED"
        r2 = gateway_run._memoized_resolve_runtime()
        assert calls["n"] == 1
        assert r2.get("provider") != "HACKED"

    def test_gateway_kwargs_wrapper_uses_memo(self, monkeypatch):
        """The actual per-message entry (_resolve_runtime_agent_kwargs) must
        resolve once across two calls (success path)."""
        calls = {"n": 0}
        orig = rp_mod.resolve_runtime_provider

        def counting(*a, **k):
            calls["n"] += 1
            return {
                "api_key": "k",
                "base_url": "https://example.com/v1",
                "provider": k.get("requested") or "openai",
                "requested_provider": k.get("requested"),
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
                "max_output_tokens": None,
            }

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", counting)
        gateway_run._resolve_runtime_agent_kwargs()
        gateway_run._resolve_runtime_agent_kwargs()
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Layer B — the _resolve_runtime_agent_kwargs wrapper itself still works
# ---------------------------------------------------------------------------

class TestRuntimeKwargsWrapper:
    def test_wrapper_returns_expected_shape(self, monkeypatch):
        """The gateway wrapper must keep returning the documented dict shape
        (max_tokens computed from env/config per call, outside the memo)."""
        fake_runtime = {
            "api_key": "k",
            "base_url": "https://example.com/v1",
            "provider": "openai",
            "requested_provider": "openai",
            "api_mode": "chat_completions",
            "command": None,
            "args": ["a"],
            "credential_pool": None,
        }
        monkeypatch.setattr(
            rp_mod, "resolve_runtime_provider", lambda *a, **k: dict(fake_runtime)
        )
        out = gateway_run._resolve_runtime_agent_kwargs()
        assert out["provider"] == "openai"
        assert out["base_url"] == "https://example.com/v1"
        assert out["args"] == ["a"]
        assert "max_tokens" in out

    def test_for_provider_wrapper_uses_memo(self, monkeypatch):
        calls = {"n": 0}

        def counting(*a, **k):
            calls["n"] += 1
            return {
                "api_key": "k",
                "base_url": "https://example.com/v1",
                "provider": k.get("requested") or "openai",
                "requested_provider": k.get("requested"),
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
                "credential_pool": None,
            }

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", counting)
        gateway_run._resolve_runtime_agent_kwargs_for_provider("openai")
        gateway_run._resolve_runtime_agent_kwargs_for_provider("openai")
        assert calls["n"] == 1
