"""Seam-test + behavior test for apex_overlay.models_dev_fast.

Pins the upstream symbol ``apex_overlay.models_dev_fast`` monkey-patches
(``agent.models_dev.fetch_models_dev``) plus the module-level cache internals
the wrapper reaches into, so an upstream rename/move turns a *silently disarmed*
non-blocking guard into a *loud CI failure* (see ``apex_overlay/README.md``).

It also proves the behavior:

* ``force_refresh=True`` still delegates to upstream (synchronous, network hit)
  — the explicit "refresh model catalog" path is untouched.
* ``force_refresh=False`` with a STALE disk cache returns immediately from disk
  and makes **zero** synchronous network calls (the picker never blocks); the
  refresh happens on a background thread.
* ``force_refresh=False`` with NO cache returns ``{}`` immediately + background
  refresh — never the ~10s synchronous block.
* CN mode raises the effective staleness threshold so a stale-but-within-window
  disk cache is served with **no** background refresh at all.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter), not a
single in-process pytest — process-wide monkey-patches behave differently under
single-process isolation.
"""

from __future__ import annotations

import inspect
import time
from unittest.mock import MagicMock, patch

from apex_overlay import models_dev_fast


def _wait_for_background(pred, timeout=2.0):
    """Spin until pred() is true or timeout — background refresh is a thread."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ---------------------------------------------------------------------------
# Seam assertions — pin the patched symbol + the internals the wrapper uses
# ---------------------------------------------------------------------------

def test_seam_target_fetch_models_dev_exists_with_compatible_signature():
    """apex_overlay patches agent.models_dev.fetch_models_dev.

    If upstream renames/moves it or drops the ``force_refresh`` param, the
    non-blocking picker guarantee silently stops working. Fail loudly here.
    """
    import agent.models_dev as md

    fn = getattr(md, models_dev_fast._TARGET_FN, None)
    assert fn is not None, (
        "agent.models_dev.fetch_models_dev is gone — apex_overlay can no longer "
        "make the picker's catalog fetch non-blocking. Update "
        "apex_overlay.models_dev_fast._TARGET_FN and the wrapper."
    )
    params = inspect.signature(fn).parameters
    assert "force_refresh" in params, (
        f"fetch_models_dev lost the force_refresh param; the apex_overlay "
        f"wrapper branches on it. Params: {list(params)!r}"
    )


def test_seam_pins_models_dev_cache_internals():
    """The wrapper reaches into module cache internals + disk helpers.

    Pin them so an upstream refactor that removes/renames any is a loud failure
    rather than a silently-disarmed fast path (the wrapper falls back to
    upstream on AttributeError, so this test is the real tripwire).
    """
    import agent.models_dev as md

    for attr in (
        "_models_dev_cache",
        "_models_dev_cache_time",
        "_MODELS_DEV_CACHE_TTL",
        "_disk_cache_age_seconds",
        "_load_disk_cache",
    ):
        assert hasattr(md, attr), (
            f"agent.models_dev.{attr} is gone — apex_overlay.models_dev_fast's "
            f"cache-first fast path depends on it."
        )


def test_apply_binds_target_and_is_idempotent():
    """apply() must bind the target and be a safe no-op on repeat."""
    import agent.models_dev as md

    models_dev_fast._APPLIED = False
    assert models_dev_fast.apply() is True
    assert getattr(md.fetch_models_dev, models_dev_fast._MARK, False)
    # Idempotent: second apply is a no-op and must not double-wrap.
    assert models_dev_fast.apply() is True
    assert getattr(md.fetch_models_dev, models_dev_fast._MARK, False)


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------

def _reset_state():
    import agent.models_dev as md

    md._models_dev_cache = {}
    md._models_dev_cache_time = 0
    models_dev_fast._APPLIED = False
    models_dev_fast._refresh_in_flight = False


def test_force_refresh_delegates_to_upstream_synchronously():
    """force_refresh=True → the original runs synchronously (network path)."""
    import agent.models_dev as md

    _reset_state()
    orig = md.fetch_models_dev
    sentinel = {"anthropic": {"models": {}}}
    stub = MagicMock(return_value=sentinel)

    # Patch the ORIGINAL that apply() will capture, then apply.
    with patch.object(md, "fetch_models_dev", stub):
        assert models_dev_fast.apply() is True
        out = md.fetch_models_dev(force_refresh=True)

    assert out == sentinel
    stub.assert_called_once()
    # Delegated with a truthy force_refresh (the explicit refresh contract),
    # whether passed positionally or by keyword.
    call = stub.call_args
    forwarded = call.args[0] if call.args else call.kwargs.get("force_refresh")
    assert forwarded is True
    md.fetch_models_dev = orig


def test_stale_disk_cache_served_immediately_no_sync_network():
    """force_refresh=False + stale disk cache → instant disk return, network
    only on a background thread (picker never blocks).

    Proven by timing: the ORIGINAL fetch is made deliberately slow. If the
    picker path blocked on it, the call couldn't return in << that latency,
    and it would return the *network* data instead of the disk data.
    """
    import agent.models_dev as md

    _reset_state()
    real_orig = md.fetch_models_dev
    disk = {"deepseek": {"models": {"deepseek-v4-pro": {}}}}
    net_latency = 0.4
    started = []
    done = []

    def slow_orig(force_refresh=False):
        started.append(time.time())
        time.sleep(net_latency)  # simulate the ~10s trans-Pacific fetch
        md._models_dev_cache = {"net": True}
        md._models_dev_cache_time = time.time()
        done.append(time.time())
        return {"net": True}

    # Stale beyond any threshold so a background refresh is triggered.
    with patch.object(md, "_disk_cache_age_seconds", return_value=10 ** 9), \
         patch.object(md, "_load_disk_cache", return_value=disk), \
         patch("apex_overlay.region.is_cn_mode", return_value=False):
        md.fetch_models_dev = models_dev_fast._wrap_fetch_models_dev(slow_orig)

        t0 = time.time()
        out = md.fetch_models_dev()
        elapsed = time.time() - t0

        # Returned the DISK data, not the (slow) network data → didn't wait.
        assert out == disk, f"expected disk data, got {out!r}"
        assert elapsed < net_latency / 2, (
            f"picker BLOCKED on the network fetch ({elapsed:.3f}s)"
        )
        assert not done, "network fetch COMPLETED before return — it blocked"
        # Background thread eventually fires (and completes) the refresh.
        assert _wait_for_background(lambda: len(done) >= 1), (
            "stale cache must trigger a background network refresh"
        )

    md.fetch_models_dev = real_orig


def test_no_cache_returns_empty_immediately_and_refreshes_async():
    """force_refresh=False + no disk cache → {} now, refresh in background."""
    import agent.models_dev as md

    _reset_state()
    real_orig = md.fetch_models_dev
    calls = []

    def stub_orig(force_refresh=False):
        calls.append(force_refresh)
        return {"net": True}

    with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
         patch.object(md, "_load_disk_cache", return_value={}), \
         patch("apex_overlay.region.is_cn_mode", return_value=False):
        md.fetch_models_dev = models_dev_fast._wrap_fetch_models_dev(stub_orig)
        out = md.fetch_models_dev()

        assert out == {}, "no-cache path must return empty immediately"
        assert _wait_for_background(lambda: len(calls) >= 1), (
            "no-cache path must still refresh in the background"
        )
        assert calls[0] is True, "background refresh must force_refresh=True"

    md.fetch_models_dev = real_orig


def test_cn_mode_serves_stale_cache_without_background_refresh():
    """In CN mode a disk cache stale by hours (but < 7d) is served with NO
    background refresh — no doomed trans-Pacific fetch every session."""
    import agent.models_dev as md

    _reset_state()
    real_orig = md.fetch_models_dev
    disk = {"alibaba": {"models": {"qwen3.7-max": {}}}}
    calls = []

    def stub_orig(force_refresh=False):
        calls.append(force_refresh)
        return {"net": True}

    # 6 hours old: past the 1h upstream TTL, but well within the CN 7-day window.
    with patch.object(md, "_disk_cache_age_seconds", return_value=6 * 3600), \
         patch.object(md, "_load_disk_cache", return_value=disk), \
         patch("apex_overlay.region.is_cn_mode", return_value=True):
        md.fetch_models_dev = models_dev_fast._wrap_fetch_models_dev(stub_orig)
        out = md.fetch_models_dev()

        assert out == disk
        # Give any (erroneous) background thread a chance, then assert none ran.
        time.sleep(0.1)
        assert calls == [], (
            "CN mode must NOT refresh a cache that is still within the 7-day "
            "staleness window"
        )

    md.fetch_models_dev = real_orig


def test_effective_threshold_is_larger_in_cn_mode():
    """CN mode raises the staleness threshold well above the upstream TTL."""
    import agent.models_dev as md

    with patch("apex_overlay.region.is_cn_mode", return_value=False):
        global_thr = models_dev_fast._effective_stale_threshold()
    with patch("apex_overlay.region.is_cn_mode", return_value=True):
        cn_thr = models_dev_fast._effective_stale_threshold()

    assert global_thr == float(md._MODELS_DEV_CACHE_TTL)
    assert cn_thr == float(models_dev_fast._CN_STALE_THRESHOLD)
    assert cn_thr > global_thr


# ---------------------------------------------------------------------------
# Wiring — the seam loads via the bundled plugin
# ---------------------------------------------------------------------------

def test_plugin_register_applies_seam():
    """The bundled apex-overlay plugin's register() applies this seam too."""
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams("_apex_overlay_plugin_under_test_mdf")
    assert "models_dev_fast" in called, (
        "plugin.register() must call models_dev_fast.apply()"
    )
