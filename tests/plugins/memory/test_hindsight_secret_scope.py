"""Regression tests: Hindsight credential handling on background threads.

Under gateway multiplexing (``set_multiplex_active(True)``) ``get_secret`` fails
closed with ``UnscopedSecretError`` whenever no profile secret scope is installed
on the calling context. Several Hindsight code paths run on background threads:

  * the retain writer thread (``_writer_loop``) — one shared thread per provider
  * the prefetch thread
  * the embedded-daemon starter thread (``_daemon_start_worker``)

Two distinct bugs are covered here.

**1. Availability** — ``_embedded_llm_api_key`` called ``get_secret`` directly.
On the daemon-starter thread that raised, which aborted embedded-daemon startup,
so every ``hindsight_retain`` blocked for the full daemon-start timeout and then
failed (logged at WARNING only, so silent in practice). A missing key must
degrade to "" and let the daemon fall back to its own profile ``.env``.

**2. Correctness / cross-profile isolation** — the writer is a single
long-lived thread, lazily started on the first retain. Capturing the context
once at thread start pins it to whichever profile retained first, so every
later profile's retain runs under that profile's secret scope. The context
snapshot must be taken per job at enqueue time instead.
"""

import contextvars
import queue
import threading

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    get_secret,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from plugins.memory.hindsight.embedded import (
    _build_embedded_profile_env,
    _embedded_llm_api_key,
)


@pytest.fixture
def multiplex_on():
    """Enable multiplex fail-closed mode for the duration of a test."""
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(False)


def _run_off_scope(fn):
    """Run ``fn`` on a bare thread (no secret scope) and capture the outcome."""
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - we assert on the type
            result["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "background thread hung"
    return result


# ---------------------------------------------------------------------------
# Bug 1: availability — daemon startup must survive a missing scope.
# ---------------------------------------------------------------------------


def test_bare_get_secret_raises_off_scope_under_multiplex(multiplex_on):
    """Baseline: confirm the fail-closed behaviour the fix has to tolerate."""
    result = _run_off_scope(lambda: get_secret("HINDSIGHT_LLM_API_KEY", ""))
    assert isinstance(result.get("error"), UnscopedSecretError)


def test_embedded_llm_api_key_off_scope_does_not_raise(multiplex_on):
    """Key resolution degrades to "" instead of aborting daemon startup."""
    result = _run_off_scope(lambda: _embedded_llm_api_key({}))
    assert "error" not in result, f"must not raise, got {result.get('error')!r}"
    assert result["value"] == ""


def test_build_env_off_scope_does_not_raise(multiplex_on):
    """The daemon-env builder is safe on the daemon-starter thread."""
    result = _run_off_scope(lambda: _build_embedded_profile_env({}))
    assert "error" not in result, f"must not raise, got {result.get('error')!r}"
    assert result["value"]["HINDSIGHT_API_LLM_API_KEY"] == ""


def test_build_env_prefers_explicitly_passed_key(multiplex_on):
    """An on-scope caller can thread the resolved key through to the builder."""
    result = _run_off_scope(
        lambda: _build_embedded_profile_env({}, llm_api_key="resolved-on-scope")
    )
    assert "error" not in result
    assert result["value"]["HINDSIGHT_API_LLM_API_KEY"] == "resolved-on-scope"


def test_config_key_short_circuits_secret_read(multiplex_on):
    """A key already in config never reaches the secret store."""
    result = _run_off_scope(
        lambda: _embedded_llm_api_key({"llm_api_key": "from-config"})
    )
    assert "error" not in result
    assert result["value"] == "from-config"


def test_env_fallback_still_works_without_multiplex(monkeypatch):
    """Single-profile installs keep reading the key from the environment."""
    set_multiplex_active(False)
    monkeypatch.setenv("HINDSIGHT_LLM_API_KEY", "env-key")
    assert _build_embedded_profile_env({})["HINDSIGHT_API_LLM_API_KEY"] == "env-key"


# ---------------------------------------------------------------------------
# Bug 2: cross-profile isolation on the shared writer thread.
# ---------------------------------------------------------------------------


def _drive_shared_writer(wrap_per_job: bool):
    """Run two profiles' retains through one shared writer thread.

    Mirrors ``_ensure_writer`` / ``_enqueue_retain``. With ``wrap_per_job`` the
    context is snapshotted at enqueue time (the fix); without it the thread
    captures one context at start (the bug).

    Returns the key each profile's job observed.
    """
    jobs: queue.Queue = queue.Queue()
    observed: list[str] = []

    def payload():
        """The retain work itself — reads the credential it was queued with."""
        try:
            observed.append(get_secret("HINDSIGHT_LLM_API_KEY", "") or "")
        except UnscopedSecretError:
            observed.append("UNSCOPED")

    def writer_loop():
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                return
            try:
                job()
            finally:
                jobs.task_done()

    def enqueue():
        if wrap_per_job:
            ctx = contextvars.copy_context()
            jobs.put(lambda: ctx.run(payload))
        else:
            jobs.put(payload)

    thread = None
    for key in ("KEY-PROFILE-A", "KEY-PROFILE-B"):
        token = set_secret_scope({"HINDSIGHT_LLM_API_KEY": key})
        try:
            if thread is None:
                # First retain lazily starts the writer, exactly as the provider does.
                if wrap_per_job:
                    thread = threading.Thread(target=writer_loop, daemon=True)
                else:
                    thread = threading.Thread(
                        target=contextvars.copy_context().run,
                        args=(writer_loop,),
                        daemon=True,
                    )
                thread.start()
            enqueue()
            jobs.join()
        finally:
            reset_secret_scope(token)

    jobs.put(None)
    thread.join(timeout=5)
    return observed


def test_shared_writer_leaks_without_per_job_context(multiplex_on):
    """Documents the bug: a thread-level context snapshot pins profile A."""
    observed = _drive_shared_writer(wrap_per_job=False)
    assert observed == ["KEY-PROFILE-A", "KEY-PROFILE-A"], (
        "expected the known leak shape from a thread-level context snapshot"
    )


def test_per_job_context_isolates_profiles(multiplex_on):
    """The fix: each job carries its own enqueue-time context."""
    observed = _drive_shared_writer(wrap_per_job=True)
    assert observed == ["KEY-PROFILE-A", "KEY-PROFILE-B"], (
        f"profiles must not share credentials, got {observed}"
    )


def test_enqueue_retain_snapshots_caller_context(multiplex_on):
    """``_enqueue_retain`` must wrap the job, not hand it over bare.

    Guards the actual provider method against a regression to
    ``self._retain_queue.put(job)``.
    """
    from plugins.memory.hindsight import HindsightMemoryProvider

    provider = HindsightMemoryProvider()
    provider._register_atexit = lambda: None
    provider._ensure_writer = lambda: None

    sentinel_seen: list[str] = []

    def job():
        try:
            sentinel_seen.append(get_secret("HINDSIGHT_LLM_API_KEY", "") or "")
        except UnscopedSecretError:
            sentinel_seen.append("UNSCOPED")

    token = set_secret_scope({"HINDSIGHT_LLM_API_KEY": "enqueue-time-key"})
    try:
        provider._enqueue_retain(job)
    finally:
        reset_secret_scope(token)

    queued = provider._retain_queue.get_nowait()
    # Run it off-scope, the way the writer thread would.
    result = _run_off_scope(queued)
    assert "error" not in result, f"queued job must not raise, got {result.get('error')!r}"
    assert sentinel_seen == ["enqueue-time-key"], (
        f"job must see the enqueueing profile's scope, got {sentinel_seen}"
    )
