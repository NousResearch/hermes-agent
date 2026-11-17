"""Remote backend lock order: same-thread re-entry, logging outside backend locks, the formatter."""
from __future__ import annotations

from hermes_cli.config_backend import ConfigBackendUnavailable, write_config_key
from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod

from . import test_remote_backend as _base

plane = _base.plane  # the stub-plane fixture
_gets = _base._gets


# --- re-entry ----------------------------------------------------------------------------

def _blocked_in_state(thread) -> bool:
    import sys
    frame = sys._current_frames().get(thread.ident)
    while frame is not None:
        if frame.f_code is backend_mod.RemoteBackend._state.__code__:
            return True
        frame = frame.f_back
    return False


def test_same_thread_reentry_raises_while_other_threads_wait(plane, monkeypatch):
    """Code on a profile's boot path (its credential before the fetch, or anything during it) that
    reads that profile's config gets a loud ConfigBackendUnavailable at once instead of waiting on
    its own load forever; another thread still waits for the fetch in flight and shares it."""
    import threading
    import time

    from tests.waiting import wait_until

    backend = remote_pkg.get_remote_backend()
    home = plane.home
    real_token, real_request = backend_mod.client.plane_token, backend_mod.client.request
    errors: list = []
    other: dict = {}
    waiter = threading.Thread(target=lambda: other.setdefault("state", backend._state(home)), daemon=True)

    def reenter():
        started = time.monotonic()
        try:
            backend.version(home)
        except ConfigBackendUnavailable as exc:
            errors.append((str(exc), time.monotonic() - started))

    def token(h):
        if not errors:
            reenter()  # before the fetch lock
        return real_token(h)

    def request(method, *args, **kwargs):
        if method == "GET" and len(errors) == 1:
            reenter()  # under the fetch lock
            waiter.start()
            wait_until(lambda: _blocked_in_state(waiter), timeout=5, what="the other thread waits")
        return real_request(method, *args, **kwargs)

    monkeypatch.setattr(backend_mod.client, "plane_token", token)
    monkeypatch.setattr(backend_mod.client, "request", request)
    plane.upper = {"display": {"skin": "ares"}}
    assert backend.read_user_doc(home)["display"]["skin"] == "ares"
    assert len(errors) == 2 and all("re-entered" in msg and secs < 1.0 for msg, secs in errors)
    waiter.join(5)
    assert not waiter.is_alive() and other["state"] is backend._state(home)
    assert len(_gets(plane)) == 1  # the waiting thread used the fetch it waited for
    backend.version(home)  # once loaded, the same thread reads normally


def test_nothing_is_logged_under_a_backend_lock(plane, monkeypatch):
    """A log handler can read config, and a thread holding the handler's lock may be waiting for a
    backend lock: the boot retry warning and the write's inherited-key warning are both emitted
    with no fetch lock or profile state lock held."""
    import logging

    backend = remote_pkg.get_remote_backend()
    seen: list = []

    class Probe(logging.Handler):
        def emit(self, record):
            held = any(lk.locked() for lk in list(backend._fetch_locks.values()))
            held = held or any(st.lock._is_owned() for st in list(backend._states.values()))
            seen.append((record.getMessage(), held))

    real_request = backend_mod.client.request

    def first_get_fails(method, *args, **kwargs):
        resp = real_request(method, *args, **kwargs)
        plane.fail_status = None
        return resp

    monkeypatch.setattr(backend_mod.client, "request", first_get_fails)
    probe = Probe(level=logging.WARNING)
    backend_mod.logger.addHandler(probe)
    try:
        plane.upper = {"display": {"personality": "concise"}}
        plane.profile("default")["values"] = {"display": {"personality": "pirate"}}
        plane.fail_status = 503  # the first GET: a retryable outage, warned about before the retry
        write_config_key(plane.home / "config.yaml", "display.personality", None)
    finally:
        backend_mod.logger.removeHandler(probe)
    messages = [m for m, _ in seen]
    assert any("retrying" in m for m in messages) and any("still set by an upper level" in m for m in messages)
    assert [m for m, held in seen if held] == []


def test_boot_retry_warning_through_the_real_redacting_formatter(plane, monkeypatch):
    """The boot's retry warning goes through RedactingFormatter on a routed profile: formatting reads
    no config, so it neither re-enters the boot nor waits for a lock a config reader holds."""
    import logging

    from agent.redact import RedactingFormatter
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    backend = remote_pkg.get_remote_backend()
    formatted: list = []

    class Handler(logging.Handler):
        def emit(self, record):
            formatted.append(self.format(record))

    real_request = backend_mod.client.request

    def first_get_fails(method, *args, **kwargs):
        resp = real_request(method, *args, **kwargs)
        plane.fail_status = None
        return resp

    monkeypatch.setattr(backend_mod.client, "request", first_get_fails)
    handler = Handler(level=logging.WARNING)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    backend_mod.logger.addHandler(handler)
    plane.upper = {"display": {"skin": "ares"}}
    plane.fail_status = 503
    token = set_hermes_home_override(plane.home)
    try:
        assert backend.read_user_doc(plane.home)["display"]["skin"] == "ares"  # booted after one retry
    finally:
        reset_hermes_home_override(token)
        backend_mod.logger.removeHandler(handler)
    assert any("retrying" in m for m in formatted), formatted
    assert len(_gets(plane)) == 2


def test_the_formatter_never_reads_config(tmp_path, monkeypatch):
    """Under a backend whose every read raises, a routed profile's records still format, redacted
    (its policy is not recorded yet: the secure default), and no config read is attempted."""
    import logging

    import hermes_cli.config_backend as cb
    from agent import redact
    from agent.redact import RedactingFormatter
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

    reads: list = []

    def refuse(*args, **kwargs):
        reads.append(args)
        raise ConfigBackendUnavailable("this test's backend refuses every read")

    monkeypatch.setattr(cb, "_route", refuse)
    monkeypatch.setattr(redact, "_ENV_POLICY_BY_HOME", {})
    monkeypatch.setattr(redact, "_CONFIG_POLICY_BY_HOME", {})
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)  # the launch profile opted out
    secret = "sk-proj-" + "c" * 32
    record = logging.LogRecord("t", logging.WARNING, __file__, 1, "key %s", (secret,), None)
    token = set_hermes_home_override(tmp_path)
    try:
        out = RedactingFormatter("%(message)s").format(record)
        redact.note_env_policy(hermes_home_key(), None)
        redact.note_config_policy(hermes_home_key(), "false")  # recorded later: the profile opted out
        out_after = RedactingFormatter("%(message)s").format(record)
    finally:
        reset_hermes_home_override(token)
    assert secret not in out and secret in out_after
    assert reads == []
