"""Job-scoped ``secret_env`` boundaries (CORP-764 formal review): the program that receives the
values is the one that was verified, profile distribution cannot retarget a protected job,
every outward view shows only validated names and supported schemes, and a stuck backend
cannot pile up threads.

Real ``run_job`` against a temp ``HERMES_HOME``; the only fake is the ``wincred`` backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

import pytest

from agent.secret_sources.base import ErrorKind

FAKE_VALUE = "FAKE-SECRET-VALUE-0123456789"
TARGET = "hermes/notice-report-slack"
NAME = "NOTICE_REPORT_SLACK_TOKEN"

_PINNED_SCRIPT = f"""\
import os, pathlib, sys
pathlib.Path(sys.argv[0]).with_suffix(".started").write_text("1")
print("pinned-program " + str(len(os.environ.get("{NAME}", ""))))
"""
_EVIL_SCRIPT = f"""\
import os
print("EVIL " + os.environ.get("{NAME}", ""))
"""


class FakeBackend:
    def __init__(self, during=None):
        self.calls: list[str] = []
        self.during = during

    def resolve(self, target):
        self.calls.append(target)
        if self.during is not None:
            self.during()
        return FAKE_VALUE


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib

    import cron.jobs
    import cron.scheduler
    import hermes_constants
    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    importlib.reload(cron.scheduler)
    return home


def _install(monkeypatch, backend):
    from cron import secret_env

    monkeypatch.setattr(secret_env, "_default_backends", lambda: {"wincred": backend})
    return backend


def _create(home, body=_PINNED_SCRIPT, name="pinned.py", **kw):
    from cron.jobs import create_job

    script = home / "scripts" / name
    script.write_text(body, encoding="utf-8")
    kw.setdefault("secret_env", {NAME: f"wincred://{TARGET}"})
    kw.setdefault("secret_env_script_sha256", hashlib.sha256(script.read_bytes()).hexdigest())
    return create_job(prompt=None, schedule="every 5m", script=name, no_agent=True,
                      deliver="local", **kw), script


# ---------------------------------------------------------------------------
# Finding 1: the verified script is the one that runs
# ---------------------------------------------------------------------------


@pytest.mark.platforms("windows")
def test_script_cannot_be_swapped_while_secrets_resolve(hermes_env, monkeypatch):
    """A concurrent same-user writer gets its chance exactly while the backend resolves (the
    window between the pin check and Popen). Whatever it tries, the pinned program runs."""
    from cron.scheduler import run_job

    attempts: dict[str, str] = {}
    script_holder: dict[str, object] = {}

    def tamper():
        script = script_holder["path"]
        decoy = script.with_name("decoy.py")
        decoy.write_text(_EVIL_SCRIPT, encoding="utf-8")
        for label, action in (("overwrite", lambda: script.write_text(_EVIL_SCRIPT, encoding="utf-8")),
                              ("replace", lambda: os.replace(decoy, script)),
                              ("rename-file", lambda: os.rename(script, script.with_name("moved.py"))),
                              ("unlink", lambda: os.unlink(script)),
                              ("rename-dir", lambda: os.rename(script.parent, script.parent.with_name("moved")))):
            try:
                action()
                attempts[label] = "succeeded"
            except OSError:
                attempts[label] = "blocked"

    backend = _install(monkeypatch, FakeBackend(during=tamper))
    job, script = _create(hermes_env)
    script_holder["path"] = script
    original = script.read_bytes()

    success, _doc, final_response, error = run_job(job)

    assert backend.calls == [TARGET]
    assert attempts == dict.fromkeys(("overwrite", "replace", "rename-file", "unlink", "rename-dir"), "blocked")
    assert "EVIL" not in final_response and FAKE_VALUE not in final_response
    assert success is True and error is None
    assert f"pinned-program {len(FAKE_VALUE)}" in final_response
    # The lock lasts for the run only: afterwards the user can edit the script again.
    assert script.read_bytes() == original
    script.write_text("print('edited later')\n", encoding="utf-8")


@pytest.mark.platforms("windows")
def test_script_reached_through_a_junction_fails_closed(tmp_path):
    """The lock protects the file it opened and that file's real folders. Through a junction
    those are not the folders Popen later walks, so the junction could be swapped back to a
    folder holding other code: a path that is not the locked file's own path never resolves."""
    import _winapi

    from cron.secret_env import SecretResolveError, resolve_for_script

    real = tmp_path / "real"
    real.mkdir()
    script = real / "s.py"
    script.write_text(_PINNED_SCRIPT, encoding="utf-8")
    _winapi.CreateJunction(str(real), str(tmp_path / "link"))
    backend = FakeBackend()
    job = {"id": "j", "no_agent": True, "secret_env": {NAME: f"wincred://{TARGET}"},
           "secret_env_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}

    with pytest.raises(SecretResolveError) as excinfo:
        with resolve_for_script(job, tmp_path / "link" / "s.py", backends={"wincred": backend}):
            pass
    assert backend.calls == []
    assert TARGET not in str(excinfo.value)
    with resolve_for_script(job, script, backends={"wincred": backend}) as values:
        assert values == {NAME: FAKE_VALUE}


@pytest.mark.platforms("windows")
def test_script_held_open_for_writing_fails_closed(hermes_env, monkeypatch):
    """A writer that already holds the script open could change it under the pin: the run must
    not resolve anything, and the script must not start."""
    from cron.scheduler import run_job

    backend = _install(monkeypatch, FakeBackend())
    job, script = _create(hermes_env)
    with open(script, "r+b"):
        success, _doc, final_response, error = run_job(job)
    assert success is False
    assert backend.calls == []
    assert not script.with_suffix(".started").exists()
    assert TARGET not in f"{final_response} {error}"


@pytest.mark.platforms("posix")
def test_secret_env_fails_closed_where_the_script_cannot_be_locked(hermes_env, monkeypatch):
    """POSIX has no way to stop a same-user writer replacing the script between the pin check
    and exec, so a secret_env job is never resolved there."""
    from cron.scheduler import run_job

    backend = _install(monkeypatch, FakeBackend())
    job, script = _create(hermes_env)
    success, _doc, _final, error = run_job(job)
    assert success is False and error
    assert backend.calls == []
    assert not script.with_suffix(".started").exists()


# ---------------------------------------------------------------------------
# Finding 2: profile distribution cannot retarget a protected job
# ---------------------------------------------------------------------------


def _protected_local(hermes_env, tmp_path):
    from cron.jobs import update_job

    job, _script = _create(hermes_env)
    workdir = tmp_path / "wd"
    workdir.mkdir()
    return update_job(job["id"], {"workdir": str(workdir), "interpreter": "C:/py/python.exe",
                                  "secret_env_script_sha256": job["secret_env_script_sha256"]})


_PINNED_KEYS = ("script", "no_agent", "workdir", "interpreter", "secret_env", "secret_env_script_sha256")


def test_distribution_cannot_change_a_protected_jobs_target(hermes_env, tmp_path):
    from cron.job_definition import merge_job_definition

    local = _protected_local(hermes_env, tmp_path)
    authored = {**local, "name": "renamed upstream", "script": "other.py", "no_agent": False,
                "workdir": str(tmp_path), "interpreter": "C:/evil/python.exe",
                "secret_env": {NAME: "wincred://hermes/attacker"},
                "secret_env_script_sha256": "b" * 64}
    merged = merge_job_definition(local, authored)
    assert {k: merged.get(k) for k in _PINNED_KEYS} == {k: local.get(k) for k in _PINNED_KEYS}
    assert merged["name"] == "renamed upstream"


def test_distribution_cannot_clear_a_protected_jobs_target(hermes_env, tmp_path):
    from cron.job_definition import merge_job_definition

    local = _protected_local(hermes_env, tmp_path)
    authored = {k: v for k, v in local.items() if k not in _PINNED_KEYS}
    authored["name"] = "renamed upstream"
    merged = merge_job_definition(local, authored)
    assert {k: merged.get(k) for k in _PINNED_KEYS} == {k: local.get(k) for k in _PINNED_KEYS}
    assert merged["name"] == "renamed upstream"


def test_distribution_still_retargets_jobs_without_secret_env(hermes_env, tmp_path):
    from cron.job_definition import merge_job_definition
    from cron.jobs import create_job

    (hermes_env / "scripts" / "plain.py").write_text("print('p')\n")
    local = create_job(prompt=None, schedule="every 5m", script="plain.py", no_agent=True)
    merged = merge_job_definition(local, {**local, "script": "other.py", "workdir": str(tmp_path)})
    assert merged["script"] == "other.py" and merged["workdir"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Finding 3: outward views show validated names and supported schemes only
# ---------------------------------------------------------------------------

_MIXED = {
    NAME: f"wincred://{TARGET}",
    "SLACK_BOT_TOKEN": "wincred://hermes/reserved-name",      # reserved name
    "lower_name": "wincred://hermes/malformed-name",           # malformed name
    "GOOD_NAME": "fake://hermes/unsupported-scheme",          # unsupported scheme
    "RAW_VALUE": FAKE_VALUE,                                    # not a reference at all
}
_TARGETS = ("hermes/reserved-name", "hermes/malformed-name", "hermes/unsupported-scheme", TARGET)


def _assert_no_target(text: str) -> None:
    for secret in (*_TARGETS, FAKE_VALUE, "wincred://", "fake://"):
        assert secret not in text


def test_display_marks_every_invalid_entry_without_detail():
    from cron.secret_env import secret_env_display

    shown = secret_env_display(_MIXED)
    assert shown == [f"{NAME} (wincred)"] + ["<invalid entry>"] * 4
    _assert_no_target(json.dumps(shown))
    assert secret_env_display("wincred://x") == ["<invalid entry>"]
    assert secret_env_display(None) == []


def test_public_job_view_keeps_storage_and_hides_targets():
    from cron.secret_env import public_job_view

    job = {"id": "j1", "name": "n", "secret_env": dict(_MIXED), "secret_env_script_sha256": "a" * 64}
    view = public_job_view(job)
    assert view["secret_env"] == [f"{NAME} (wincred)"] + ["<invalid entry>"] * 4
    assert view["name"] == "n"
    _assert_no_target(json.dumps(view))
    assert job["secret_env"] == _MIXED  # the stored record is untouched
    plain = {"id": "j2", "name": "p"}
    assert public_job_view(plain) == plain


def test_tool_view_hides_invalid_entries():
    from tools.cronjob_job_args import _format_job

    formatted = _format_job({"id": "j1", "no_agent": True, "script": "s.py", "secret_env": dict(_MIXED)})
    assert formatted["secret_env"] == [f"{NAME} (wincred)"] + ["<invalid entry>"] * 4
    _assert_no_target(json.dumps(formatted))


# ---------------------------------------------------------------------------
# Non-blocking: a stuck backend cannot pile up threads or hold shutdown
# ---------------------------------------------------------------------------


def test_stuck_backend_threads_are_bounded_and_daemonic():
    from cron.secret_env import SECRET_ENV_MAX_STUCK_RESOLVERS, SecretResolveError, resolve_job_secret_env

    release = threading.Event()
    calls: list[str] = []

    class Stuck:
        def resolve(self, target):
            calls.append(target)
            release.wait(30)
            return FAKE_VALUE

    for leftover in [t for t in threading.enumerate() if t.name.startswith("cron-secret-env")]:
        leftover.join(10)  # an earlier test's timed-out call still counts as stuck until it ends
    job = {"id": "j", "no_agent": True, "secret_env": {NAME: f"wincred://{TARGET}"}}
    try:
        for _ in range(SECRET_ENV_MAX_STUCK_RESOLVERS + 3):
            with pytest.raises(SecretResolveError) as excinfo:
                resolve_job_secret_env(job, backends={"wincred": Stuck()}, timeout=0.05)
            assert excinfo.value.kind is ErrorKind.TIMEOUT
        workers = [t for t in threading.enumerate() if t.name.startswith("cron-secret-env")]
        assert len(workers) <= SECRET_ENV_MAX_STUCK_RESOLVERS
        assert all(t.daemon for t in workers)
        assert len(calls) == SECRET_ENV_MAX_STUCK_RESOLVERS
    finally:
        release.set()
    for t in [t for t in threading.enumerate() if t.name.startswith("cron-secret-env")]:
        t.join(5)
    # Once the stuck calls return, resolution works again.
    assert resolve_job_secret_env(job, backends={"wincred": FakeBackend()}, timeout=1) == {NAME: FAKE_VALUE}
