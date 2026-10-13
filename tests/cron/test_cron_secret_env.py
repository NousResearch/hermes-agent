"""Job-scoped ``secret_env`` for no_agent cron scripts: references, resolution, scrubbing, storage.

Only fake backends are used; the real Windows Credential Manager is never touched (the
WinCred tests drive a fake ``advapi32``).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading

import pytest

from agent.secret_sources.base import ErrorKind

FAKE_VALUE = "FAKE-SECRET-VALUE-0123456789"
OTHER_VALUE = "FAKE-OTHER-VALUE-9876543210"
TARGET = "hermes/notice-report-slack"


class FakeBackend:
    def __init__(self, values=None, error=None, delay=None):
        self.values = dict(values or {})
        self.error = error
        self.delay = delay
        self.calls: list[str] = []

    def resolve(self, target: str) -> str:
        self.calls.append(target)
        if self.delay is not None:
            self.delay.wait(5)
        if self.error is not None:
            raise self.error
        return self.values[target]


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib

    import cron.jobs
    import hermes_constants
    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    return home


def _job(mapping, **extra):
    return {"id": "j1", "no_agent": True, "script": "s.py", "secret_env": mapping, **extra}


# ---------------------------------------------------------------------------
# validate_secret_env
# ---------------------------------------------------------------------------


def test_validate_accepts_and_normalizes_references():
    from cron.secret_env import validate_secret_env

    result = validate_secret_env({"NOTICE_REPORT_SLACK_TOKEN": f"  wincred://{TARGET} "})
    assert result == {"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"}


@pytest.mark.parametrize("empty", [None, {}])
def test_validate_empty_mapping_means_no_secret_env(empty):
    from cron.secret_env import validate_secret_env

    assert validate_secret_env(empty) is None


@pytest.mark.parametrize("name", ["lower_case", "1LEADING_DIGIT", "A" * 65, "HAS-DASH", "", "_LEADING"])
def test_validate_rejects_malformed_env_names(name):
    from cron.secret_env import validate_secret_env

    with pytest.raises(ValueError):
        validate_secret_env({name: f"wincred://{TARGET}"})


@pytest.mark.parametrize("name", [
    "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY", "PATH", "PATHEXT", "SYSTEMROOT",
    "HOME", "USERPROFILE", "TEMP", "VIRTUAL_ENV", "HERMES_HOME", "OP_SERVICE_ACCOUNT_TOKEN",
    "BWS_ACCESS_TOKEN", "PYTHONPATH", "PYTHONHOME", "GH_TOKEN", "AUXILIARY_VISION_API_KEY",
    "GATEWAY_RELAY_SECRET", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "NODE_OPTIONS", "BASH_ENV",
])
def test_validate_rejects_reserved_env_names(name):
    from cron.secret_env import validate_secret_env

    with pytest.raises(ValueError, match="reserved"):
        validate_secret_env({name: f"wincred://{TARGET}"})


def test_validate_allows_the_notice_report_token_name():
    from cron.secret_env import validate_secret_env

    assert validate_secret_env({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})


@pytest.mark.parametrize("ref", [
    "xoxb-1234-5678-abcdefghijklmnop",      # a raw token: no scheme
    FAKE_VALUE,
    "op://vault/item/field",                # unregistered scheme
    "wincred://",                           # empty target
    "wincred://has space",
    "wincred://" + "a" * 129,
    "wincred://bad\\backslash",
    "WINCRED://" + TARGET,
])
def test_validate_rejects_raw_values_and_bad_references(ref):
    from cron.secret_env import validate_secret_env

    with pytest.raises(ValueError) as excinfo:
        validate_secret_env({"NOTICE_REPORT_SLACK_TOKEN": ref})
    # The error names the variable, never echoes what was typed (it may be the token itself);
    # a bare "wincred://" legitimately appears in the format hint.
    assert ref == "wincred://" or ref not in str(excinfo.value)
    assert "NOTICE_REPORT_SLACK_TOKEN" in str(excinfo.value)


def test_validate_rejects_more_than_eight_entries():
    from cron.secret_env import validate_secret_env

    mapping = {f"JOB_SECRET_{i}": f"wincred://{TARGET}{i}" for i in range(9)}
    with pytest.raises(ValueError, match="8"):
        validate_secret_env(mapping)
    del mapping["JOB_SECRET_8"]
    assert len(validate_secret_env(mapping)) == 8


@pytest.mark.parametrize("bad", ["not-a-dict", ["A=wincred://x"], {"A": 1}])
def test_validate_rejects_non_mapping_shapes(bad):
    from cron.secret_env import validate_secret_env

    with pytest.raises(ValueError):
        validate_secret_env(bad)


# ---------------------------------------------------------------------------
# resolve_job_secret_env
# ---------------------------------------------------------------------------


def test_resolve_returns_every_value_when_all_succeed():
    from cron.secret_env import resolve_job_secret_env

    backend = FakeBackend({TARGET: FAKE_VALUE, "hermes/other": OTHER_VALUE})
    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}",
                "JOB_OTHER_TOKEN": "wincred://hermes/other"})
    assert resolve_job_secret_env(job, backends={"wincred": backend}) == {
        "NOTICE_REPORT_SLACK_TOKEN": FAKE_VALUE, "JOB_OTHER_TOKEN": OTHER_VALUE}


def test_resolve_is_all_or_nothing():
    from cron.secret_env import SecretResolveError, resolve_job_secret_env

    backend = FakeBackend({TARGET: FAKE_VALUE})
    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}",
                "JOB_MISSING_TOKEN": "wincred://hermes/missing"})
    with pytest.raises(SecretResolveError) as excinfo:
        resolve_job_secret_env(job, backends={"wincred": backend})
    assert excinfo.value.env_name == "JOB_MISSING_TOKEN"


@pytest.mark.parametrize("value, kind", [("", ErrorKind.EMPTY_VALUE), ("  \r\n", ErrorKind.EMPTY_VALUE)])
def test_resolve_rejects_empty_values(value, kind):
    from cron.secret_env import SecretResolveError, resolve_job_secret_env

    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    with pytest.raises(SecretResolveError) as excinfo:
        resolve_job_secret_env(job, backends={"wincred": FakeBackend({TARGET: value})})
    assert excinfo.value.kind is kind


def test_resolve_maps_backend_errors_and_unexpected_exceptions():
    from cron.secret_env import SecretBackendError, SecretResolveError, resolve_job_secret_env

    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    for error, kind in (
        (SecretBackendError(ErrorKind.REF_INVALID), ErrorKind.REF_INVALID),
        (SecretBackendError(ErrorKind.NOT_CONFIGURED), ErrorKind.NOT_CONFIGURED),
        (RuntimeError(f"boom {FAKE_VALUE} {TARGET}"), ErrorKind.INTERNAL),
    ):
        with pytest.raises(SecretResolveError) as excinfo:
            resolve_job_secret_env(job, backends={"wincred": FakeBackend(error=error)})
        assert excinfo.value.kind is kind
        _assert_safe(excinfo.value)


def test_resolve_times_out_fail_closed():
    from cron.secret_env import SecretResolveError, resolve_job_secret_env

    release = threading.Event()
    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    backend = FakeBackend({TARGET: FAKE_VALUE}, delay=release)
    try:
        with pytest.raises(SecretResolveError) as excinfo:
            resolve_job_secret_env(job, backends={"wincred": backend}, timeout=0.2)
    finally:
        release.set()
    assert excinfo.value.kind is ErrorKind.TIMEOUT


def test_resolve_revalidates_hand_edited_references():
    from cron.secret_env import SecretResolveError, resolve_job_secret_env

    backend = FakeBackend({TARGET: FAKE_VALUE})
    for mapping, label in (
        ({"NOTICE_REPORT_SLACK_TOKEN": FAKE_VALUE}, "NOTICE_REPORT_SLACK_TOKEN"),
        ({"NOTICE_REPORT_SLACK_TOKEN": f"fake://{TARGET}"}, "NOTICE_REPORT_SLACK_TOKEN"),
        ({"SLACK_BOT_TOKEN": f"wincred://{TARGET}"}, "SLACK_BOT_TOKEN"),
        ({FAKE_VALUE.lower(): f"wincred://{TARGET}"}, "<invalid name>"),
    ):
        with pytest.raises(SecretResolveError) as excinfo:
            resolve_job_secret_env(_job(mapping), backends={"wincred": backend})
        assert excinfo.value.kind is ErrorKind.REF_INVALID
        assert excinfo.value.env_name == label
        _assert_safe(excinfo.value)
    assert backend.calls == []


def test_resolve_never_caches():
    from cron.secret_env import resolve_job_secret_env

    backend = FakeBackend({TARGET: FAKE_VALUE})
    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    resolve_job_secret_env(job, backends={"wincred": backend})
    resolve_job_secret_env(job, backends={"wincred": backend})
    assert backend.calls == [TARGET, TARGET]


def test_resolve_error_message_carries_only_name_and_kind():
    from cron.secret_env import SecretResolveError, resolve_job_secret_env

    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    with pytest.raises(SecretResolveError) as excinfo:
        resolve_job_secret_env(job, backends={"wincred": FakeBackend({})})
    assert str(excinfo.value) == "Secret env resolution failed: NOTICE_REPORT_SLACK_TOKEN (internal)"


def _assert_safe(exc: BaseException) -> None:
    text = f"{exc} {exc!r} {exc.args}"
    assert FAKE_VALUE not in text and TARGET not in text
    # No chained backend exception (its text may name the target) in a logged traceback.
    assert exc.__cause__ is None and (exc.__context__ is None or exc.__suppress_context__)


# ---------------------------------------------------------------------------
# Script digest pin
# ---------------------------------------------------------------------------


@pytest.mark.platforms("windows")
def test_resolve_for_script_checks_digest_before_any_backend_call(tmp_path):
    from cron.secret_env import SecretResolveError, resolve_for_script

    script = tmp_path / "s.py"
    script.write_text("print('pinned')\n")
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    backend = FakeBackend({TARGET: FAKE_VALUE})
    job = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"}, secret_env_script_sha256=digest)

    with resolve_for_script(job, script, backends={"wincred": backend}) as values:
        assert values == {"NOTICE_REPORT_SLACK_TOKEN": FAKE_VALUE}

    script.write_text("print('tampered')\n")
    backend.calls.clear()
    with pytest.raises(SecretResolveError, match="repin-script"):
        with resolve_for_script(job, script, backends={"wincred": backend}):
            pass
    assert backend.calls == []

    unpinned = _job({"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    with pytest.raises(SecretResolveError, match="repin-script"):
        with resolve_for_script(unpinned, script, backends={"wincred": backend}):
            pass
    assert backend.calls == []


# ---------------------------------------------------------------------------
# scrub_secret_values
# ---------------------------------------------------------------------------


def test_scrub_replaces_exact_and_line_variants_longest_first():
    from cron.secret_env import SECRET_ENV_REDACTION, scrub_secret_values

    short, long_value = "FAKE-SECRET", FAKE_VALUE
    text = f"a={long_value} b={short} c={FAKE_VALUE[:5]}"
    scrubbed = scrub_secret_values(text, [short, long_value])
    assert long_value not in scrubbed and short not in scrubbed
    assert scrubbed.count(SECRET_ENV_REDACTION) == 2
    assert scrub_secret_values(f"x{FAKE_VALUE}y", [FAKE_VALUE + "\r\n"]) == f"x{SECRET_ENV_REDACTION}y"
    assert scrub_secret_values("abc", ["", "  "]) == "abc"
    assert scrub_secret_values("abc", []) == "abc"


# ---------------------------------------------------------------------------
# WinCredBackend (fake advapi32 — the real credential store is never read)
# ---------------------------------------------------------------------------


class _FakeAdvapi:
    def __init__(self, blob: bytes | None, error: int = 0):
        self.blob = blob
        self.error = error
        self.freed: list[int] = []
        self.read_args: list[tuple] = []
        self._keep: list = []

    def CredReadW(self, target, cred_type, flags, pcred_ref):
        from cron.secret_env_wincred import CREDENTIALW

        self.read_args.append((target, cred_type, flags))
        if self.blob is None:
            return False
        buf = ctypes.create_string_buffer(self.blob, len(self.blob)) if self.blob else None
        cred = CREDENTIALW()
        cred.CredentialBlobSize = len(self.blob)
        cred.CredentialBlob = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)) if buf else None
        self._keep += [buf, cred]
        pcred_ref[0] = ctypes.pointer(cred)
        return True

    def CredFree(self, pointer):
        self.freed.append(1)


def _wincred(fake: _FakeAdvapi, monkeypatch):
    from cron import secret_env_wincred

    monkeypatch.setattr(secret_env_wincred, "_load_advapi32", lambda: fake)
    monkeypatch.setattr(secret_env_wincred, "_last_error", lambda: fake.error)
    return secret_env_wincred.WinCredBackend()


def test_wincred_decodes_utf16le_and_frees(monkeypatch):
    fake = _FakeAdvapi(FAKE_VALUE.encode("utf-16-le"))
    assert _wincred(fake, monkeypatch).resolve(TARGET) == FAKE_VALUE
    assert fake.read_args == [(TARGET, 1, 0)]
    assert fake.freed == [1]


@pytest.mark.parametrize("blob, error, kind", [
    (None, 1168, ErrorKind.REF_INVALID),           # ERROR_NOT_FOUND
    (None, 5, ErrorKind.INTERNAL),                 # anything else
    (b"\x00\xd8", 0, ErrorKind.INTERNAL),          # lone surrogate: not valid UTF-16LE
    (b"abc", 0, ErrorKind.INTERNAL),               # odd length
    (b"", 0, ErrorKind.EMPTY_VALUE),
])
def test_wincred_failures_are_classified(monkeypatch, blob, error, kind):
    from cron.secret_env import SecretBackendError

    fake = _FakeAdvapi(blob, error)
    with pytest.raises(SecretBackendError) as excinfo:
        _wincred(fake, monkeypatch).resolve(TARGET)
    assert excinfo.value.kind is kind
    assert TARGET not in str(excinfo.value)
    assert fake.freed == ([] if blob is None else [1])


@pytest.mark.platforms("not windows")
def test_wincred_is_not_configured_off_windows():
    from cron.secret_env import SecretBackendError
    from cron.secret_env_wincred import WinCredBackend

    with pytest.raises(SecretBackendError) as excinfo:
        WinCredBackend().resolve(TARGET)
    assert excinfo.value.kind is ErrorKind.NOT_CONFIGURED


# ---------------------------------------------------------------------------
# jobs layer
# ---------------------------------------------------------------------------

_DIGEST = "a" * 64


def _create(hermes_env, **kw):
    from cron.jobs import create_job

    (hermes_env / "scripts" / "s.py").write_text("print('x')\n")
    kw.setdefault("secret_env", {"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"})
    kw.setdefault("secret_env_script_sha256", _DIGEST)
    return create_job(prompt=None, schedule="every 5m", script="s.py", no_agent=True, **kw)


def test_create_job_stores_only_references(hermes_env):
    job = _create(hermes_env)
    raw = (hermes_env / "cron" / "jobs.json").read_text(encoding="utf-8")
    stored = json.loads(raw)["jobs"][0]
    assert stored["secret_env"] == {"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"}
    assert stored["secret_env_script_sha256"] == _DIGEST
    assert job["secret_env"] == stored["secret_env"]
    assert FAKE_VALUE not in raw


def test_create_job_rejects_raw_value_before_storing(hermes_env):
    with pytest.raises(ValueError) as excinfo:
        _create(hermes_env, secret_env={"NOTICE_REPORT_SLACK_TOKEN": FAKE_VALUE})
    assert FAKE_VALUE not in str(excinfo.value)
    jobs_file = hermes_env / "cron" / "jobs.json"
    assert not jobs_file.exists() or FAKE_VALUE not in jobs_file.read_text(encoding="utf-8")


def test_secret_env_requires_no_agent_and_a_pin(hermes_env):
    from cron.jobs import create_job, update_job

    (hermes_env / "scripts" / "s.py").write_text("print('x')\n")
    with pytest.raises(ValueError, match="no_agent"):
        create_job(prompt="p", schedule="every 5m", script="s.py",
                   secret_env={"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"},
                   secret_env_script_sha256=_DIGEST)
    with pytest.raises(ValueError, match="pin"):
        _create(hermes_env, secret_env_script_sha256=None)
    with pytest.raises(ValueError):
        _create(hermes_env, secret_env_script_sha256="not-hex")

    job = _create(hermes_env)
    with pytest.raises(ValueError, match="no_agent"):
        update_job(job["id"], {"no_agent": False})
    plain = create_job(prompt="p", schedule="every 5m")
    with pytest.raises(ValueError, match="no_agent"):
        update_job(plain["id"], {"secret_env": {"NOTICE_REPORT_SLACK_TOKEN": f"wincred://{TARGET}"},
                                 "secret_env_script_sha256": _DIGEST})


def test_jobs_without_secret_env_keep_their_stored_shape(hermes_env):
    from cron.jobs import create_job, update_job

    job = create_job(prompt="p", schedule="every 5m")
    update_job(job["id"], {"name": "renamed"})
    stored = json.loads((hermes_env / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"][0]
    assert "secret_env" not in stored and "secret_env_script_sha256" not in stored


def test_update_job_adds_replaces_and_clears(hermes_env):
    from cron.jobs import get_job, update_job

    job = _create(hermes_env)
    update_job(job["id"], {"secret_env": {"NOTICE_REPORT_SLACK_TOKEN": "wincred://hermes/new",
                                          "JOB_OTHER_TOKEN": "wincred://hermes/other"}})
    assert get_job(job["id"])["secret_env"] == {
        "NOTICE_REPORT_SLACK_TOKEN": "wincred://hermes/new", "JOB_OTHER_TOKEN": "wincred://hermes/other"}
    with pytest.raises(ValueError):
        update_job(job["id"], {"secret_env": {"NOTICE_REPORT_SLACK_TOKEN": FAKE_VALUE}})

    update_job(job["id"], {"secret_env": None})
    stored = json.loads((hermes_env / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"][0]
    assert "secret_env" not in stored and "secret_env_script_sha256" not in stored


def test_unknown_keys_survive_load_and_normalize(hermes_env):
    from cron.jobs import get_job, load_jobs

    job = _create(hermes_env)
    assert load_jobs()[0]["secret_env"] == job["secret_env"]
    assert get_job(job["id"])["secret_env_script_sha256"] == _DIGEST


def test_profile_distribution_neither_carries_nor_overwrites_references(hermes_env):
    from cron.job_definition import merge_job_definition

    local = _create(hermes_env)
    authored = {k: v for k, v in local.items() if k not in {"secret_env", "secret_env_script_sha256"}}
    authored["secret_env"] = {"NOTICE_REPORT_SLACK_TOKEN": "wincred://hermes/attacker"}
    authored["secret_env_script_sha256"] = "b" * 64
    merged = merge_job_definition(local, authored)
    assert merged["secret_env"] == local["secret_env"]
    assert merged["secret_env_script_sha256"] == _DIGEST

    fresh = merge_job_definition({"id": "new", "enabled": False, "state": "paused"}, authored)
    assert "secret_env" not in fresh and "secret_env_script_sha256" not in fresh


def test_retargeting_a_secret_env_job_requires_a_fresh_pin(hermes_env, tmp_path):
    """The pin covers the script file; changing which program runs it (interpreter), where it
    runs (workdir) or which script it is must come with an explicit repin, from any door."""
    from cron.jobs import get_job, update_job

    job = _create(hermes_env)
    (hermes_env / "scripts" / "other.py").write_text("print('o')\n")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    for updates in ({"interpreter": "C:/elsewhere/python.exe"}, {"workdir": str(workdir)},
                    {"script": "other.py"}):
        with pytest.raises(ValueError, match="repin-script"):
            update_job(job["id"], dict(updates))
        assert get_job(job["id"]).get(next(iter(updates))) == job.get(next(iter(updates)))

    update_job(job["id"], {"interpreter": "C:/elsewhere/python.exe",
                           "secret_env_script_sha256": "c" * 64})
    assert get_job(job["id"])["interpreter"] == "C:/elsewhere/python.exe"
    # Re-sending an unchanged value is not a retarget.
    update_job(job["id"], {"script": "s.py", "name": "renamed"})


@pytest.mark.parametrize("name", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY", "GIT_SSH_COMMAND",
                                  "NODE_PATH", "PERL5LIB", "JAVA_TOOL_OPTIONS", "PROMPT_COMMAND", "IFS"])
def test_validate_rejects_more_runtime_steering_names(name):
    from cron.secret_env import validate_secret_env

    with pytest.raises(ValueError, match="reserved"):
        validate_secret_env({name: f"wincred://{TARGET}"})
