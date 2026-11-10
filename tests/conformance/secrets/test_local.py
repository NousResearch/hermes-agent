"""The secret store conformance suite against the local backend (today's profile ``.env`` and
``auth.json``).

Agents are profile homes under ``tmp_path``. The local backend holds the agent level only, so
only the pinning part of the suite runs here; the owner-wide and tenant-wide resolution invariants
run against a backend that holds every level (``test_reference.py``). The suite's list of
invariants it does NOT cover applies here too (``hermes_conformance.secrets.suite``).
"""

import json
import logging
import os
import traceback
from contextlib import contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from agent.secret_scope import load_env_file
from hermes_cli.auth import _load_auth_store
from hermes_conformance.secrets import Backend, SecretPinningConformance
from hermes_conformance.secrets.suite import (
    NEW_NAME, SEED_PROVIDER, WRITE_FAILURES, _plaintext_forms, assert_no_frame_holds,
)
from hermes_scope import LOCAL_OWNER, Group, Principal, Scope
from hermes_secret_port import SecretAdminRefused, SecretWriteFailed
from hermes_secret_port_local import LocalSecretStoreProvider

#: The key the managed-key hook pins, as an administrator would (the managed ``.env`` defines
#: it, so the write lock refuses changing it).
PINNED_KEY = "ZZ_ADMINISTRATOR_PINNED_KEY"


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`, and the profile homes it serves."""
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir(exist_ok=True)
    provider = LocalSecretStoreProvider()

    def bind_member(agent):
        return bind_state(Scope("local", str(homes[agent]), Principal("member"),
                                owner=LOCAL_OWNER, can_administer=False))

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"),
                                owner=Group("g"), can_administer=False))

    @contextmanager
    def unbound():
        # No scope: a local store's pin is the ambient profile, here one that owns no store.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    @contextmanager
    def managed_key():
        """While active, :data:`PINNED_KEY` is defined by the managed ``.env`` (the
        administrator's scope the write lock already honors), so writing it is refused with a
        plain ``ValueError`` — not the store's pin refusal."""
        managed = tmp_path / "managed"
        managed.mkdir(exist_ok=True)
        (managed / ".env").write_text(f"{PINNED_KEY}=set-by-the-administrator\n", encoding="utf-8")
        previous = os.environ.get("HERMES_MANAGED_DIR")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        try:
            yield PINNED_KEY
        finally:
            if previous is None:
                monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
            else:
                monkeypatch.setenv("HERMES_MANAGED_DIR", previous)

    @contextmanager
    def fail_writes(kind: str):
        """While active, the seeded agent's writes fail part-way through at the failure point
        *kind* names (``hermes_conformance.secrets.suite.WRITE_FAILURES``) — after the caller's
        value (or record) is bound on their frames — with a plain error, never the store's
        refusal. ``"prepare"`` breaks the store the write prepares against, before anything
        is serialized: today's unreadable directories, so the write's own load of the
        ``.env`` or ``auth.json`` it is about to change fails with an ``OSError`` (and today's
        auth loader logs the failed load with its live exception); ``"encode"`` breaks the
        serialization of the value or record into what gets written (today's env-line quoter
        with a ``ValueError``, today's JSON dumper with a ``TypeError``); ``"replace"`` breaks
        the atomic rename of the written file (an ``OSError`` in today's replace, both
        writers')."""
        if kind == "prepare":
            broken = []
            for filename in (".env", "auth.json"):
                path = homes["a"] / filename
                if path.is_dir():
                    path.rmdir()
                elif path.exists():
                    path.unlink()
                path.mkdir()
                broken.append(path)
            try:
                yield
            finally:
                for path in broken:
                    path.rmdir()
            return
        import utils
        from hermes_cli import config as env_writer
        if kind == "encode":
            def quoted(value: str):
                raise ValueError("the value cannot be written as an env line")

            def dumped(data, f, *, indent, ensure_ascii, dump_kwargs):
                raise TypeError("the record cannot be written as JSON")

            saved = (env_writer._quote_env_value, utils._dump_json)
            env_writer._quote_env_value, utils._dump_json = quoted, dumped
            try:
                yield
            finally:
                env_writer._quote_env_value, utils._dump_json = saved
        elif kind == "replace":
            def replaced(tmp_path, target):
                raise OSError("the written file could not be replaced atomically")

            saved = (env_writer.atomic_replace, utils.atomic_replace)
            env_writer.atomic_replace, utils.atomic_replace = replaced, replaced
            try:
                yield
            finally:
                env_writer.atomic_replace, utils.atomic_replace = saved
        else:
            raise LookupError(f"no such failure point: {kind!r}")

    backend = Backend(open=lambda agent: provider.open(homes[agent]),
                      bind=lambda agent: bind_profile_scope(homes[agent]),
                      bind_member=bind_member, foreign_tenant=foreign_tenant, unbound=unbound,
                      managed_key=managed_key, fail_writes=fail_writes,
                      levels=provider.levels)
    return backend, homes, provider


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    return backend


class TestLocalSecrets(SecretPinningConformance):
    pass


# --- today's semantics, unchanged ---------------------------------------------------------------


def test_a_set_succeeds_through_todays_env_file_and_reads_back(tmp_path, monkeypatch):
    backend, homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    with backend.bind("a"):
        store.set("ANTHROPIC_API_KEY", "sk-file-value", level="agent")
        assert store.get("ANTHROPIC_API_KEY") == "sk-file-value"
        assert "ANTHROPIC_API_KEY" in store.names()
    # Today's writer and today's tokenizer see the same file the port wrote.
    assert load_env_file(homes["a"] / ".env")["ANTHROPIC_API_KEY"] == "sk-file-value"
    with backend.bind("a"):
        assert store.delete("ANTHROPIC_API_KEY", level="agent") is True
        assert store.get("ANTHROPIC_API_KEY") is None
    assert "ANTHROPIC_API_KEY" not in load_env_file(homes["a"] / ".env")


def test_the_launch_profiles_env_only_credential_succeeds_for_the_process_home_only(tmp_path, monkeypatch):
    """A credential the launcher injected into the environment (no file to rebuild it from)
    resolves for the process's own home, exactly as the bound turn's secret scope does, and for
    no other profile."""
    backend, homes, _provider = local_backend(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(homes["a"]))  # a is this process's own home
    monkeypatch.setenv("ZZ_LAUNCHER_INJECTED_KEY", "injected-by-the-launcher")
    with backend.bind("a"):
        assert backend.open("a").get("ZZ_LAUNCHER_INJECTED_KEY") == "injected-by-the-launcher"
    with backend.bind("b"):
        assert backend.open("b").get("ZZ_LAUNCHER_INJECTED_KEY") is None


def test_an_auth_record_roundtrips_succeeds_through_todays_auth_json(tmp_path, monkeypatch):
    backend, homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    record = {"api_key": "sk-auth-value", "type": "api_key"}
    with backend.bind("a"):
        store.set_auth("openai", record)
        assert store.get_auth("openai") == record
    # Today's auth store file, written the way every local writer writes it.
    raw = _load_auth_store(homes["a"] / "auth.json")
    assert raw["providers"]["openai"] == record
    assert not raw.get("active_provider")  # a record write is not a provider choice
    with open(homes["a"] / "auth.json", encoding="utf-8") as handle:
        assert json.load(handle)["providers"]["openai"]["api_key"] == "sk-auth-value"


# --- a failed write publishes nothing an observer can keep --------------------------------------


class _KeepingHandler(logging.Handler):
    """An observer's logger: keeps every LogRecord that carries ``exc_info`` — the live
    exception, frames and all — for later rendering, the way a structured-log collector or a
    crash reporter does."""

    def __init__(self):
        super().__init__()
        self.exceptions: list[BaseException] = []

    def emit(self, record: logging.LogRecord) -> None:
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            self.exceptions.append(exc)


@pytest.mark.parametrize("kind", WRITE_FAILURES)
@pytest.mark.parametrize("operation", ["set", "set_auth"])
def test_a_failed_write_leaves_no_plaintext_in_a_log_handlers_kept_exception(tmp_path, monkeypatch, operation, kind):
    """Today's loaders log the failures they handle while a write is in flight: the local auth
    loader logs an unreadable store with the live exception, ``exc_info`` and all. A log handler
    that keeps that record keeps the exception itself, whose traceback grew through every frame
    the failure unwound — rendered later with frame locals, it must hold no trace of the value
    (or record) the caller asked to store: every load and validation that can fail runs before
    any frame binds it, and the write that does bind it calls nothing that logs, at whichever
    of its failure points the fixture breaks (the preparation, the serialization, the
    replace)."""
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    marker = 'zKept\nCaptured"\\Secret46'
    record = {"token": marker, "refresh": marker}
    kept = _KeepingHandler()
    logging.getLogger().addHandler(kept)
    try:
        with backend.bind("a"), backend.fail_writes(kind):
            if operation == "set":
                with pytest.raises(OSError):
                    store.set(NEW_NAME, marker, level="agent")
            else:
                with pytest.raises(OSError):
                    store.set_auth(SEED_PROVIDER, record)
    finally:
        logging.getLogger().removeHandler(kept)
    if operation == "set_auth" and kind == "prepare":
        # Today's auth loader logs the unreadable store with its live exception: the kept record
        # this test renders is the real artifact the fix answers, not a hypothesis. The other
        # failure points keep no record on the unbroken backend — nothing there logs — so a
        # passing check on them is exactly what the mutants that do log there must break.
        assert kept.exceptions, "the auth loader no longer logs the load failure with its exception"
    for kept_exc in kept.exceptions:
        rendered = traceback.TracebackException.from_exception(kept_exc, capture_locals=True)
        for frame in rendered.stack:
            locals_text = "\n".join((frame.locals or {}).values())
            for form in _plaintext_forms(marker):
                assert form not in locals_text, (
                    f"a kept log record's exception holds the plaintext in frame {frame.name!r} "
                    f"({frame.filename}:{frame.lineno})")


#: How an injected writer fails, each channel naming the value it saw: a plain exception's
#: message, an OSError's strerror, an OSError's filename, a refusal typed as the store's own,
#: an OSError whose errno slot holds whatever the failing code put there — Python allows a
#: string, so a writer that fails as ``OSError(value, ...)`` names the value in ``errno`` — and
#: an exception whose very class is named after the value (``type(value, (OSError,), {})``):
#: the class name is the failing code's own text too, and a fresh error that copies it names
#: the value as surely as one that copies a message.
_WRITER_FAILURES = {
    "plain": lambda value: ValueError("validation saw " + value),
    "strerror": lambda value: OSError(5, "writer saw " + value),
    "filename": lambda value: OSError(5, "I/O failed", value),
    "refusal": lambda value: SecretAdminRefused("plugin refused " + value),
    "string errno": lambda value: OSError(value, "write failed"),
    "class name": lambda value: type(value, (OSError,), {})(5, "the writer failed"),
}


@pytest.mark.parametrize("kind", sorted(_WRITER_FAILURES))
def test_a_failed_writes_fresh_error_names_none_of_the_writers_text(tmp_path, monkeypatch, kind):
    """A writer (or a backend plugin) that fails naming the value it saw — in its message, in
    an OSError's strerror, in an OSError's filename, in a refusal it typed as the store's own,
    in the errno slot of an OSError it built with the value's text, in an exception class it
    named after the value —
    puts nothing in the error the caller reads: whatever fails part-way through the write is
    re-raised as the port's own fresh error, whose message the port wrote — a fixed failure
    category (``os-error``, ``encode-error`` or ``write-error``) and, for an OSError, its
    errno number — never the failing code's own text, which nothing vouches free of the
    value, its class name included."""
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    marker = 'zFresh\nMessage"\\Secret53'

    def writer(name, value):
        raise _WRITER_FAILURES[kind](value)

    monkeypatch.setattr("hermes_cli.config.save_env_value", writer)
    with backend.bind("a"):
        with pytest.raises(SecretWriteFailed) as failed:
            store.set(NEW_NAME, marker, level="agent")
    assert failed.value.__context__ is None and failed.value.__cause__ is None
    text = str(failed.value) + repr(failed.value) + repr(failed.value.args) + "".join(
        str(arg) for arg in failed.value.args)
    for form in _plaintext_forms(marker):
        assert form not in text, f"the fresh error carries the writer's own text: {failed.value!r}"


def test_a_failed_auth_writes_fresh_error_names_none_of_the_writers_text(tmp_path, monkeypatch):
    """As the env writer's test, through the auth store's own writers: a refusal the merge's
    helper types as the store's own, and an OSError carrying the record in its strerror and
    filename, surface as the port's fresh error carrying none of the failing code's text."""
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    marker = 'zFresh\nAuth"\\Record61'
    record = {"token": marker, "refresh": marker}

    def refusing_merge(auth_store, provider_id, state, *, set_active=True):
        raise SecretAdminRefused("plugin refused " + marker)

    def failing_save(auth_store, target_path=None):
        raise OSError(5, "writer saw " + marker, marker)

    with backend.bind("a"):
        monkeypatch.setattr("hermes_cli.auth._store_provider_state", refusing_merge)
        with pytest.raises(SecretWriteFailed) as refused:
            store.set_auth(SEED_PROVIDER, record)
        monkeypatch.setattr("hermes_cli.auth._save_auth_store", failing_save)
        with pytest.raises(SecretWriteFailed) as failed:
            store.set_auth(SEED_PROVIDER, record)
    for raised in (refused.value, failed.value):
        assert raised.__context__ is None and raised.__cause__ is None
        text = str(raised) + repr(raised) + repr(raised.args) + "".join(str(arg) for arg in raised.args)
        for form in _plaintext_forms(marker):
            assert form not in text, f"the fresh error carries the writer's own text: {raised!r}"


def test_the_newline_stripped_form_of_a_short_marker_is_not_hunted():
    """The newline-stripped form of a short marker is a one- or two-character fragment that
    appears in unrelated text: the stripped form is hunted only for a marker at least 16
    characters long, so a frame holding 'cat' is not flagged by a marker of 'a\\n'."""
    assert "a" not in _plaintext_forms("a\n")
    long_marker = "z" * 15 + "\nq"
    assert "z" * 15 + "q" in _plaintext_forms(long_marker)

    def unrelated():
        label = "cat"
        raise ValueError(f"not a secret: {label}")

    with pytest.raises(ValueError) as failed:
        unrelated()
    assert_no_frame_holds(failed.value, "a\n")  # a short marker's stripped form flags nothing


# --- each failure point breaks the write where it claims to -------------------------------------


@pytest.mark.parametrize("kind", WRITE_FAILURES)
@pytest.mark.parametrize("operation", ["set", "set_auth"])
def test_each_failure_point_breaks_a_real_writer_seam(tmp_path, monkeypatch, kind, operation):
    """Each failure point the fixture can break must reach the real writer seam it names —
    the preparation in today's env writer or auth store loader, the serialization in today's
    env-line quoter or JSON dumper, the atomic replace in today's rename — so the
    no-plaintext tests exercise the write path where it actually works, never a stand-in a
    broken backend could satisfy."""
    import utils
    from hermes_cli import auth, config
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    seam = {
        ("set", "prepare"): (config, "save_env_value"),
        ("set_auth", "prepare"): (auth, "_load_auth_store"),
        ("set", "encode"): (config, "_quote_env_value"),
        ("set_auth", "encode"): (utils, "_dump_json"),
        ("set", "replace"): (config, "atomic_replace"),
        ("set_auth", "replace"): (utils, "atomic_replace"),
    }[operation, kind]
    reached = []
    with backend.bind("a"), backend.fail_writes(kind):
        original = getattr(*seam)

        def spy(*args, **kwargs):
            reached.append(True)
            return original(*args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(*seam, spy)
            with pytest.raises(SecretWriteFailed):
                if operation == "set":
                    store.set(NEW_NAME, "private-hook-marker", level="agent")
                else:
                    store.set_auth(SEED_PROVIDER, {"token": "private-hook-marker"})
    assert reached, f"the {kind!r} failure point never reached the writer seam it names"


@pytest.mark.parametrize("operation", ["set", "set_auth"])
def test_the_prepare_failure_point_fails_the_write_before_the_serialization(tmp_path, monkeypatch, operation):
    """The prepare failure point is a failure of the write's preparation — the store the
    write is about to change is unreadable — not a second serialization point: a healthy
    write reaches the serialization exactly once, and with the store broken the
    serialization is never reached at all, so the preparation and serialization failure
    paths cannot silently collapse into one (a hook that broke the write only at the
    serialization would pass a suite that believed it had tested the preparation). The
    serialization is watched through whatever the fixture installed on the seam, so a
    fixture that breaks the write there instead of the store is caught in the act."""
    import utils
    from hermes_cli import config
    backend, _homes, _provider = local_backend(tmp_path, monkeypatch)
    store = backend.open("a")
    seam = (config, "_quote_env_value") if operation == "set" else (utils, "_dump_json")
    calls = []

    def reached(call):
        def counting(*args, **kwargs):
            calls.append(True)
            return call(*args, **kwargs)
        return counting

    with backend.bind("a"):
        with monkeypatch.context() as scoped:
            scoped.setattr(*seam, reached(getattr(*seam)))
            if operation == "set":
                store.set("CONTROL", "works", level="agent")
            else:
                store.set_auth("probe", {"token": "works"})
            assert calls, "a healthy write never reached its serialization"
        calls.clear()
        with backend.fail_writes(WRITE_FAILURES[0]):
            # Whatever the fixture installed on the serialization seam (nothing, when it
            # truly broke the write's preparation) is wrapped, so a write that reaches the
            # serialization before failing is seen however the fixture broke it.
            with monkeypatch.context() as scoped:
                scoped.setattr(*seam, reached(getattr(*seam)))
                with pytest.raises(SecretWriteFailed):
                    if operation == "set":
                        store.set(NEW_NAME, "secret", level="agent")
                    else:
                        store.set_auth("probe", {"token": "secret"})
        assert calls == [], "the prepare failure point broke the write after the serialization"
