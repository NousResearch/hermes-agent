"""The offline recovery errors are contract classes: callers catch them without importing a backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from persistence.core.admin import (
    PHYSICAL_ADMIN, AdminUnsupported, SessionRecoveryError, SessionRecoverySafetyError, SessionRecoverySourceError,
)


def test_the_recovery_errors_keep_their_runtime_error_family():
    # CONTRACT: callers catch the base for every recovery refusal and RuntimeError keeps its meaning.
    assert SessionRecoveryError.__bases__ == (RuntimeError,)
    assert SessionRecoverySafetyError.__bases__ == (SessionRecoveryError,)
    assert SessionRecoverySourceError.__bases__ == (SessionRecoveryError,)


def _admin(backend, backend_name, scope_a):
    admin = backend.open_store_admin(scope_a)
    if not backend.capabilities.physical_admin:
        # CONTRACT: a backend without physical admin answers the door, so recovery never runs there.
        assert admin == AdminUnsupported(PHYSICAL_ADMIN, backend_name)
        return None
    return admin


def test_recovery_refusals_raise_the_contract_classes(backend, backend_name, scope_a, tmp_path):
    admin = _admin(backend, backend_name, scope_a)
    if admin is None:
        return
    source = tmp_path / "state.db"
    source.write_bytes(b"")
    existing = tmp_path / "exists.db"
    existing.write_bytes(b"keep")
    not_a_dir = tmp_path / "work-file"
    not_a_dir.write_bytes(b"")
    cases = [
        (dict(source=source, output=tmp_path / "a.db", chunk_size=0), "chunk_size must be greater than zero"),
        (dict(source=tmp_path, output=tmp_path / "b.db"), "Source is not a file: "),
        (dict(source=source, output=source), "The recovery output must not be the source database"),
        (dict(source=source, output=Path(f"{source}-wal")), "The recovery output must not be the source database"),
        (dict(source=source, output=existing), f"Refusing to overwrite existing recovery output: {existing}"),
        (dict(source=source, output=tmp_path / "c.db", work_dir=not_a_dir), "Recovery work directory is not a directory: "),
        (dict(source=tmp_path, output=tmp_path / "d.db", inspect_only=True), "Source is not a file: "),
    ]
    for kwargs, message in cases:
        with pytest.raises(SessionRecoverySafetyError) as caught:
            admin.recover_transcripts(**kwargs)
        # CONTRACT: each path or overwrite guard refuses with the contract's safety class and its text.
        assert type(caught.value) is SessionRecoverySafetyError
        assert str(caught.value).startswith(message)
    assert existing.read_bytes() == b"keep"


def test_an_unrecoverable_source_raises_the_contract_source_error(backend, backend_name, scope_a, tmp_path):
    admin = _admin(backend, backend_name, scope_a)
    if admin is None:
        return
    source = tmp_path / "state.db"
    source.write_bytes(b"")
    output = tmp_path / "out.db"
    with pytest.raises(SessionRecoverySourceError) as caught:
        admin.recover_transcripts(source, output)
    # CONTRACT: a source without the readable canonical tables refuses with the source class and writes nothing.
    assert type(caught.value) is SessionRecoverySourceError
    assert str(caught.value).startswith("Required canonical tables are not readable: ")
    assert not output.exists()
