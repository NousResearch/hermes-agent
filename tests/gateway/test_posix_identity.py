from types import SimpleNamespace

import pytest

from gateway import posix_identity


def test_posix_identity_preserves_exact_host_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        posix_identity,
        "_OS",
        SimpleNamespace(
            getuid=lambda: 1201,
            getgid=lambda: 1202,
            geteuid=lambda: 1203,
            getegid=lambda: 1204,
        ),
    )

    assert posix_identity.real_uid() == 1201
    assert posix_identity.real_gid() == 1202
    assert posix_identity.effective_uid() == 1203
    assert posix_identity.effective_gid() == 1204


@pytest.mark.parametrize(
    "reader",
    (
        posix_identity.real_uid,
        posix_identity.real_gid,
        posix_identity.effective_uid,
        posix_identity.effective_gid,
    ),
)
def test_posix_identity_fails_closed_when_windows_getters_are_absent(
    monkeypatch: pytest.MonkeyPatch,
    reader,
) -> None:
    monkeypatch.setattr(posix_identity, "_OS", SimpleNamespace())

    with pytest.raises(
        posix_identity.PosixIdentityUnavailable,
        match="POSIX process identity is unavailable",
    ):
        reader()


@pytest.mark.parametrize("value", (True, -1, "1201", None))
def test_posix_identity_rejects_non_exact_identity_values(
    monkeypatch: pytest.MonkeyPatch,
    value,
) -> None:
    monkeypatch.setattr(
        posix_identity,
        "_OS",
        SimpleNamespace(getuid=lambda: value),
    )

    with pytest.raises(
        posix_identity.PosixIdentityUnavailable,
        match="POSIX process identity is invalid",
    ):
        posix_identity.real_uid()


def test_posix_identity_normalizes_os_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> int:
        raise OSError("identity backend unavailable")

    monkeypatch.setattr(
        posix_identity,
        "_OS",
        SimpleNamespace(geteuid=unavailable),
    )

    with pytest.raises(
        posix_identity.PosixIdentityUnavailable,
        match="POSIX process identity is unavailable",
    ):
        posix_identity.effective_uid()
