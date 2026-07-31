"""Read/write guard coverage for the credential residence.

A genuinely distinct ``HERMES_AUTH_HOME`` is one protected tree — root,
active and sibling profiles, shared Nous store, lock, corrupt-quarantine, and
atomic-write temp files. An ordinary no-override ``HERMES_HOME/auth.json``
stays requested-writable, and a path-equal override changes nothing.
"""

from __future__ import annotations

import json

import pytest

from agent.file_safety import (
    build_write_denied_prefixes,
    get_read_block_error,
    is_write_denied,
    raise_if_read_blocked,
)


_RESIDENCE_FILES = (
    "auth.json",
    "auth.lock",
    "auth.json.corrupt",
    "auth.json.tmp.4242.deadbeef",
    ".anthropic_oauth.json",
    "profiles/work/auth.json",
    "profiles/other/.anthropic_oauth.json",
    "shared/nous_auth.json",
    "shared/nous_auth.lock",
    "unclassified-note.txt",
)


def test_distinct_residence_is_denied_as_a_whole_tree(monkeypatch, tmp_path):
    residence = tmp_path / "auth-residence"
    runtime = tmp_path / "runtime" / "profiles" / "work"
    runtime.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    for rel in _RESIDENCE_FILES:
        target = residence / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret", encoding="utf-8")
        assert get_read_block_error(str(target)) is not None, rel
        assert is_write_denied(str(target)), rel
        with pytest.raises(ValueError, match="Access denied"):
            raise_if_read_blocked(str(target))
    assert is_write_denied(str(residence))

    # Runtime state outside the residence stays usable.
    note = runtime / "notes.txt"
    note.write_text("plain", encoding="utf-8")
    assert get_read_block_error(str(note)) is None
    assert not is_write_denied(str(note))


def test_no_override_home_auth_json_stays_requested_writable(monkeypatch, tmp_path):
    home = tmp_path / "runtime"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)

    auth_json = home / "auth.json"
    auth_json.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    # Deliberate main behavior: an explicitly requested write to the ordinary
    # store is allowed; reads stay blocked.
    assert not is_write_denied(str(auth_json))
    assert get_read_block_error(str(auth_json)) is not None

    # Lock, quarantine, temp, and PKCE files carry the same tokens but are
    # never legitimate write targets.
    for name in (
        "auth.lock",
        "auth.json.corrupt",
        "auth.json.tmp.77.cafe",
        ".anthropic_oauth.json",
    ):
        target = home / name
        target.write_text("secret", encoding="utf-8")
        assert is_write_denied(str(target)), name


def test_path_equal_override_is_a_total_guard_no_op(monkeypatch, tmp_path):
    home = tmp_path / "runtime"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def snapshot() -> tuple:
        note = home / "notes.txt"
        note.write_text("plain", encoding="utf-8")
        auth_json = home / "auth.json"
        auth_json.write_text("{}", encoding="utf-8")
        return (
            is_write_denied(str(auth_json)),
            get_read_block_error(str(auth_json)) is not None,
            is_write_denied(str(note)),
            get_read_block_error(str(note)) is None,
            sorted(build_write_denied_prefixes(str(tmp_path))),
        )

    monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    without_override = snapshot()
    monkeypatch.setenv("HERMES_AUTH_HOME", str(home))
    with_path_equal_override = snapshot()

    assert with_path_equal_override == without_override
    # In particular the runtime tree did not become a denied residence.
    assert not with_path_equal_override[0]
    assert not with_path_equal_override[2]


def test_context_image_references_inside_the_residence_are_not_loaded(
    monkeypatch, tmp_path
):
    """The attachment/context reference loader refuses residence files.

    ``_file_to_data_url`` is what turns a locally referenced image into
    model-visible bytes; an arbitrary (non-credential-named) file inside a
    distinct residence must never make that trip, while the same bytes
    outside the residence still load.
    """
    from agent.image_routing import _file_to_data_url

    residence = tmp_path / "auth-residence"
    residence.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    inside = residence / "diagram.png"
    inside.write_bytes(png_bytes)
    outside = runtime / "diagram.png"
    outside.write_bytes(png_bytes)

    assert _file_to_data_url(inside) is None
    loaded = _file_to_data_url(outside)
    assert loaded is not None and loaded.startswith("data:image/png;base64,")


def test_read_guard_fails_closed_when_resolution_machinery_breaks(
    monkeypatch, tmp_path
):
    """Regression: a broken resolver must degrade to a name-only deny.

    ``raise_if_read_blocked`` used to swallow any resolver error and allow the
    read — one malformed environment disabled the credential read-deny at
    every provider input-loading site at once.
    """
    import agent.file_safety as file_safety

    def broken(_path: str):
        raise RuntimeError("resolution broken")

    monkeypatch.setattr(file_safety, "get_read_block_error", broken)

    for name in (
        "auth.json",
        "auth.lock",
        "auth.json.corrupt",
        ".anthropic_oauth.json",
        "nous_auth.json",
        "auth.json.tmp.99.beef",
        ".env",
    ):
        with pytest.raises(ValueError, match="Access denied"):
            file_safety.raise_if_read_blocked(str(tmp_path / name))

    file_safety.raise_if_read_blocked(str(tmp_path / "notes.txt"))
