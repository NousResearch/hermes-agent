"""End-to-end tests for shared Hermes PKCE grants across real profile homes.

Unlike ``test_credential_pool_shared_pkce.py`` (unit tests around the pool
object), these tests drive the real persistence path: two ``HERMES_HOME``
layouts (``<root>`` and ``<root>/profiles/<name>``), real ``auth.json`` files,
real ``load_pool()`` / ``select()`` / ``_refresh_entry()``.  Only the network
refresh call (``refresh_anthropic_oauth_pure``) is mocked.
"""

import json
import os
from pathlib import Path

import pytest

ROOT_NAME = "hermes-root"
WORKER_NAME = "worker"
LABEL = "cg.chae-claude"
PAST_MS = 1_000
FUTURE_MS = 4_102_444_800_000  # 2100-01-01


def _copy_entry(label=LABEL, access="copied-access", refresh="copied-refresh", expires=FUTURE_MS):
    return {
        "id": f"id-{label}",
        "label": label,
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:hermes_pkce",
        "access_token": access,
        "refresh_token": refresh,
        "expires_at_ms": expires,
    }


def _write_store(home: Path, entries: list) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"anthropic": entries}}, indent=2)
    )


def _read_store(home: Path) -> list:
    return json.loads((home / "auth.json").read_text())["credential_pool"]["anthropic"]


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Real root + worker profile layout. Returns (root, worker)."""
    root = tmp_path / ROOT_NAME
    worker = root / "profiles" / WORKER_NAME
    root.mkdir(parents=True)
    worker.mkdir(parents=True)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)
    # Keep the real ~/.claude and legacy singleton readers out of the picture.
    monkeypatch.setattr("agent.anthropic_adapter.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr("agent.anthropic_adapter.read_hermes_oauth_credentials", lambda: None)
    return root, worker


def _use_home(monkeypatch, home: Path):
    monkeypatch.setenv("HERMES_HOME", str(home))


def _load(monkeypatch, home: Path):
    _use_home(monkeypatch, home)
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


def _shared_file(root: Path, label: str = LABEL) -> Path:
    # Resolve through the adapter so the test follows the real (sanitized +
    # digest-suffixed) filename instead of re-implementing it.
    from agent.anthropic_adapter import _shared_pkce_label_filename

    return root / ".anthropic_oauth" / _shared_pkce_label_filename(label)


def _mock_refresh(monkeypatch, *, valid: set, counter: list):
    """Refresh succeeds only for tokens in ``valid``; each success rotates."""

    def _refresh(refresh_token, *, use_json=False):
        counter.append(refresh_token)
        if refresh_token not in valid:
            raise RuntimeError("HTTP Error 404: Not Found")
        new_refresh = f"{refresh_token}-rot"
        valid.discard(refresh_token)
        valid.add(new_refresh)
        return {
            "access_token": f"access-from-{refresh_token}",
            "refresh_token": new_refresh,
            "expires_at_ms": FUTURE_MS,
        }

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", _refresh)


# ---------------------------------------------------------------------------


def test_root_seeds_shared_grant_and_worker_copy_becomes_reference(homes, monkeypatch):
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale-access", refresh="stale-refresh")])

    root_pool = _load(monkeypatch, root)
    assert [e.source for e in root_pool.entries()] == ["hermes_pkce_shared"]
    shared = json.loads(_shared_file(root).read_text())
    assert shared["refreshToken"] == "root-refresh"
    assert oct(_shared_file(root).stat().st_mode & 0o777) == oct(0o600)

    worker_pool = _load(monkeypatch, worker)
    rows = _read_store(worker)
    assert [r["source"] for r in rows] == ["hermes_pkce_shared"]
    assert rows[0]["label"] == LABEL
    # The worker's stale copy must never leak into the shared source.
    assert json.loads(_shared_file(root).read_text())["refreshToken"] == "root-refresh"

    selected = worker_pool.select()
    assert selected is not None
    assert selected.label == LABEL
    assert selected.access_token == "root-access"
    assert selected.refresh_token == "root-refresh"


def test_worker_loaded_before_root_keeps_its_copy_usable(homes, monkeypatch):
    """Order independence: no shared grant yet -> copy stays a copy, not a dangling ref."""
    root, worker = homes
    _write_store(worker, [_copy_entry(access="copy-access", refresh="copy-refresh")])

    pool = _load(monkeypatch, worker)
    assert not _shared_file(root).exists()
    rows = _read_store(worker)
    assert rows[0]["source"] == "manual:hermes_pkce"
    assert rows[0]["access_token"] == "copy-access"
    selected = pool.select()
    assert selected is not None and selected.access_token == "copy-access"

    # Root comes online later and publishes; the worker converts on its next load.
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _load(monkeypatch, root)
    pool2 = _load(monkeypatch, worker)
    assert _read_store(worker)[0]["source"] == "hermes_pkce_shared"
    assert pool2.select().access_token == "root-access"


def test_refresh_in_one_profile_is_visible_to_the_other_without_refreshing(homes, monkeypatch):
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh", expires=PAST_MS)])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale", expires=PAST_MS)])
    calls: list = []
    _mock_refresh(monkeypatch, valid={"root-refresh"}, counter=calls)

    root_pool = _load(monkeypatch, root)
    _load(monkeypatch, worker)  # migrate the worker row to a shared reference

    # Root refreshes (expired) -> publishes rotated pair to the shared file.
    chosen = root_pool.select()
    assert chosen is not None and chosen.access_token == "access-from-root-refresh"
    assert calls == ["root-refresh"]
    shared = json.loads(_shared_file(root).read_text())
    assert shared["refreshToken"] == "root-refresh-rot"

    # Worker's next select() picks up the rotated pair from the file: no refresh call.
    worker_pool = _load(monkeypatch, worker)
    picked = worker_pool.select()
    assert picked is not None
    assert picked.access_token == "access-from-root-refresh"
    assert picked.refresh_token == "root-refresh-rot"
    assert calls == ["root-refresh"]
    # Reference rows never persist token material into the profile auth.json.
    row = _read_store(worker)[0]
    assert row["source"] == "hermes_pkce_shared"
    assert not row.get("access_token") and not row.get("refresh_token")


def test_rotated_shared_grant_is_adopted_under_lock_without_posting(homes, monkeypatch):
    """Concurrent-refresh loser: the shared file already moved on -> adopt, no POST."""
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale")])
    _load(monkeypatch, root)
    worker_pool = _load(monkeypatch, worker)
    hydrated = worker_pool.select()  # reference rows hydrate from the shared file on selection
    assert hydrated is not None and hydrated.source == "hermes_pkce_shared"
    assert hydrated.refresh_token == "root-refresh"

    # Another profile rotates the grant after this pool hydrated.
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "newer-access", "refreshToken": "newer-refresh", "expiresAt": FUTURE_MS},
        label=LABEL,
    )
    _use_home(monkeypatch, worker)
    calls: list = []
    _mock_refresh(monkeypatch, valid={"newer-refresh"}, counter=calls)

    refreshed = worker_pool._refresh_entry(hydrated, force=True)
    assert refreshed is not None
    assert calls == []  # in-lock re-sync picked up the rotated pair; no network call
    assert refreshed.access_token == "newer-access"
    assert refreshed.refresh_token == "newer-refresh"


def test_revoked_refresh_token_syncs_from_shared_and_retries_once(homes, monkeypatch):
    """Server-side revocation with an out-of-band rotation -> 404, re-sync, single retry."""
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale")])
    _load(monkeypatch, root)
    worker_pool = _load(monkeypatch, worker)
    hydrated = worker_pool.select()
    assert hydrated is not None and hydrated.refresh_token == "root-refresh"

    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    calls: list = []

    def _refresh(refresh_token, *, use_json=False):
        calls.append(refresh_token)
        if refresh_token == "root-refresh":
            # An external writer (not holding our lock) rotated the grant and
            # published it between our sync and our POST.
            _use_home(monkeypatch, root)
            assert write_shared_hermes_oauth_credentials(
                {"accessToken": "ext-access", "refreshToken": "ext-refresh", "expiresAt": FUTURE_MS},
                label=LABEL,
            )
            _use_home(monkeypatch, worker)
            raise RuntimeError("HTTP Error 404: Not Found")
        assert refresh_token == "ext-refresh"
        return {"access_token": "access-from-ext", "refresh_token": "ext-refresh-rot", "expires_at_ms": FUTURE_MS}

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", _refresh)

    refreshed = worker_pool._refresh_entry(hydrated, force=True)
    assert refreshed is not None
    assert calls == ["root-refresh", "ext-refresh"]  # 404 -> sync -> exactly one retry
    assert refreshed.access_token == "access-from-ext"
    assert json.loads(_shared_file(root).read_text())["refreshToken"] == "ext-refresh-rot"
    row = _read_store(worker)[0]
    assert row["source"] == "hermes_pkce_shared"
    assert not row.get("access_token") and not row.get("refresh_token")


def test_two_labels_migrate_independently_without_cross_hydration(homes, monkeypatch):
    root, worker = homes
    other = "second-account"
    _write_store(
        root,
        [
            _copy_entry(access="a-access", refresh="a-refresh"),
            {**_copy_entry(other, access="b-access", refresh="b-refresh"), "id": "id-b", "priority": 1},
        ],
    )
    _write_store(
        worker,
        [
            _copy_entry(access="old-a", refresh="old-a"),
            {**_copy_entry(other, access="old-b", refresh="old-b"), "id": "id-b", "priority": 1},
        ],
    )
    _load(monkeypatch, root)
    assert json.loads(_shared_file(root).read_text())["refreshToken"] == "a-refresh"
    assert json.loads(_shared_file(root, other).read_text())["refreshToken"] == "b-refresh"

    worker_pool = _load(monkeypatch, worker)
    available = worker_pool._available_entries()  # hydrates every reference row
    by_label = {e.label: e for e in available}
    assert by_label[LABEL].access_token == "a-access"
    assert by_label[other].access_token == "b-access"
    assert {e.source for e in available} == {"hermes_pkce_shared"}


def test_malformed_shared_file_skips_reference_with_warning_and_keeps_row(homes, monkeypatch, caplog):
    root, worker = homes
    _write_store(root, [_copy_entry(access="root-access", refresh="root-refresh")])
    _write_store(worker, [_copy_entry(access="stale", refresh="stale")])
    _load(monkeypatch, root)
    _load(monkeypatch, worker)

    _shared_file(root).write_text("{not json")
    with caplog.at_level("WARNING"):
        pool = _load(monkeypatch, worker)
        assert pool.select() is None
    assert any("shared OAuth source is missing" in r.getMessage() for r in caplog.records)
    # Row is preserved (not pruned) so it recovers once the source is restored.
    assert _read_store(worker)[0]["source"] == "hermes_pkce_shared"

    _use_home(monkeypatch, root)
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "restored", "refreshToken": "restored-r", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert _load(monkeypatch, worker).select().access_token == "restored"


def test_empty_profile_discovers_every_shared_label_without_copies(homes, monkeypatch):
    """A profile with no pool rows (e.g. a freshly created worker) must see all
    published grants by label -- no `auth add`, no copied tokens."""
    root, worker = homes
    other = "second account"  # label with a character the filename sanitizer rewrites
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "a-access", "refreshToken": "a-refresh", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "b-access", "refreshToken": "b-refresh", "expiresAt": FUTURE_MS}, label=other
    )
    _write_store(worker, [])

    pool = _load(monkeypatch, worker)
    available = pool._available_entries()
    by_label = {e.label: e for e in available}
    assert set(by_label) == {LABEL, other}
    assert {e.source for e in available} == {"hermes_pkce_shared"}
    assert by_label[LABEL].access_token == "a-access"
    assert by_label[other].access_token == "b-access"
    assert pool.select() is not None

    rows = _read_store(worker)
    assert sorted(r["label"] for r in rows) == sorted([LABEL, other])
    assert all(r["source"] == "hermes_pkce_shared" for r in rows)
    assert all(not r.get("access_token") and not r.get("refresh_token") for r in rows)

    # Reloading must not duplicate rows or collapse the two labels into one.
    pool2 = _load(monkeypatch, worker)
    assert len(_read_store(worker)) == 2
    assert {e.label for e in pool2._available_entries()} == {LABEL, other}


def test_empty_profile_discovers_legacy_grants_map_file(homes, monkeypatch):
    """Pre-directory layout: `<root>/.anthropic_oauth.json` holding a grants map."""
    root, worker = homes
    (root / ".anthropic_oauth.json").write_text(
        json.dumps({"grants": {LABEL: {"accessToken": "legacy-access", "refreshToken": "legacy-refresh", "expiresAt": FUTURE_MS}}})
    )
    _write_store(worker, [])
    pool = _load(monkeypatch, worker)
    picked = pool.select()
    assert picked is not None
    assert picked.label == LABEL and picked.access_token == "legacy-access"
    assert _read_store(worker)[0]["source"] == "hermes_pkce_shared"


def _publish(monkeypatch, root: Path, label: str, access: str, refresh: str) -> None:
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": access, "refreshToken": refresh, "expiresAt": FUTURE_MS}, label=label
    )


def test_auth_remove_retires_one_shared_label_without_reseeding(homes, monkeypatch, capsys):
    """`hermes auth remove anthropic <N>` on a shared row must suppress exactly
    that label for this profile: the next load_pool() must not resurrect it,
    the other label keeps seeding, and the grant file stays for other profiles."""
    import types

    from hermes_cli.auth_commands import auth_remove_command

    root, worker = homes
    other = "second-account"
    _publish(monkeypatch, root, LABEL, "a-access", "a-refresh")
    _publish(monkeypatch, root, other, "b-access", "b-refresh")
    _write_store(worker, [])

    pool = _load(monkeypatch, worker)
    assert {e.label for e in pool._available_entries()} == {LABEL, other}

    auth_remove_command(types.SimpleNamespace(provider="anthropic", target=LABEL))
    out = capsys.readouterr().out
    assert "Suppressed shared Anthropic OAuth grant" in out

    store = json.loads((worker / "auth.json").read_text())
    assert store["suppressed_sources"]["anthropic"] == [f"hermes_pkce_shared:{LABEL}"]
    assert _shared_file(root, LABEL).exists(), "removing from one profile must not revoke the grant"

    pool2 = _load(monkeypatch, worker)
    assert {e.label for e in pool2._available_entries()} == {other}
    assert [r["label"] for r in _read_store(worker)] == [other]

    # The grant is untouched for every other profile.
    sibling = root / "profiles" / "sibling"
    sibling.mkdir(parents=True)
    _write_store(sibling, [])
    assert {e.label for e in _load(monkeypatch, sibling)._available_entries()} == {LABEL, other}


def test_sanitized_labels_never_share_a_file(homes, monkeypatch):
    """`team a` and `team_a` sanitize to the same prefix; they must still live
    in distinct files, round-trip their own tokens, and both be discovered."""
    from agent.anthropic_adapter import (
        _shared_pkce_label_filename,
        list_shared_hermes_oauth_credentials,
        read_shared_hermes_oauth_credentials,
    )

    root, worker = homes
    a, b = "team a", "team_a"
    assert _shared_pkce_label_filename(a) != _shared_pkce_label_filename(b)
    _publish(monkeypatch, root, a, "a-access", "a-refresh")
    _publish(monkeypatch, root, b, "b-access", "b-refresh")

    assert read_shared_hermes_oauth_credentials(a)["accessToken"] == "a-access"
    assert read_shared_hermes_oauth_credentials(b)["accessToken"] == "b-access"
    listed = list_shared_hermes_oauth_credentials()
    assert set(listed) == {a, b}
    assert listed[a]["refreshToken"] == "a-refresh" and listed[b]["refreshToken"] == "b-refresh"

    _write_store(worker, [])
    pool = _load(monkeypatch, worker)
    assert {e.label: e.access_token for e in pool._available_entries()} == {a: "a-access", b: "b-access"}


def test_refresh_and_publish_lock_the_same_label_target_from_every_profile(homes, monkeypatch):
    """Two profiles refreshing one label must serialize on one flock, and it
    must be the same lock the publisher takes -- not the legacy singleton path."""
    import agent.credential_pool as pool_mod
    from agent.anthropic_adapter import (
        _get_shared_hermes_oauth_file,
        _get_shared_hermes_oauth_label_file,
    )

    root, worker = homes
    _publish(monkeypatch, root, LABEL, "old-access", "old-refresh")
    for home in (root, worker):
        _write_store(home, [])

    counter: list = []
    _mock_refresh(monkeypatch, valid={"old-refresh"}, counter=counter)

    observed: list = []
    real_lock = pool_mod._auth_store_lock

    def spy(*args, **kwargs):
        observed.append(kwargs.get("target_path"))
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(pool_mod, "_auth_store_lock", spy)

    targets = []
    for home in (root, worker):
        pool = _load(monkeypatch, home)
        entry = next(e for e in pool._available_entries() if e.label == LABEL)
        observed.clear()
        assert pool._refresh_entry(entry, force=True) is not None
        label_targets = [p for p in observed if p is not None]
        assert label_targets, "shared refresh must take a target-path lock"
        targets.append(label_targets[0])
        # The second profile syncs the rotated grant from the shared file
        # before refreshing, so no re-publish is needed here.

    expected = _get_shared_hermes_oauth_label_file(LABEL)
    assert targets[0] == targets[1] == expected
    assert expected != _get_shared_hermes_oauth_file()
    assert expected.with_suffix(".lock").exists()


def test_pre_digest_alias_file_is_read_listed_once_and_retired_on_publish(homes, monkeypatch):
    """A grant published by an earlier build under ``<sanitized>.json`` (no
    digest) must still resolve for its label, appear once in discovery, and be
    retired the first time the canonical file is published."""
    from agent.anthropic_adapter import (
        list_shared_hermes_oauth_credentials,
        read_shared_hermes_oauth_credentials,
        write_shared_hermes_oauth_credentials,
    )

    root, worker = homes
    _use_home(monkeypatch, root)
    alias = root / ".anthropic_oauth" / f"{LABEL}.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.write_text(json.dumps(
        {"accessToken": "alias-access", "refreshToken": "alias-refresh", "expiresAt": FUTURE_MS, "label": LABEL}
    ))
    canonical = _shared_file(root, LABEL)
    assert alias != canonical and not canonical.exists()

    got = read_shared_hermes_oauth_credentials(LABEL)
    assert got and got["accessToken"] == "alias-access"
    listed = list_shared_hermes_oauth_credentials()
    assert list(listed) == [LABEL] and listed[LABEL]["accessToken"] == "alias-access"

    # An empty profile hydrates from the alias too.
    _write_store(worker, [])
    pool = _load(monkeypatch, worker)
    assert {e.label: e.access_token for e in pool._available_entries() if e.label == LABEL} == {LABEL: "alias-access"}

    # Publishing (e.g. after a refresh) moves the label to the canonical file.
    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "canon-access", "refreshToken": "canon-refresh", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert canonical.exists() and not alias.exists()
    assert read_shared_hermes_oauth_credentials(LABEL)["accessToken"] == "canon-access"
    assert list(list_shared_hermes_oauth_credentials()) == [LABEL]


def test_canonical_file_wins_over_stale_alias_in_discovery(homes, monkeypatch):
    from agent.anthropic_adapter import list_shared_hermes_oauth_credentials, read_shared_hermes_oauth_credentials

    root, _worker = homes
    _use_home(monkeypatch, root)
    canonical = _shared_file(root, LABEL)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(json.dumps({"accessToken": "canon-access", "refreshToken": "canon-refresh", "label": LABEL}))
    alias = root / ".anthropic_oauth" / f"{LABEL}.json"
    alias.write_text(json.dumps({"accessToken": "stale-access", "refreshToken": "stale-refresh", "label": LABEL}))

    assert read_shared_hermes_oauth_credentials(LABEL)["accessToken"] == "canon-access"
    assert list_shared_hermes_oauth_credentials()[LABEL]["accessToken"] == "canon-access"


def test_refresh_of_unhydrated_shared_row_hydrates_instead_of_marking_exhausted(homes, monkeypatch):
    """A raw shared reference row (no token material on disk) handed to
    _refresh_entry(force=True) must hydrate from the shared file and refresh,
    not be marked exhausted for lacking a refresh token."""
    import agent.anthropic_adapter as adapter
    from agent.credential_pool import STATUS_EXHAUSTED

    root, worker = homes
    _publish(monkeypatch, root, LABEL, "shared-access", "shared-refresh")
    _write_store(worker, [])
    pool = _load(monkeypatch, worker)
    raw = next(e for e in pool.entries() if e.label == LABEL)
    assert raw.source == "hermes_pkce_shared" and not raw.refresh_token

    calls = []
    def fake_refresh(refresh_token, **_):
        calls.append(refresh_token)
        return {"access_token": "rot-access", "refresh_token": "rot-refresh", "expires_at_ms": FUTURE_MS}
    monkeypatch.setattr(adapter, "refresh_anthropic_oauth_pure", fake_refresh)

    out = pool._refresh_entry(raw, force=True)
    assert calls == ["shared-refresh"]
    assert out is not None and out.access_token == "rot-access"
    assert all(e.last_status != STATUS_EXHAUSTED for e in pool.entries() if e.label == LABEL)
    assert adapter.read_shared_hermes_oauth_credentials(LABEL)["refreshToken"] == "rot-refresh"


def test_grant_file_without_payload_label_is_never_trusted_by_filename(homes, monkeypatch):
    """A ``<label>.json`` whose payload declares no ``label`` must not be
    served for that label, must not be listed, and must not be retired as an
    alias when the canonical file is published (Codex review 8)."""
    from agent.anthropic_adapter import (
        list_shared_hermes_oauth_credentials,
        read_shared_hermes_oauth_credentials,
        write_shared_hermes_oauth_credentials,
    )

    root, worker = homes
    _use_home(monkeypatch, root)
    stray = root / ".anthropic_oauth" / f"{LABEL}.json"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text(json.dumps({"accessToken": "stray-access", "refreshToken": "stray-refresh", "expiresAt": FUTURE_MS}))

    assert read_shared_hermes_oauth_credentials(LABEL) is None
    assert list_shared_hermes_oauth_credentials() == {}
    _write_store(worker, [])
    assert [e for e in _load(monkeypatch, worker)._available_entries() if e.label == LABEL] == []

    _use_home(monkeypatch, root)
    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "canon-access", "refreshToken": "canon-refresh", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert stray.exists(), "an unlabelled file is not an alias and must not be retired"
    assert read_shared_hermes_oauth_credentials(LABEL)["accessToken"] == "canon-access"


def test_alias_with_a_different_payload_label_is_not_matched(homes, monkeypatch):
    from agent.anthropic_adapter import (
        list_shared_hermes_oauth_credentials,
        read_shared_hermes_oauth_credentials,
        write_shared_hermes_oauth_credentials,
    )

    root, _worker = homes
    _use_home(monkeypatch, root)
    other = root / ".anthropic_oauth" / f"{LABEL}.json"  # filename says LABEL ...
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps(
        {"accessToken": "other-access", "refreshToken": "other-refresh", "expiresAt": FUTURE_MS, "label": "someone-else"}
    ))  # ... payload says otherwise

    assert read_shared_hermes_oauth_credentials(LABEL) is None
    assert read_shared_hermes_oauth_credentials("someone-else")["accessToken"] == "other-access"
    assert set(list_shared_hermes_oauth_credentials()) == {"someone-else"}

    assert write_shared_hermes_oauth_credentials(
        {"accessToken": "canon-access", "refreshToken": "canon-refresh", "expiresAt": FUTURE_MS}, label=LABEL
    )
    assert other.exists(), "a different account's grant must not be retired as LABEL's alias"
    assert set(list_shared_hermes_oauth_credentials()) == {LABEL, "someone-else"}


@pytest.mark.parametrize(
    "bad_label",
    [
        pytest.param("attacker", id="mismatch"),
        pytest.param(None, id="missing"),
        pytest.param(42, id="non-string"),
        pytest.param("   ", id="whitespace"),
    ],
)
def test_canonical_file_is_not_trusted_unless_payload_label_matches(homes, monkeypatch, bad_label):
    """Codex review 8-2: the canonical ``<sanitized>-<digest>.json`` filename
    is not an identity either -- a victim's canonical path holding another
    (or no) label must not be served, listed under the victim, or selected."""
    from agent.anthropic_adapter import (
        _get_shared_hermes_oauth_label_file,
        list_shared_hermes_oauth_credentials,
        read_shared_hermes_oauth_credentials,
    )

    root, worker = homes
    _use_home(monkeypatch, root)
    canonical = _get_shared_hermes_oauth_label_file(LABEL)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    payload = {"accessToken": "bad-access", "refreshToken": "bad-refresh", "expiresAt": FUTURE_MS}
    if bad_label is not None:
        payload["label"] = bad_label
    canonical.write_text(json.dumps(payload))

    assert read_shared_hermes_oauth_credentials(LABEL) is None
    listed = list_shared_hermes_oauth_credentials()
    assert LABEL not in listed
    if bad_label == "attacker":
        # A well-formed grant that merely lives under the wrong filename is
        # still discoverable under the label it actually declares.
        assert listed["attacker"]["accessToken"] == "bad-access"
        assert read_shared_hermes_oauth_credentials("attacker")["accessToken"] == "bad-access"
    else:
        assert listed == {}
    _write_store(worker, [])
    assert [e for e in _load(monkeypatch, worker)._available_entries() if e.label == LABEL] == []


def test_padded_payload_label_normalizes_to_an_exact_match(homes, monkeypatch):
    """Both sides are stripped before the exact comparison (same rule the
    publisher and pool use), so surrounding whitespace is not a mismatch."""
    from agent.anthropic_adapter import _get_shared_hermes_oauth_label_file, read_shared_hermes_oauth_credentials

    root, _worker = homes
    _use_home(monkeypatch, root)
    canonical = _get_shared_hermes_oauth_label_file(LABEL)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(json.dumps(
        {"accessToken": "ok-access", "refreshToken": "ok-refresh", "expiresAt": FUTURE_MS, "label": f"  {LABEL} "}
    ))
    assert read_shared_hermes_oauth_credentials(LABEL)["accessToken"] == "ok-access"
    assert read_shared_hermes_oauth_credentials(f" {LABEL}")["accessToken"] == "ok-access"


def test_read_validates_the_bytes_it_returns_under_the_label_lock(homes, monkeypatch):
    """The alias list is re-read and re-validated inside the label lock, so a
    file that changed between discovery and read cannot leak a foreign grant."""
    import agent.anthropic_adapter as adapter

    root, _worker = homes
    _use_home(monkeypatch, root)
    alias = root / ".anthropic_oauth" / f"{LABEL}.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    good = {"accessToken": "good-access", "refreshToken": "good-refresh", "expiresAt": FUTURE_MS, "label": LABEL}
    alias.write_text(json.dumps(good))

    real_find = adapter._find_shared_hermes_oauth_label_aliases
    lock_seen = []

    def swapping_find(label):
        found = real_find(label)  # discovery sees the good alias ...
        lock_seen.append(adapter.shared_hermes_oauth_lock_target(label).with_suffix(".lock").exists())
        alias.write_text(json.dumps({**good, "label": "attacker"}))  # ... then it is swapped
        return found

    monkeypatch.setattr(adapter, "_find_shared_hermes_oauth_label_aliases", swapping_find)
    assert adapter.read_shared_hermes_oauth_credentials(LABEL) is None
    assert lock_seen == [True], "alias discovery must run under the label lock"
