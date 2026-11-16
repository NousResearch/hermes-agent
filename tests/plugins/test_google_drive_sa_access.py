"""Tests for plugins/google_drive_sa/access.py — per-user Drive access check."""

from __future__ import annotations

import pytest

from plugins.google_drive_sa import access


def _cfg(**over):
    base = dict(
        enabled=True,
        cache_ttl=300.0,
        everyone_groups=frozenset({"all@everafter.ai"}),
        group_members={"sales@everafter.ai": frozenset({"alice@everafter.ai"})},
    )
    base.update(over)
    return access.AccessConfig(**base)


ALICE = "alice@everafter.ai"


# --------------------------------------------------------------------------- #
# evaluate()
# --------------------------------------------------------------------------- #

def test_evaluate_user_entry_grants_role():
    acl = [{"type": "user", "emailAddress": "Alice@EverAfter.ai", "role": "reader"}]
    d = access.evaluate(acl, ALICE, _cfg())
    assert d.granted_role == "reader"
    assert d.satisfies(access.READER) is True
    assert d.satisfies(access.WRITER) is False


def test_evaluate_domain_entry_matches_requester_domain():
    acl = [{"type": "domain", "domain": "EverAfter.ai", "role": "writer"}]
    assert access.evaluate(acl, ALICE, _cfg()).granted_role == "writer"


def test_evaluate_domain_entry_does_not_match_other_domain():
    acl = [{"type": "domain", "domain": "base.ai", "role": "writer"}]
    assert access.evaluate(acl, ALICE, _cfg()).granted_role is None


def test_evaluate_anyone_entry_always_grants():
    acl = [{"type": "anyone", "role": "commenter"}]
    d = access.evaluate(acl, ALICE, _cfg())
    assert d.granted_role == "commenter"
    assert d.satisfies(access.READER) and not d.satisfies(access.WRITER)


def test_evaluate_everyone_group_grants():
    acl = [{"type": "group", "emailAddress": "ALL@everafter.ai", "role": "reader"}]
    assert access.evaluate(acl, ALICE, _cfg()).granted_role == "reader"


def test_evaluate_mapped_group_grants_only_listed_members():
    acl = [{"type": "group", "emailAddress": "sales@everafter.ai", "role": "writer"}]
    assert access.evaluate(acl, ALICE, _cfg()).granted_role == "writer"
    d = access.evaluate(acl, "bob@everafter.ai", _cfg())
    assert d.granted_role is None
    assert d.unmapped_groups == ()  # mapped group, just not a member — not "unmapped"


def test_evaluate_unmapped_group_is_ignored_and_reported():
    acl = [
        {"type": "group", "emailAddress": "finance@everafter.ai", "role": "writer"},
        {"type": "group", "emailAddress": "finance@everafter.ai", "role": "reader"},
        {"type": "group", "emailAddress": "legal@everafter.ai", "role": "reader"},
    ]
    d = access.evaluate(acl, ALICE, _cfg())
    assert d.granted_role is None
    assert d.unmapped_groups == ("finance@everafter.ai", "legal@everafter.ai")


def test_evaluate_highest_role_wins_across_entries():
    acl = [
        {"type": "user", "emailAddress": ALICE, "role": "reader"},
        {"type": "domain", "domain": "everafter.ai", "role": "writer"},
    ]
    assert access.evaluate(acl, ALICE, _cfg()).granted_role == "writer"


@pytest.mark.parametrize(
    "role,reader,writer",
    [
        ("owner", True, True),
        ("organizer", True, True),
        ("fileOrganizer", True, True),
        ("writer", True, True),
        ("commenter", True, False),
        ("reader", True, False),
        ("bogus", False, False),
    ],
)
def test_role_ladder(role, reader, writer):
    acl = [{"type": "user", "emailAddress": ALICE, "role": role}]
    d = access.evaluate(acl, ALICE, _cfg())
    assert d.satisfies(access.READER) is reader
    assert d.satisfies(access.WRITER) is writer


def test_evaluate_empty_acl_denies():
    d = access.evaluate([], ALICE, _cfg())
    assert d.granted_role is None and d.satisfies(access.READER) is False


def test_evaluate_ignores_malformed_entries():
    acl = [None, "x", {"type": "user"}, {"role": "writer"}]
    assert access.evaluate(acl, ALICE, _cfg()).granted_role is None


# --------------------------------------------------------------------------- #
# load_access_config()
# --------------------------------------------------------------------------- #

def test_load_access_config_defaults_when_block_missing(monkeypatch):
    monkeypatch.setattr(access, "_raw_config", lambda: {})
    cfg = access.load_access_config()
    assert cfg.enabled is True
    assert cfg.cache_ttl == 300.0
    assert cfg.everyone_groups == frozenset()
    assert cfg.group_members == {}


def test_load_access_config_normalises_case_and_shapes(monkeypatch):
    monkeypatch.setattr(
        access,
        "_raw_config",
        lambda: {
            "google_drive": {
                "access_check": False,
                "acl_cache_ttl_seconds": "45",
                "everyone_groups": ["All@EverAfter.ai", " staff@everafter.ai "],
                "group_members": {
                    "Sales@everafter.ai": ["Alice@EverAfter.ai", "bob@everafter.ai"],
                    "broken": "not-a-list",
                },
            }
        },
    )
    cfg = access.load_access_config()
    assert cfg.enabled is False
    assert cfg.cache_ttl == 45.0
    assert cfg.everyone_groups == frozenset({"all@everafter.ai", "staff@everafter.ai"})
    assert cfg.group_members == {
        "sales@everafter.ai": frozenset({"alice@everafter.ai", "bob@everafter.ai"}),
    }


def test_load_access_config_survives_read_error(monkeypatch):
    def boom():
        raise RuntimeError("no config")

    monkeypatch.setattr(access, "_raw_config", boom)
    assert access.load_access_config().enabled is True


# --------------------------------------------------------------------------- #
# Requester resolution
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _reset_access_cache():
    access.reset_cache()
    yield
    access.reset_cache()


@pytest.fixture
def engaged(monkeypatch):
    monkeypatch.setattr(access, "_engaged", lambda: True)
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg())


def _session(monkeypatch, platform="slack", user_id="U1", source=""):
    from gateway.session_context import set_session_vars, clear_session_vars

    tokens = set_session_vars(platform=platform, user_id=user_id, source=source)
    return lambda: clear_session_vars(tokens)


def test_resolve_requester_from_session(monkeypatch, engaged):
    monkeypatch.setattr(access, "_resolve_email", lambda p, u: "alice@everafter.ai" if (p, u) == ("slack", "U1") else None)
    undo = _session(monkeypatch)
    try:
        r = access.resolve_requester()
    finally:
        undo()
    assert r == access.Requester(platform="slack", user_id="U1", email="alice@everafter.ai")


def test_resolve_requester_none_when_email_unresolvable(monkeypatch, engaged):
    monkeypatch.setattr(access, "_resolve_email", lambda p, u: None)
    undo = _session(monkeypatch)
    try:
        assert access.resolve_requester() is None
    finally:
        undo()


def test_resolve_requester_none_when_resolve_email_raises(monkeypatch, engaged):
    def _boom(platform, user_id):
        raise RuntimeError("slack API down")

    monkeypatch.setattr(access, "_resolve_email", _boom)
    undo = _session(monkeypatch)
    try:
        assert access.resolve_requester() is None
    finally:
        undo()


def test_resolve_requester_none_without_identity(monkeypatch, engaged):
    monkeypatch.setattr(access, "_resolve_email", lambda p, u: "x@y.z")
    monkeypatch.setattr(access, "_cron_owner", lambda: None)
    undo = _session(monkeypatch, platform="", user_id="")
    try:
        assert access.resolve_requester() is None
    finally:
        undo()


def test_resolve_requester_from_cron_owner(monkeypatch, engaged):
    monkeypatch.setattr(access, "_cron_owner", lambda: ("slack", "U9"))
    monkeypatch.setattr(access, "_resolve_email", lambda p, u: "owner@everafter.ai" if u == "U9" else None)
    undo = _session(monkeypatch, platform="", user_id="")
    try:
        r = access.resolve_requester()
    finally:
        undo()
    assert r.email == "owner@everafter.ai" and r.user_id == "U9"


def test_cron_owner_reads_ownership_registry(monkeypatch):
    from cron.tool_approval_context import set_cron_tool_context, clear_cron_tool_context
    from agent import automation_ownership as ao

    monkeypatch.setattr(
        ao, "get_record",
        lambda key: {"owner": {"platform": "slack", "user_id": "U7"}} if key == "cron:j1" else None,
    )
    tok = set_cron_tool_context(owner_grant=None, acked_tools=[], job_id="j1")
    try:
        assert access._cron_owner() == ("slack", "U7")
    finally:
        clear_cron_tool_context(tok)
    assert access._cron_owner() is None  # no job id → no owner


def test_cron_owner_none_for_ownerless_job(monkeypatch):
    from cron.tool_approval_context import set_cron_tool_context, clear_cron_tool_context
    from agent import automation_ownership as ao

    monkeypatch.setattr(ao, "get_record", lambda key: None)
    tok = set_cron_tool_context(owner_grant=None, acked_tools=[], job_id="j2")
    try:
        assert access._cron_owner() is None
    finally:
        clear_cron_tool_context(tok)


# --------------------------------------------------------------------------- #
# is_check_active()
# --------------------------------------------------------------------------- #

def test_check_inactive_when_not_engaged(monkeypatch):
    monkeypatch.setattr(access, "_engaged", lambda: False)
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg())
    assert access.is_check_active() is False


def test_check_inactive_when_disabled(monkeypatch):
    monkeypatch.setattr(access, "_engaged", lambda: True)
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg(enabled=False))
    assert access.is_check_active() is False


def test_check_active(monkeypatch, engaged):
    assert access.is_check_active() is True


def test_check_inactive_for_local_tui_session(monkeypatch, engaged):
    undo = _session(monkeypatch, platform="", user_id="", source="tui")
    try:
        assert access.is_check_active() is False
    finally:
        undo()


def test_check_active_for_api_server_session(monkeypatch, engaged):
    undo = _session(monkeypatch, platform="", user_id="", source="api_server")
    try:
        assert access.is_check_active() is True
    finally:
        undo()


def test_check_active_when_platform_set_even_with_local_source(monkeypatch, engaged):
    undo = _session(monkeypatch, platform="slack", user_id="U1", source="tui")
    try:
        assert access.is_check_active() is True
    finally:
        undo()


# --------------------------------------------------------------------------- #
# ACL fetch + cache
# --------------------------------------------------------------------------- #

class _Req:
    def __init__(self, result):
        self._r = result

    def execute(self):
        if isinstance(self._r, Exception):
            raise self._r
        return self._r


class _FakeDrive:
    """files().get / permissions().list / permissions().create recorder."""

    def __init__(self, *, get=None, perms=None, perm_pages=None):
        self.get_calls = []
        self.list_calls = []
        self.create_calls = []
        self._get = get
        self._perm_pages = perm_pages if perm_pages is not None else (
            [{"permissions": perms or []}]
        )

    def files(self):
        outer = self

        class _F:
            def get(self, **kw):
                outer.get_calls.append(kw)
                return _Req(outer._get)

        return _F()

    def permissions(self):
        outer = self

        class _P:
            def list(self, **kw):
                outer.list_calls.append(kw)
                page = len(outer.list_calls) - 1
                return _Req(outer._perm_pages[min(page, len(outer._perm_pages) - 1)])

            def create(self, **kw):
                outer.create_calls.append(kw)
                return _Req({"id": "perm1"})

        return _P()


@pytest.fixture
def drive(monkeypatch):
    def _install(fake):
        from plugins.google_drive_sa import client as gd_client

        monkeypatch.setattr(gd_client, "get_service", lambda: fake)
        return fake

    return _install


def test_fetch_acl_uses_inline_permissions(drive):
    fake = drive(_FakeDrive(get={"id": "f1", "name": "Doc", "permissions": [
        {"type": "user", "emailAddress": ALICE, "role": "reader"}]}))
    name, acl = access.fetch_acl("f1")
    assert name == "Doc" and acl[0]["emailAddress"] == ALICE
    assert fake.list_calls == []
    assert fake.get_calls[0]["supportsAllDrives"] is True
    assert "permissions(" in fake.get_calls[0]["fields"]


def test_fetch_acl_falls_back_to_permissions_list_and_pages(drive):
    fake = drive(_FakeDrive(
        get={"id": "f1", "name": "SD", "driveId": "D1"},
        perm_pages=[
            {"permissions": [{"type": "user", "emailAddress": "a@x", "role": "reader"}], "nextPageToken": "p2"},
            {"permissions": [{"type": "domain", "domain": "x", "role": "writer"}]},
        ],
    ))
    _, acl = access.fetch_acl("f1")
    assert [e["type"] for e in acl] == ["user", "domain"]
    assert len(fake.list_calls) == 2
    assert fake.list_calls[1]["pageToken"] == "p2"
    assert fake.list_calls[0]["supportsAllDrives"] is True


def test_fetch_acl_cached_within_ttl(drive, monkeypatch):
    fake = drive(_FakeDrive(get={"id": "f1", "name": "Doc", "permissions": []}))
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg(cache_ttl=100.0))
    t = [1000.0]
    monkeypatch.setattr(access, "_now", lambda: t[0])
    access.fetch_acl("f1")
    t[0] += 50
    access.fetch_acl("f1")
    assert len(fake.get_calls) == 1
    t[0] += 60  # past TTL
    access.fetch_acl("f1")
    assert len(fake.get_calls) == 2


def test_fetch_acl_falls_back_when_inline_permissions_empty(drive):
    """An empty inline `permissions: []` can never be a file's real ACL (the
    SA's own entry is always on it) — it means files.get didn't return one,
    so fetch_acl must fall back to permissions.list rather than treating []
    as "no one has access"."""
    fake = drive(_FakeDrive(
        get={"id": "f1", "name": "x", "permissions": []},
        perms=[{"type": "user", "emailAddress": ALICE, "role": "reader"}],
    ))
    name, acl = access.fetch_acl("f1")
    assert name == "x"
    assert acl == [{"type": "user", "emailAddress": ALICE, "role": "reader"}]
    assert len(fake.list_calls) == 1


def test_fetch_acl_raises_on_api_error(drive):
    drive(_FakeDrive(get=RuntimeError("403")))
    with pytest.raises(RuntimeError):
        access.fetch_acl("f1")


# --------------------------------------------------------------------------- #
# require_access()
# --------------------------------------------------------------------------- #

@pytest.fixture
def alice(monkeypatch, engaged):
    monkeypatch.setattr(
        access, "resolve_requester",
        lambda: access.Requester(platform="slack", user_id="U1", email=ALICE),
    )
    audit = []
    monkeypatch.setattr(access, "_audit_denied", lambda **kw: audit.append(kw))
    return audit


def test_require_access_returns_none_when_inactive(monkeypatch, drive):
    fake = drive(_FakeDrive(get={"id": "f1", "permissions": []}))
    monkeypatch.setattr(access, "is_check_active", lambda: False)
    assert access.require_access("f1", access.READER) is None
    assert fake.get_calls == []


def test_require_access_allows_reader(drive, alice):
    drive(_FakeDrive(get={"id": "f1", "name": "Doc", "permissions": [
        {"type": "user", "emailAddress": ALICE, "role": "reader"}]}))
    r = access.require_access("f1", access.READER)
    assert r.email == ALICE
    assert alice == []


def test_require_access_acl_cache_is_per_file_not_per_decision(drive, engaged, monkeypatch):
    fake = drive(_FakeDrive(get={"id": "f1", "name": "Doc", "permissions": [
        {"type": "user", "emailAddress": ALICE, "role": "reader"}]}))
    audit = []
    monkeypatch.setattr(access, "_audit_denied", lambda **kw: audit.append(kw))

    monkeypatch.setattr(
        access, "resolve_requester",
        lambda: access.Requester(platform="slack", user_id="U1", email=ALICE),
    )
    r = access.require_access("f1", access.READER)
    assert r.email == ALICE

    bob = access.Requester(platform="slack", user_id="U2", email="bob@everafter.ai")
    monkeypatch.setattr(access, "resolve_requester", lambda: bob)
    with pytest.raises(access.DriveAccessDenied):
        access.require_access("f1", access.READER)

    assert len(fake.get_calls) == 1  # served from cache; Bob still denied


def test_require_access_denies_writer_for_reader(drive, alice):
    drive(_FakeDrive(get={"id": "f1", "name": "Doc", "permissions": [
        {"type": "user", "emailAddress": ALICE, "role": "reader"}]}))
    with pytest.raises(access.DriveAccessDenied) as ei:
        access.require_access("f1", access.WRITER)
    assert ALICE in str(ei.value)
    assert "edit" in str(ei.value)
    assert alice[0]["level"] == "writer" and alice[0]["granted_role"] == "reader"


def test_require_access_denial_never_names_groups_but_audits_them(drive, alice):
    drive(_FakeDrive(get={"id": "f1", "name": "ARR", "permissions": [
        {"type": "group", "emailAddress": "finance@everafter.ai", "role": "reader"}]}))
    with pytest.raises(access.DriveAccessDenied) as ei:
        access.require_access("f1", access.READER)
    assert "finance@" not in str(ei.value)
    assert alice[0]["unmapped_groups"] == ("finance@everafter.ai",)
    assert alice[0]["file_id"] == "f1" and alice[0]["name"] == "ARR"


def test_require_access_denies_when_no_requester(monkeypatch, drive, engaged):
    fake = drive(_FakeDrive(get={"id": "f1", "permissions": []}))
    monkeypatch.setattr(access, "resolve_requester", lambda: None)
    audit = []
    monkeypatch.setattr(access, "_audit_denied", lambda **kw: audit.append(kw))
    with pytest.raises(access.DriveAccessDenied) as ei:
        access.require_access("f1", access.READER)
    assert "requesting user" in str(ei.value)
    assert fake.get_calls == []  # no identity → no API call
    assert audit[0]["reason"] == "no_requester"


def test_require_access_denies_on_fetch_error(drive, alice):
    drive(_FakeDrive(get=RuntimeError("boom")))
    with pytest.raises(access.DriveAccessDenied):
        access.require_access("f1", access.READER)
    assert alice[0]["reason"] == "acl_fetch_failed"


def test_audit_denied_writes_record_access(monkeypatch):
    calls = []
    import agent.data_access_audit as daa

    monkeypatch.setattr(daa, "record_access", lambda **kw: calls.append(kw))
    access._audit_denied(
        file_id="f1", name="ARR", level="reader", requester=ALICE,
        granted_role=None, unmapped_groups=("finance@everafter.ai",), reason="denied",
    )
    assert calls[0]["tool"] == "google_drive"
    assert calls[0]["action"] == "drive_access_denied"
    assert "finance@everafter.ai" in calls[0]["target"]
    assert "f1" in calls[0]["target"] and ALICE in calls[0]["target"]


def test_audit_denied_keeps_unmapped_groups_when_name_is_long(monkeypatch):
    calls = []
    import agent.data_access_audit as daa

    monkeypatch.setattr(daa, "record_access", lambda **kw: calls.append(kw))
    long_name = "x" * 300
    access._audit_denied(
        file_id="f1", name=long_name, level="reader", requester=ALICE,
        granted_role=None, unmapped_groups=("finance@everafter.ai",), reason="denied",
    )
    target = calls[0]["target"]
    assert "finance@everafter.ai" in target
    assert len(target) <= 500


# --------------------------------------------------------------------------- #
# share_with_requester()
# --------------------------------------------------------------------------- #

def test_share_with_requester_creates_writer_permission(drive):
    fake = drive(_FakeDrive())
    r = access.Requester(platform="slack", user_id="U1", email=ALICE)
    assert access.share_with_requester("new1", r) is None
    kw = fake.create_calls[0]
    assert kw["fileId"] == "new1"
    assert kw["body"] == {"type": "user", "role": "writer", "emailAddress": ALICE}
    assert kw["sendNotificationEmail"] is False
    assert kw["supportsAllDrives"] is True


def test_share_with_requester_noop_without_requester(drive):
    fake = drive(_FakeDrive())
    assert access.share_with_requester("new1", None) is None
    assert fake.create_calls == []


def test_share_with_requester_reports_failure(drive, monkeypatch):
    fake = drive(_FakeDrive())

    class _BadP:
        def create(self, **kw):
            return _Req(RuntimeError("quota"))

    monkeypatch.setattr(fake, "permissions", lambda: _BadP())
    r = access.Requester(platform="slack", user_id="U1", email=ALICE)
    msg = access.share_with_requester("new1", r)
    assert msg and ALICE in msg and "quota" in msg


# --------------------------------------------------------------------------- #
# folder_access — operator grants for files whose ACL the SA cannot read
# (a shared-drive folder shared with the SA as a non-member viewer shows an
# empty permission list, which denied everyone — 2026-09-14/16).
# --------------------------------------------------------------------------- #

_TREE = {"pdf1": ["agreements"], "agreements": ["customer"], "customer": ["active"], "active": []}


@pytest.fixture
def tree(monkeypatch):
    calls = []

    def fetch(fid):
        calls.append(fid)
        return _TREE.get(fid, [])

    monkeypatch.setattr(access, "_fetch_parents", fetch)
    access.reset_cache()
    return calls


def test_load_access_config_parses_folder_access(monkeypatch):
    monkeypatch.setattr(access, "_raw_config", lambda: {"google_drive": {"folder_access": {
        "active": {"name": "Active Customers", "readers": ["Pazit@Base.ai"], "writers": ["tal@base.ai"]},
        "bad": "not-a-dict",
    }}})
    cfg = access.load_access_config()
    assert cfg.folder_access == {"active": {
        access.READER: frozenset({"pazit@base.ai"}), access.WRITER: frozenset({"tal@base.ai"})}}


def test_folder_grant_walks_ancestors(tree):
    cfg = _cfg(folder_access={"active": {access.READER: frozenset({ALICE})}})
    assert access.folder_grant("pdf1", ALICE, cfg) == access.READER
    assert access.folder_grant("pdf1", "bob@everafter.ai", cfg) is None


def test_folder_grant_writer_beats_reader(tree):
    cfg = _cfg(folder_access={
        "active": {access.READER: frozenset({ALICE})},
        "customer": {access.WRITER: frozenset({ALICE})},
    })
    assert access.folder_grant("pdf1", ALICE, cfg) == access.WRITER


def test_folder_grant_no_api_calls_when_unconfigured(tree):
    assert access.folder_grant("pdf1", ALICE, _cfg()) is None
    assert tree == []


def test_require_access_folder_grant_rescues_unreadable_acl(drive, alice, tree, monkeypatch):
    drive(_FakeDrive(get={"id": "pdf1", "name": "PO.pdf"}, perms=[]))
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg(
        folder_access={"active": {access.READER: frozenset({ALICE})}}))
    assert access.require_access("pdf1", access.READER).email == ALICE
    with pytest.raises(access.DriveAccessDenied):
        access.require_access("pdf1", access.WRITER)   # readers can't write


def test_unreadable_acl_denial_names_the_real_cause(drive, alice, tree, monkeypatch):
    drive(_FakeDrive(get={"id": "pdf1", "name": "PO.pdf"}, perms=[]))
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg())
    with pytest.raises(access.DriveAccessDenied) as exc:
        access.require_access("pdf1", access.READER)
    assert "sharing settings" in str(exc.value)
    assert "Ask the file's owner" not in str(exc.value)
    assert alice[-1]["reason"] == "acl_unreadable"


def test_filter_listing_honours_folder_grant(drive, alice, tree, monkeypatch):
    drive(_FakeDrive(get={"id": "x"}, perms=[]))
    monkeypatch.setattr(access, "load_access_config", lambda: _cfg(
        folder_access={"agreements": {access.READER: frozenset({ALICE})}}))
    files = [{"id": "pdf1", "name": "PO.pdf", "permissions": []},
             {"id": "other", "name": "elsewhere.pdf", "permissions": []}]
    assert [f["id"] for f in access.filter_listing(files)] == ["pdf1"]
