"""Unit tests for the Mattermost approval-button callback logic (pure)."""

from plugins.platforms.mattermost import approval as mm


def _ctx(**kw):
    base = {"kind": "thread", "token": "secret", "post_ref": "ref1",
            "choice": "once", "session_key": "sk"}
    base.update(kw)
    return base


def _secret_for(expected):
    return lambda ref: expected if ref == "ref1" else None


class TestAuth:
    def test_empty_allowlist_denies(self):
        assert mm.is_user_authorized("u1", set()) is False

    def test_wildcard_allows(self):
        assert mm.is_user_authorized("u1", {"*"}) is True

    def test_explicit_user(self):
        assert mm.is_user_authorized("u1", {"u1"}) is True
        assert mm.is_user_authorized("u2", {"u1"}) is False


class TestCallback:
    def test_unauthorized_user_rejected(self):
        store = {"ref1": False}
        r = mm.handle_callback(_ctx(), "intruder",
                               allowed_users={"alice"},
                               expected_secret_for=_secret_for("secret"),
                               resolved_store=store)
        assert r["ok"] is False and r["status"] == "unauthorized"
        # Store untouched (no resolution happened).
        assert store == {"ref1": False}

    def test_bad_token_rejected(self):
        store = {"ref1": False}
        r = mm.handle_callback(_ctx(token="wrong"), "alice",
                               allowed_users={"alice"},
                               expected_secret_for=_secret_for("secret"),
                               resolved_store=store)
        assert r["ok"] is False and r["status"] == "bad_token"
        assert store == {"ref1": False}

    def test_double_click_guard(self):
        store = {"ref1": False}
        resolved = []
        kw = dict(allowed_users={"alice"},
                  expected_secret_for=_secret_for("secret"),
                  resolved_store=store,
                  resolve_fn=lambda sk, ch: resolved.append((sk, ch)))
        r1 = mm.handle_callback(_ctx(), "alice", **kw)
        r2 = mm.handle_callback(_ctx(), "alice", **kw)
        assert r1["ok"] is True
        assert r2["status"] == "already_resolved"
        # resolve_fn invoked exactly once.
        assert resolved == [("sk", "once")]

    def test_thread_resolve_passes_choice(self):
        store = {"ref1": False}
        resolved = []
        r = mm.handle_callback(_ctx(choice="session"), "alice",
                               allowed_users={"alice"},
                               expected_secret_for=_secret_for("secret"),
                               resolved_store=store,
                               resolve_fn=lambda sk, ch: resolved.append((sk, ch)))
        assert r["ok"] is True
        assert resolved == [("sk", "session")]
        assert "session" in r["update_text"].lower()

    def test_card_approve_calls_approve_fn(self):
        store = {"ref1": False}
        approved = []
        r = mm.handle_callback(
            _ctx(kind="card", pending_id="p1", choice="once", session_key=""),
            "alice",
            allowed_users={"alice"},
            expected_secret_for=_secret_for("secret"),
            resolved_store=store,
            approve_fn=lambda pid: approved.append(pid) or {"ok": True},
        )
        assert r["ok"] is True
        assert approved == ["p1"]

    def test_card_deny_calls_discard(self):
        store = {"ref1": False}
        discarded = []
        r = mm.handle_callback(
            _ctx(kind="card", pending_id="p1", choice="deny", session_key=""),
            "alice",
            allowed_users={"alice"},
            expected_secret_for=_secret_for("secret"),
            resolved_store=store,
            discard_fn=lambda pid: discarded.append(pid),
        )
        assert r["ok"] is True
        assert discarded == ["p1"]
        assert "deni" in r["update_text"].lower()


class TestBuilders:
    def test_thread_attachment_has_scope_buttons(self):
        att = mm.build_approval_attachment(
            text="x", callback_url="http://cb", kind="thread",
            token="t", post_ref="r", session_key="sk")
        names = [a["name"] for a in att["actions"]]
        assert names == ["Approve", "Approve (session)", "Approve (always)", "Deny"]
        for a in att["actions"]:
            assert a["integration"]["url"] == "http://cb"
            assert a["integration"]["context"]["token"] == "t"
            assert a["integration"]["context"]["session_key"] == "sk"

    def test_card_attachment_is_approve_deny_only(self):
        att = mm.build_approval_attachment(
            text="x", callback_url="http://cb", kind="card",
            token="t", post_ref="r", pending_id="p1")
        names = [a["name"] for a in att["actions"]]
        assert names == ["Approve", "Deny"]
        assert att["actions"][0]["integration"]["context"]["pending_id"] == "p1"
