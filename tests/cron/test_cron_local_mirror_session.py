"""A local-origin cron mirror must address the chat's transcript directly.

A ``local`` conversation's ``chat_id`` IS its session id (``gateway/session_local``), and the
mirror's origin scan admits only OPEN, keyed sessions (``SessionDB.find_session_by_origin``). A
desktop chat that has been closed — the app was quit overnight, or the chat sat idle past an end —
therefore dropped every later cron delivery, which is exactly the "continuable report" the job was
delivering: the chat showed nothing and a reply had no report in context.

The desktop resumes a closed local chat under the SAME id (``SessionDB.reopen_session`` clears the
end stamp), so the pinned id stays the reply surface. The mirror must append to it whether it is
open or not; every other platform keeps the origin scan.
"""

from cron.scheduler_delivery import _local_mirror_session_id, _maybe_mirror_cron_delivery

SID = "local-" + "a" * 64


class TestLocalMirrorSessionId:
    def test_local_chat_id_is_the_session_id(self):
        assert _local_mirror_session_id("local", SID) == SID

    def test_platform_name_is_case_insensitive(self):
        assert _local_mirror_session_id("LOCAL", SID) == SID

    def test_other_platforms_keep_the_origin_scan(self):
        assert _local_mirror_session_id("telegram", "12345") is None
        assert _local_mirror_session_id("slack", "D0AAA") is None
        assert _local_mirror_session_id("bot-chat", "default") is None

    def test_missing_chat_id_is_not_addressed(self):
        assert _local_mirror_session_id("local", None) is None
        assert _local_mirror_session_id("local", "") is None


class TestMirrorCallShape:
    def _capture(self, monkeypatch):
        calls = []
        import gateway.mirror as mirror_mod
        monkeypatch.setattr(
            mirror_mod, "mirror_to_session",
            lambda platform, chat_id, text, **kw: calls.append((platform, chat_id, text, kw)) or True)
        return calls

    def test_local_delivery_passes_the_session_id(self, monkeypatch):
        calls = self._capture(monkeypatch)
        _maybe_mirror_cron_delivery(
            {"id": "j1", "name": "hermes-sync"}, "local", SID, "the brief",
            user_id="uid:502", enabled=True)
        assert len(calls) == 1
        platform, chat_id, text, kw = calls[0]
        assert (platform, chat_id) == ("local", SID)
        assert kw["session_id"] == SID
        assert kw["role"] == "user"
        assert "the brief" in text

    def test_platform_delivery_leaves_the_scan_to_the_mirror(self, monkeypatch):
        calls = self._capture(monkeypatch)
        _maybe_mirror_cron_delivery(
            {"id": "j1", "name": "hermes-sync"}, "telegram", "12345", "the brief", enabled=True)
        assert len(calls) == 1
        assert calls[0][3]["session_id"] is None

    def test_disabled_mirror_writes_nothing(self, monkeypatch):
        calls = self._capture(monkeypatch)
        _maybe_mirror_cron_delivery({"id": "j1"}, "local", SID, "the brief", enabled=False)
        assert calls == []

class TestLocalDeliveryFallback:
    """A local fire outside the owning gateway still lands in the chat's transcript.

    ``deliver_to_local_chat`` needs the owning gateway in-process; a manual fire from the dashboard
    process or a CLI ``cron run`` has none, and refusing the run there loses the brief the chat was
    supposed to show. The history append is the delivery in that case.
    """

    JOB = {
        "id": "j-local", "name": "hermes-sync", "deliver": "origin", "attach_to_session": True,
        "origin": {"platform": "local", "chat_id": SID, "chat_type": "dm", "user_id": "uid:502"},
    }

    def test_history_append_stands_in_when_no_gateway_owns_the_chat(self, monkeypatch):
        from cron.scheduler import _deliver_result
        from gateway import session_cron

        monkeypatch.setattr(session_cron, "deliver_to_local_chat",
                            lambda *a, **k: f"local chat {SID}: no gateway in this process owns its session")
        appended = []
        import gateway.mirror as mirror_mod
        monkeypatch.setattr(mirror_mod, "mirror_to_session",
                            lambda platform, chat_id, text, **kw: appended.append(
                                (platform, chat_id, kw.get("session_id"))) or True)

        err = _deliver_result(dict(self.JOB), "the brief", adapters=None, loop=None)

        assert err is None
        assert appended == [("local", SID, SID)]

    def test_the_error_survives_when_the_append_lands_nowhere(self, monkeypatch):
        from cron.scheduler import _deliver_result
        from gateway import session_cron
        import gateway.mirror as mirror_mod

        monkeypatch.setattr(session_cron, "deliver_to_local_chat",
                            lambda *a, **k: f"local chat {SID}: no gateway in this process owns its session")
        monkeypatch.setattr(mirror_mod, "mirror_to_session", lambda *a, **k: False)

        err = _deliver_result(dict(self.JOB), "the brief", adapters=None, loop=None)

        assert err is not None and "no gateway" in err
