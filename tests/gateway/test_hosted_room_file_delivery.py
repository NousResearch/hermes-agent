"""Finite file inputs survive reply limits and transcript eviction."""
import pytest

from tests.gateway.test_hosted_discussion_file_context import _settle, plan, roster, send, upload
from tests.gateway.test_hosted_room_log_file_context import decision


@pytest.mark.parametrize("recovery", ["live", "reopen", 4, 5])
def test_previously_unaddressed_bot_gets_all_17_files(tmp_path, monkeypatch, recovery):
    with roster(tmp_path, monkeypatch, "abcdef") as service:
        received = {member: set() for member in "abcdef"}
        inputs = set()
        for start in range(0, 17, 8):
            files = [upload(service, n) for n in range(start, min(start + 8, 17))]
            inputs.update(item["attachment_id"] for item in files)
            send(service, start, files, text="@a @b @c @d @e Review")
        while (task := decision(service, recovery).task) is not None:
            received[task.member.member_id].update(a["attachment_id"] for a in task.payload["attachments"])
            _settle(service, task, "Reviewed.")
        assert received == {member: inputs if member != "f" else set() for member in "abcdef"}
        send(service, "all", text="@all Update your reports")
        for n in range(50):
            current = decision(service, recovery)
            if current.task is None:
                break
            task = current.task
            assert task.seen_through_seq > task.payload["input_context"]["watermark"]
            received[task.member.member_id].update(
                a["attachment_id"] for a in task.payload.get("attachments", []) if a["attachment_id"] in inputs)
            _settle(service, task, "@all Updated report", [upload(service, f"report-{n}")])
        else:
            pytest.fail("finite inputs and repeated reports did not stop")
        assert received == dict.fromkeys("abcdef", inputs)
        assert current.status == "bounded" and current.reason == "max_messages"


def test_unread_bot_file_keeps_its_original_discussion_receipt(tmp_path, monkeypatch):
    with roster(tmp_path, monkeypatch, "ab") as service:
        send(service, "original", text="@a Write a report")
        report = upload(service, "report")
        _settle(service, plan(service), "Report ready.", [report])
        for n in range(24):
            send(service, n, text="@a Keep working")
        send(service, "reader", text="@b Read the earlier report")
        task = plan(service)
        assert task.member.member_id == "b"
        assert [item["attachment_id"] for item in task.payload["attachments"]] == [report["attachment_id"]]
        _settle(service, task, "Read it.")
        send(service, "release", text="Thanks")
        assert plan(service) is not None
