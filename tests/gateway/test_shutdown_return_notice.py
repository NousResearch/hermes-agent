"""Chats warned that the gateway is going down are told when it comes back.

Before this, only an in-chat ``/restart`` requester (``.restart_notify.json``) or a configured
home channel learned the gateway was back. A restart from outside the process — ``systemctl
restart``, an update, a reboot — left the active chats that DID get the "going down" notice
silent, so the user had to ask "are you back?".
"""

import json

import gateway.shutdown_notice as rs


def test_marker_round_trip(tmp_path):
    targets = [
        {"platform": "telegram", "chat_id": "8607228256", "thread_id": None},
        {"platform": "whatsapp", "chat_id": "30094903001245@lid", "thread_id": "7"},
    ]
    rs.mark_shutdown_notified(tmp_path, targets)
    assert rs.read_shutdown_notice_marker(tmp_path) == targets
    rs.clear_shutdown_notice_marker(tmp_path)
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_missing_marker_reads_empty(tmp_path):
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_corrupt_marker_reads_empty(tmp_path):
    rs._shutdown_notice_marker(tmp_path).write_text("{not json", encoding="utf-8")
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_marker_without_targets_list_reads_empty(tmp_path):
    rs._shutdown_notice_marker(tmp_path).write_text(json.dumps({"targets": "nope"}), encoding="utf-8")
    assert rs.read_shutdown_notice_marker(tmp_path) == []


def test_non_dict_entries_are_dropped(tmp_path):
    rs._shutdown_notice_marker(tmp_path).write_text(
        json.dumps({"targets": [{"platform": "telegram", "chat_id": "1"}, "junk", 3]}), encoding="utf-8"
    )
    assert rs.read_shutdown_notice_marker(tmp_path) == [{"platform": "telegram", "chat_id": "1"}]


def test_empty_list_does_not_create_a_marker(tmp_path):
    rs.mark_shutdown_notified(tmp_path, [])
    assert not rs._shutdown_notice_marker(tmp_path).exists()


def test_clear_is_idempotent(tmp_path):
    rs.clear_shutdown_notice_marker(tmp_path)  # no marker yet: must not raise
    rs.mark_shutdown_notified(tmp_path, [{"platform": "telegram", "chat_id": "1"}])
    rs.clear_shutdown_notice_marker(tmp_path)
    rs.clear_shutdown_notice_marker(tmp_path)
    assert rs.read_shutdown_notice_marker(tmp_path) == []
