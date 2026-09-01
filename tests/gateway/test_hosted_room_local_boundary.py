"""Dependency boundary for the same-gateway hosted-room backend."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOM_MODULES = (
    "gateway/hosted_room_discussion.py",
    "gateway/hosted_room_driver.py",
    "gateway/hosted_room_policy_checkpoint.py",
    "tui_gateway/hosted_room_driver.py",
    "tui_gateway/hosted_room_service.py",
    "tui_gateway/hosted_room_server_rpc.py",
    "tui_gateway/methods_groups.py",
)
FORBIDDEN_SURFACES = (
    "apps/desktop",
    "attachments",
    "artifact",
    "gateway.platforms",
    "hosted_room_links",
    "hosted_room_peer",
    "messaging_refs",
    "roomlink",
    "transport_resolver",
    "turn.handoff",
)


def test_local_room_modules_do_not_depend_on_excluded_surfaces():
    violations = {}
    for relative_path in LOCAL_ROOM_MODULES:
        source = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        found = [token for token in FORBIDDEN_SURFACES if token in source]
        if found:
            violations[relative_path] = found

    assert violations == {}
