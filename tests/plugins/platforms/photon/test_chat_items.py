"""Photon chat actions are conversation events, not messages.

iMessage stores Name & Photo sharing, group renames and member changes as chat
items. spectrum-ts turns a content-less one into ``custom`` with
``imessage_type: "unsupported-message"``, which used to reach the agent as
"[Photon content type not handled: custom]". The sidecar now drops them in
``normalizeEvent`` using ``plugins/platforms/photon/sidecar/chat-items.mjs``.

These tests execute that real module under node, in the style of
test_url_send_path.py, and check the mirror carries every module index.mjs
imports.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from plugins.platforms.photon import sidecar_paths

_SIDECAR = Path("plugins/platforms/photon/sidecar").resolve()
_MODULE = _SIDECAR / "chat-items.mjs"

_UNSUPPORTED = {"type": "custom", "raw": {"imessage_type": "unsupported-message"}}

_CASES = {
    "name_and_photo_share_is_dropped": ({"itemType": "chatAction", "content": _UNSUPPORTED}, "chatAction"),
    "group_rename_is_dropped": ({"itemType": "groupNameChange", "content": _UNSUPPORTED}, "groupNameChange"),
    "member_change_is_dropped": ({"itemType": "participantChange", "content": _UNSUPPORTED}, "participantChange"),
    "normal_text_is_forwarded": ({"itemType": "normal", "content": {"type": "text", "text": "hi"}}, None),
    "normal_custom_is_forwarded": ({"itemType": "normal", "content": _UNSUPPORTED}, None),
    "chat_action_with_readable_content_is_forwarded": (
        {"itemType": "chatAction", "content": {"type": "text", "text": "renamed"}}, None,
    ),
    "unknown_item_type_is_forwarded": ({"itemType": "unknown", "content": _UNSUPPORTED}, None),
    "missing_item_type_is_forwarded": ({"content": _UNSUPPORTED}, None),
}


def _run(messages):
    harness = (
        f"import {{ nonMessageItemType }} from {json.dumps(_MODULE.as_uri())};\n"
        f"const messages = {json.dumps(messages)};\n"
        "console.log(JSON.stringify(messages.map(nonMessageItemType)));\n"
    )
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        text=True, capture_output=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_chat_items_classification():
    names = list(_CASES)
    got = _run([_CASES[n][0] for n in names])
    assert dict(zip(names, got)) == {n: _CASES[n][1] for n in names}


def test_null_message_is_forwarded():
    assert _run([None]) == [None]


def test_mirror_carries_every_local_module_index_imports():
    source = (_SIDECAR / "index.mjs").read_text(encoding="utf-8")
    local = set(re.findall(r'from\s+"\./([^"]+)"', source))
    assert local, "expected index.mjs to import local modules"
    assert local <= set(sidecar_paths._MIRROR_FILES)
