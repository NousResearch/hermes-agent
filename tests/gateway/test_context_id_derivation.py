"""Tests for gateway per-context memory id derivation.

`_context_id_for_source` turns a messaging SessionSource into the context id
used to partition memory. The guarantees under test:

  - Every chat type is scoped, DMs included (fixes the leak where DMs were
    left unscoped and shared one global memory).
  - Ids are platform-qualified, so the same raw chat_id on two platforms — or
    the same id used for a DM and a group — never collide.
  - No chat_id ⇒ None ⇒ unscoped/global memory (interactive CLI, no-chat).
  - Path-unsafe chat_ids are sanitized by MemoryStore and stay under contexts/.

Uses the real Platform enum plus a lightweight fake source; no real chat ids.
"""

import sys
from dataclasses import dataclass
from typing import Optional

import pytest

from gateway.run import _context_id_for_source
from gateway.config import Platform
from tools.memory_tool import MemoryStore, derive_context_id

# Patch get_memory_dir on the exact module object MemoryStore came from, not by
# dotted string: tests/tools/test_memory_tool_import_fallback.py reimports
# tools.memory_tool, so a later sys.modules entry can be a different object.
_MEM_MOD = sys.modules[MemoryStore.__module__]


@dataclass
class FakeSource:
    """Minimal stand-in for gateway SessionSource: only the attributes the
    derivation reads."""
    platform: object
    chat_type: str
    chat_id: Optional[str]


def test_dm_is_scoped():
    """A DM is scoped (non-None) — the leak this fixes was DMs going unscoped."""
    cid = _context_id_for_source(FakeSource(Platform.TELEGRAM, "dm", "u1"))
    assert cid is not None
    assert cid == "telegram:dm:u1"


def test_group_is_scoped():
    cid = _context_id_for_source(FakeSource(Platform.TELEGRAM, "group", "123"))
    assert cid == "telegram:group:123"


def test_no_chat_id_is_unscoped():
    """Missing chat_id ⇒ None ⇒ global/unscoped memory."""
    assert _context_id_for_source(FakeSource(Platform.LOCAL, "dm", None)) is None
    assert _context_id_for_source(FakeSource(Platform.LOCAL, "dm", "")) is None


def test_platform_qualified_ids_dont_collide():
    """Same chat_id '123' as a group on TELEGRAM vs SLACK → distinct ids."""
    tg = _context_id_for_source(FakeSource(Platform.TELEGRAM, "group", "123"))
    sl = _context_id_for_source(FakeSource(Platform.SLACK, "group", "123"))
    assert tg != sl
    assert tg == "telegram:group:123"
    assert sl == "slack:group:123"


def test_dm_and_group_same_chat_id_distinct():
    """Same chat_id '123' as a DM vs a group → distinct ids (no cross-type leak)."""
    dm = _context_id_for_source(FakeSource(Platform.TELEGRAM, "dm", "123"))
    grp = _context_id_for_source(FakeSource(Platform.TELEGRAM, "group", "123"))
    assert dm != grp


def test_string_platform_supported():
    """A plugin platform passed as a bare string (no .value) still derives."""
    cid = _context_id_for_source(FakeSource("irc", "group", "42"))
    assert cid == "irc:group:42"


def test_context_id_path_safe(tmp_path, monkeypatch):
    """A chat_id containing path-traversal sequences is sanitized by the store
    and stays under contexts/ — never escapes the memories dir."""
    monkeypatch.setattr(_MEM_MOD, "get_memory_dir", lambda: tmp_path)

    cid = _context_id_for_source(FakeSource(Platform.SLACK, "group", "../../etc/x"))
    assert cid is not None

    store = MemoryStore(context_id=cid)
    store.load_from_disk()
    store.add("memory", "trapped")

    # No escape outside the memories tree.
    assert not (tmp_path.parent.parent / "etc" / "x" / "MEMORY.md").exists()
    # The scoped file lives under contexts/.
    contexts = tmp_path / "contexts"
    assert contexts.exists()
    written = list(contexts.rglob("MEMORY.md"))
    assert written, "expected a scoped MEMORY.md under contexts/"
    for p in written:
        assert contexts in p.parents


class TestLiveDerivationPath:
    """agent/agent_init.py is the LIVE construction site: it calls
    tools.memory_tool.derive_context_id directly with the agent's stored
    platform string / chat_type / chat_id (the gateway threads these onto the
    agent). These tests exercise that shared core so the wired path — not only
    the gateway-typed _context_id_for_source adapter — is covered, and assert
    the two entry points agree."""

    def test_derive_context_id_matches_source_helper(self):
        """The bare-string core and the SessionSource adapter agree, so the
        gateway and agent_init compute identical ids."""
        via_core = derive_context_id("telegram", "dm", "u1")
        via_source = _context_id_for_source(FakeSource(Platform.TELEGRAM, "dm", "u1"))
        assert via_core == via_source == "telegram:dm:u1"

    def test_cli_like_source_is_unscoped(self):
        """The interactive CLI passes platform='cli' with no chat_id → None,
        so a scoped store is never built even if scoping is enabled."""
        assert derive_context_id("cli", None, None) is None
        assert derive_context_id("cli", "dm", "") is None

    def test_dm_scoped_via_core(self):
        assert derive_context_id("slack", "dm", "u9") == "slack:dm:u9"


def test_derivation_matches_stored_context_id(tmp_path, monkeypatch):
    """The colon-delimited id resolves to a single directory under contexts/
    (the ':' separators are kept verbatim, not treated as path boundaries)."""
    monkeypatch.setattr(_MEM_MOD, "get_memory_dir", lambda: tmp_path)
    cid = _context_id_for_source(FakeSource(Platform.TELEGRAM, "dm", "u1"))
    store = MemoryStore(context_id=cid)
    store.load_from_disk()
    store.add("memory", "hi")
    assert (tmp_path / "contexts" / "telegram:dm:u1" / "MEMORY.md").exists()
