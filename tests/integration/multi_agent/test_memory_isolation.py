"""B (memory half) — per-agent MEMORY isolation.

The existing suite proves HOME + SOUL + credential isolation. This closes the
loop on the *persistent* per-agent surface: the memory store. A memory written
while running as agent A must land under A's own home (``profiles/A/memories``)
and must NOT be readable by a run scoped to agent B.

We exercise the REAL memory subsystem (``tools.memory_tool.MemoryStore``, whose
``get_memory_dir()`` resolves through ``get_hermes_home()``) under the SAME
profile+secret scope a real run enters (``_use_profile_and_secret_scope``, the
wrapper ``APIServerAdapter._run_agent`` uses inside the executor thread). So the
redirection under test is the production one, not a stand-in.
"""
from pathlib import Path

import pytest

from gateway.platforms.api_server import _use_profile_and_secret_scope
from tools.memory_tool import MemoryStore, get_memory_dir

CODER_MEMO = "Coder private note alpha-7: the widget build lives in module X."
RESEARCH_MEMO = "Research private note beta-3: the survey cohort is N=42."


def _write_memory(text: str) -> Path:
    """Add a MEMORY.md entry under the currently-scoped agent home; return the
    resolved memory dir."""
    store = MemoryStore()
    result = store.add("memory", text)
    assert result.get("success"), f"memory write failed: {result}"
    return get_memory_dir()


def _read_memory_entries() -> list:
    store = MemoryStore()
    store.load_from_disk()
    return list(store.memory_entries)


def test_memory_written_as_A_is_isolated_from_B(integ):
    """A memory written as coder lands in coder's home and is absent from a
    research-scoped read; and vice-versa — no cross-agent leakage."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"},
                multiplex=True)
    coder = env.adapter._profile_for_agent_id("coder")
    research = env.adapter._profile_for_agent_id("research")
    assert coder is not None and research is not None

    # --- write as coder ------------------------------------------------
    with _use_profile_and_secret_scope(coder):
        coder_dir = _write_memory(CODER_MEMO)
        # Artifact physically lands under coder's OWN home.
        assert coder_dir == coder.resolved_home / "memories"
        coder_file = coder_dir / "MEMORY.md"
        assert coder_file.exists()
        assert CODER_MEMO in coder_file.read_text()

    # --- write as research (its own distinct memo) ---------------------
    with _use_profile_and_secret_scope(research):
        research_dir = _write_memory(RESEARCH_MEMO)
        assert research_dir == research.resolved_home / "memories"
        assert research_dir != coder_dir

    # --- read as research: coder's memo must NOT surface ---------------
    with _use_profile_and_secret_scope(research):
        research_entries = _read_memory_entries()
    assert CODER_MEMO not in research_entries, (
        "coder's memory leaked into research's scope")
    assert RESEARCH_MEMO in research_entries  # research sees its own

    # --- read as coder: research's memo must NOT surface ---------------
    with _use_profile_and_secret_scope(coder):
        coder_entries = _read_memory_entries()
    assert RESEARCH_MEMO not in coder_entries, (
        "research's memory leaked into coder's scope")
    assert CODER_MEMO in coder_entries  # coder sees its own


def test_agent_memory_file_absent_under_other_agent_home(integ):
    """Filesystem ground truth: coder's MEMORY.md exists under profiles/coder
    and there is no such file under profiles/research (research never wrote)."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"},
                multiplex=True)
    coder = env.adapter._profile_for_agent_id("coder")
    research = env.adapter._profile_for_agent_id("research")

    with _use_profile_and_secret_scope(coder):
        _write_memory(CODER_MEMO)

    coder_file = coder.resolved_home / "memories" / "MEMORY.md"
    research_file = research.resolved_home / "memories" / "MEMORY.md"
    assert coder_file.exists() and CODER_MEMO in coder_file.read_text()
    # research wrote nothing → no file, and certainly not coder's content.
    assert not research_file.exists()
