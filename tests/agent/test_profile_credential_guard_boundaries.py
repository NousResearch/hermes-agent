"""The cross-profile boundary, proven in BOTH directions.

DENY    — another profile's credential stores are unreachable through the file tools
          (``write_file`` / ``read_file`` / ``patch``): its ``.env`` (every bot token and
          API key), its ``vault/`` (key + ciphertext side by side) and its ``cron/`` store
          (a ``jobs.json`` write is scheduled execution as that profile).

ALLOW   — the sanctioned sibling lifecycle route stays permitted: ``hermes -p <other>
          gateway restart|stop`` operates on a sibling's gateway and is a normal fleet
          operation (guard #78028). The credential widening must not swallow it, so the
          deny half and the allow half are asserted *side by side* in one test below: a
          future patch that tightens the deny set into profile isolation fails here.

CONTROL — the ACTIVE profile's own ``.env`` keeps exactly its pre-patch contract: read
          blocked, write denied. Measured on pristine ``origin/main`` before this patch
          (``is_write_denied`` already True there, because ``.env`` is in ``hermes_files``
          and that set was applied to the active home), so the assertions below *preserve*
          the contract rather than change it. The supported path for a credential change
          is the user's own edit / ``hermes config``, not an agent's ``write_file`` — the
          point of the patch is that the same bar now applies to a sibling profile, which
          had NO bar at all.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import agent.file_safety as fs


@pytest.fixture()
def hermes_layout(tmp_path, monkeypatch):
    """Active profile ``sol`` and sibling profile ``min`` under one root."""
    root = tmp_path / "hermes_root"
    active = root / "profiles" / "sol"
    sibling = root / "profiles" / "min"
    for home in (active, sibling):
        home.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: active)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, active, sibling


def _touch(base: Path, rel: str, body: str = "dummy") -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# Secret stores a sibling profile keeps in parallel. Measured on pristine origin/main:
# every one of these was writable through write_file.
SIBLING_WRITE_DENIED = (".env", "vault/vault.key", os.path.join("cron", "jobs.json"))
# Of those, the ones whose CONTENT is also a credential (write-only denial would still
# leak the live token). cron/jobs.json is deliberately absent: a job list is not secret,
# and the exposure closed there is the write (scheduled execution as that profile).
SIBLING_READ_DENIED = (".env", "vault/vault.key")


def test_sibling_credential_stores_are_denied_through_the_file_tools(hermes_layout):
    """The exposure this patch closes: another profile's .env / vault.key / cron/jobs.json
    were writable (create path, so the read-before-write guard is not what refuses)."""
    _root, _active, sibling = hermes_layout
    from tools.file_tools import write_file_tool

    for rel in SIBLING_WRITE_DENIED:
        target = sibling / rel  # does not exist: isolates the credential guard
        result = json.loads(write_file_tool(str(target), "TELEGRAM_BOT_TOKEN=attacker\n"))
        assert "denied" in (result.get("error") or "").lower(), (
            f"write_file may plant/replace a sibling profile store: {rel}: {result}"
        )
        assert not target.exists(), f"sibling store written despite the denial: {target}"

    # The read half — same stores, real files.
    from tools.file_tools import read_file_tool

    for rel in SIBLING_READ_DENIED:
        stored = _touch(sibling, rel, "TELEGRAM_BOT_TOKEN=sibling-secret\n")
        result = json.loads(read_file_tool(str(stored)))
        assert "denied" in (result.get("error") or "").lower(), (
            f"read_file hands the agent a sibling profile store: {rel}: {result}"
        )


def test_own_profile_env_contract_is_unchanged(hermes_layout):
    """Preserved, not introduced: an agent's own .env was already read-blocked and
    write-denied before this patch. Widening the set to siblings must not relax it, and
    the control fails loudly if a later change opens the active home instead."""
    _root, active, _sibling = hermes_layout
    from tools.file_tools import write_file_tool

    # Fresh path in the active home: the credential guard, not read-before-write, is what
    # has to refuse it.
    oauth_store = active / ".anthropic_oauth.json"
    result = json.loads(write_file_tool(str(oauth_store), "{\"token\": \"attacker\"}\n"))
    assert "denied" in (result.get("error") or "").lower(), (
        f"own credential store writable through the tool: {result}"
    )
    assert not oauth_store.exists()

    own_env = _touch(active, ".env", "TELEGRAM_BOT_TOKEN=own-secret\n")
    assert fs.is_write_denied(str(own_env)) is True, "own .env became writable"
    assert fs.get_read_block_error(str(own_env)), "own .env became readable"


def test_sibling_gateway_restart_stays_permitted(monkeypatch):
    """THE ALLOW DIRECTION. ``hermes -p <other> gateway restart`` must keep working: it is
    how one agent bounces a sibling's gateway (#78028). Only the self-targeting shape is
    the foot-gun the lifecycle guard blocks."""
    from cron import lifecycle_guard as guard

    monkeypatch.setattr(guard, "_current_profile_name", lambda: "sol")

    for command in (
        "hermes -p min gateway restart",
        "hermes --profile min gateway restart",
    ):
        assert guard.contains_gateway_lifecycle_command(command) is False, (
            f"sibling lifecycle route blocked: {command}"
        )
        assert guard.contains_gateway_lifecycle_command_or_referenced_script(command) is False

    # Self-target still blocked — the allow is scoped, not a hole.
    assert guard.contains_gateway_lifecycle_command("hermes -p sol gateway restart") is True
    assert guard.contains_gateway_lifecycle_command("hermes gateway restart") is True

    # Permitted, but still consent-gated at the terminal layer: sanctioned ≠ silent.
    from tools.approval_detection import detect_dangerous_command

    flagged, reason, _key = detect_dangerous_command("hermes -p min gateway restart")
    assert flagged is True, "sibling restart should still prompt for approval"
    assert "gateway" in (reason or "").lower()


def test_both_directions_hold_at_once(hermes_layout, monkeypatch):
    """One boundary, two truths. A patch that closes the credential hole by isolating
    profiles (the retired #97165 approach) cannot pass this test."""
    _root, _active, sibling = hermes_layout
    from cron import lifecycle_guard as guard

    monkeypatch.setattr(guard, "_current_profile_name", lambda: "sol")
    sibling_env = sibling / ".env"

    assert fs.is_write_denied(str(sibling_env)) is True, "sibling .env writable"
    assert guard.contains_gateway_lifecycle_command("hermes -p min gateway restart") is False, (
        "sibling gateway restart blocked by the widening"
    )

    # And the fleet route does not depend on the guarded file layer at all: a sibling's
    # own skills/notes stay writable, which is the contract the widening must keep.
    from tools.file_tools import write_file_tool

    notes = sibling / "workspace" / "new-note.md"  # fresh path: read-before-write does not fire
    result = json.loads(write_file_tool(str(notes), "new\n"))
    assert not result.get("error"), f"sibling workspace write refused: {result}"
    assert notes.read_text() == "new\n"
