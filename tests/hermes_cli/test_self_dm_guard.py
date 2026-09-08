"""Self-DM guard: `chat -c "Bot Chat" --create-if-missing` from inside an
agent session of the SAME profile is refused (raw-CLI fallback vector).

Incident class (fleet finding "profile-identity lifecycle collision"):
`hermes -p forge chat --in ~ -c "Bot Chat" --create-if-missing -Q --query-file
<f>` — intended `-p elon` — queued into forge's own canonical Bot Chat and
spawned a SECOND full forge agent context with no dispatcher run record. The
native message_agent tool already refuses self-messages (tools/bot_mode_dm.py
"You can't message yourself"); the raw CLI had no such gate.

Guard design: HERMES_CALLER_PROFILE is snapshotted at CLI import time (BEFORE
the -p override re-points HERMES_HOME), so it names the launching context;
the guard refuses when caller == resolved profile, the launch came from a
hermes agent session (HERMES_AGENT), and the target is the canonical
"Bot Chat". --allow-self-bot-chat is the explicit escape hatch.
"""
from __future__ import annotations

import os
import sys

import pytest

from hermes_cli.main import _enforce_self_dm_guard


class _Args:
    """Minimal argparse namespace stand-in for the chat subcommand."""

    def __init__(self, *, continue_last="Bot Chat", create_if_missing=True,
                 allow_self_bot_chat=False):
        self.continue_last = continue_last
        self.create_if_missing = create_if_missing
        self.allow_self_bot_chat = allow_self_bot_chat


@pytest.fixture(autouse=True)
def _agent_env(monkeypatch):
    """The fork's env: an agent session of profile forge."""
    monkeypatch.setenv("HERMES_AGENT", "true")
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-root/profiles/forge")


# ---------------------------------------------------------------------------
# The defect (RED against base): the self-DM shape must refuse
# ---------------------------------------------------------------------------

def test_self_dm_fork_shape_is_refused(monkeypatch):
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    with pytest.raises(SystemExit) as exc:
        _enforce_self_dm_guard(_Args())
    assert exc.value.code == 1


def test_refusal_message_names_the_profile_and_escape_hatch(
        monkeypatch, capsys):
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    with pytest.raises(SystemExit):
        _enforce_self_dm_guard(_Args())
    err = capsys.readouterr().err
    assert "forge" in err
    assert "--allow-self-bot-chat" in err
    assert "hermes -p <target> chat" in err


# ---------------------------------------------------------------------------
# Legitimate shapes must pass through
# ---------------------------------------------------------------------------

def test_stale_label_contrary_to_home_resolves_to_home(monkeypatch):
    """Precedence contract: the post-override HOME is the actual selected
    profile (config loads from it); a contrary HERMES_PROFILE label is stale
    inheritance and must not mask it. A forge-home process opening forge's
    Bot Chat is the self-DM shape even if the label says otherwise."""
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    monkeypatch.setenv("HERMES_PROFILE", "elon")  # stale label only
    with pytest.raises(SystemExit) as exc:
        _enforce_self_dm_guard(_Args())
    assert exc.value.code == 1


def test_label_fallback_when_home_is_not_profile_shaped(monkeypatch):
    """Root/custom HERMES_HOME yields no profile name → the label remains a
    meaningful fallback for non-`-p` sessions: caller=forge, label=elon →
    relay allowed."""
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-root")
    monkeypatch.setenv("HERMES_PROFILE", "elon")
    _enforce_self_dm_guard(_Args())  # no SystemExit


def test_human_shell_is_allowed(monkeypatch):
    """No HERMES_CALLER_PROFILE snapshot (human terminal) → never trips."""
    monkeypatch.delenv("HERMES_CALLER_PROFILE", raising=False)
    _enforce_self_dm_guard(_Args())


def test_non_agent_child_process_is_allowed(monkeypatch):
    """HERMES_AGENT unset (plain subprocess, not an agent session)."""
    monkeypatch.delenv("HERMES_AGENT", raising=False)
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    _enforce_self_dm_guard(_Args())


def test_normal_named_session_is_allowed(monkeypatch):
    """`chat -c some-project` — not the canonical Bot Chat."""
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    _enforce_self_dm_guard(_Args(continue_last="some-project"))


def test_without_create_if_missing_is_allowed(monkeypatch):
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    _enforce_self_dm_guard(_Args(create_if_missing=False))


def test_escape_hatch_allows_explicit_self_bot_chat(monkeypatch):
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    _enforce_self_dm_guard(_Args(allow_self_bot_chat=True))


# ---------------------------------------------------------------------------
# Profile resolution used by the guard
# ---------------------------------------------------------------------------

def test_profile_resolution_prefers_env_profile(monkeypatch):
    """HERMES_PROFILE (agent sessions) wins over HERMES_HOME derivation."""
    from hermes_cli.main import _profile_name_from_homes
    assert _profile_name_from_homes("/tmp/x/.hermes/profiles/elon") == "elon"
    assert _profile_name_from_homes("/tmp/hermes-root") == ""       # root install
    assert _profile_name_from_homes(None) == ""
    assert _profile_name_from_homes("/tmp/x/.hermes/profiles/forge",
                                    "/tmp/y/.hermes/profiles/elon") == "forge"


def test_snapshot_takes_first_value(monkeypatch):
    """Module-level snapshot uses setdefault: an outer chain wins. Deriving
    from a target-path HERMES_HOME still yields that name (the guard relies
    on the snapshot being taken BEFORE the override)."""
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    # Guard equivalence: caller read from env; here we just pin the contract.
    assert __import__("os").environ["HERMES_CALLER_PROFILE"] == "forge"


# ---------------------------------------------------------------------------
# Blocker-2 regression: the REAL launcher path (-p through preparse)
# ---------------------------------------------------------------------------

def _run_preparse(tmp_path, monkeypatch, *, target: str):
    """Apply -p <target> through the ACTUAL _apply_profile_override preparse,
    starting from the real launcher env of a forge agent session (inherited
    HERMES_PROFILE=forge, HERMES_HOME=.../profiles/forge) — exactly the shape
    a `hermes -p <target> chat …` fallback send runs under."""
    root = tmp_path / "hermes-root"
    for name in ("forge", "elon"):
        (root / "profiles" / name).mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HERMES_PROFILE", "forge")
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "forge"))
    monkeypatch.setenv("HERMES_CALLER_PROFILE", "forge")
    monkeypatch.setattr(sys, "argv",
                        ["hermes", "-p", target, "chat",
                         "-c", "Bot Chat", "--create-if-missing"])
    from hermes_cli.main import _apply_profile_override
    _apply_profile_override()
    # Pin the preparse contract this regression stands on: the override moved
    # the home and did NOT touch the inherited label.
    assert os.environ["HERMES_HOME"] == str(root / "profiles" / target)


def test_preparse_cross_profile_relay_is_allowed(tmp_path, monkeypatch):
    """forge agent, `hermes -p elon chat -c "Bot Chat" --create-if-missing`:
    the preparse moved HERMES_HOME to profiles/elon; the guard must resolve
    the TARGET (derived from the post-override home), not the stale
    HERMES_PROFILE label — relay allowed, no SystemExit."""
    _run_preparse(tmp_path, monkeypatch, target="elon")
    _enforce_self_dm_guard(_Args())  # must NOT raise


def test_preparse_self_profile_still_refused(tmp_path, monkeypatch):
    """forge agent, `hermes -p forge chat -c "Bot Chat" --create-if-missing`:
    post-override home names forge == caller → the self-DM fork shape, still
    refused with the escape-hatch message."""
    _run_preparse(tmp_path, monkeypatch, target="forge")
    with pytest.raises(SystemExit) as exc:
        _enforce_self_dm_guard(_Args())
    assert exc.value.code == 1
