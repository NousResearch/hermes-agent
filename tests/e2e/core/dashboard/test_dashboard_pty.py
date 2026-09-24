"""Dashboard embedded chat: the ``/api/pty`` WebSocket drives a REAL ``hermes --tui`` end to end.

Issue class: the dashboard Chat tab talking to the wrong profile, or losing the conversation on a
reconnect. Every cell runs the real ``hermes dashboard``, which
spawns the real Ink TUI (``ui-tui/dist/entry.js``) and its ``tui_gateway`` under a real PTY; the
client is what the browser's xterm does (raw PTY bytes, the ``\\x1b[RESIZE:c;r]`` control frame, the
SPA's ``?attach=`` keep-alive token, ``?profile=`` and a per-(profile) ``?channel=``). Each profile
has its own fake provider that only accepts that profile's key and echoes the prompt's canary, so
every recorded request proves which profile's TUI sent it, and state.db proves where it landed.

Contract under test (``hermes_cli/web_routers/chat_ws.py::pty_ws`` + ``hermes_cli/pty_session.py``):
  * ``?profile=<name>`` scopes the whole chat to that profile (HERMES_HOME=<profile dir>);
  * with ``?attach=T`` the PTY outlives the socket; reconnecting with the same T + profile replays
    the buffer and forces a redraw of the SAME TUI/session;
  * the registry key includes the profile, so the same T under another profile gets a fresh TUI.

Not a cell: "dashboard shutdown leaves no TUI behind". It holds, but it cannot be made red from the
dashboard side: with ``PtySessionRegistry.close_all`` disabled (and even with the child ignoring
SIGHUP) the TUI still exits when the dying dashboard's PTY master closes (kernel hangup / EIO) and
its ``tui_gateway`` exits on stdin EOF. The fixture still waits for, reports and reaps any
descendant that outlives the dashboard, so a leak cannot poison later tests.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.terminal._vt import Screen
from tests.fakes.fake_llm_provider import Text

from . import _helpers as H
from ._pty_helpers import ProcessLedger, WsTerm, canon, poll

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX PTY dashboard")

_CANARY = re.compile(r"canary-[0-9a-f]{12}")
ROWS, COLS = 40, 120
TURN_TIMEOUT = 90.0


def _require_tui() -> None:
    if shutil.which("node") and (H.REPO_ROOT / "ui-tui" / "dist" / "entry.js").is_file():
        return
    if os.environ.get("HERMES_E2E_REQUIRE_TUI") == "1":
        pytest.fail("ui-tui/dist/entry.js or node missing but HERMES_E2E_REQUIRE_TUI=1")
    pytest.skip("Ink TUI not built (cd ui-tui && npm run build) or node missing")


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def _last_user(body: dict) -> str:
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    return _text_of(users[-1].get("content")) if users else ""


def _echo_responder(p: H.Profile):
    """Main turns answer ``reply-from-<profile>-<tag> ack-<canary of the prompt>``."""
    def respond(record: dict) -> Text:
        m = _CANARY.search(_last_user(record["body"]))
        return Text(f"reply-from-{p.name}-{p.tag} ack-{m.group(0) if m else 'none'}")
    return respond


def _reply(p: H.Profile, canary: str) -> str:
    return f"reply-from-{p.name}-{p.tag} ack-{canary}"


def _main_requests(p: H.Profile) -> list[dict]:
    assert p.srv is not None
    return [r for r in list(p.srv.requests) if r["kind"] == "main"]


def _user_rows(p: H.Profile, needle: str) -> list[tuple[str, str]]:
    if not p.db.exists():
        return []
    return [(str(s), str(c)) for s, c in H.db_rows(
        p.db, "SELECT session_id, COALESCE(content, '') FROM messages WHERE role = 'user' AND content LIKE ?",
        (f"%{needle}%",))]


def _user_sessions(p: H.Profile) -> set[str]:
    if not p.db.exists():
        return set()
    return {str(s) for (s,) in H.db_rows(p.db, "SELECT DISTINCT session_id FROM messages WHERE role = 'user'")}


class Chat:
    """The dashboard + one tag-scoped attach token, as a single browser tab would hold them."""

    def __init__(self, dash: H.Dashboard, ledger: ProcessLedger) -> None:
        self.dash = dash
        self.ledger = ledger
        self.attach = f"e2e-attach-{secrets.token_hex(6)}"
        self.terms: list[WsTerm] = []

    def open(self, profile: str) -> WsTerm:
        url = self.dash.ws_url("/api/pty", token=self.dash.token, profile=profile, attach=self.attach,
                               channel=f"e2e-{profile}-{self.attach}")
        term = WsTerm(url, rows=ROWS, cols=COLS)
        self.terms.append(term)
        return term

    def wait_composer(self, term: WsTerm) -> None:
        """The TUI painted its composer prompt and session banner, and its status bar no longer
        reports a boot phase ("summoning hermes…", "starting agent…"). Profile-agnostic on purpose:
        which profile answered is judged by the provider/state.db assertions, not by this wait."""
        def ready() -> bool:
            s = term.text()
            return term.closed or ("❯" in s and "Session:" in s
                                   and "startingagent" not in s and "summoninghermes" not in s)
        try:
            poll(ready, 120, "the TUI composer to be ready")
            assert not term.closed, f"/api/pty closed (code={term.close_code}) during TUI startup"
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n--- screen ---\n{term.dump()}") from None
        term.wait_quiet(0.5)
        self.ledger.snapshot()

    def turn(self, term: WsTerm, p: H.Profile, canary: str, n_main_before: int) -> dict:
        """Submit a prompt carrying ``canary``; returns the provider record of that turn. Waits for
        the canary to reach ANY sandbox provider, so a mis-scoped chat fails fast and by name."""
        term.submit(f"please echo {canary}")

        def carriers() -> list[str]:
            return [q.name for q in self.dash.sb.profiles.values()
                    if any(canary in _last_user(r["body"]) for r in _main_requests(q))]
        poll(lambda: carriers() or term.closed, TURN_TIMEOUT, f"a provider to receive the turn carrying {canary}")
        rec = next((r for r in _main_requests(p)[n_main_before:] if canary in _last_user(r["body"])), None)
        assert rec is not None, (
            f"the profile={p.name} turn carrying {canary} reached provider(s) {carriers()} instead of "
            f"{p.name}'s\n--- screen ---\n{term.dump()}")
        term.wait_for(_reply(p, canary), timeout=TURN_TIMEOUT)
        poll(lambda: _user_rows(p, canary), TURN_TIMEOUT, f"{canary} persisted in {p.name}'s state.db")
        self.ledger.snapshot()
        return rec


def _private_tui(tmp_path: Path) -> Path:
    """A private, syntax-checked copy of the prebuilt bundle, handed to the dashboard as
    ``HERMES_TUI_DIR`` (the prebuilt-install path of ``_make_tui_argv``). On a source checkout every
    PTY spawn otherwise re-runs ``npm run build``, rewriting ``ui-tui/dist/entry.js`` in place, and a
    TUI starting during a concurrent build (another worker, another dashboard) crashes on the
    truncated file (#121286). The bundle is self-contained (node builtins only)."""
    src = H.REPO_ROOT / "ui-tui" / "dist" / "entry.js"
    dst = tmp_path / "tui" / "dist" / "entry.js"
    dst.parent.mkdir(parents=True)
    # Same module type as ui-tui/package.json, so older Node versions without ESM syntax detection
    # load the copy exactly like the original.
    (dst.parent.parent / "package.json").write_text('{"type": "module"}\n', encoding="utf-8")
    def copied_intact() -> bool:
        shutil.copyfile(src, dst)
        return subprocess.run([shutil.which("node") or "node", "--check", str(dst)], capture_output=True,
                              timeout=60, stdin=subprocess.DEVNULL).returncode == 0
    poll(copied_intact, 180, "an intact copy of ui-tui/dist/entry.js (a concurrent build may be writing it)",
         interval=1.0)
    return dst.parent.parent


@pytest.fixture
def dash_env(tmp_path: Path):
    _require_tui()
    tui_dir = _private_tui(tmp_path)
    sb = H.make_sandbox(tmp_path, ("default", "alpha", "beta"), responder=_echo_responder)
    dash = H.Dashboard(sb, tmp_path / "dashboard.log", extra_env={"HERMES_TUI_DIR": str(tui_dir)})
    ledger = ProcessLedger(dash.proc.pid)
    try:
        yield sb, dash, ledger
    finally:
        try:
            dash.close()
        finally:
            try:
                poll(lambda: not ledger.survivors(), 30, "dashboard descendants to exit")
            except AssertionError:
                pass
            left = ledger.kill_survivors()
            if left:
                print("teardown killed dashboard leftovers:\n  " + "\n  ".join(left), file=sys.stderr)
            sb.stop()


def _assert_labelled(term: WsTerm, profile: str) -> None:
    assert canon(f"{profile} ❯") in term.text(), (
        f"the TUI composer is not labelled with profile {profile!r}\n--- screen ---\n{term.dump()}")


def _first_turn(chat: Chat, alpha: H.Profile, default: H.Profile, canary: str) -> WsTerm:
    """Scenario 1: one chat turn through /api/pty lands on alpha's provider, screen and state.db."""
    term = chat.open("alpha")
    chat.wait_composer(term)
    rec = chat.turn(term, alpha, canary, n_main_before=0)
    assert rec["auth"] == f"Bearer {alpha.provider_key}", f"turn authenticated with {rec['auth']!r}, not alpha's key"
    assert rec["body"].get("model") == alpha.model, f"turn used model {rec['body'].get('model')!r}, not alpha's"
    rows = _user_rows(alpha, canary)
    assert len({s for s, _ in rows}) == 1, f"alpha's state.db has the canary in {rows}"
    assert not _main_requests(default), (
        f"the DEFAULT profile's provider served a profile=alpha chat: "
        f"{[_last_user(r['body'])[:80] for r in _main_requests(default)]}")
    assert not _user_rows(default, canary), "profile=alpha's prompt was persisted in the DEFAULT profile's state.db"
    _assert_labelled(term, "alpha")
    return term


def _replayed_screen(term: WsTerm) -> str:
    """Render ONLY the first binary frame of a reattached socket: the server sends the ring-buffer
    snapshot as one frame before it asks the TUI for a redraw, so this is the replay on its own."""
    poll(lambda: term.first_frames or term.closed, 30, "the first frame of the reattached socket")
    assert term.first_frames, f"reattached socket closed (code={term.close_code}) before any output"
    screen = Screen(ROWS, COLS)
    screen.feed(term.first_frames[0])
    return canon("".join(screen.transcript() + screen.display()))


def _reconnect_turn(chat: Chat, first: WsTerm, alpha: H.Profile, canary1: str, canary2: str) -> WsTerm:
    """Scenario 2: a client-side close + reconnect with the same attach token resumes the SAME TUI."""
    session_id = _user_rows(alpha, canary1)[0][0]
    first.close()
    assert first.closed, "client close did not complete"
    term = chat.open("alpha")
    assert canon(_reply(alpha, canary1)) in _replayed_screen(term), (
        "reattach did not replay the buffered PTY output (the first frame lacks the earlier reply): "
        f"{term.first_frames[0][:300]!r}")
    # ...and the forced redraw leaves the live screen showing the earlier turn and a usable composer.
    term.wait_for(_reply(alpha, canary1), timeout=60)
    chat.wait_composer(term)
    assert term.close_code is None, f"reattached socket closed with {term.close_code}"
    n_before = len(_main_requests(alpha))
    rec = chat.turn(term, alpha, canary2, n_main_before=n_before)
    history = " ".join(_text_of(m.get("content")) for m in rec["body"].get("messages", []))
    assert canary1 in history and _reply(alpha, canary1) in history, (
        "the post-reconnect turn did not carry the first turn in its history (a fresh TUI/session "
        f"answered): {history[-600:]!r}")
    sid2 = {s for s, _ in _user_rows(alpha, canary2)}
    assert sid2 == {session_id}, f"post-reconnect prompt landed in session(s) {sid2}, not the original {session_id}"
    assert _user_sessions(alpha) == {session_id}, (
        f"reconnect created a second session in alpha's state.db: {_user_sessions(alpha)}")
    return term


def _other_profile_same_token(chat: Chat, alpha: H.Profile, beta: H.Profile,
                              alpha_canaries: tuple[str, ...], canary3: str) -> None:
    """Scenario 3: the same attach token under profile=beta must not reattach alpha's PTY."""
    alpha_main = len(_main_requests(alpha))
    term = chat.open("beta")
    chat.wait_composer(term)
    screen = term.text()
    leaked = [c for c in (*alpha_canaries, f"reply-from-alpha-{alpha.tag}") if canon(c) in screen]
    assert not leaked, (f"profile=beta with alpha's attach token reattached alpha's TUI (showed {leaked})"
                        f"\n--- screen ---\n{term.dump()}")
    rec = chat.turn(term, beta, canary3, n_main_before=0)
    assert rec["auth"] == f"Bearer {beta.provider_key}", f"beta's turn authenticated with {rec['auth']!r}"
    assert len(_main_requests(alpha)) == alpha_main, "beta's prompt reached alpha's provider"
    assert not _user_rows(alpha, canary3), "beta's prompt was persisted in alpha's state.db"
    assert not any(c in _last_user(r["body"]) for r in _main_requests(beta) for c in alpha_canaries), (
        "alpha's prompts reached beta's provider")
    _assert_labelled(term, "beta")


def test_pty_chat_turn_reconnect_and_profile_scoped_keepalive(dash_env) -> None:
    sb, dash, ledger = dash_env
    alpha, beta, default = sb.profiles["alpha"], sb.profiles["beta"], sb.profiles["default"]
    chat = Chat(dash, ledger)
    c1, c2, c3 = (f"canary-{secrets.token_hex(6)}" for _ in range(3))

    term = _first_turn(chat, alpha, default, c1)
    _reconnect_turn(chat, term, alpha, c1, c2)
    _other_profile_same_token(chat, alpha, beta, (c1, c2), c3)

    assert not _main_requests(default), "the DEFAULT profile's provider saw traffic from profile-scoped chats"
    assert not _user_rows(default, "canary-"), "a profile-scoped prompt landed in the DEFAULT profile's state.db"
    assert dash.proc.poll() is None, f"dashboard died:\n{dash.log_tail()}"
