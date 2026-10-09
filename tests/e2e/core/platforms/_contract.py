"""The messaging-adapter contract: scenarios every real adapter must pass through a real gateway.

Each scenario takes a running ``Rig`` (gateway child + stand-in driver + fake LLM director) and
asserts user-visible outcomes at the stand-in (what a human in the chat now sees), the next wire
request Hermes sent the model, or persisted state (state.db). Tokens ``[in:<id>]`` tie every inbound
to the model turns it started, so duplicate or dropped turns are counted, never guessed.

Negative claims ("no second reply", "no turn for the unmentioned message") are settled by a BARRIER:
a later inbound whose reply proves the gateway already processed everything queued before it — never
by sleeping. Streaming scenarios run each in a private chat of their own (``Rig.fresh_user``); their
same-chat barrier is re-sent if the first one is dropped while the adapter closes the streamed turn
(#121393: no platform-visible signal marks that window).

Open bugs are ``Rig.gate()`` blocks around a scenario's FINAL assertions only (see
``tests/e2e/core/_pending_fixes.known_failure``): the cell xfails only on that bug's own message,
fails on anything else, and simply passes once the fix lands.
"""

from __future__ import annotations

import contextlib
import html
import io
import itertools
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Tuple

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.fakes.platforms._standin import Visible, wait_until

# A turn takes ~3 s even under 8-way parallel load; the budget only bounds how long one red scenario
# can hold the file (the CI runner kills a file at 300 s, a module-scoped rig serves ~12 scenarios).
TURN_TIMEOUT = 30.0


@dataclass
class Rig:
    gw: GatewayUnderTest
    drv: Any
    director: Director
    llm: FakeLLMServer
    known: Optional[Tuple[str, str]] = None  # (pattern, reason) for the running scenario's open bug
    _users: Optional[Iterator[str]] = field(default=None, repr=False)

    def ctx(self) -> str:
        return f"{self.drv.describe()}\n{self.gw.tail(4000)}"

    def gate(self) -> ContextManager[None]:
        """Wrap a scenario's final assertions: xfail only on the running scenario's KNOWN bug signature."""
        return known_failure(*self.known) if self.known else contextlib.nullcontext()

    def fresh_user(self) -> str:
        """An allowlisted user whose private chat no earlier scenario used (``drv.stream_users``)."""
        if self._users is None:
            self._users = itertools.cycle(self.drv.stream_users)
        return next(self._users)


def head(aid: str) -> str:
    return f"<<{aid}>>"


def foot(aid: str) -> str:
    return f"<<end-{aid}>>"


def answer(aid: str, body: str, **kw: Any) -> Text:
    return Text(f"{head(aid)} {body} {foot(aid)}", **kw)


def norm(text: str) -> str:
    # Renderers escape markdown punctuation (``\\.``) or HTML (``&lt;``) and may re-wrap whitespace.
    return re.sub(r"\s+", " ", html.unescape(text).replace("\\", "")).strip()


def copies(visible: List[Visible], aid: str) -> List[Visible]:
    return [v for v in visible if head(aid) in norm(v.text) or foot(aid) in norm(v.text)]


def complete(visible: List[Visible], aid: str) -> List[Visible]:
    return [v for v in visible if head(aid) in norm(v.text) and foot(aid) in norm(v.text)]


def wait_reply(rig: Rig, chat_id: str, aid: str, what: str = "", timeout: float = TURN_TIMEOUT) -> None:
    """Until the reply's LAST piece (footer) is visible in ``chat_id``."""
    wait_until(lambda: any(foot(aid) in norm(v.text) for v in rig.drv.visible(chat_id)),
               what or f"reply {aid} visible in {chat_id}", timeout=timeout, on_timeout=rig.ctx)


def barrier(rig: Rig, token: str, *, group: bool = False, user_id: Optional[str] = None, attempts: int = 1) -> None:
    """One more inbound in the same chat, answered: everything queued before it has been handled.

    The gateway serializes a chat: an inbound arriving while a turn runs is drained only after that
    turn's final delivery, so the barrier's reply orders after it. ``user_id`` picks that user's
    private chat. ``attempts > 1`` is for streamed turns: a same-chat inbound landing while the
    adapter closes such a turn can be dropped (#121393; no platform-visible signal marks that
    window), so an unanswered barrier is re-sent under a new token after a short wait.
    """
    rig.gw.wait_idle()
    for n in range(attempts):
        tok = token if n == 0 else f"{token}r{n}"
        aid = f"B-{tok}"
        rig.director.script(tok, answer(aid, "barrier"))
        text = f"barrier [in:{tok}]"
        inbound = rig.drv.group(text, mention=True) if group else rig.drv.dm(text, user_id=user_id)
        if n == attempts - 1:
            wait_reply(rig, inbound.chat_id, aid, f"barrier {tok} answered")
            return
        try:
            wait_until(lambda: complete(rig.drv.visible(inbound.chat_id), aid), f"barrier {tok}", timeout=8.0)
            return
        except AssertionError:
            continue  # dropped in the #121393 window: the next barrier still orders after the turn


def turns(rig: Rig, token: str) -> int:
    return rig.director.turns.get(token, 0)


# 1. inbound DM -> exactly one reply --------------------------------------------------------------
def dm_gets_exactly_one_reply(rig: Rig, tag: str) -> None:
    token, aid = f"dm-{tag}", f"A-dm-{tag}"
    # punctuation every markup dialect reserves: the adapter must escape it, not lean on a fallback
    rig.director.script(token, answer(aid, "hello back (v1.2) - ok! #1 = {x}"))
    inbound = rig.drv.dm(f"hello [in:{token}]")
    wait_reply(rig, inbound.chat_id, aid)
    barrier(rig, f"dmb-{tag}")
    vis = copies(rig.drv.visible(inbound.chat_id), aid)
    with rig.gate():
        assert len(vis) == 1 and complete(vis, aid), f"expected one complete reply, got {vis}\n{rig.ctx()}"
        assert "hello back (v1.2) - ok! #1 = {x}" in norm(vis[0].text), f"reply text mangled: {vis[0].text!r}"
        assert turns(rig, token) == 1, f"inbound started {turns(rig, token)} model turns\n{rig.ctx()}"
        assert len(rig.gw.user_rows(f"[in:{token}]")) == 1, "inbound persisted != once"
        refused = rig.drv.format_rejections(inbound.chat_id)
        assert not refused, (f"the platform refused the reply's formatting (a plain-text fallback hid it): "
                             f"{[c.response for c in refused]}\n{rig.ctx()}")


# 2. group without mention obeys require_mention --------------------------------------------------
def group_obeys_require_mention(rig: Rig, tag: str) -> None:
    quiet, loud = f"gq-{tag}", f"gl-{tag}"
    rig.director.script(quiet, answer(f"A-{quiet}", "should never be said"))
    rig.director.script(loud, answer(f"A-{loud}", "mentioned, so answered"))
    rig.drv.group(f"just chatting [in:{quiet}]", mention=False)
    inbound = rig.drv.group(f"please answer [in:{loud}]", mention=True)
    wait_reply(rig, inbound.chat_id, f"A-{loud}")
    barrier(rig, f"gb-{tag}", group=True)
    assert turns(rig, quiet) == 0, f"unmentioned group message started a model turn\n{rig.ctx()}"
    # Rapid messages may be batched into one turn: the quiet text must not ride along in any prompt.
    leaked = [t for t, reqs in rig.director.requests.items()
              if any(f"[in:{quiet}]" in str(m.get("content")) for r in reqs for m in r.get("messages", [])
                     if m.get("role") == "user")]
    assert not leaked, f"unmentioned group message reached the model inside turn(s) {leaked}\n{rig.ctx()}"
    assert not copies(rig.drv.visible(inbound.chat_id), f"A-{quiet}"), "unmentioned message was answered"
    assert len(complete(rig.drv.visible(inbound.chat_id), f"A-{loud}")) == 1, rig.ctx()


# 3. reply longer than the platform limit -> split, in order, nothing lost ------------------------
def long_body(limit: int, factor: float = 2.4) -> str:
    n = int(limit * factor / 7) + 1
    return " ".join(f"w{i:05d}" for i in range(n))


def _words(text: str) -> List[str]:
    return re.findall(r"w\d{5}", text)


def long_reply_is_split_in_order(rig: Rig, tag: str) -> None:
    token, aid = f"long-{tag}", f"A-long-{tag}"
    body = long_body(rig.drv.limit)
    rig.director.script(token, answer(aid, body, chunk_chars=4000))
    inbound = rig.drv.dm(f"write a lot [in:{token}]")
    wait_reply(rig, inbound.chat_id, aid)
    barrier(rig, f"longb-{tag}")
    parts = [v for v in rig.drv.visible(inbound.chat_id) if _words(v.text) or head(aid) in norm(v.text)]
    assert len(parts) >= 2, f"a {len(body)}-char reply was not split\n{rig.ctx()}"
    too_long = [len(v.text) for v in parts if len(v.text) > rig.drv.limit]
    assert not too_long, f"chunks over the {rig.drv.limit} cap: {too_long}"
    seen = [w for v in parts for w in _words(v.text)]
    assert seen == _words(body), (f"split reply lost/reordered/duplicated words: got {len(seen)} of "
                                  f"{len(_words(body))}\n{rig.ctx()}")


# 4. failed continuation chunk is not silently dropped (#120073) ----------------------------------
def failed_continuation_is_retried_or_reported(rig: Rig, tag: str) -> None:
    token, aid = f"gap-{tag}", f"A-gap-{tag}"
    body = long_body(rig.drv.limit)
    words = _words(body)
    marker = words[int(len(words) * 0.75)]  # lands in a continuation chunk, never the first
    rig.director.script(token, answer(aid, body, chunk_chars=4000))
    # every attempt at that chunk is rejected (a plain-text/markdown fallback or a short retry ladder
    # must not paper over it): only a later retry or an honest notice satisfies the contract
    rig.drv.fail_send(times=3, match=lambda text: marker in text)
    inbound = rig.drv.dm(f"write a lot again [in:{token}]")
    wait_until(lambda: turns(rig, token) >= 1, "model turn started", timeout=TURN_TIMEOUT, on_timeout=rig.ctx)
    barrier(rig, f"gapb-{tag}")
    visible = rig.drv.visible(inbound.chat_id)
    seen = [w for v in visible for w in _words(v.text)]
    missing = [w for w in words if w not in seen]
    if not missing:
        return  # the rejected continuation was re-sent: nothing lost
    tail_idx = max(i for i, v in enumerate(visible) if _words(v.text))
    notices = [v.text for v in visible[tail_idx + 1:]
               if not _words(v.text) and "barrier" not in v.text and head(aid) not in norm(v.text)]
    assert notices, (f"{len(missing)} words of the reply (from {missing[0]}) were never delivered and the "
                     f"user was not told\n{rig.ctx()}")


# 5. stream finalize rejected -> exactly one complete copy (#121108 pattern) -----------------------
def rejected_finalize_leaves_one_copy(rig: Rig, tag: str, *, group: bool = False) -> None:
    """The platform refuses the call that would complete a streamed reply (``drv.fail_finalize``:
    the finalize edit, the final send after draft previews, or the closing stream append)."""
    token, aid = f"fin-{tag}", f"A-fin-{tag}"
    body = " ".join(f"q{i:03d}" for i in range(120))
    rig.director.script(token, answer(aid, body, chunk_chars=12, delay_per_chunk=0.03))
    faults = rig.drv.fail_finalize(lambda text: foot(aid) in norm(text), group=group)
    text, user = f"stream it [in:{token}]", rig.fresh_user()
    inbound = rig.drv.group(text, mention=True) if group else rig.drv.dm(text, user_id=user)
    # settled once the ending is visible: whole, or completed by a continuation message
    wait_until(lambda: (any(foot(aid) in norm(v.text) for v in rig.drv.visible(inbound.chat_id))
                        or _split_seam(rig.drv.visible(inbound.chat_id), aid)),
               f"reply {aid} visible in {inbound.chat_id}", timeout=TURN_TIMEOUT, on_timeout=rig.ctx)
    barrier(rig, f"finb-{tag}", group=group, user_id=user, attempts=3)
    rig.drv.standin.clear_faults()
    assert any(f.fired for f in faults), f"the finalize fault never fired: nothing was rejected\n{rig.ctx()}"
    with rig.gate():
        seam = _split_seam(rig.drv.visible(inbound.chat_id), aid)
        assert seam is None, (f"the continuation after a rejected finalize resumed mid-word: the answer's "
                              f"last word is split across two messages ({seam})\n{rig.ctx()}")
    shown = _shown(rig, inbound.chat_id, aid, "q")
    ctx = f"visible: {shown[:300]!r} ... {shown[-300:]!r}\n{rig.ctx()}"
    # the ending is shown exactly once: neither lost nor re-sent by a second final delivery
    assert shown.count(foot(aid)) == 1, (f"answer ending shown {shown.count(foot(aid))}x after a rejected "
                                         f"finalize (lost or re-sent)\n{ctx}")
    assert turns(rig, token) == 1, rig.ctx()
    with rig.gate():
        # ... and so is everything before it, in order: no stale partial copy left beside it
        assert shown.count(head(aid)) == 1 and _seamless(_pwords(shown, "q")) == _pwords(body, "q"), (
            f"a partial copy of the answer was left visible after a rejected finalize "
            f"(head shown {shown.count(head(aid))}x)\n{ctx}")


def _split_seam(visible: List[Visible], aid: str) -> Optional[str]:
    """The seam, as the user reads it, where the answer's ending is cut mid-word across two
    consecutive messages (a continuation that resumed mid-word instead of at a word boundary)."""
    texts, end = [norm(v.text) for v in visible], foot(aid)
    for a, b in zip(texts, texts[1:]):
        for i in range(1, len(end)):
            if a.endswith(end[:i]) and b.startswith(end[i:]):
                return f"{a[-40:]!r} | {b[:40]!r}"
    return None


def _pwords(text: str, prefix: str) -> List[str]:
    return re.findall(rf"\b{prefix}\d{{3}}\b", text)


def _seamless(words: List[str]) -> List[str]:
    """Collapse a word repeated across a message seam: the edit-fallback continuation backs its cut up
    to the previous word boundary (``_continuation_text``), so the stuck preview's last word can
    reappear at the start of the continuation. Tolerated here: it is not a second copy."""
    return [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]


def _shown(rig: Rig, chat_id: str, aid: str, prefix: str) -> str:
    """Everything visible that carries a piece of THIS answer (the chat is shared across scenarios)."""
    return " ".join(norm(v.text) for v in rig.drv.visible(chat_id)
                    if copies([v], aid) or _pwords(norm(v.text), prefix))


def streamed_reply_shown_once(rig: Rig, tag: str) -> None:
    """The plain streamed turn: previews then a final, and the user ends up with ONE copy (the
    stream consumer's final-delivered bookkeeping is what stops the gateway's normal final send)."""
    token, aid = f"st-{tag}", f"A-st-{tag}"
    body = " ".join(f"s{i:03d}" for i in range(80))
    rig.director.script(token, answer(aid, body, chunk_chars=12, delay_per_chunk=0.03))
    user = rig.fresh_user()
    inbound = rig.drv.dm(f"stream it [in:{token}]", user_id=user)
    wait_reply(rig, inbound.chat_id, aid)
    barrier(rig, f"stb-{tag}", user_id=user, attempts=3)
    shown = _shown(rig, inbound.chat_id, aid, "s")
    with rig.gate():
        assert shown.count(head(aid)) == 1 and shown.count(foot(aid)) == 1, (
            f"a streamed reply is shown != once: {shown[:300]!r}\n{rig.ctx()}")
        assert _pwords(shown, "s") == _pwords(body, "s"), f"streamed words lost/duplicated: {shown[:300]!r}"
        assert turns(rig, token) == 1, rig.ctx()


def streamed_reply_ending_in_whitespace_shown_once(rig: Rig, tag: str) -> None:
    """Models routinely end on a newline; the streamed transport must still finalize ONE message."""
    token, aid = f"ws-{tag}", f"A-ws-{tag}"
    body = " ".join(f"w{i:03d}" for i in range(80))
    rig.director.script(token, Text(f"{head(aid)} {body} {foot(aid)}\n\n", chunk_chars=12, delay_per_chunk=0.03))
    user = rig.fresh_user()
    inbound = rig.drv.dm(f"stream with a trailing newline [in:{token}]", user_id=user)
    wait_reply(rig, inbound.chat_id, aid)
    barrier(rig, f"wsb-{tag}", user_id=user, attempts=3)
    shown = _shown(rig, inbound.chat_id, aid, "w")
    with rig.gate():
        assert shown.count(head(aid)) == 1 and shown.count(foot(aid)) == 1, (
            f"a streamed reply ending in whitespace is shown != once: {shown[:400]!r}\n{rig.ctx()}")


# 6. the platform redelivers the same inbound -> one reply (#119848 pattern) -----------------------
def redelivered_inbound_gets_one_reply(rig: Rig, tag: str) -> None:
    token, aid = f"re-{tag}", f"A-re-{tag}"
    rig.director.script(token, answer(aid, "first and only"), answer(f"{aid}-dup", "DUPLICATE"))
    inbound = rig.drv.dm(f"once please [in:{token}]")
    wait_reply(rig, inbound.chat_id, aid)
    rig.drv.redeliver(inbound)
    barrier(rig, f"reb-{tag}")
    with rig.gate():
        assert turns(rig, token) == 1, f"redelivered inbound started {turns(rig, token)} turns\n{rig.ctx()}"
        assert not copies(rig.drv.visible(inbound.chat_id), f"{aid}-dup"), "a duplicate reply was sent"
        assert len(copies(rig.drv.visible(inbound.chat_id), aid)) == 1, rig.ctx()
        # A replay that slips past dedup can ride INTO the barrier's turn (text batching merges the
        # two), which the turn/reply counts above cannot see: the transcript can.
        rows = rig.gw.user_rows(f"[in:{token}]")
        assert len(rows) == 1, f"the redelivered inbound reached the transcript {len(rows)}x: {rows}\n{rig.ctx()}"


# 7. approval button click: allowlisted user accepted, stranger refused ----------------------------
def _button(rig: Rig, chat_id: str, label_re: str) -> Dict[str, Any]:
    def pick() -> Any:
        for b in rig.drv.buttons(chat_id):
            label = str(b.get("text") or b.get("label") or "")
            label = str(label.get("text", "")) if isinstance(label, dict) else label
            if re.search(label_re, label, re.I):
                return b
        return None
    return wait_until(pick, f"a {label_re!r} button under the approval prompt", timeout=TURN_TIMEOUT,
                      on_timeout=rig.ctx)


def approval_click_by_allowlisted_user_runs_command(rig: Rig, tag: str, victim: Path) -> None:
    """A stranger's DENY must be refused (else it would deny the command); the owner's approve runs it."""
    token, aid = f"ap-{tag}", f"A-ap-{tag}"
    victim.mkdir(parents=True, exist_ok=True)
    (victim / "f.txt").write_text("x")
    rig.director.script(token, ToolCall("terminal", {"command": f"rm -rf {victim}"}), answer(aid, "cleaned"))
    inbound = rig.drv.dm(f"clean up [in:{token}]")
    approve = _button(rig, inbound.chat_id, r"once")
    deny = _button(rig, inbound.chat_id, r"deny")
    assert victim.exists(), "dangerous command ran before approval"
    stranger = rig.drv.click(inbound.chat_id, deny, user_id=rig.drv.other_user_id)
    # the adapter has handled (and answered) the stranger's click before the owner clicks
    wait_until(lambda: rig.drv.click_answered(stranger), "the stranger's click answered", timeout=TURN_TIMEOUT,
               on_timeout=rig.ctx)
    rig.drv.click(inbound.chat_id, approve)
    wait_reply(rig, inbound.chat_id, aid, "turn completed after the approval click")
    assert not victim.exists(), (f"approved command did not run: the stranger's deny was obeyed or the "
                                 f"owner's approval refused\n{rig.ctx()}")


# 8. platform_toolsets / disabled_toolsets honored (#121089) --------------------------------------
def _offered_tools(rig: Rig, tag: str) -> set:
    token, aid = f"ts-{tag}", f"A-ts-{tag}"
    rig.director.script(token, answer(aid, "tools checked"))
    inbound = rig.drv.dm(f"what tools [in:{token}]")
    wait_reply(rig, inbound.chat_id, aid)
    req = rig.director.requests[token][0]
    names = {t.get("function", {}).get("name") for t in req.get("tools") or []}
    assert names, "the model request carried no tools at all"
    return names


def platform_toolsets_are_honored(rig: Rig, tag: str) -> None:
    """``platform_toolsets.<platform>: [file]``: the file tools, and never the terminal."""
    names = _offered_tools(rig, tag)
    assert "read_file" in names, f"the configured toolset (file) is missing: {sorted(names)}"
    assert "terminal" not in names, f"terminal offered although platform_toolsets omits it: {sorted(names)}"


def disabled_toolsets_are_honored(rig: Rig, tag: str) -> None:
    """``agent.disabled_toolsets: [file]``: the file tools gone, the rest of the platform preset kept."""
    names = _offered_tools(rig, tag)
    assert "terminal" in names, f"control: the platform preset lost terminal: {sorted(names)}"
    leaked = sorted(names & {"read_file", "write_file", "patch", "search_files"})
    assert not leaked, f"disabled toolset 'file' still offered: {leaked}"


def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# 9. inbound photo sent as a file (HEIC) reaches the agent as an image (#119593) -------------------
def _image(fmt: str) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (32, 32), (200, 30, 30))
    buf = io.BytesIO()
    if fmt == "HEIF":
        import pillow_heif

        pillow_heif.register_heif_opener()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _model_saw_image(req: dict) -> bool:
    for m in req.get("messages", []):
        content = m.get("content")
        if isinstance(content, list) and any(p.get("type") in ("image_url", "input_image") for p in content
                                             if isinstance(p, dict)):
            return True
    return False


def heic_document_reaches_agent_as_image(rig: Rig, tag: str) -> None:
    got: Dict[str, bool] = {}
    for fmt, name, mime in (("PNG", "control.png", "image/png"), ("HEIF", "IMG_0001.HEIC", "image/heic")):
        token, aid = f"img-{fmt}-{tag}", f"A-img-{fmt}-{tag}"
        rig.director.script(token, answer(aid, "saw it"))
        inbound = rig.drv.document(name, _image(fmt), mime, caption=f"look [in:{token}]")
        try:
            wait_reply(rig, inbound.chat_id, aid)
        except AssertionError:
            got[fmt] = False  # the photo never started a turn at all
            continue
        got[fmt] = _model_saw_image(rig.director.requests[token][0])
    log = rig.gw.grep(r"image|attach|heic|png|cache")
    assert got["PNG"], f"control: a PNG sent as a file did not reach the model as an image\n{log}\n{rig.ctx()}"
    with rig.gate():
        assert got["HEIF"], f"a HEIC photo sent as a file did not reach the model as an image\n{log}"


# 10. planned restart: notice delivered once; a redelivered /restart does not loop ----------------
def planned_restart_notice_once(rig: Rig, tag: str, restart: Callable[[], None]) -> None:
    inbound = rig.drv.dm("/restart")
    wait_until(lambda: not rig.gw.alive(), "gateway exits for the planned restart", timeout=TURN_TIMEOUT,
               on_timeout=rig.ctx)
    rc = rig.gw.proc.returncode if rig.gw.proc else None
    assert rc == 75, f"planned restart under a supervisor must exit 75 (got {rc})\n{rig.gw.tail()}"
    reborn = time.monotonic()
    restart()  # what the supervisor does on exit 75

    def notices() -> List[str]:
        texts = [str(c.params.get("text") or c.params.get("content") or c.params) for c in rig.drv.sends(inbound.chat_id)
                 if c.at > reborn]
        return [t for t in texts if "restart" in t.lower() and "<<" not in norm(t)]  # not a scripted reply

    wait_until(notices, "restart notice after the gateway came back", timeout=TURN_TIMEOUT, on_timeout=rig.ctx)
    rig.drv.redeliver(inbound)  # the pre-restart /restart replayed (lost ack on the way out)
    bid = f"B-rsb-{tag}"
    rig.director.script(f"rsb-{tag}", answer(bid, "barrier"))
    probe = rig.drv.dm(f"barrier [in:rsb-{tag}]")
    wait_until(lambda: complete(rig.drv.visible(probe.chat_id), bid) or not rig.gw.alive(),
               "barrier answered (or the gateway went down again)", timeout=TURN_TIMEOUT, on_timeout=rig.ctx)
    rc = rig.gw.proc.returncode if rig.gw.proc else None
    # a crash is not the replay bug: only a second planned exit (75) is
    assert rig.gw.alive() or rc == 75, f"gateway died (rc={rc}) after the replayed /restart\n{rig.ctx()}"
    with rig.gate():
        assert rig.gw.alive(), f"a redelivered /restart restarted the gateway again (exit 75)\n{rig.ctx()}"
        assert len(notices()) == 1, (f"expected exactly one restart notice from the new process, got "
                                     f"{notices()} (a second restart ack means the replayed /restart was obeyed)"
                                     f"\n{rig.ctx()}")
