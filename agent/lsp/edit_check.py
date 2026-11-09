"""Post-write LSP check that never holds the write.

``write_file`` / ``patch`` used to block on two language-server round trips per edit: a pre-write
baseline (open + wait) and the post-write verdict (open + save + wait), each up to ``lsp.wait_timeout``.
A push-only server that stays silent on a clean edit (typescript-language-server publishes nothing
when a file's diagnostics stay empty) ran both waits to the deadline: 10s per clean TypeScript edit.

Here the check is one coroutine on the service loop: sync the edit's post-edit snapshot and wait for
its verdict; only when that verdict has diagnostics, get the verdict on the pre-edit snapshot (or reuse
the one an earlier check left for exactly that text) and return the diagnostics the edit introduced.
Both snapshots are the bytes the write itself held, never a later read of the file: by the time a
queued check runs, another session may have written the file again, and that session's errors are not
this edit's.
The caller waits for it only within its latency budget (:func:`inline_wait`); a slower verdict is delivered later through
:mod:`agent.lsp.deferred`.  Checks of one file run one at a time (two checks syncing one document at
once would satisfy each other's waits).  A newer check by the SAME session takes an unfinished one over
(its baseline, the undelivered notes it carried, and its result when the newer check gets no verdict),
so a superseded edit's errors are still caught unless the newer edit fixed them; a check by another
session only queues behind it, so each session hears about its own edits only.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agent.lsp import eventlog, sync_diagnostics
from agent.lsp.client import _diagnostic_key, as_disk_read
from agent.lsp.servers import language_id_for

if TYPE_CHECKING:
    from agent.lsp.manager import LSPService

logger = logging.getLogger("agent.lsp.edit_check")

# A write waits for its verdict only while it stays inside WRITE_BUDGET (from write entry).  A warm
# tsserver answers its synchronous request in ~10-30ms and a warm pyright its pull in ~10-90ms, so the
# common verdict is reported inline; a slower one is delivered later (agent.lsp.deferred).
WRITE_BUDGET = 0.08
INLINE_GRACE = 0.08
_VERDICT_CAP = 256  # per-file last verdicts kept for baseline reuse
_SPAWN_CEILING = 60.0  # outer bound on one check beyond its two waits (cold spawn + initialize)

_Diags = list[dict[str, Any]]


@dataclass
class CheckResult:
    diagnostics: Optional[_Diags]  # introduced by the edit (baseline removed); [] = clean; None = no verdict
    baseline_text: Optional[str]  # the pre-edit text this result was diffed against
    superseded: bool = False  # a newer check of the file by the same session took this one over
    # Undelivered older notes this check took over and did not settle (no verdict): the caller queues
    # them again.  Empty whenever there is a verdict (it was diffed against the oldest baseline).
    carry: tuple = ()


@dataclass
class _Inflight:
    task: asyncio.Task
    baseline_text: Optional[str]
    carry: tuple
    superseded: bool = False


class EditChecks:
    """Per-service state, touched only on the loop thread: the last verdict per file (baseline reuse),
    the newest unfinished check per file (``tail``: serializes every session's checks of it) and per
    ``(session, file)`` (``inflight``: the one a newer edit by that session takes over)."""

    def __init__(self) -> None:
        self.verdicts: OrderedDict[str, tuple[str, _Diags]] = OrderedDict()
        self.tail: dict[str, _Inflight] = {}
        self.inflight: dict[tuple[Optional[str], str], _Inflight] = {}
        # abs path -> content of the newest edit scheduled for it (set on the tool thread at schedule
        # time; a single dict store): what the server should hold once a check has diagnosed its own,
        # possibly older, snapshot.
        self.newest: dict[str, str] = {}

    def remember(self, abs_path: str, text: str, diags: _Diags) -> None:
        self.verdicts.pop(abs_path, None)
        self.verdicts[abs_path] = (_digest(text), list(diags))
        while len(self.verdicts) > _VERDICT_CAP:
            self.verdicts.popitem(last=False)

    def recall(self, abs_path: str, text: str) -> Optional[_Diags]:
        hit = self.verdicts.get(abs_path)
        return list(hit[1]) if hit is not None and hit[0] == _digest(text) else None


def inline_wait(started: Optional[float]) -> float:
    """Seconds the write may still wait for its verdict inline."""
    if started is None:
        return INLINE_GRACE
    return max(0.0, min(INLINE_GRACE, started + WRITE_BUDGET - time.monotonic()))


def _digest(text: str) -> str:
    return hashlib.sha256(as_disk_read(text).encode("utf-8", "surrogateescape")).hexdigest()


def start(svc: LSPService, path: str, *, pre_text: Optional[str], post_text: Optional[str] = None,
          owner: Optional[str] = None, carry: tuple = ()) -> Optional[concurrent.futures.Future]:
    """Schedule the check of ``path`` (already written) on the service loop; the Future resolves to a
    :class:`CheckResult`.  ``pre_text`` None = the file is new; ``post_text`` = the content this edit
    wrote (None: read the file when the check runs).  ``owner`` = the editing session (None:
    no later delivery, nothing to take over); ``carry`` = that session's undelivered notes for this
    file (``agent.lsp.deferred``) this edit supersedes: the oldest one's baseline becomes ours."""
    from agent.async_utils import safe_schedule_threadsafe
    loop = svc._loop._loop
    if loop is None:
        return None
    abs_path = os.path.abspath(path)
    checks: EditChecks = svc._edit_checks
    if post_text is not None:
        checks.newest[abs_path] = post_text
    else:
        checks.newest.pop(abs_path, None)
    return safe_schedule_threadsafe(
        _serialized(svc, abs_path, pre_text, post_text, owner, tuple(carry)), loop)


async def _serialized(svc: LSPService, abs_path: str, pre_text: Optional[str], post_text: Optional[str],
                      owner: Optional[str], carry: tuple) -> CheckResult:
    """Run after any unfinished check of the same file; take over the same session's unfinished check:
    its baseline and carried notes become ours, and its result is ours to report when we get no
    verdict.  Never cancelled: cancelling mid-spawn would restart a cold server on every rapid edit."""
    checks: EditChecks = svc._edit_checks
    baseline_text = carry[0].baseline_text if carry else pre_text
    tail = checks.tail.get(abs_path)
    tail = tail if tail is not None and not tail.task.done() else None
    prev = checks.inflight.get((owner, abs_path)) if owner else None
    if prev is not None and not prev.task.done():
        prev.superseded = True
        baseline_text, carry = prev.baseline_text, prev.carry + carry
    else:
        prev = None
    current = asyncio.current_task()
    assert current is not None
    me = _Inflight(current, baseline_text, carry)
    checks.tail[abs_path] = me
    if owner:
        checks.inflight[(owner, abs_path)] = me
    try:
        prev_result = None
        waits = {t.task for t in (prev, tail) if t is not None}
        if waits:
            await asyncio.wait(waits)
        if prev is not None and not prev.task.cancelled() and prev.task.exception() is None:
            prev_result = prev.task.result()
            if prev_result.diagnostics is None:
                prev_result = None
        if me.superseded:
            # A newer edit by this session queued behind us: it checks the newest bytes against our
            # baseline and owns our carried notes, so our own round trip would only delay it.
            return CheckResult(prev_result.diagnostics if prev_result else None,
                               prev_result.baseline_text if prev_result else baseline_text, True)
        budget = svc._wait_budget(abs_path)
        try:
            diags = await asyncio.wait_for(_check(svc, abs_path, baseline_text, post_text, budget),
                                           timeout=2 * budget + _SPAWN_CEILING)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # timeout / dead server: same treatment as the old blocking path
            logger.debug("LSP edit check failed for %s", abs_path, exc_info=True)
            await svc._mark_broken_for_file_async(abs_path, e)
            diags = None
        if diags is not None:
            result = CheckResult(diags, baseline_text)
        elif prev_result is not None:
            result = CheckResult(prev_result.diagnostics, prev_result.baseline_text)
        else:
            result = CheckResult(None, baseline_text, carry=carry)
        result.superseded = me.superseded
        if me.superseded:
            result.carry = ()  # the newer check took them over
        return result
    finally:
        if checks.tail.get(abs_path) is me:
            del checks.tail[abs_path]
            if post_text is not None and checks.newest.get(abs_path) is post_text:
                del checks.newest[abs_path]  # no newer edit scheduled: the server holds these bytes
        if owner and checks.inflight.get((owner, abs_path)) is me:
            del checks.inflight[(owner, abs_path)]


async def _check(svc: LSPService, abs_path: str, baseline_text: Optional[str], post_text: Optional[str],
                 budget: float) -> Optional[_Diags]:
    """Verdict on ``post_text`` (the bytes this edit wrote) first; only when it has diagnostics, the
    verdict on ``baseline_text`` (cached from an earlier check when that text was checked before) to drop
    the pre-existing ones.  The server is left holding the file's current disk content afterwards.
    A clean edit costs one round trip, and push-only servers answer the baseline fast in the common
    case (errors -> fewer errors is a change they publish)."""
    client = await svc._get_or_spawn(abs_path)
    if client is None:
        return None
    srv = svc._server_for(abs_path)
    server_id = srv.server_id if srv is not None else "lsp"
    language_id = language_id_for(abs_path, srv)
    checks: EditChecks = svc._edit_checks
    use_sync = sync_diagnostics.supported(client)
    if client.seeds_first_push and client.document_text(abs_path) is None and not use_sync:
        # First sync of this document on a seeding server: its first push is a verdict on the didOpen
        # text that never counts as fresh.  Open the pre-edit text so that push becomes the baseline;
        # a new file is opened as written, so the seed is the verdict itself.
        seed = await client.wait_for_seed(abs_path, budget) if await client.open_file(
            abs_path, language_id=language_id, text=post_text if baseline_text is None else baseline_text
        ) == 0 else None
        if baseline_text is None:
            await _resync_disk(svc, client, abs_path, language_id)
            svc._touch(client)
            return _report(server_id, abs_path, seed)
        if seed is not None:
            checks.remember(abs_path, baseline_text, seed)
    version = await client.open_file(abs_path, language_id=language_id, text=post_text)
    await client.save_file(abs_path)
    diags = await _verdict(svc, client, abs_path, version, budget, use_sync)
    svc._touch(client)
    if diags is None:
        await _resync_disk(svc, client, abs_path, language_id)
        # Alive but no verdict in budget: "no data", never stale stores (ghost diagnostics).
        return _report(server_id, abs_path, None)
    synced = client.document_text(abs_path)
    verdict = diags
    if diags and baseline_text is not None and synced is not None:
        # Read the cached pre-edit verdict (an earlier check's, or the seed above) BEFORE this edit's
        # verdict replaces the file's cache entry.
        baseline = checks.recall(abs_path, baseline_text)
        if baseline is None:
            baseline = []
            base_version = await client.open_file(abs_path, language_id=language_id, text=baseline_text)
            # No verdict on the pre-edit text -> empty baseline: the filter removes less, never more.
            base = await _verdict(svc, client, abs_path, base_version, budget, use_sync)
            if base is not None:
                baseline = base
        from agent.lsp.range_shift import build_line_shift, shift_baseline
        seen = {_diagnostic_key(d) for d in shift_baseline(baseline, build_line_shift(baseline_text, synced))}
        diags = [d for d in diags if _diagnostic_key(d) not in seen]
    if synced is not None:
        checks.remember(abs_path, synced, verdict)  # the next edit's baseline is these bytes
    await _resync_disk(svc, client, abs_path, language_id)
    return _report(server_id, abs_path, diags)


async def _resync_disk(svc: LSPService, client: Any, abs_path: str, language_id: str) -> None:
    """Leave the server holding the newest content written to the file (a newer edit, or this check's
    baseline sync, may differ from the snapshot just diagnosed); no-op when it already does.  Unknown
    newest content (a caller without a snapshot) re-reads the file, as every sync used to."""
    newest = svc._edit_checks.newest.get(abs_path)
    if newest is None:
        try:
            await client.open_file(abs_path, language_id=language_id)
        except Exception:  # the file vanished: the next check syncs whatever is there then
            logger.debug("LSP resync of %s failed", abs_path, exc_info=True)
    elif as_disk_read(newest) != client.document_text(abs_path):
        await client.open_file(abs_path, language_id=language_id, text=newest)


async def _verdict(svc: LSPService, client: Any, abs_path: str, version: int, budget: float,
                   use_sync: bool) -> Optional[_Diags]:
    """Diagnostics on the text synced at ``version``: a synchronous request where the server has one
    (``agent.lsp.sync_diagnostics``, ~10ms, so the write can report it inline), else the push/pull
    wait.  None = no verdict within ``budget``."""
    if use_sync:
        try:
            diags = await sync_diagnostics.request(client, abs_path, timeout=budget)
        except TimeoutError:
            return None
        if diags is not None:
            return diags
    if not await client.wait_for_diagnostics(abs_path, version, mode=svc._wait_mode, timeout=budget):
        return None
    return list(client.diagnostics_for(abs_path, fresh_only=True))


def _report(server_id: str, abs_path: str, diags: Optional[_Diags]) -> Optional[_Diags]:
    if diags is None:
        eventlog.log_timeout(server_id, abs_path, kind="fresh diagnostics")
    elif diags:
        eventlog.log_diagnostics(server_id, abs_path, len(diags))
    else:
        eventlog.log_clean(server_id, abs_path)
    return diags


__all__ = ["INLINE_GRACE", "WRITE_BUDGET", "CheckResult", "EditChecks", "inline_wait", "start"]
