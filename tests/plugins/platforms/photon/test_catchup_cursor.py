"""Inbound catch-up across sidecar restarts.

spectrum-ts keeps its resume cursor in memory, so every new sidecar process
starts live-only and messages sent while no sidecar ran are lost. The sidecar
now keeps the cursor on disk (``sidecar/catchup.mjs``), hands it back to
spectrum through three hook lines (``sidecar/patch-spectrum-resume-cursor.mjs``)
and reports a refused cursor as an ``inbound_gap`` control line, which the
adapter logs and never dispatches. Related: #100032.

The decision module and the patch run under node; the end-to-end case drives
the real, patched ``resumableOrderedStream`` across two node processes.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter

_SIDECAR = Path("plugins/platforms/photon/sidecar").resolve()
_CATCHUP = (_SIDECAR / "catchup.mjs").as_uri()
_PATCHER = (_SIDECAR / "patch-spectrum-resume-cursor.mjs").as_uri()
_SPECTRUM_CORE = _SIDECAR / "node_modules" / "@spectrum-ts" / "core"


def _node(script: str, cwd: Path | None = None) -> Any:
    run = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=cwd or Path.cwd(), text=True, capture_output=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _catchup(script: str, state: Path) -> Any:
    return _node(
        f"import {{ createCatchUp, catchUpDisabled }} from {json.dumps(_CATCHUP)};\n"
        f"const statePath = {json.dumps(str(state))};\n"
        "const logs = []; const gaps = [];\n"
        "const make = (extra = {}) => createCatchUp({ statePath, projectId: 'p1',"
        " log: (m) => logs.push(m), onGap: (g) => gaps.push(g), ...extra });\n"
        "const msg = (id, ts) => ({ id, timestamp: ts || '2026-10-01T10:00:00.000Z' });\n"
        + script
    )


# -- Decision module ----------------------------------------------------------

def test_cursor_commits_only_after_delivery_and_survives_restart(tmp_path: Path) -> None:
    state = tmp_path / "runtime" / "photon-catchup.json"
    out = _catchup(
        """
        const a = make(); a.load();
        const L = 'imessage.messages:shared';
        const first = a.hook.initial(L);
        a.hook.note(L, { cursor: '5', values: [msg('m5')] });
        const beforeSettle = a.snapshot().cursors[L] ?? null;
        a.settle(msg('m5'), { delivered: true });
        const afterSettle = a.snapshot().cursors[L];
        a.hook.note(L, { cursor: '6', values: [msg('m6')] });  // emitted, never written

        const b = make(); b.load();
        const resumed = b.hook.initial(L);
        process.stdout.write(JSON.stringify({ first: first ?? null, beforeSettle,
          afterSettle, resumed, replay: b.isReplay(msg('m5')), fresh: b.isReplay(msg('m6')),
          snap: b.snapshot() }));
        """,
        state,
    )
    assert out["first"] is None
    assert out["beforeSettle"] is None
    assert out["afterSettle"] == "5"
    assert out["resumed"] == "5"  # m6 was never delivered, so it is replayed
    assert out["replay"] is True and out["fresh"] is False
    assert out["snap"]["resumedFrom"] == {"imessage.messages:shared": "5"}
    assert out["snap"]["replaysSkipped"] == 1
    assert stat.S_IMODE(state.stat().st_mode) == 0o600


def test_cursor_only_advance_waits_for_open_items(tmp_path: Path) -> None:
    out = _catchup(
        """
        const a = make(); a.load();
        const L = 'imessage.messages:shared';
        a.hook.note(L, { cursor: '7', values: [{ id: 'g', content: { type: 'group',
          items: [msg('g/0'), msg('g/1')] } }] });
        a.hook.note(L, { cursor: '9', values: [] });  // e.g. catchup.complete
        const held = a.snapshot().cursors[L] ?? null;
        a.settle(msg('g/0'), { delivered: true });
        const half = a.snapshot().cursors[L] ?? null;
        a.settle(msg('g/1'), { delivered: false });
        process.stdout.write(JSON.stringify({ held, half, done: a.snapshot().cursors[L] }));
        """,
        tmp_path / "state.json",
    )
    assert out == {"held": None, "half": None, "done": "9"}


def test_rejected_cursor_reports_gap_once_for_messages(tmp_path: Path) -> None:
    out = _catchup(
        """
        const a = make(); a.load();
        const L = 'imessage.messages:shared';
        a.hook.note(L, { cursor: '3', values: [msg('m3', '2026-10-01T09:00:00.000Z')] });
        a.settle(msg('m3', '2026-10-01T09:00:00.000Z'), { delivered: true });
        a.hook.rejected(L);
        a.hook.rejected('imessage.polls:shared');
        process.stdout.write(JSON.stringify({ gaps, snap: a.snapshot() }));
        """,
        tmp_path / "state.json",
    )
    assert len(out["gaps"]) == 1
    assert out["gaps"][0]["since"] == "2026-10-01T09:00:00.000Z"
    assert out["gaps"][0]["until"]
    assert out["snap"]["cursors"] == {}
    assert out["snap"]["gaps"] == 2


def test_state_from_another_project_and_opt_out_are_ignored(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"project": "someone-else", "cursors": {"L": "42"}}))
    out = _catchup(
        """
        const a = make(); a.load();
        const off = createCatchUp({ statePath: '' });
        process.stdout.write(JSON.stringify({ cursor: a.hook.initial('L') ?? null,
          offEnabled: off.enabled, offCursor: off.hook.initial('L') ?? null,
          flags: ['off', 'OFF', '0', 'false', 'on', ''].map(catchUpDisabled) }));
        """,
        state,
    )
    assert out["cursor"] is None
    assert out["offEnabled"] is False and out["offCursor"] is None
    assert out["flags"] == [True, True, True, True, False, False]


# -- spectrum-ts patch --------------------------------------------------------

def _patch(root: Path) -> Any:
    return _node(
        f"import {{ patchSpectrumResumeCursor }} from {json.dumps(_PATCHER)};\n"
        f"const root = {json.dumps(str(root))};\n"
        "let out;\n"
        "try { out = patchSpectrumResumeCursor(root); } catch (e) { out = { error: e.message }; }\n"
        "process.stdout.write(JSON.stringify(out));"
    )


def test_patch_refuses_unknown_spectrum_source_without_writing(tmp_path: Path) -> None:
    dist = tmp_path / "node_modules" / "@spectrum-ts" / "core" / "dist"
    dist.mkdir(parents=True)
    source = "const resumableOrderedStream = () => { let lastCursor = 1; };\n"
    (dist / "authoring.js").write_text(source)
    assert "unexpected spectrum-ts" in _patch(tmp_path)["error"]
    assert (dist / "authoring.js").read_text() == source


def _isolated_spectrum(tmp_path: Path) -> Path:
    """A node_modules whose @spectrum-ts/core is a private copy (patched freely)
    and whose other packages link to the installed sidecar deps."""
    if not _SPECTRUM_CORE.is_dir():
        pytest.skip("sidecar node_modules not installed")
    modules = tmp_path / "node_modules"
    (modules / "@spectrum-ts").mkdir(parents=True)
    for entry in (_SIDECAR / "node_modules").iterdir():
        if entry.name != "@spectrum-ts":
            os.symlink(entry, modules / entry.name)
    shutil.copytree(_SPECTRUM_CORE, modules / "@spectrum-ts" / "core")
    return tmp_path


def test_patch_applies_to_pinned_spectrum_and_is_idempotent(tmp_path: Path) -> None:
    root = _isolated_spectrum(tmp_path)
    assert _patch(root)["patched"] is True
    patched = (root / "node_modules/@spectrum-ts/core/dist/authoring.js").read_text()
    assert patched.count("globalThis.__hermesPhotonResumeCursor") == 3
    assert _patch(root)["patched"] is False


def test_restarted_process_resumes_through_spectrum_catch_up(tmp_path: Path) -> None:
    """Process 1 delivers m5 live and dies; process 2 resumes after cursor 5,
    gets m6 from the catch-up replay, and a refused cursor surfaces as a gap."""
    root = _isolated_spectrum(tmp_path)
    assert "error" not in _patch(root)
    state = tmp_path / "state.json"

    def run(live: list, missed: list | None, reject: bool = False) -> Any:
        return _node(
            f"import {{ createCatchUp, RESUME_HOOK }} from {json.dumps(_CATCHUP)};\n"
            "const gaps = [];\n"
            f"const c = createCatchUp({{ statePath: {json.dumps(str(state))}, projectId: 'p',"
            " log: () => {}, onGap: (g) => gaps.push(g) });\n"
            "c.load(); globalThis[RESUME_HOOK] = c.hook;\n"
            "const { resumableOrderedStream } = await import('@spectrum-ts/core/authoring');\n"
            f"const live = {json.dumps(live)}; const missed = {json.dumps(missed)};\n"
            f"const reject = {json.dumps(reject)}; let askedFrom = null;\n"
            # A live source that stays open after its events until spectrum close()s it, like Photon's.
            "const liveSource = () => { const q = [...live]; let wake; let closed = false; return {"
            " [Symbol.asyncIterator]() { return this; },"
            " next: () => closed ? Promise.resolve({ done: true }) : q.length"
            " ? Promise.resolve({ value: q.shift(), done: false }) : new Promise((r) => { wake = r; }),"
            " close: async () => { closed = true; wake?.({ done: true }); } }; };\n"
            "const item = (seq) => ({ cursor: String(seq), id: `e${seq}`,"
            " values: seq < 0 ? [] : [{ id: `m${seq}`, timestamp: '2026-10-01T10:00:00.000Z' }] });\n"
            "const s = resumableOrderedStream({ label: 'imessage.messages:shared',"
            " initialRetryDelayMs: 1, jitter: (d) => d,\n"
            "  isCursorRejectedError: (e) => e.message === 'gone',\n"
            "  fetchMissed: async function* (cursor) { askedFrom = cursor;"
            "    if (reject) throw new Error('gone'); for (const seq of missed) yield seq; },\n"
            "  processMissed: async (seq) => item(seq), processLive: async (seq) => item(seq),\n"
            "  subscribeLive: liveSource });\n"
            "const got = []; const want = live.length + (reject ? 0 : (missed || []).length);\n"
            # Exit from inside the loop: closing the stream would wait on the hung live source.
            "for await (const m of s) { got.push(m.id); c.settle(m, { delivered: true });\n"
            "  if (got.length === want) { process.stdout.write(JSON.stringify("
            "{ got, askedFrom, gaps, snap: c.snapshot() })); process.exit(0); } }\n",
            cwd=root,
        )

    first = run(live=[5], missed=None)
    assert first["got"] == ["m5"] and first["askedFrom"] is None
    assert first["snap"]["cursors"] == {"imessage.messages:shared": "5"}

    second = run(live=[], missed=[6])
    assert second["askedFrom"] == "5"
    assert second["got"] == ["m6"]
    assert second["snap"]["cursors"] == {"imessage.messages:shared": "6"}

    third = run(live=[7], missed=[], reject=True)
    assert third["askedFrom"] == "6"
    assert third["got"] == ["m7"]
    assert len(third["gaps"]) == 1


# -- Adapter ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inbound_gap_line_is_logged_never_dispatched(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))
    dispatch = AsyncMock()
    monkeypatch.setattr(adapter, "_dispatch_inbound", dispatch)
    line = json.dumps({"control": "inbound_gap", "since": "2026-10-01T09:00:00Z",
                       "until": "2026-10-01T11:00:00Z"})
    with caplog.at_level("WARNING", logger="plugins.platforms.photon.adapter"):
        await adapter._on_inbound_line(line)
        await adapter._on_inbound_line(json.dumps({"control": "something_new"}))
    dispatch.assert_not_awaited()
    assert any("2026-10-01T09:00:00Z" in r.getMessage() for r in caplog.records)
