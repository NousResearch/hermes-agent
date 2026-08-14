"""Measure the gateway's attach-RPC dispatch, against the real dispatcher.

Every attach handler (image.attach, image.attach_bytes, file.attach,
clipboard.paste, pdf.attach) resolves its session through ``_sess()``, which
blocks on the deferred agent build. None of them is in ``_LONG_HANDLERS``, so
that block happens INLINE on the socket reader thread.

This drives the real ``tui_gateway.server.dispatch`` with a session whose
agent build has not completed, and times it. ``prompt.submit`` (which uses
``_sess_nowait``) is timed alongside as the control — it is the path that
stays instant today.

    python3 scripts/perf/gateway_attach_bench.py [--build-seconds 8] [--rounds 3]
"""

from __future__ import annotations

import argparse
import base64
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="hermes-bench-home-"))


class CollectTransport:
    """Stand-in for the WS transport: records frames, never touches a socket."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.lock = threading.Lock()

    def write(self, obj: dict) -> bool:
        with self.lock:
            self.frames.append(obj)
        return True

    def close(self) -> None:
        return None


def make_session(server, sid: str, *, build_seconds: float, home: Path) -> dict:
    """A session whose deferred agent build is still running.

    Mirrors the shape ``_deferred_build`` leaves behind: an unset ``agent_ready``
    event plus a live build thread. That is exactly the state a session is in
    for the first seconds after ``session.create`` — which is when a user
    pastes their first image.
    """
    ready = threading.Event()
    session: dict = {
        "agent": None,
        "agent_ready": ready,
        "agent_error": None,
        "attached_images": [],
        "cwd": str(home),
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "image_counter": 0,
        "profile_home": str(home),
        "running": False,
        "session_key": sid,
        "transport": None,
    }

    def build() -> None:
        time.sleep(build_seconds)
        ready.set()

    thread = threading.Thread(target=build, daemon=True)
    session["_agent_build_thread"] = thread
    thread.start()

    server._sessions[sid] = session
    return session


def png_bytes(kb: int) -> bytes:
    body = bytearray(b"\x89PNG\r\n\x1a\n")
    body.extend(bytes((i * 37) & 0xFF for i in range(kb * 1024)))
    return bytes(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-seconds", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kb", type=int, default=900)
    args = ap.parse_args()

    from tui_gateway import server

    # The build is already in flight for these sessions (that is the state the
    # bench recreates), so the "start one if none is running" call is a no-op.
    # Without this stub the real builder races the bench's controlled one and
    # completes instantly, hiding the very wait being measured.
    server._start_agent_build = lambda sid, session: None

    # Keep the run readable: session.info frames go to the transport, not stdout.
    server._emit = lambda *a, **k: None

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)

    content_b64 = base64.b64encode(png_bytes(args.kb)).decode("ascii")

    scratch = home / "scratch.txt"
    scratch.write_text("hello from the bench\n")

    image_on_disk = home / "on-disk.png"
    image_on_disk.write_bytes(png_bytes(args.kb))

    pdf_on_disk = home / "doc.pdf"
    pdf_on_disk.write_bytes(b"%PDF-1.4\n" + b"0" * 2048 + b"\n%%EOF\n")

    calls = [
        (
            "image.attach_bytes",
            lambda sid: {
                "session_id": sid,
                "content_base64": content_b64,
                "filename": "bench.png",
            },
        ),
        (
            "image.attach",
            lambda sid: {"session_id": sid, "path": str(image_on_disk)},
        ),
        (
            "file.attach",
            lambda sid: {
                "session_id": sid,
                "name": "scratch.txt",
                "path": str(scratch),
            },
        ),
        (
            "pdf.attach",
            lambda sid: {"session_id": sid, "path": str(pdf_on_disk)},
        ),
        (
            "clipboard.paste",
            lambda sid: {"session_id": sid},
        ),
        (
            "image.detach",
            lambda sid: {"session_id": sid, "path": "/tmp/nothing.png"},
        ),
        (
            "prompt.submit",
            lambda sid: {"session_id": sid, "text": "control: plain text"},
        ),
    ]

    print(
        f"agent build takes {args.build_seconds:.1f}s; "
        f"image is {args.kb} KB; {args.rounds} rounds\n"
    )
    print(f"{'rpc':<22} {'in _LONG_HANDLERS':<19} {'mean':>8} {'max':>8}   blocks reader?")

    for method, build_params in calls:
        samples: list[float] = []

        for round_index in range(args.rounds):
            sid = f"bench-{method}-{round_index}"
            make_session(server, sid, build_seconds=args.build_seconds, home=home)
            transport = CollectTransport()
            req = {
                "jsonrpc": "2.0",
                "id": round_index,
                "method": method,
                "params": build_params(sid),
            }

            start = time.perf_counter()
            try:
                server.dispatch(req, transport)
            except Exception as exc:  # noqa: BLE001 - report, don't mask
                print(f"  ! {method} raised {type(exc).__name__}: {exc}")
            samples.append(time.perf_counter() - start)

            server._sessions.pop(sid, None)

        pooled = method in server._LONG_HANDLERS
        mean = statistics.mean(samples)
        worst = max(samples)
        verdict = "no (pooled)" if pooled else ("YES" if mean > 1.0 else "no")

        print(
            f"{method:<22} {str(pooled):<19} {mean:>7.2f}s {worst:>7.2f}s   {verdict}"
        )

    print(
        "\ndispatch() returns immediately for pooled handlers, so a pooled timing\n"
        "is the enqueue cost — the work still happens, just off the reader thread."
    )

    _report_surfaces()
    return 0


def _report_surfaces() -> None:
    """Which surfaces can even reach this code path.

    The stall lives in the gateway's session resolver, so a surface is exposed
    only if it attaches over the gateway. That is a fact about the call graph
    rather than a timing, so it is read out of the source — and it moves if
    the call graph moves.
    """
    print("\n\n=== which surfaces reach the gateway attach RPCs ===\n")

    root = Path(__file__).resolve().parents[4]
    attach_rpcs = ("image.attach", "image.attach_bytes", "file.attach", "clipboard.paste")

    surfaces = {
        "CLI (cli.py)": [root / "cli.py"],
        "TUI (ui-tui)": sorted((root / "ui-tui" / "src").rglob("*.ts")),
        "Desktop (apps/desktop)": sorted((root / "apps" / "desktop" / "src").rglob("*.ts")),
    }

    for label, paths in surfaces.items():
        hits: set[str] = set()

        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for rpc in attach_rpcs:
                if f"'{rpc}'" in text or f'"{rpc}"' in text:
                    hits.add(rpc)

        if hits:
            print(f"  {label:<24} EXPOSED — calls {', '.join(sorted(hits))}")
        else:
            print(f"  {label:<24} not exposed — no gateway attach RPC")

    print(
        "\n  CLI attaches inline in its own turn path (cli.py → image_routing) with\n"
        "  the agent already constructed. There is no gateway session to resolve,\n"
        "  so the stall is structurally unreachable — matching the ~4s report.\n"
        "\n  The TUI calls the SAME RPCs and was equally exposed. What differed was\n"
        "  hit rate, not code path: Desktop mints sessions constantly (new chat,\n"
        "  tabs, tiles), so a paste routinely lands inside the seconds-long window\n"
        "  while a fresh session's agent is still building. A TUI user launches\n"
        "  once and the build finishes while they type."
    )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
