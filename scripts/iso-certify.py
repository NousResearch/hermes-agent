#!/usr/bin/env python3
"""iso-certify — AC-4 dashboard turn-isolation certify harness.

Certifies Mechanism-B process isolation
(``docs/desktop/2026-07-04-dashboard-process-isolation-PRD.md``, AC-4): under
6 concurrent heavy agent turns, the dashboard's HTTP/ws SERVING plane must stay
responsive (p99 < 1s) with zero event-loop stalls.

What it does
------------
1. Spawns a SCRATCH dashboard (``hermes dashboard``) bound to loopback on a
   free port, with an ISOLATED ``HERMES_HOME`` (temp dir, minimal seeded state).
   It NEVER touches the live :9119 dashboard / ai.hermes.dashboard / live
   state.db. Loopback bind ⇒ no auth gate (web_server.should_require_auth).
2. Arms the synthetic GIL-heavy turn seam (``HERMES_ISO_CERTIFY_SYNTH_TURN=1``,
   see ``tui_gateway/synthetic_turn.py``) so 6 concurrent turns reproduce the
   ``take_gil`` interpreter-contention regime WITHOUT real model calls. A
   network/sleep stub would release the GIL and NOT reproduce the incident, so
   it would be a fake green — the synthetic turn is pure-Python CPU on purpose.
3. Drives 6 concurrent heavy turns over ws (session.create → prompt.submit),
   and CONCURRENTLY probes the serving path — a ws ``session.list`` round-trip
   AND a REST ``GET /api/status`` — every 500ms, timing each.
4. Reports p50/p95/p99 latency for both probes + the count of probes over the
   1s threshold ("serving stalls") + the count of ``event loop stalled`` /
   ``ws write slow`` lines the dashboard logged during the run.

Verdict
-------
The AC-4 verdict comes ONLY from the heavy run: PASS iff serving p99 < 1s AND
zero serving stalls AND zero ``event loop stalled`` log lines over the sustained
window. ``--dry-run`` runs ONE short light turn as a plumbing smoke test and is
explicitly NOT a verdict (a dry-run green is a fake green — the spec says so).

Turn isolation is controlled by the ``dashboard.turn_isolation`` config knob in
the scratch HERMES_HOME; ``--isolation on|off`` sets it. Run BOTH:
    iso-certify --isolation off   # baseline: expect stalls
    iso-certify --isolation on    # the measurement that decides AC-4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from websockets.sync.client import connect as ws_connect
except Exception as exc:  # pragma: no cover - dependency guard
    print(f"iso-certify requires the 'websockets' package: {exc}", file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parents[1]
_READY_RE = re.compile(r"HERMES_(?:DASHBOARD|BACKEND)_READY port=(\d+)")
_STALL_LOG_RE = re.compile(r"event loop stalled|ws write slow \(loop stalled")


# ── stats ──────────────────────────────────────────────────────────────
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values) if values else 0.0,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ── scratch HERMES_HOME ─────────────────────────────────────────────────
def seed_scratch_home(home: Path, *, isolation: str, heartbeat_secs: int, respawn_max: int) -> None:
    """Write a minimal config.yaml with the isolation knob set."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "state").mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(parents=True, exist_ok=True)
    cfg = {
        # Pin the model/provider to what SyntheticHeavyAgent reports so the
        # per-turn _sync_agent_model_with_config sees a match and no-ops — a
        # mismatch would try (and fail) a real model switch, erroring the turn
        # before its heavy loop runs (which would be a FALSE green: serving
        # stays responsive because no heavy compute happened). The synthetic
        # seam means no real API call is ever made regardless.
        "provider": "synthetic",
        "model": "synthetic-heavy",
        "dashboard": {
            "turn_isolation": (isolation == "on"),
            "compute_host_heartbeat_secs": heartbeat_secs,
            "compute_host_respawn_max": respawn_max,
        },
        # Keep memory/mem0/skills side-machinery from reaching out.
        "memory": {"enabled": False},
    }
    # config.yaml is the canonical config; write it directly.
    import yaml  # provided by the runtime venv

    (home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    # A stub .env so credential resolution doesn't spelunk the real home.
    (home / ".env").write_text("OPENAI_API_KEY=sk-synthetic-not-used\n", encoding="utf-8")


# ── dashboard process ───────────────────────────────────────────────────
class ScratchDashboard:
    def __init__(
        self,
        *,
        home: Path,
        port: int,
        isolation: str,
        env_extra: dict[str, str],
    ) -> None:
        self.home = home
        self.port = port
        self.isolation = isolation
        self.env_extra = env_extra
        self.proc: subprocess.Popen[str] | None = None
        self.actual_port = port
        self.log_lines: list[str] = []
        self._log_lock = threading.Lock()
        self._ready = threading.Event()

    @property
    def stall_log_count(self) -> int:
        with self._log_lock:
            return sum(1 for ln in self.log_lines if _STALL_LOG_RE.search(ln))

    def _drain(self, stream: Any) -> None:
        for raw in stream:
            line = raw.rstrip("\n")
            with self._log_lock:
                self.log_lines.append(line)
            m = _READY_RE.search(line)
            if m:
                self.actual_port = int(m.group(1))
                self._ready.set()

    def __enter__(self) -> "ScratchDashboard":
        venv_py = REPO_ROOT / "venv" / "bin" / "python"
        python = str(venv_py) if venv_py.exists() else sys.executable
        env = dict(os.environ)
        env.update(self.env_extra)
        env["HERMES_HOME"] = str(self.home)
        env["HOME"] = str(self.home.parent) if str(self.home.parent) else env.get("HOME", "")
        env["HERMES_HOME"] = str(self.home)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["HERMES_ISO_CERTIFY_SYNTH_TURN"] = "1"
        cmd = [
            python, "-m", "hermes_cli.main", "dashboard",
            "--no-open", "--host", "127.0.0.1", "--port", str(self.port),
        ]
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        threading.Thread(target=self._drain, args=(self.proc.stdout,), name="dash-log", daemon=True).start()
        if not self._ready.wait(timeout=90.0):
            self._dump_tail()
            raise RuntimeError("scratch dashboard did not become ready within 90s")
        # Give uvicorn a beat to actually bind the ws route.
        time.sleep(1.0)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def _dump_tail(self, n: int = 40) -> None:
        with self._log_lock:
            tail = self.log_lines[-n:]
        sys.stderr.write("---- scratch dashboard log tail ----\n")
        for ln in tail:
            sys.stderr.write(ln + "\n")
        sys.stderr.write("------------------------------------\n")


# ── ws client (one connection = one lane) ───────────────────────────────
class WSClient:
    def __init__(self, port: int, token: str) -> None:
        self.url = f"ws://127.0.0.1:{port}/api/ws?token={token}"
        self.ws = ws_connect(
            self.url,
            open_timeout=15,
            max_size=None,
            additional_headers={"Origin": f"http://127.0.0.1:{port}"},
        )
        self._id = 0
        self._lock = threading.Lock()
        # Drain the gateway.ready event.
        self._recv_until(lambda o: o.get("method") == "event", timeout=10)

    def _next_id(self) -> str:
        with self._lock:
            self._id += 1
            return f"r{self._id}"

    def _recv_until(self, pred, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                raw = self.ws.recv(timeout=max(0.05, deadline - time.monotonic()))
            except TimeoutError:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if pred(obj):
                return obj
        raise TimeoutError("ws recv predicate timed out")

    def rpc(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        rid = self._next_id()
        self.ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
        return self._recv_until(lambda o: o.get("id") == rid, timeout=timeout)

    def send_only(self, method: str, params: dict) -> str:
        rid = self._next_id()
        self.ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
        return rid

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


# ── heavy-turn lane ─────────────────────────────────────────────────────
def drive_heavy_turn(port: int, token: str, turn_spec: dict, stop_at: float, results: list[dict]) -> None:
    """One lane: create a session, submit heavy turns until the deadline."""
    lane: dict[str, Any] = {"turns": 0, "errors": [], "turn_durations_s": [], "min_deltas": None}
    try:
        cli = WSClient(port, token)
    except Exception as exc:
        lane["errors"].append(f"connect: {exc}")
        results.append(lane)
        return
    try:
        resp = cli.rpc("session.create", {"cols": 80, "source": "iso-certify"}, timeout=30)
        sid = ((resp.get("result") or {}).get("session_id")) or ((resp.get("result") or {}).get("id"))
        if not sid:
            lane["errors"].append(f"session.create bad resp: {resp}")
            results.append(lane)
            return
        lane["sid"] = sid
        spec_text = json.dumps(turn_spec)
        while time.monotonic() < stop_at:
            # prompt.submit returns immediately ("streaming"); the turn runs async.
            r = cli.rpc("prompt.submit", {"session_id": sid, "text": spec_text}, timeout=30)
            if r.get("error"):
                lane["errors"].append(f"prompt.submit: {r['error']}")
                break
            # Wait for the REAL turn boundary. The isolated path emits
            # session.info MID-turn (metadata mirror), so waiting on it would
            # false-complete a turn in <1s and inflate the count while NO heavy
            # compute ran — the acceptance-gate "proxy not effect" trap. The
            # turn is done only on message.complete. Count deltas + duration so
            # a fast-erroring turn (e.g. a failed model switch) is caught, not
            # masked as sustained load.
            per_turn_budget = turn_spec.get("duration_s", 8.0) + 30.0
            deadline = time.monotonic() + per_turn_budget
            turn_start = time.monotonic()
            deltas = 0
            started = False
            done = False
            errored = False
            while time.monotonic() < deadline:
                try:
                    o = cli._recv_until(
                        lambda o: o.get("method") == "event"
                        and (o.get("params") or {}).get("type")
                        in {"message.start", "message.delta", "message.complete", "error"},
                        timeout=max(0.5, deadline - time.monotonic()),
                    )
                except TimeoutError:
                    break
                ptype = (o.get("params") or {}).get("type")
                if ptype == "message.start":
                    started = True
                    turn_start = time.monotonic()
                    deltas = 0
                    continue
                # prompt.submit's RPC response may race with a duplicate terminal
                # event from the previous turn. Only events after this turn's
                # message.start can satisfy or score its load boundary.
                if not started:
                    continue
                if ptype == "message.delta":
                    deltas += 1
                    continue
                if ptype == "error":
                    msg = str(((o.get("params") or {}).get("payload") or {}).get("message") or "")
                    lane["errors"].append(f"turn error: {msg[:160]}")
                    errored = True
                    break
                if ptype == "message.complete":
                    done = True
                    break
            turn_dur = time.monotonic() - turn_start
            if done:
                lane["turns"] += 1
                lane["turn_durations_s"].append(round(turn_dur, 2))
                lane["min_deltas"] = deltas if lane["min_deltas"] is None else min(lane["min_deltas"], deltas)
            if errored:
                break
    except Exception as exc:
        lane["errors"].append(f"lane: {exc}")
    finally:
        cli.close()
        results.append(lane)


# ── serving-path probes ─────────────────────────────────────────────────
def warmup_serving(port: int, token: str, rounds: int = 6) -> None:
    """Prime serving-path caches before the measured window.

    The first ``/api/status`` on a freshly-booted process pays one-time
    cold-start cost (config-version check, gateway-health probe, DB connect) that
    a real dashboard — warm for hours before the incident regime — never pays
    during a stall. AC-4 measures sustained-load responsiveness, not cold boot,
    so we hit both serving endpoints a few times UNMEASURED first. This is not a
    green-washing shortcut: the measured window still runs the full 6-lane heavy
    load; warmup only removes a one-time boot artifact from the p99.
    """
    rest_url = f"http://127.0.0.1:{port}/api/status"
    warm_ws: WSClient | None = None
    try:
        warm_ws = WSClient(port, token)
    except Exception:
        warm_ws = None
    for _ in range(rounds):
        try:
            with urllib.request.urlopen(rest_url, timeout=30) as fh:
                fh.read()
        except Exception:
            pass
        if warm_ws is not None:
            try:
                warm_ws.rpc("session.list", {"limit": 20}, timeout=30)
            except Exception:
                pass
        time.sleep(0.2)
    if warm_ws is not None:
        warm_ws.close()


def probe_loop(port: int, token: str, stop_at: float, cadence_s: float, ws_samples: list[float], rest_samples: list[float]) -> None:
    """Probe ws session.list + REST /api/status every ``cadence_s`` seconds."""
    probe_ws: WSClient | None = None
    try:
        probe_ws = WSClient(port, token)
    except Exception:
        probe_ws = None
    rest_url = f"http://127.0.0.1:{port}/api/status"
    next_tick = time.monotonic()
    while time.monotonic() < stop_at:
        # ws round-trip (session.list — the serving-plane DB read AC-4 protects).
        if probe_ws is not None:
            t0 = time.perf_counter()
            try:
                probe_ws.rpc("session.list", {"limit": 20}, timeout=30)
                ws_samples.append((time.perf_counter() - t0) * 1000.0)
            except Exception:
                # A failed/timed-out probe is itself a serving stall; record the
                # elapsed as the sample so it counts against p99.
                ws_samples.append((time.perf_counter() - t0) * 1000.0)
                try:
                    probe_ws.close()
                    probe_ws = WSClient(port, token)
                except Exception:
                    probe_ws = None
        # REST round-trip.
        t1 = time.perf_counter()
        try:
            with urllib.request.urlopen(rest_url, timeout=30) as fh:
                fh.read()
            rest_samples.append((time.perf_counter() - t1) * 1000.0)
        except Exception:
            rest_samples.append((time.perf_counter() - t1) * 1000.0)
        next_tick += cadence_s
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()
    if probe_ws is not None:
        probe_ws.close()


# ── run ─────────────────────────────────────────────────────────────────
def run_certify(args: argparse.Namespace) -> dict[str, Any]:
    port = free_port()
    import secrets
    token = secrets.token_urlsafe(24)
    parent_tmp = Path(tempfile.mkdtemp(prefix="iso-certify-"))
    home = parent_tmp / "hermes-home"
    seed_scratch_home(
        home,
        isolation=args.isolation,
        heartbeat_secs=args.heartbeat_secs,
        respawn_max=args.respawn_max,
    )

    concurrency = 1 if args.dry_run else args.concurrency
    duration_s = 3.0 if args.dry_run else args.duration_s
    turn_duration = 0.5 if args.dry_run else args.turn_duration_s
    threshold_ms = args.threshold_ms

    turn_spec = {
        "duration_s": turn_duration,
        "delta_interval_s": args.delta_interval_s,
        "tokens_per_delta": args.tokens_per_delta,
        "chunk": args.chunk,
    }

    result: dict[str, Any] = {
        "mode": "dry-run" if args.dry_run else "heavy",
        "isolation": args.isolation,
        "concurrency": concurrency,
        "run_duration_s": duration_s,
        "turn_spec": turn_spec,
        "threshold_ms": threshold_ms,
        "scratch_home": str(home),
        "port": port,
    }

    try:
        with ScratchDashboard(
            home=home, port=port, isolation=args.isolation,
            env_extra={"HERMES_DASHBOARD_SESSION_TOKEN": token},
        ) as dash:
            actual_port = dash.actual_port
            result["port"] = actual_port
            ws_samples: list[float] = []
            rest_samples: list[float] = []
            lane_results: list[dict] = []
            stall_before = dash.stall_log_count

            # Prime serving-path caches so a one-time cold-start artifact does
            # not count as a serving stall (only for the real heavy run; a
            # dry-run stays a raw plumbing smoke).
            if not args.dry_run:
                warmup_serving(actual_port, token)
                stall_before = dash.stall_log_count

            stop_at = time.monotonic() + duration_s
            probe_thread = threading.Thread(
                target=probe_loop,
                args=(actual_port, token, stop_at, args.cadence_s, ws_samples, rest_samples),
                name="probe", daemon=True,
            )
            probe_thread.start()

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for _ in range(concurrency):
                    pool.submit(drive_heavy_turn, actual_port, token, turn_spec, stop_at, lane_results)
                # Pool context waits for all lanes.
            probe_thread.join(timeout=30)

            stall_after = dash.stall_log_count
            total_turns = sum(l.get("turns", 0) for l in lane_results)
            lane_errors = [e for l in lane_results for e in l.get("errors", [])]
            all_durations = [d for l in lane_results for d in l.get("turn_durations_s", [])]
            min_deltas = [l["min_deltas"] for l in lane_results if l.get("min_deltas") is not None]
            lanes_with_turn = sum(1 for l in lane_results if l.get("turns", 0) > 0)

            ws_stat = summarize(ws_samples)
            rest_stat = summarize(rest_samples)
            ws_over = sum(1 for v in ws_samples if v > threshold_ms)
            rest_over = sum(1 for v in rest_samples if v > threshold_ms)
            serving_stalls = ws_over + rest_over
            log_stalls = stall_after - stall_before

            # Load validity — the run only certifies anything if the offered load
            # was REAL: every lane completed ≥1 turn, and the completed turns
            # actually held ~the requested heavy duration and streamed deltas.
            # A fast-erroring/short turn is NOT sustained GIL load, so a green off
            # it would be a proxy (serving stays fast because nothing burned).
            median_turn_dur = percentile(all_durations, 50) if all_durations else 0.0
            min_turn_dur = min(all_durations) if all_durations else 0.0
            worst_min_deltas = min(min_deltas) if min_deltas else 0
            expected_turn_s = float(turn_spec.get("duration_s", 8.0))
            load_valid = (
                lanes_with_turn >= concurrency
                and total_turns >= concurrency
                and min_turn_dur >= 0.7 * expected_turn_s
                and worst_min_deltas >= 5
            )

            result.update({
                "ws_probe": ws_stat,
                "rest_probe": rest_stat,
                "ws_probes_over_threshold": ws_over,
                "rest_probes_over_threshold": rest_over,
                "serving_stalls": serving_stalls,
                "event_loop_stall_log_lines": log_stalls,
                "heavy_turns_completed": total_turns,
                "lanes_with_turn": lanes_with_turn,
                "median_turn_duration_s": round(median_turn_dur, 2),
                "min_turn_duration_s": round(min_turn_dur, 2),
                "worst_lane_min_deltas": worst_min_deltas,
                "load_valid": load_valid,
                "lane_errors": lane_errors[:20],
            })

            # AC-4 verdict — heavy run only. A dry-run reports but never PASSes.
            serving_p99 = max(ws_stat["p99_ms"], rest_stat["p99_ms"])
            result["serving_p99_ms"] = serving_p99
            if args.dry_run:
                result["verdict"] = "SMOKE-OK" if (total_turns > 0 and not lane_errors) else "SMOKE-FAIL"
                result["is_verdict"] = False
            else:
                serving_ok = (
                    serving_p99 < threshold_ms
                    and serving_stalls == 0
                    and log_stalls == 0
                    and probe_thread_samples_ok(ws_samples, rest_samples)
                )
                if not load_valid:
                    # Cannot certify: the offered load was not the incident
                    # regime. Never a PASS; report INCONCLUSIVE, not FAIL, so it
                    # is not read as "isolation broke serving".
                    result["verdict"] = "INCONCLUSIVE"
                    result.setdefault("notes", []).append(
                        "load invalid: lanes/turn-duration/deltas below the sustained-heavy-load floor — "
                        "not the AC-4 incident regime, verdict cannot certify"
                    )
                else:
                    result["verdict"] = "PASS" if serving_ok else "FAIL"
                result["is_verdict"] = True
    finally:
        if not args.keep_home:
            shutil.rmtree(parent_tmp, ignore_errors=True)
        else:
            result["scratch_home_kept"] = str(parent_tmp)

    return result


def probe_thread_samples_ok(ws_samples: list[float], rest_samples: list[float]) -> bool:
    """Guard against a blind gate: require the probes actually ran.

    A run that produced no probe samples saw NOTHING — it must not PASS. This is
    the tri-state INCONCLUSIVE floor (an empty timeline is never a green).
    """
    return len(ws_samples) >= 3 and len(rest_samples) >= 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AC-4 dashboard turn-isolation certify harness")
    p.add_argument("--isolation", choices=["on", "off"], default="on",
                   help="set dashboard.turn_isolation in the scratch HERMES_HOME")
    p.add_argument("--dry-run", action="store_true",
                   help="1 short light turn plumbing smoke — NOT an AC-4 verdict")
    p.add_argument("--concurrency", type=int, default=6, help="concurrent heavy-turn lanes (AC-4: 6)")
    p.add_argument("--duration-s", type=float, default=600.0, dest="duration_s",
                   help="sustained run window seconds (AC-4: ~600 = 10 min)")
    p.add_argument("--turn-duration-s", type=float, default=12.0, dest="turn_duration_s",
                   help="wall seconds of GIL-holding compute per heavy turn")
    p.add_argument("--delta-interval-s", type=float, default=0.05, dest="delta_interval_s",
                   help="streamed-delta cadence per heavy turn")
    p.add_argument("--tokens-per-delta", type=int, default=512, dest="tokens_per_delta")
    p.add_argument("--chunk", type=int, default=20000, help="pure-Python ops per interrupt-check chunk")
    p.add_argument("--cadence-s", type=float, default=0.5, dest="cadence_s",
                   help="serving-path probe cadence (AC-4: 500ms)")
    p.add_argument("--threshold-ms", type=float, default=1000.0, dest="threshold_ms",
                   help="serving p99 threshold (AC-4: <1s)")
    p.add_argument("--heartbeat-secs", type=int, default=15, dest="heartbeat_secs")
    p.add_argument("--respawn-max", type=int, default=3, dest="respawn_max")
    p.add_argument("--keep-home", action="store_true", help="do not delete the scratch HERMES_HOME on exit")
    p.add_argument("--json-out", type=Path, help="write JSON metrics to this path")
    args = p.parse_args(argv)

    result = run_certify(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")

    # Exit code: 0 only on a real PASS (or a clean dry-run smoke); non-zero
    # otherwise. A dry-run is never treated as a verdict for automation.
    if result.get("is_verdict"):
        return 0 if result.get("verdict") == "PASS" else 1
    return 0 if result.get("verdict") == "SMOKE-OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
