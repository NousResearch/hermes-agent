#!/usr/bin/env python3
"""Live key_cmd catalog A/B: local authenticated HTTP, real helper, CLI PTY.

Run with the Hermes venv Python and --repo CHECKOUT --output RECEIPT_DIR.
No user environment/config is inherited by Hermes children. Unix PTY required.
"""
import argparse
import http.server
import json
import os
from pathlib import Path
import pty
import re
import select
import shlex
import subprocess
import sys
import tempfile
import threading
import time

CATALOG = ["live-configured", "live-discovered-b", "live-discovered-c"]
TOKEN = "local-eval-token-not-a-secret"


def rows_worker():
    from hermes_cli.config import load_config
    from hermes_cli.model_switch_providers import (
        _PickerBuild, _lap_custom_provider_rows, _lap_user_provider_rows,
    )
    cfg = load_config()
    b = _PickerBuild(current_provider="", current_base_url="", current_model="",
                     max_models=None, for_picker=True, force_fresh_nous_tier=False,
                     probe_custom_providers=True, probe_current_custom_provider=False,
                     refresh=False, excluded=set(), curated={})
    if cfg.get("providers"):
        _lap_user_provider_rows(b, cfg["providers"])
    else:
        _lap_custom_provider_rows(b, cfg["custom_providers"])
    print("RECEIPT_ROWS=" + json.dumps(b.results))


def cli_pty(repo, env):
    master, slave = pty.openpty()
    import fcntl
    import struct
    import termios
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 45, 140, 0, 0))
    proc = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "model"],
                            cwd=repo, env=env, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True)
    os.close(slave)
    data = bytearray()
    selected = False
    deadline = time.monotonic() + 45
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.2)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                data.extend(chunk)
            text = data.decode(errors="replace")
            if not selected and ("Select provider:" in text or "Choice [1-" in text):
                os.write(master, b"\r")
                selected = True
            if selected and (re.search(r"Found \d+ model", text)
                             or "Could not fetch models" in text):
                # Let the actual picker render before cancelling; never save a selection.
                end = time.monotonic() + 0.7
                while time.monotonic() < end:
                    if select.select([master], [], [], 0.1)[0]:
                        data.extend(os.read(master, 65536))
                os.write(master, b"\x03")
                break
            if proc.poll() is not None:
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=10)
        os.close(master)
    text = data.decode(errors="replace")
    found = re.search(r"Found (\d+) model", text)
    if not found:
        raise RuntimeError("CLI failed to reach model picker:\n" + text[-12000:])
    return int(found.group(1)), text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect", choices=["before", "after"])
    args = ap.parse_args()
    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    requests = []
    phase = ""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            authorized = self.headers.get("Authorization") == "Bearer " + TOKEN
            requests.append({"phase": phase, "path": self.path, "authorized": authorized})
            body = json.dumps({"data": [{"id": m} for m in CATALOG]} if authorized
                              else {"error": "authorization required"}).encode()
            self.send_response(200 if authorized else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = {}
    try:
        for schema in ("providers", "custom_providers"):
            for surface in ("cli", "rows"):
                phase = f"{schema}/{surface}"
                with tempfile.TemporaryDirectory(prefix="hermes-keycmd-live-") as tmp:
                    home = Path(tmp)
                    state = home / ".hermes"
                    state.mkdir()
                    mint_log = home / "helper-invocations"
                    helper = home / "mint.py"
                    helper.write_text("from pathlib import Path\n"
                                      f"with Path({str(mint_log)!r}).open('a') as f: f.write('mint\\n')\n"
                                      f"print({TOKEN!r})\n")
                    entry = {"name": "Live Keycmd", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                             "key_cmd": f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))}",
                             "model": CATALOG[0], "models": {CATALOG[0]: {}},
                             "models_discovered": True}
                    slug = "live-keycmd" if schema == "providers" else "custom:live-keycmd"
                    cfg = {"model": {"provider": slug, "default": CATALOG[0]},
                           schema: {slug: entry} if schema == "providers" else [entry]}
                    # JSON is valid YAML; no third-party harness dependencies.
                    (state / "config.yaml").write_text(json.dumps(cfg))
                    env = {"HOME": str(home), "HERMES_HOME": str(state),
                           "PATH": "/usr/bin:/bin", "TERM": "xterm-256color", "LANG": "C.UTF-8",
                           "PYTHONPATH": str(repo), "PYTHONUNBUFFERED": "1"}
                    if surface == "cli":
                        count, transcript = cli_pty(repo, env)
                        results[phase] = {"model_count": count}
                    else:
                        code = ("import runpy; ns=runpy.run_path(" + repr(str(Path(__file__).resolve()))
                                + "); ns['rows_worker']()")
                        proc = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
                        transcript = proc.stdout + proc.stderr
                        if proc.returncode:
                            raise RuntimeError(transcript)
                        rows = json.loads(next(s.split("=", 1)[1] for s in proc.stdout.splitlines()
                                               if s.startswith("RECEIPT_ROWS=")))
                        assert len(rows) == 1, rows
                        results[phase] = {"model_count": rows[0]["total_models"], "models": rows[0]["models"]}
                    results[phase]["helper_invocations"] = len(mint_log.read_text().splitlines()) if mint_log.exists() else 0
                    (args.output / (phase.replace("/", "-") + ".txt")).write_text(transcript)
        receipt = {"repo": str(repo), "sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "server_catalog": CATALOG, "results": results, "requests": requests}
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        if args.expect:
            expected = 1 if args.expect == "before" else len(CATALOG)
            assert all(row["model_count"] == expected for row in results.values()), results
            assert all(row["helper_invocations"] > 0 for row in results.values()) if args.expect == "after" else True
            assert all(any(r["phase"] == p and r["authorized"] == (args.expect == "after")
                           for r in requests) for p in results), requests
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
