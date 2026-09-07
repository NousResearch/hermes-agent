"""Live PTY auth controls with isolated homes and a loopback OAuth server.

Usage: python evals/auth_pool_controls.py REPO OUTPUT_JSON
No vendor requests or real credentials are used. The token URL alone is redirected;
the production parser, command, pool refresh, HTTP client, and persistence execute.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import errno
import pty
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("output")
    args = parser.parse_args()
    requests = []
    response_status = 200

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            requests.append({"grant_type": payload.get("grant_type"),
                             "target_grant": payload.get("refresh_token") == ["fixture-refresh-1"]})
            body = ({"access_token": "fixture-new-access", "refresh_token": "fixture-new-refresh"}
                    if response_status == 200 else {"error": "invalid_grant" if response_status == 401 else "unavailable"})
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format, *values):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cases = [
        ("target-reset", "openrouter", ["reset", "openrouter", "row1"], None),
        ("all-reset", "openrouter", ["reset", "openrouter"], None),
        ("ambiguous-reset", "openrouter", ["reset", "openrouter", "shared"], None),
        ("priority", "openrouter", ["priority", "openrouter", "row1", "0"], None),
        ("priority-clamp", "openrouter", ["priority", "openrouter", "row0", "99"], None),
        ("priority-missing", "openrouter", ["priority", "openrouter", "missing", "0"], None),
        ("add-priority", "openrouter", ["add", "openrouter", "--api-key", "fixture-new", "--label", "new", "--priority", "0"], None),
        ("refresh-api-key", "openrouter", ["refresh", "openrouter", "row1"], None),
        ("refresh-ambiguous", "openai-codex", ["refresh", "openai-codex"], None),
        *[(f"refresh-{status}", "openai-codex", ["refresh", "openai-codex", "row1"], status)
          for status in (200, 503, 401)],
    ]
    results = []
    try:
        for name, provider, command, status in cases:
            response_status = status or 200
            requests.clear()
            with tempfile.TemporaryDirectory(prefix="auth-pool-pty-") as temp:
                home = Path(temp)
                now = time.time()
                rows = [dict(id=f"row{i}", label="shared" if name == "ambiguous-reset" else f"account{i}",
                             source="manual:device_code" if provider == "openai-codex" else "manual",
                             auth_type="oauth" if provider == "openai-codex" else "api_key",
                             access_token=f"fixture-access-{i}", refresh_token=f"fixture-refresh-{i}",
                             priority=i, last_status="exhausted", last_status_at=now,
                             last_error_code=429, last_error_reset_at=now+3600) for i in range(2)]
                store = home / "auth.json"
                store.write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {provider: rows}}), encoding="utf-8")
                env = {k: v for k, v in os.environ.items() if not any(t in k for t in
                       ("TOKEN", "API_KEY", "SECRET", "PASSWORD", "HERMES", "PYTEST"))}
                env.update(HOME=temp, HERMES_HOME=temp, PYTHONPATH=str(Path(args.repo).absolute()), TERM="xterm")
                bootstrap = ("import sys; from hermes_cli import auth_codex; "
                             f"auth_codex.CODEX_OAUTH_TOKEN_URL='http://127.0.0.1:{server.server_port}/token'; "
                             "from hermes_cli.main import main; "
                             f"sys.argv=['hermes','auth',*{command!r}]; main()")
                master, slave = pty.openpty()
                child = subprocess.Popen([sys.executable, "-c", bootstrap], cwd=args.repo,
                                         env=env, stdin=slave, stdout=slave, stderr=slave)
                os.close(slave)
                chunks = []
                while True:
                    try:
                        chunk = os.read(master, 65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        break
                os.close(master)
                output = b"".join(chunks).decode("utf-8")
                exit_code = child.wait(timeout=30)
                disk = json.loads(store.read_text(encoding="utf-8"))["credential_pool"][provider]
                results.append({"case": name, "exit": exit_code, "output": output,
                                "wire": list(requests), "secrets_printed": "fixture-" in output,
                                "disk": [{k: e.get(k) for k in ("id", "priority", "last_status", "last_error_reset_at", "request_count")}
                                         for e in disk],
                                "target_rotated": any(e["id"] == "row1" and e.get("access_token") == "fixture-new-access" for e in disk),
                                "sibling_cooldown_preserved": next(e for e in disk if e["id"] == "row0").get("last_error_reset_at") == now+3600})
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    Path(args.output).write_text(json.dumps({"repo": args.repo, "cases": results}, indent=2), encoding="utf-8")
    print(json.dumps([{"case": r["case"], "exit": r["exit"], "posts": len(r["wire"]),
                       "target_rotated": r["target_rotated"], "secrets_printed": r["secrets_printed"]} for r in results], indent=2))


if __name__ == "__main__":
    main()
