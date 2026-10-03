"""Live E2E for #55712: real uvicorn, stub rotating IdP with reuse detection.

Proves (a) N concurrent stale-RT requests on BOTH refresh paths rotate exactly once, and
(b) /api/status answers while a slow provider refresh is in flight (the event-loop wedge).
Run against origin/main to see both fail; against the fix to see both pass.
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="hermes-e2e-55712-")
for m in [k for k in sys.modules if k.startswith(("hermes", "tools", "plugins"))]:
    del sys.modules[m]

import uvicorn  # noqa: E402

from hermes_cli import web_server  # noqa: E402
from hermes_cli.dashboard_auth import register_provider  # noqa: E402
from hermes_cli.dashboard_auth.base import RefreshExpiredError, Session  # noqa: E402
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider  # noqa: E402


class SlowRotatingIdP(StubAuthProvider):
    name = "stub"

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.rotated = set()
        self.delay = 0.0
        self.lock = threading.Lock()

    def verify_session(self, *, access_token):
        return None if access_token.startswith("expired") else super().verify_session(access_token=access_token)

    def refresh_session(self, *, refresh_token):
        with self.lock:
            self.calls += 1
            if refresh_token in self.rotated:
                raise RefreshExpiredError("reuse detected -> session revoked")
            self.rotated.add(refresh_token)
        time.sleep(self.delay)
        return Session(user_id="u1", email="u@x.test", display_name="U", org_id="o", provider="stub",
                       expires_at=int(time.time()) + 900, access_token="fresh-" + refresh_token,
                       refresh_token="rotated-" + refresh_token)


idp = SlowRotatingIdP()
register_provider(idp)
app = web_server.app
app.state.bound_host = "127.0.0.1"
app.state.auth_required = True

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
base = f"http://127.0.0.1:{port}"
for _ in range(100):
    try:
        with urllib.request.urlopen(base + "/api/status", timeout=1):
            break
    except Exception:
        time.sleep(0.1)


def http(path, *, method="GET", body=None, headers=None):
    req = urllib.request.Request(base + path, method=method, headers=headers or {},
                                 data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


results = {}

# (1) native bearer path: 4 concurrent refreshes with the same stale RT
idp.calls = 0
with ThreadPoolExecutor(4) as pool:
    codes = sorted(pool.map(lambda _: http("/auth/native/refresh", method="POST",
                                            body={"refresh_token": "native-stale", "provider": "stub"}), range(4)))
results["native_burst"] = {"codes": codes, "provider_calls": idp.calls}

# (2) cookie gate path: 4 concurrent gated requests with an expired AT + one stale RT
idp.calls = 0
ck = "hermes_session_at=expired-at; hermes_session_rt=cookie-stale; hermes_session_provider=stub"
with ThreadPoolExecutor(4) as pool:
    codes = sorted(pool.map(lambda _: http("/api/auth/me", headers={"cookie": ck}), range(4)))
results["cookie_burst"] = {"codes": codes, "provider_calls": idp.calls}

# (3) event-loop wedge: a 3s provider refresh must not block /api/status
idp.delay = 3.0
slow = threading.Thread(target=lambda: http("/auth/native/refresh", method="POST",
                                             body={"refresh_token": "slow-stale", "provider": "stub"}))
slow.start(); time.sleep(0.3)
t0 = time.monotonic(); code = http("/api/status"); dt = time.monotonic() - t0
slow.join()
results["status_during_slow_refresh"] = {"code": code, "seconds": round(dt, 2)}

ok = (results["native_burst"] == {"codes": [200] * 4, "provider_calls": 1}
      and results["cookie_burst"] == {"codes": [200] * 4, "provider_calls": 1}
      and code == 200 and dt < 1.0)
print(json.dumps(results, indent=1))
print("VERDICT:", "PASS" if ok else "FAIL")
server.should_exit = True
