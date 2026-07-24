"""Two-OS-process regression for the shared xAI OAuth store (#65394 / PR #67261).

Acceptance boundary this file alone covers:

* Two **separate OS processes** (distinct interpreter PIDs), each with its own
  ``HERMES_HOME``, both pointing at one shared ``HERMES_SHARED_AUTH_DIR``.
* Real kernel advisory locking (``fcntl.flock`` on ``xai_oauth.json.lock``),
  not an in-process ``threading`` stand-in.
* Exactly **one** refresh POST to a single-use local HTTP stub; the loser
  must adopt the winner's rotated tokens instead of re-POSTing the burned RT.

This deliberately does **not** monkeypatch the shared lock, the resolve/adopt
path, or ``refresh_xai_oauth_pure``'s HTTP body. The only worker-side patch is
``_xai_validate_oauth_endpoint`` so a loopback stub URL is accepted (production
pins ``*.x.ai`` HTTPS — that pin is unit-tested elsewhere).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

# Repo root so worker subprocesses can import hermes_cli with PYTHONPATH=.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_FLAG = "--xai-shared-auth-two-process-worker"


# ---------------------------------------------------------------------------
# Helpers (parent + worker)
# ---------------------------------------------------------------------------


def _jwt(exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    )
    return f"{header}.{payload}.sig"


def _wait_for_path(path: Path, timeout: float, *, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"timed out after {timeout}s waiting for {label}: {path}")


def _wait_for_paths(paths: Tuple[Path, ...], timeout: float, *, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(p.exists() for p in paths):
            return
        time.sleep(0.01)
    missing = [str(p) for p in paths if not p.exists()]
    raise TimeoutError(f"timed out after {timeout}s waiting for {label}; missing={missing}")


def _scrub_credential_env(env: Dict[str, str]) -> Dict[str, str]:
    """Drop credential-shaped vars so workers cannot inherit live keys."""
    suffixes = (
        "_API_KEY",
        "_TOKEN",
        "_SECRET",
        "_PASSWORD",
        "_CREDENTIALS",
        "_ACCESS_KEY",
        "_SECRET_ACCESS_KEY",
        "_PRIVATE_KEY",
        "_OAUTH_TOKEN",
    )
    # Keep our coordination env vars even if a suffix would match by accident.
    keep_prefixes = ("HERMES_XAI_TP_",)
    out: Dict[str, str] = {}
    for key, value in env.items():
        if any(key.startswith(p) for p in keep_prefixes):
            out[key] = value
            continue
        if key in {
            "XAI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_TOKEN",
            "GITHUB_TOKEN",
            "GH_TOKEN",
        }:
            continue
        if any(key.endswith(suf) for suf in suffixes):
            continue
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Single-use refresh stub (parent process)
# ---------------------------------------------------------------------------


class _SingleUseRefreshStub:
    """Local token endpoint that burns the seeded RT on first successful POST.

    * First POST presenting ``seed_rt`` → 200 with rotated tokens (gen2).
      The response is **held** until ``release_path`` exists so the second
      OS process can queue on the shared flock while the winner still holds it.
    * Any later POST with the same consumed RT → 400 ``invalid_grant``.
    * Thread-safe counters for successful / invalid_grant POSTs.
    """

    def __init__(
        self,
        *,
        seed_rt: str,
        rotated_access: str,
        rotated_refresh: str,
        first_post_path: Path,
        release_path: Path,
        hold_timeout: float = 20.0,
    ) -> None:
        self.seed_rt = seed_rt
        self.rotated_access = rotated_access
        self.rotated_refresh = rotated_refresh
        self.first_post_path = first_post_path
        self.release_path = release_path
        self.hold_timeout = hold_timeout
        self._lock = threading.Lock()
        self.success_posts = 0
        self.invalid_grant_posts = 0
        self.other_posts = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.token_endpoint = ""

    def start(self) -> str:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                params = urllib.parse.parse_qs(raw, keep_blank_values=True)
                rt = (params.get("refresh_token") or [""])[0]
                grant = (params.get("grant_type") or [""])[0]

                hold_after_claim = False
                status = 400
                body: Dict[str, Any] = {
                    "error": "invalid_request",
                    "error_description": "unexpected",
                }

                with stub._lock:
                    if grant != "refresh_token" or not rt:
                        stub.other_posts += 1
                        body = {
                            "error": "invalid_request",
                            "error_description": "missing grant_type or refresh_token",
                        }
                    elif rt == stub.seed_rt:
                        if stub.success_posts >= 1:
                            # Seeded RT already consumed — single-use enforcement.
                            stub.invalid_grant_posts += 1
                            body = {
                                "error": "invalid_grant",
                                "error_description": "refresh_token already used",
                            }
                        else:
                            stub.success_posts += 1
                            hold_after_claim = True
                            status = 200
                            body = {
                                "access_token": stub.rotated_access,
                                "refresh_token": stub.rotated_refresh,
                                "token_type": "Bearer",
                                "expires_in": 3600,
                            }
                    else:
                        stub.other_posts += 1
                        body = {
                            "error": "invalid_grant",
                            "error_description": "unknown refresh_token",
                        }

                if hold_after_claim:
                    # Signal parent that the winner is inside pure-refresh HTTP
                    # while still holding the shared flock.
                    stub.first_post_path.write_text("1", encoding="utf-8")
                    deadline = time.monotonic() + stub.hold_timeout
                    while not stub.release_path.exists():
                        if time.monotonic() >= deadline:
                            # Fail loud rather than release early and race.
                            status = 500
                            body = {
                                "error": "server_error",
                                "error_description": "hold timeout waiting for release",
                            }
                            break
                        time.sleep(0.01)

                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return  # silence access log noise in pytest output

        # Bind ephemeral port on loopback only.
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server = server
        host, port = server.server_address[:2]
        self.token_endpoint = f"http://{host}:{port}/oauth/token"
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self.token_endpoint

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


# ---------------------------------------------------------------------------
# Worker entrypoint (runs in a child OS process)
# ---------------------------------------------------------------------------


def _worker_main() -> int:
    """Child process: barrier → real resolve/refresh → write result JSON."""
    label = os.environ["HERMES_XAI_TP_LABEL"]
    coord = Path(os.environ["HERMES_XAI_TP_COORD"])
    result_path = Path(os.environ["HERMES_XAI_TP_RESULT"])
    stub_url = os.environ["HERMES_XAI_TP_TOKEN_ENDPOINT"]
    expired_at = os.environ["HERMES_XAI_TP_EXPIRED_AT"]
    barrier_timeout = float(os.environ.get("HERMES_XAI_TP_BARRIER_TIMEOUT", "25"))

    def _write_result(payload: Dict[str, Any], *, exit_code: int) -> int:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return exit_code

    # Phase 1: filesystem barrier — both children must arrive before either
    # enters resolve, so they genuinely contend on the flock.
    try:
        (coord / f"ready_{label}").write_text(str(os.getpid()), encoding="utf-8")
        _wait_for_path(coord / "go", barrier_timeout, label="parent go signal")
    except Exception as exc:
        return _write_result(
            {
                "ok": False,
                "pid": os.getpid(),
                "error": f"barrier: {type(exc).__name__}: {exc}",
            },
            exit_code=2,
        )

    # Allow the loopback stub as token_endpoint. Production validation pins
    # https://*.x.ai; we do NOT patch the lock, pure-refresh HTTP, or adopt path.
    from hermes_cli import auth

    real_validate = auth._xai_validate_oauth_endpoint

    def _allow_stub_endpoint(url: str, *, field: str) -> str:
        if url == stub_url:
            return url
        return real_validate(url, field=field)

    auth._xai_validate_oauth_endpoint = _allow_stub_endpoint  # type: ignore[assignment]

    (coord / f"starting_{label}").write_text(str(os.getpid()), encoding="utf-8")

    try:
        creds = auth.resolve_xai_oauth_runtime_credentials(
            force_refresh=True,
            rejected_access_token=expired_at,
            expected_generation=1,
        )
    except Exception as exc:
        return _write_result(
            {
                "ok": False,
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
                "code": getattr(exc, "code", None),
            },
            exit_code=1,
        )

    return _write_result(
        {
            "ok": True,
            "pid": os.getpid(),
            "api_key": creds.get("api_key"),
            "generation": creds.get("generation"),
            "source": creds.get("source"),
            "auth_store": creds.get("auth_store"),
            "error": None,
            "code": None,
        },
        exit_code=0,
    )


# ---------------------------------------------------------------------------
# Parent test
# ---------------------------------------------------------------------------


def test_two_os_processes_shared_store_one_refresh_loser_adopts(tmp_path: Path) -> None:
    """#65394 multi-profile/multi-gateway death scenario — real two-process path.

    Spawns two genuine child interpreters with distinct HERMES_HOME roots and
    one shared auth dir. A single-use HTTP stub + filesystem barrier force
    flock contention so exactly one process POSTs and the other adopts.
    """
    import subprocess

    from hermes_cli import auth

    shared_dir = tmp_path / "shared"
    root_a = tmp_path / "rootA"
    root_b = tmp_path / "rootB"
    home_sandbox = tmp_path / "home"
    coord = tmp_path / "coord"
    for path in (shared_dir, root_a, root_b, home_sandbox, coord):
        path.mkdir()

    # Empty profile-root under sandbox HOME so fleet enumeration never
    # walks the real user's ~/.hermes/profiles (HOME-anchored).
    (home_sandbox / ".hermes" / "profiles").mkdir(parents=True)

    seed_rt = "seed-rt-single-use-gen1"
    rotated_access = "rotated-access-token-gen2"
    rotated_refresh = "rotated-refresh-token-gen2"
    expired_access = _jwt(int(time.time()) - 120)

    first_post_path = coord / "first_post_received"
    release_path = coord / "release_first_response"

    stub = _SingleUseRefreshStub(
        seed_rt=seed_rt,
        rotated_access=rotated_access,
        rotated_refresh=rotated_refresh,
        first_post_path=first_post_path,
        release_path=release_path,
        hold_timeout=20.0,
    )
    token_endpoint = stub.start()
    try:
        # Seed canonical shared store: generation 1, expired AT, single-use RT.
        store_path = shared_dir / "xai_oauth.json"
        store_path.write_text(
            json.dumps(
                {
                    "_schema": 1,
                    "generation": 1,
                    "access_token": expired_access,
                    "refresh_token": seed_rt,
                    "token_type": "Bearer",
                    "auth_mode": "oauth_device_code",
                    "last_refresh": "2026-07-01T00:00:00Z",
                    "discovery": {"token_endpoint": token_endpoint},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(store_path, 0o600)

        # Clean per-root auth.json — no local RT forks to start with.
        empty_auth = json.dumps({"version": 1, "providers": {}}, indent=2) + "\n"
        for root in (root_a, root_b):
            (root / "auth.json").write_text(empty_auth, encoding="utf-8")
            os.chmod(root / "auth.json", 0o600)

        parent_pid = os.getpid()
        result_a = coord / "result_a.json"
        result_b = coord / "result_b.json"

        base_env = _scrub_credential_env(dict(os.environ))
        base_env.update(
            {
                "PYTHONPATH": str(_REPO_ROOT),
                "TZ": "UTC",
                "LANG": "C.UTF-8",
                "HOME": str(home_sandbox),
                "HERMES_XAI_SHARED_AUTH": "1",
                "HERMES_SHARED_AUTH_DIR": str(shared_dir),
                "HERMES_XAI_TP_COORD": str(coord),
                "HERMES_XAI_TP_TOKEN_ENDPOINT": token_endpoint,
                "HERMES_XAI_TP_EXPIRED_AT": expired_access,
                "HERMES_XAI_TP_BARRIER_TIMEOUT": "25",
                # Keep refresh + lock windows bounded for the test.
                "HERMES_XAI_REFRESH_TIMEOUT_SECONDS": "15",
                # Seat belt: refuse real-user shared path if misconfigured.
                "PYTEST_CURRENT_TEST": (
                    "tests/hermes_cli/test_xai_shared_auth_two_process.py::"
                    "test_two_os_processes_shared_store_one_refresh_loser_adopts"
                ),
            }
        )
        # Do not inherit a parent HERMES_HOME into children; each sets its own.
        base_env.pop("HERMES_HOME", None)

        worker_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            _WORKER_FLAG,
        ]
        env_a = {
            **base_env,
            "HERMES_HOME": str(root_a),
            "HERMES_XAI_TP_LABEL": "a",
            "HERMES_XAI_TP_RESULT": str(result_a),
        }
        env_b = {
            **base_env,
            "HERMES_HOME": str(root_b),
            "HERMES_XAI_TP_LABEL": "b",
            "HERMES_XAI_TP_RESULT": str(result_b),
        }

        proc_a = subprocess.Popen(
            worker_cmd,
            env=env_a,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc_b = subprocess.Popen(
            worker_cmd,
            env=env_b,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Barrier: both children ready, then release them together.
            _wait_for_paths(
                (coord / "ready_a", coord / "ready_b"),
                20.0,
                label="both workers ready",
            )
            (coord / "go").write_text("1", encoding="utf-8")

            # Winner must enter pure-refresh (first POST) AND both must have
            # entered resolve so the loser is blocked on the advisory lock
            # while the stub still holds the winner's HTTP response.
            _wait_for_paths(
                (
                    first_post_path,
                    coord / "starting_a",
                    coord / "starting_b",
                ),
                20.0,
                label="first POST held + both workers in resolve",
            )
            # Tiny beat so the non-holder has entered the flock wait loop
            # (non-blocking flock retries every 50ms in _file_lock).
            time.sleep(0.08)
            release_path.write_text("1", encoding="utf-8")

            try:
                out_a, err_a = proc_a.communicate(timeout=35)
            except subprocess.TimeoutExpired:
                proc_a.kill()
                out_a, err_a = proc_a.communicate(timeout=5)
            try:
                out_b, err_b = proc_b.communicate(timeout=35)
            except subprocess.TimeoutExpired:
                proc_b.kill()
                out_b, err_b = proc_b.communicate(timeout=5)
        finally:
            if proc_a.poll() is None:
                proc_a.kill()
            if proc_b.poll() is None:
                proc_b.kill()

        def _load_result(path: Path, *, name: str, rc: int, err: str) -> Dict[str, Any]:
            if not path.is_file():
                pytest.fail(
                    f"{name}: no result file (rc={rc}). stderr={err!r}"
                )
            return json.loads(path.read_text(encoding="utf-8"))

        payload_a = _load_result(result_a, name="worker-a", rc=proc_a.returncode, err=err_a)
        payload_b = _load_result(result_b, name="worker-b", rc=proc_b.returncode, err=err_b)

        # --- PID sanity: genuine separate OS processes ---
        pid_a = int(payload_a["pid"])
        pid_b = int(payload_b["pid"])
        assert pid_a != pid_b, "workers must have distinct PIDs"
        assert pid_a != parent_pid and pid_b != parent_pid, (
            "workers must not share the parent PID"
        )
        assert pid_a == proc_a.pid and pid_b == proc_b.pid

        # --- Both resolve successfully (loser must not hit invalid_grant) ---
        if not payload_a.get("ok") or not payload_b.get("ok"):
            pytest.fail(
                "one or both workers failed resolve/refresh:\n"
                f"  A: {payload_a}\n"
                f"  B: {payload_b}\n"
                f"  stub success={stub.success_posts} invalid_grant={stub.invalid_grant_posts}\n"
                f"  stderr_a={err_a!r}\n"
                f"  stderr_b={err_b!r}\n"
                f"  stdout_a={out_a!r}\n"
                f"  stdout_b={out_b!r}"
            )

        # --- Exactly one successful refresh POST; no invalid_grant ---
        assert stub.success_posts == 1, (
            f"expected exactly one successful refresh POST, got {stub.success_posts}; "
            f"invalid_grant={stub.invalid_grant_posts} other={stub.other_posts}"
        )
        assert stub.invalid_grant_posts == 0, (
            "loser re-POSTed the consumed RT (invalid_grant) — adopt/lock regressed"
        )
        assert stub.other_posts == 0

        # --- Both processes use the winner's rotated access token ---
        assert payload_a["api_key"] == rotated_access
        assert payload_b["api_key"] == rotated_access
        assert payload_a["generation"] == 2
        assert payload_b["generation"] == 2
        assert payload_a["source"] == auth.XAI_SHARED_SOURCE
        assert payload_b["source"] == auth.XAI_SHARED_SOURCE

        # --- Shared store advanced exactly once; one rotated RT ---
        shared_state = json.loads(store_path.read_text(encoding="utf-8"))
        assert shared_state["generation"] == 2
        assert shared_state["access_token"] == rotated_access
        assert shared_state["refresh_token"] == rotated_refresh
        assert shared_state["refresh_token"] != seed_rt

        # --- Neither HERMES_HOME retains a durable local xAI RT fork ---
        for root, name in ((root_a, "rootA"), (root_b, "rootB")):
            auth_path = root / "auth.json"
            assert auth_path.is_file(), f"{name} missing auth.json after resolve"
            store = json.loads(auth_path.read_text(encoding="utf-8"))
            assert not auth._auth_store_holds_durable_xai_refresh_token(store), (
                f"{name} retained a durable xAI refresh_token fork: {store}"
            )
            state = (store.get("providers") or {}).get("xai-oauth") or {}
            assert state.get("source") == auth.XAI_SHARED_SOURCE, (
                f"{name} profile source should be shared reference, got {state!r}"
            )
            assert "tokens" not in state or not (
                isinstance(state.get("tokens"), dict)
                and str(state["tokens"].get("refresh_token") or "").strip()
            )
            assert not str(state.get("refresh_token") or "").strip()
            pool = (store.get("credential_pool") or {}).get("xai-oauth") or []
            for entry in pool:
                if not isinstance(entry, dict):
                    continue
                assert not str(entry.get("refresh_token") or "").strip(), (
                    f"{name} credential_pool retained refresh_token: {entry}"
                )
    finally:
        stub.stop()


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        raise SystemExit(_worker_main())
    raise SystemExit(
        f"This module is a pytest file. Worker mode: python {sys.argv[0]} {_WORKER_FLAG}"
    )
