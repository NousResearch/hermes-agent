"""Two profiles must not spend one single-use OAuth refresh token twice.

Profiles share the root login store: ``HERMES_HOME=<root>/profiles/<name>`` borrows the grant in
``<root>/auth.json`` and writes a refresh back there under the root's ``auth.lock``. Providers
that rotate single-use refresh tokens (openai-codex, xai-oauth) treat a replay as theft:
the loser gets ``refresh_token_reused`` and the provider may revoke the whole grant.

These tests run two real Hermes processes against one temporary root and a loopback token
endpoint that spends each refresh token on first use. The endpoint waits before it answers,
standing in for provider latency, so a refresh that does not hold the root lock for its whole
read-POST-write is caught. Every test asserts that the root token was spent exactly once, that
no process ended signed out, and that the root holds a token the provider will still accept.

Covered: the credential-pool refresh (openai-codex and xai-oauth), the xAI singleton resolver,
and the Codex pool replay by a process that loaded the pool before a peer rotated the token.

Subprocesses are used because the race is between processes (file locks, one store); an
in-process thread test would share module state the real race does not.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from tests.waiting import wait_until

_WORKER = Path(__file__).resolve().parents[1] / "_fixtures" / "oauth_refresh_worker.py"
_PROVIDER_LATENCY = 0.4  # seconds the endpoint holds a spent token before it answers
_STEP_TIMEOUT = 90.0  # a deadline for a worker to import Hermes or finish, not a delay
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _jwt(exp: float, subject: str = "fake") -> str:
    def encode(part: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=").decode()

    claims = {
        "exp": int(exp),
        "iat": int(time.time()),
        "sub": subject,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-fake"},
    }
    return f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode(claims)}.sig"


@dataclass
class _TokenEndpoint:
    """Loopback OAuth token endpoint with single-use rotating refresh tokens (fake credentials)."""

    latency: float
    live: set = field(default_factory=set)
    spent: set = field(default_factory=set)
    refreshes: list = field(default_factory=list)  # one dict per refresh_token POST
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _server: ThreadingHTTPServer | None = None
    port: int = 0

    def start(self) -> None:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if "openid-configuration" in self.path:
                    return self._reply(200, {
                        "issuer": "https://auth.x.ai",
                        "authorization_endpoint": "https://auth.x.ai/oauth2/auth",
                        "token_endpoint": "https://auth.x.ai/oauth2/token",
                    })
                self._reply(404, {"error": "not_found"})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                if "json" in (self.headers.get("Content-Type") or ""):  # Anthropic pkce rows post JSON
                    form = json.loads(body or "{}")
                else:
                    form = {k: v[0] for k, v in parse_qs(body).items()}
                if form.get("grant_type") != "refresh_token":
                    return self._reply(404, {"error": "not_found"})
                worker = self.headers.get("X-Worker", "?")
                code, reply = endpoint.spend(form.get("refresh_token", ""), worker)
                # Provider latency between consuming the token and answering.
                threading.Event().wait(endpoint.latency)  # health: allow HX013 -- the provider latency is the behaviour under test
                self._reply(code, reply)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def spend(self, refresh_token: str, worker: str) -> tuple[int, dict]:
        with self._lock:
            valid = refresh_token in self.live
            reused = refresh_token in self.spent
            self.live.discard(refresh_token)
            self.spent.add(refresh_token)
            rotated = f"rotated-{len(self.refreshes) + 1}-{uuid.uuid4().hex[:8]}" if valid else None
            access = _jwt(time.time() + 86400, rotated) if valid else None
            if valid:
                self.live.add(rotated)
            self.refreshes.append({
                "refresh_token": refresh_token, "valid": valid, "reused": reused,
                "rotated": rotated, "access_token": access, "worker": worker,
            })
        if not valid:
            return 400, {
                "error": "invalid_grant",
                "error_description": "refresh_token_reused" if reused else "unknown refresh token",
            }
        return 200, {
            "access_token": access,
            "refresh_token": rotated,
            "id_token": _jwt(time.time() + 86400, worker),
            "expires_in": 3600,
            "token_type": "Bearer",
        }


@pytest.fixture
def endpoint():
    server = _TokenEndpoint(latency=_PROVIDER_LATENCY)
    server.start()
    try:
        yield server
    finally:
        server.stop()


@dataclass
class _Root:
    path: Path
    profiles: list
    initial_refresh_token: str


@pytest.fixture
def shared_root(tmp_path, endpoint):
    """A root login store with an expired grant for each provider, and two profiles that borrow it."""

    def make(
        provider: str, access_valid_for: float = -600, pool_row: bool = False,
        row_source: str = "manual:device_code", singleton: bool = False,
    ) -> _Root:
        root = tmp_path / "hermes-root"
        profiles = [root / "profiles" / name for name in ("alpha", "beta")]
        for home in (root, *profiles):
            home.mkdir(parents=True, exist_ok=True)
            # No adopting of the OS user's own Codex or xAI login into the temporary stores.
            (home / "config.yaml").write_text(
                json.dumps({"auth": {"adopt_external_logins": False}}) + "\n", encoding="utf-8"
            )
        refresh_token = f"initial-{uuid.uuid4().hex}"
        state = {
            "tokens": {"access_token": _jwt(time.time() + access_valid_for), "refresh_token": refresh_token},
            "last_refresh": "2026-01-01T00:00:00Z",
            "auth_mode": "chatgpt" if provider == "openai-codex" else "oauth_device_code",
        }
        if provider == "xai-oauth":
            state["discovery"] = {
                "token_endpoint": "https://auth.x.ai/oauth2/token",
                "authorization_endpoint": "https://auth.x.ai/oauth2/auth",
            }
        store = {"version": 1, "providers": {provider: state}, "active_provider": provider}
        if pool_row:  # a login added to the pool by hand, with no singleton behind it
            store = {"version": 1, "credential_pool": {provider: [{
                "id": "account", "source": row_source, "auth_type": "oauth",
                "access_token": state["tokens"]["access_token"], "refresh_token": refresh_token,
                "last_refresh": state["last_refresh"],
            }]}}
            if provider == "anthropic":  # Anthropic rows are due by their expiry stamp, not a JWT
                store["credential_pool"][provider][0]["expires_at_ms"] = int((time.time() + access_valid_for) * 1000)
        (root / "auth.json").write_text(json.dumps(store), encoding="utf-8")
        endpoint.live.add(refresh_token)
        if singleton:  # the legacy singleton file the pool row was seeded from
            (root / ".anthropic_oauth.json").write_text(json.dumps({
                "accessToken": state["tokens"]["access_token"], "refreshToken": refresh_token,
                "expiresAt": int((time.time() + access_valid_for) * 1000),
            }), encoding="utf-8")
        return _Root(root, profiles, refresh_token)

    return make


class _Fleet:
    """Worker processes of one test; whatever is still running at the end is killed."""

    def __init__(self, tmp_path: Path, port: int):
        self._dir = tmp_path / "workers"
        self._dir.mkdir()
        self._home = tmp_path / "fake-home"
        for sub in ("AppData/Local", "AppData/Roaming", "tmp"):
            (self._home / sub).mkdir(parents=True, exist_ok=True)
        self._port = port
        self.procs: list[subprocess.Popen] = []

    def _env(self, hermes_home: Path) -> dict:
        keep = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT")
        env = {k: v for k, v in os.environ.items() if k.upper() in keep}
        env.update(
            HERMES_HOME=str(hermes_home), PYTHONUTF8="1", NO_PROXY="*",
            HOME=str(self._home), USERPROFILE=str(self._home),
            LOCALAPPDATA=str(self._home / "AppData/Local"), APPDATA=str(self._home / "AppData/Roaming"),
            TEMP=str(self._home / "tmp"), TMP=str(self._home / "tmp"), TMPDIR=str(self._home / "tmp"),
            CODEX_HOME=str(self._home / ".codex"),
        )
        return env

    def start(self, mode: str, hermes_home: Path, name: str) -> _Worker:
        out = self._dir / f"{name}.json"
        err = self._dir / f"{name}.err"
        with err.open("w") as err_file:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(_WORKER), mode, str(self._port), str(out), name],
                env=self._env(hermes_home), cwd=str(hermes_home), stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=err_file, text=True, creationflags=_NO_WINDOW,
            )
        self.procs.append(proc)
        worker = _Worker(proc, out, err, name)
        wait_until(
            lambda: Path(str(out) + ".ready").exists() or proc.poll() is not None,
            timeout=_STEP_TIMEOUT, what=f"{name} to load Hermes",
        )
        return worker

    def kill_all(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=30)
            if proc.stdin:
                proc.stdin.close()


@dataclass
class _Worker:
    proc: subprocess.Popen
    out: Path
    err: Path
    name: str

    def go(self) -> None:
        self.proc.stdin.write("go\n")
        self.proc.stdin.flush()

    def result(self) -> dict:
        try:
            self.proc.wait(timeout=_STEP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            pytest.fail(f"{self.name} did not finish within {_STEP_TIMEOUT}s")
        if not self.out.exists():
            pytest.fail(f"{self.name} left no result: {self.err.read_text(errors='replace')[-1500:]}")
        return json.loads(self.out.read_text(encoding="utf-8"))


@pytest.fixture
def fleet(tmp_path, endpoint):
    workers = _Fleet(tmp_path, endpoint.port)
    try:
        yield workers
    finally:
        workers.kill_all()


def _stored_tokens(root: Path, key: str) -> list[str]:
    """Every ``key`` token the root login store holds, singleton state and pool rows alike."""
    store = json.loads((root / "auth.json").read_text(encoding="utf-8"))
    found = [(state.get("tokens") or {}).get(key) for state in (store.get("providers") or {}).values()]
    for rows in (store.get("credential_pool") or {}).values():
        found.extend(row.get(key) for row in rows)
    return [token for token in found if token]


def _assert_rotated_exactly_and_nobody_signed_out(endpoint, root: _Root, results: list[dict]) -> None:
    """The root grant rotated once, and every process ended on the current, live credential."""
    initial = [r for r in endpoint.refreshes if r["refresh_token"] == root.initial_refresh_token]
    replays = [r for r in endpoint.refreshes if r["reused"]]
    assert len(initial) == 1 and initial[0]["valid"], f"root refresh token spent {len(initial)} times: {endpoint.refreshes}"
    assert not replays, f"a spent refresh token was replayed: {replays}"
    # Exactly one rotation in every scenario: the loser of a race, or a process holding a stale
    # copy, adopts the winner's tokens instead of refreshing again. (The xAI resolver would refresh
    # again inside its one-hour expiry skew, so the endpoint issues 24-hour access tokens.)
    assert len(endpoint.refreshes) == 1, f"expected exactly one rotation: {endpoint.refreshes}"
    winner = initial[0]
    stored_refresh = _stored_tokens(root.path, "refresh_token")
    stored_access = _stored_tokens(root.path, "access_token")
    assert stored_refresh, "the root store lost its refresh token"
    # The root ends on the one token pair the winning rotation issued; nothing stale is kept.
    assert set(stored_refresh) == {winner["rotated"]} == endpoint.live, (stored_refresh, endpoint.live)
    assert set(stored_access) == {winner["access_token"]}, "the root holds an access token the winner did not issue"
    for result in results:
        assert result["ok"], f"{result['name']} failed: {result.get('error')}\n{result.get('trace')}"
        assert result["selected"], f"{result['name']} ended with no credential: {result}"
        assert result.get("status") != "exhausted", f"{result['name']} marked its credential exhausted"
        # Every process returns what the one winning rotation issued, not the token it started with.
        assert result["access_token"] == winner["access_token"], f"{result['name']} returned a stale access token"
        if "refresh_token" in result:
            assert result["refresh_token"] == winner["rotated"], f"{result['name']} holds a stale refresh token"
    # A failed borrowed refresh quarantines the profile with its own signed-out auth.json.
    for profile in root.profiles:
        assert not (profile / "auth.json").exists(), f"{profile.name} was signed out into its own store"
    assert "relogin_required" not in (root.path / "auth.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
def test_pool_refresh_by_two_profiles_spends_the_root_token_once(provider, shared_root, fleet, endpoint):
    root = shared_root(provider)
    workers = [fleet.start(f"pool-{provider}", home, f"{provider}-{home.name}") for home in root.profiles]

    for worker in workers:
        worker.go()
    results = [worker.result() for worker in workers]

    _assert_rotated_exactly_and_nobody_signed_out(endpoint, root, results)


def test_xai_singleton_resolve_by_two_profiles_spends_the_root_token_once(shared_root, fleet, endpoint):
    root = shared_root("xai-oauth")
    workers = [fleet.start("single-xai", home, f"xai-singleton-{home.name}") for home in root.profiles]

    for worker in workers:
        worker.go()
    results = [worker.result() for worker in workers]

    _assert_rotated_exactly_and_nobody_signed_out(endpoint, root, results)


def test_codex_pool_force_refreshed_after_a_peer_rotated_does_not_replay_the_spent_token(
    shared_root, fleet, endpoint
):
    # A usable grant: each process loads the pool, serves a request, and gets a 401 for it.
    root = shared_root("openai-codex", access_valid_for=86400, pool_row=True)
    first, second = (fleet.start("pool-openai-codex-401", home, f"codex-stale-{home.name}") for home in root.profiles)

    first.go()
    first_result = first.result()  # rotates the root token and writes it back
    assert first_result["ok"], first_result
    second.go()  # its in-memory pool still holds the token the first process spent
    second_result = second.result()

    _assert_rotated_exactly_and_nobody_signed_out(endpoint, root, [first_result, second_result])


def test_anthropic_pool_refresh_by_two_profiles_spends_the_root_token_once(shared_root, fleet, endpoint):
    # A hermes_pkce login added to the shared root's pool (JSON refresh body, urllib transport).
    root = shared_root("anthropic", pool_row=True, row_source="manual:hermes_pkce")
    workers = [fleet.start("pool-anthropic", home, f"anthropic-{home.name}") for home in root.profiles]

    for worker in workers:
        worker.go()
    results = [worker.result() for worker in workers]

    _assert_rotated_exactly_and_nobody_signed_out(endpoint, root, results)


def test_anthropic_singleton_backed_pool_row_refresh_spends_the_root_token_once(shared_root, fleet, endpoint):
    # Only the legacy singleton and the pool row seeded from it (source exactly hermes_pkce) exist.
    root = shared_root("anthropic", pool_row=True, row_source="hermes_pkce", singleton=True)
    workers = [fleet.start("pool-anthropic", home, f"anthropic-singleton-{home.name}") for home in root.profiles]

    for worker in workers:
        worker.go()
    results = [worker.result() for worker in workers]

    _assert_rotated_exactly_and_nobody_signed_out(endpoint, root, results)
    winner = next(r for r in endpoint.refreshes if r["refresh_token"] == root.initial_refresh_token)
    singleton = json.loads((root.path / ".anthropic_oauth.json").read_text(encoding="utf-8"))
    assert (singleton["accessToken"], singleton["refreshToken"]) == (winner["access_token"], winner["rotated"])
    assert not any((profile / ".anthropic_oauth.json").exists() for profile in root.profiles)
