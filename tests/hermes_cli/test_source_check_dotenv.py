"""Regression test: the update probe must authenticate even without an inherited dotenv.

The Desktop app's "check for updates" button does NOT go through the Python
backend's ``/api/hermes/update/check`` route. It goes:

    renderer -> window.hermesDesktop.updates.check()
      -> Electron main: checkUpdates() -> updater/checkout.ts
      -> readSourceUpdate() (updater/checkout-source.ts)
      -> spawn: python -c 'runpy.run_path("hermes_cli/source_check.py").get("main")()'
         --install-root <root> --home <home> --git <git> --channel stable --force

with ``env = sourceUpdateEnvironment(...)`` = the Electron process environment plus
``buildDesktopBackendEnv()``.

Two things make that child anonymous:

1. The Electron main process never has ``GITHUB_TOKEN`` in its environment — only
   the Python backend reads the home dotenv, and the Electron side never does.
2. ``runpy.run_path("hermes_cli/source_check.py")`` runs the probe WITHOUT importing
   ``hermes_cli.main``, so the module-level ``load_hermes_dotenv()`` never executes.

The probe therefore reaches api.github.com with no token and spends the anonymous
60 requests/hour **per exit IP** bucket — which, behind a shared egress (office NAT,
a Clash/VPN node), every other client behind the same address also spends. The check
then fails with an anonymous-quota 403 instead of reporting "no update available".

This test drives the REAL probe entrypoint the way the Desktop spawns it — a fresh
subprocess, ``runpy.run_path``, and an environment with no token — against a LOCAL
api.github.com stand-in, and asserts the request carried the home dotenv's token.

A plain ``check_for_updates()`` call from a shell loads the dotenv through
``hermes_cli.main`` and therefore never reproduces the bug; that is why the
subprocess is required.

The stand-in is injected with a ``sitecustomize`` on the child's ``PYTHONPATH``,
which redirects ``urllib.request.urlopen`` for api.github.com only — the same
technique ``test_source_check.py`` uses in-process, applied across the process
boundary the Desktop actually has.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The marker stands in for a real token: the assertion is about the probe finding
# the dotenv, never about a secret's value.
_DOTENV_MARKER = "hermes-desktop-probe-dotenv-marker"

# Injected into the child's interpreter. It redirects api.github.com at the local
# stand-in and records the Authorization header of every redirected request.
_SITECUSTOMIZE = '''
import json, os, urllib.request
from urllib.parse import urlsplit

_BASE = os.environ["HERMES_TEST_GITHUB_API_BASE"]
_RECORD = os.environ["HERMES_TEST_GITHUB_API_RECORD"]
_original = urllib.request.urlopen


def _redirected(request, *args, **kwargs):
    url = urlsplit(request.full_url)
    if url.hostname != "api.github.com":
        return _original(request, *args, **kwargs)
    with open(_RECORD, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "path": url.path,
            "authorization": request.headers.get("Authorization")
            if hasattr(request, "headers") else request.get_header("Authorization"),
        }) + "\\n")
    rewritten = urllib.request.Request(
        _BASE + url.path + (("?" + url.query) if url.query else ""),
        headers=dict(request.header_items()),
    )
    return _original(rewritten, *args, **kwargs)


urllib.request.urlopen = _redirected
'''


def _spawn_probe(env_home: Path, api_base: str, record: Path, git_bin: str) -> dict:
    """Run the probe exactly as ``updater/checkout-source.ts`` spawns it.

    Same entrypoint (``runpy.run_path`` on ``hermes_cli/source_check.py``, so
    ``hermes_cli.main`` is never imported), same "no token in the environment"
    starting state, with api.github.com redirected at a local stand-in.

    Returns the probe's parsed status.
    """
    probe = (
        "from pathlib import Path; import runpy; p = Path('hermes_cli/source_check.py'); "
        "entry = runpy.run_path(str(p)).get('main') if p.is_file() else None; "
        "entry() if callable(entry) else print('null')"
    )
    # The sitecustomize that redirects api.github.com lives beside the probe's
    # sys.path entry, so the interpreter imports it at startup.
    shim = env_home / "_shim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")

    child_env = dict(os.environ)
    child_env["HERMES_HOME"] = str(env_home)
    # The Electron process carries no token: strip it so the probe can only find
    # one by reading the home dotenv itself.
    child_env.pop("GITHUB_TOKEN", None)
    child_env.pop("GH_TOKEN", None)
    child_env["PYTHONPATH"] = os.pathsep.join([str(shim), str(ROOT)])
    child_env["HERMES_TEST_GITHUB_API_BASE"] = api_base
    child_env["HERMES_TEST_GITHUB_API_RECORD"] = str(record)
    # Git probes must never reach the external network.
    child_env["GIT_ALLOW_PROTOCOL"] = "file"

    completed = subprocess.run(
        [sys.executable, "-c", probe,
         "--install-root", str(ROOT),
         "--home", str(env_home),
         "--git", git_bin,
         "--channel", "stable",
         "--force"],
        capture_output=True,
        text=True,
        env=child_env,
        cwd=str(ROOT),
        timeout=300,
        check=False,
    )
    out = (completed.stdout or "").strip()
    if not out:
        raise AssertionError(
            "the probe produced no output.\n"
            f"exit={completed.returncode}\n"
            f"stdout tail: {completed.stdout[-400:]}\n"
            f"stderr tail: {completed.stderr[-800:]}"
        )
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"the probe did not emit JSON: {out[:300]!r}\n"
            f"stderr tail: {completed.stderr[-800:]}"
        ) from exc


class _ApiHandler(BaseHTTPRequestHandler):
    """A stand-in for api.github.com: answers every GET with a release payload."""

    def do_GET(self):
        body = json.dumps({
            "tag_name": "v9.9.9",
            "draft": False,
            "prerelease": False,
            "sha": "0" * 40,
            "ahead_by": 1,
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


def _start_api() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _read_record(record: Path) -> list[dict]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_probe_authenticates_without_inherited_dotenv(tmp_path):
    """A probe spawned with no token in its environment must still authenticate."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / ".env").write_text(f"GITHUB_TOKEN={_DOTENV_MARKER}\n", encoding="utf-8")

    record = tmp_path / "requests.jsonl"
    server, api_base = _start_api()
    try:
        _spawn_probe(home, api_base, record, "git")
    finally:
        server.shutdown()
        server.server_close()

    requests = _read_record(record)
    assert requests, "the probe made no api.github.com request"
    assert requests[0]["authorization"] == f"Bearer {_DOTENV_MARKER}", (
        "the probe sent no Authorization header (or an anonymous one) even though the "
        "home dotenv carries a token. Without it the request spends the anonymous "
        "60/hour per-IP bucket, which every client behind a shared exit also spends."
    )


def test_probe_without_any_token_stays_anonymous(tmp_path):
    """No token anywhere must not turn the update check into a hard failure.

    The probe degrades to the anonymous quota — exactly the pre-fix behaviour —
    but it still returns a status the caller can act on rather than raising.
    """
    home = tmp_path / "hermes-home-empty"
    home.mkdir()
    (home / ".env").write_text("# no credentials here\n", encoding="utf-8")

    record = tmp_path / "requests.jsonl"
    server, api_base = _start_api()
    try:
        status = _spawn_probe(home, api_base, record, "git")
    finally:
        server.shutdown()
        server.server_close()

    assert isinstance(status, dict), "the probe must return a status object"
    assert "error" in status, "the probe's status must carry an error field when it fails"
    for request in _read_record(record):
        assert not request["authorization"], (
            "a request carried an Authorization header although the dotenv defines no token"
        )
