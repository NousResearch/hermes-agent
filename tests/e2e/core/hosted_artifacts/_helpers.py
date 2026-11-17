"""Harness for the hosted artifacts end-to-end tests.

A real ``hermes chat -q`` turn runs against a temp ``HERMES_HOME`` that is signed in to Nous
(``auth.json`` ``providers.nous`` with a long-lived unsigned JWT-shaped access token whose
``portal_base_url`` is a loopback fake Portal), with ``artifacts.base_url`` pointing at the hosted
artifacts MCP server: by default the in-repo fake (``tests/fakes/hosted_artifacts_server.py``, run as
a subprocess), or a real plane named by ``HERMES_ARTIFACTS_E2E_*`` (see README.md). The model is the
repo's scripted fake LLM provider, so assertions read what the model was offered and saw, and what
the artifacts server received.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from tests.e2e.core.mcp_plugins._helpers import REPO_ROOT, E2EHome, build_home, wait_until
from tests.fakes.nous_portal_account import FakePortal
from tests.fakes.providers.catalog_oauth import make_jwt

FAKE_SERVER = REPO_ROOT / "tests" / "fakes" / "hosted_artifacts_server.py"
SERVER = "hermes-artifacts"
TOOLS = ("artifact_create", "artifact_update", "artifact_get", "artifact_list", "artifact_share", "artifact_show")

# Test-only knobs for an external harness pointing this suite at a real artifacts plane. The token
# may come as a file path (``_TOKEN_FILE``), which is what scripts/run_tests.sh can forward: it never
# forwards a credential-valued variable.
ENV_BASE_URL = "HERMES_ARTIFACTS_E2E_BASE_URL"
ENV_PORTAL_URL = "HERMES_ARTIFACTS_E2E_PORTAL_URL"
ENV_TOKEN = "HERMES_ARTIFACTS_E2E_TOKEN"
ENV_TOKEN_FILE = "HERMES_ARTIFACTS_E2E_TOKEN_FILE"

_PASSTHROUGH_ENV = frozenset({"PATH", "LANG", "LANGUAGE", "USER", "LOGNAME", "SHELL", "TMPDIR", "TZ"})


def tool(name: str) -> str:
    return f"mcp__hermes_artifacts__{name}"


@dataclass(frozen=True)
class RealPlane:
    base_url: str
    portal_url: str
    token: str


def real_plane() -> Optional[RealPlane]:
    """The real plane named by the ``HERMES_ARTIFACTS_E2E_*`` variables, or None unless the base URL,
    the Portal URL and a token (``_TOKEN``, else the contents of ``_TOKEN_FILE``) are all set."""
    base_url = os.environ.get(ENV_BASE_URL, "").strip()
    portal_url = os.environ.get(ENV_PORTAL_URL, "").strip()
    token = os.environ.get(ENV_TOKEN, "").strip()
    token_file = os.environ.get(ENV_TOKEN_FILE, "").strip()
    if not token and token_file:
        token = Path(token_file).read_text(encoding="utf-8-sig").strip()
    if not (base_url and portal_url and token):
        return None
    return RealPlane(base_url.rstrip("/"), portal_url.rstrip("/"), token)


def _iso(delta_s: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=delta_s)).isoformat()


def _jwt_exp(token: str) -> Optional[float]:
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        exp = claims.get("exp")
        return float(exp) if isinstance(exp, (int, float)) else None
    except (IndexError, ValueError, AttributeError):
        return None


def nous_token(ttl_s: int = 6 * 3600) -> str:
    """A long-lived unsigned JWT-shaped Nous access token (the client only decodes its claims)."""
    return make_jwt("hosted-artifacts-e2e", ttl_s=ttl_s)


def seed_nous_login(eh: E2EHome, portal_url: str, token: str) -> None:
    """Sign the temp profile in to Nous: ``providers.nous`` with *token*, valid until its ``exp``.

    The active model provider stays the fake LLM (``config.yaml``); this login only feeds the
    account read and the artifacts bearer."""
    exp = _jwt_exp(token)
    ttl = (exp - time.time()) if exp else 6 * 3600
    state = {"access_token": token, "refresh_token": "rt-hosted-artifacts-e2e-unused", "token_type": "Bearer",
             "scope": "inference:invoke", "client_id": "hermes-cli", "portal_base_url": portal_url,
             "inference_base_url": f"{portal_url}/v1", "obtained_at": _iso(-60), "expires_at": _iso(ttl),
             "tls": {"insecure": False, "ca_bundle": None}}
    (eh.hermes_home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {"nous": state}}, indent=2), encoding="utf-8")


def closed_port_url() -> str:
    """``http://127.0.0.1:<port>`` for a port nothing listens on (bound, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def refuses(url: str) -> bool:
    """True when nothing accepts a TCP connection at *url*'s host:port."""
    host, port = url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return False
    except OSError:
        return True


class FakeArtifactsServer:
    """``tests/fakes/hosted_artifacts_server.py`` as a subprocess accepting only *token*, logging
    every POSTed JSON-RPC body with its bearer."""

    def __init__(self, root: Path, token: str, portal_url: str, tag: str) -> None:
        self.root = root
        self.token = token
        self.portal_url = portal_url
        self.tag = tag
        self.log = root / "artifacts_inbound.jsonl"
        self.port_file = root / "artifacts_port"
        self.stderr_path = root / "artifacts_server.stderr"
        self.port = 0
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> FakeArtifactsServer:
        env = {k: v for k, v in os.environ.items() if k in _PASSTHROUGH_ENV}
        env.update({"PYTHONPATH": str(REPO_ROOT), "ARTIFACTS_FAKE_PORT_FILE": str(self.port_file),
                    "ARTIFACTS_FAKE_TOKEN": self.token, "ARTIFACTS_FAKE_LOG": str(self.log),
                    "ARTIFACTS_FAKE_PORTAL": self.portal_url, "PARITY_TREE_TAG": self.tag})
        with open(self.stderr_path, "ab") as err:
            self.proc = subprocess.Popen([sys.executable, str(FAKE_SERVER)], env=env, cwd=str(REPO_ROOT),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
        text = wait_until(lambda: (self.port_file.exists() and self.port_file.read_text(encoding="utf-8-sig").strip())
                          or (self.proc is not None and self.proc.poll() is not None and "exited"), 60,
                          f"fake artifacts server bind ({self.stderr_path})")
        assert text != "exited", self.stderr_path.read_text(encoding="utf-8-sig", errors="replace")
        self.port = int(text.split()[0])
        return self

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)

    def entries(self) -> list[dict[str, Any]]:
        """Each logged POST as ``{"bearer", "msg"}`` (``msg`` the parsed JSON-RPC body)."""
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                raw = json.loads(line)
                out.append({"bearer": raw["bearer"], "msg": json.loads(raw["body"]) if raw["body"] else None})
        return out

    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [(e["msg"]["params"]["name"], e["msg"]["params"].get("arguments") or {})
                for e in self.entries() if isinstance(e["msg"], dict) and e["msg"].get("method") == "tools/call"]


def build_artifacts_home(root: Path, llm_base_url: str) -> E2EHome:
    """A temp home routing the model to the fake LLM; :func:`configure` signs it in and points it at a plane."""
    eh = build_home(root, llm_base_url)
    eh.extra_env["HERMES_SHARED_AUTH_DIR"] = str(eh.home / "shared")
    return eh


def configure(eh: E2EHome, *, base_url: str, portal_url: str, token: str, **artifacts: Any) -> None:
    """Sign *eh* in to Nous at *portal_url* and set ``artifacts`` (``base_url`` plus *artifacts*)."""
    eh.update_config(lambda c: c.__setitem__("artifacts", {"base_url": base_url, **artifacts}))
    seed_nous_login(eh, portal_url, token)


@contextlib.contextmanager
def fake_portal(body: Optional[dict[str, Any]] = None) -> Iterator[FakePortal]:
    with FakePortal(body=body) as portal:
        yield portal


def offered(body: dict[str, Any]) -> str:
    """Everything a provider request offers the model as tools (names, descriptions, and any Tool
    Search listing), as one string."""
    return json.dumps(body.get("tools") or [])
