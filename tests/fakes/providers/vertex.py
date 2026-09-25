"""Loopback fake of Google Vertex AI (Gemini behind the OpenAI-compatible endpoint) + Google OAuth.

Two external boundaries Hermes does not own, both faked for real:

* **Google OAuth2 token endpoint** (``POST /token`` on plain loopback HTTP). A generated
  service-account JSON names it as ``token_uri``, so the REAL ``google-auth`` library signs an
  RS256 JWT assertion with the SA private key and exchanges it. The fake verifies the signature
  against the SA public key and the claims Google checks (``iss``/``aud``/``scope``/``iat``/``exp``)
  and mints ``ya29.``-style access tokens with a scripted ``expires_in``.
* **Vertex AI** at ``https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/
  {region}/endpoints/openapi/chat/completions``. Hermes has no base-URL override for Vertex, so the
  fake is an HTTPS ``CONNECT`` proxy that terminates TLS with a leaf cert signed by a generated CA:
  the child trusts it through the standard ``SSL_CERT_FILE`` and reaches it through the standard
  ``HTTPS_PROXY`` (the corporate-proxy channel Hermes documents). CONNECTs to any other host are
  refused (recorded), so nothing can leak to the real network.

Every Vertex request is recorded (host, path, headers, body) and validated against the published
OpenAI-compatibility contract before a scripted response is chosen; a malformed request gets the
error Vertex returns (``[{"error": {"code", "message", "status"}}]``) and is marked ``rejected``.
Checks: URL scheme (project/location path segments), a live minted bearer, the ``google/<model>``
publisher form, tool-call/tool-result pairing, JSON-object function arguments, function-name rules,
and Gemini 3 thought signatures (``extra_content.google.thought_signature``) on the first function
call of every step in the current turn, byte-identical to one the fake issued.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import re
import secrets
import socketserver
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
# Google's token endpoint requires this audience whatever host the SA file's token_uri names.
GOOGLE_TOKEN_AUDIENCE = "https://oauth2.googleapis.com/token"
UNAUTHENTICATED_MESSAGE = (
    "Request had invalid authentication credentials. Expected OAuth 2 access token, login cookie or "
    "other valid authentication credential. See https://developers.google.com/identity/sign-in/web/devconsole-project.")
MISSING_SIGNATURE_MESSAGE = (
    "Function call is missing a thought_signature in functionCall parts. This is required for tools to work "
    "correctly, and missing thought_signature may lead to degraded model performance. Additional data, function "
    "call `default_api:{name}` , position {pos}. Please refer to https://ai.google.dev/gemini-api/docs/"
    "thought-signatures for more details.")
# HTTP status -> google.rpc canonical status name (https://cloud.google.com/apis/design/errors).
GRPC_STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
               429: "RESOURCE_EXHAUSTED", 500: "INTERNAL", 503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED"}
_MODEL_RE = re.compile(r"^google/gemini-[0-9a-z.\-]+$")
# FunctionDeclaration.name: letter/underscore first, then [a-zA-Z0-9_.:-], at most 64 chars.
_FUNC_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:\-]{0,63}$")
_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


# Scripted responses --------------------------------------------------------------------------------


@dataclass
class Say:
    """A final assistant answer streamed in chunks."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int = 20
    chunk_chars: int = 16
    expire_tokens_after: bool = False  # the ~1h boundary passes right after this response


@dataclass
class Call:
    """One assistant step issuing function calls; Gemini 3 signs the first call of each step."""

    calls: list[tuple[str, dict[str, Any]]]
    text: str | None = None
    signed: bool = True
    prompt_tokens: int | None = None
    expire_tokens_after: bool = False


@dataclass
class Fail:
    """A Vertex error response; ``list_body`` is the openapi endpoint's list-wrapped envelope."""

    status: int
    message: str
    list_body: bool = True
    retry_after: float | None = None


@dataclass
class Drop:
    """Open the SSE stream, send ``text[:after_chars]``, then reset the TLS connection mid-body."""

    text: str
    after_chars: int = 10


Response = Union[Say, Call, Fail, Drop]
Responder = Callable[[dict[str, Any]], Response]


# Certificates and service account --------------------------------------------------------------------


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _write_pem(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def make_tls_material(root: Path, hosts: list[str]) -> tuple[Path, Path, Path]:
    """A throwaway CA (PEM for ``SSL_CERT_FILE``) and a leaf for ``hosts`` signed by it."""
    root.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
    ca_cert = (
        x509.CertificateBuilder().subject_name(_name("hermes-e2e fake Google CA")).issuer_name(_name("hermes-e2e fake Google CA"))
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=5)).not_valid_after(now + _dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                     key_encipherment=False, data_encipherment=False, key_agreement=False,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(ca_ski, critical=False)
        .sign(ca_key, hashes.SHA256()))
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_cert = (
        x509.CertificateBuilder().subject_name(_name(hosts[0])).issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=5)).not_valid_after(now + _dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(h) for h in hosts]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski), critical=False)
        .sign(ca_key, hashes.SHA256()))
    ca_pem = _write_pem(root / "fake-google-ca.pem", ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_pem = _write_pem(root / "leaf.pem", leaf_cert.public_bytes(serialization.Encoding.PEM))
    key_pem = _write_pem(root / "leaf.key", leaf_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return ca_pem, cert_pem, key_pem


@dataclass
class ServiceAccount:
    path: Path
    client_email: str
    project_id: str
    private_key_id: str
    public_key: rsa.RSAPublicKey


def make_service_account(path: Path, token_uri: str, project_id: str) -> ServiceAccount:
    """A service-account key file shaped like the one the Cloud console downloads."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    email = f"hermes-e2e@{project_id}.iam.gserviceaccount.com"
    kid = secrets.token_hex(20)
    info = {
        "type": "service_account", "project_id": project_id, "private_key_id": kid,
        "private_key": key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()).decode(),
        "client_email": email, "client_id": str(secrets.randbelow(10**20)),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": token_uri,
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "universe_domain": "googleapis.com",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return ServiceAccount(path, email, project_id, kid, key.public_key())


def _b64url_decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def verify_jwt_assertion(assertion: str, sa: ServiceAccount) -> tuple[dict[str, Any] | None, str]:
    """RS256 signature + the claims Google's token endpoint enforces; (claims, "") or (None, reason)."""
    try:
        head_b64, body_b64, sig_b64 = assertion.split(".")
        header, claims = json.loads(_b64url_decode(head_b64)), json.loads(_b64url_decode(body_b64))
        sa.public_key.verify(_b64url_decode(sig_b64), f"{head_b64}.{body_b64}".encode(), padding.PKCS1v15(), hashes.SHA256())
    except Exception as exc:  # noqa: BLE001 - any parse/verify failure is Google's "Invalid JWT Signature."
        return None, f"Invalid JWT Signature. ({type(exc).__name__})"
    now = time.time()
    problems = {
        "alg": header.get("alg") != "RS256",
        "kid": header.get("kid") not in (None, sa.private_key_id),
        "iss": claims.get("iss") != sa.client_email,
        "aud": claims.get("aud") != GOOGLE_TOKEN_AUDIENCE,
        "scope": CLOUD_PLATFORM_SCOPE not in str(claims.get("scope", "")).split(),
        "iat": not isinstance(claims.get("iat"), int) or abs(claims["iat"] - now) > 300,
        "exp": not isinstance(claims.get("exp"), int) or not 0 < claims["exp"] - claims.get("iat", 0) <= 3600,
    }
    bad = sorted(k for k, v in problems.items() if v)
    return (None, f"Invalid JWT: bad claims {bad}") if bad else (claims, "")


# Request validation ----------------------------------------------------------------------------------


def _first_signature(tool_call: dict[str, Any]) -> Any:
    extra = tool_call.get("extra_content")
    google = extra.get("google") if isinstance(extra, dict) else None
    return google.get("thought_signature") if isinstance(google, dict) else None


def _check_tool_call_shape(tc: Any) -> str | None:
    if not isinstance(tc, dict) or tc.get("type", "function") != "function" or not isinstance(tc.get("function"), dict):
        return "Invalid value at 'messages[].tool_calls[]': expected a function tool call"
    fn = tc["function"]
    if not isinstance(tc.get("id"), str) or not tc["id"]:
        return "tool_calls[].id must be a non-empty string"
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (TypeError, json.JSONDecodeError):
        return f"Invalid JSON payload in function call arguments for `{fn.get('name')}`"
    return None if isinstance(args, dict) else "function call arguments must be a JSON object (google.protobuf.Struct)"


def _check_pairing(messages: list[dict[str, Any]]) -> str | None:
    """Every function-call turn is answered by exactly its function responses, immediately after."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "tool":
            return ("Please ensure that function response turn comes immediately after a function call turn. "
                    f"(orphan tool message at index {i}, tool_call_id={msg.get('tool_call_id')!r})")
        calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if not calls:
            i += 1
            continue
        expected = [tc.get("id") for tc in calls]
        j = i + 1
        answered: list[Any] = []
        while j < len(messages) and messages[j].get("role") == "tool":
            answered.append(messages[j].get("tool_call_id"))
            j += 1
        if sorted(map(str, answered)) != sorted(map(str, expected)):
            return ("Please ensure that the number of function response parts is equal to the number of function "
                    f"call parts of the function call turn. (calls {expected} at index {i}, responses {answered})")
        i = j
    return None


def _check_signatures(messages: list[dict[str, Any]], issued: set[str]) -> str | None:
    """Gemini 3: the first call of every step in the current turn carries a signature we issued;
    any replayed signature (current or historical) must be byte-identical to one we issued."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    step = 0
    for i, msg in enumerate(messages):
        calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if not calls:
            continue
        for pos, tc in enumerate(calls):
            sig = _first_signature(tc)
            if sig is not None and sig not in issued:
                return "Corrupted thought signature."
            if i > last_user and pos == 0 and sig is None:
                return MISSING_SIGNATURE_MESSAGE.format(name=tc.get("function", {}).get("name"), pos=step + 1)
        step += 1
    return None


def _merged_anyof_error(node: Any, where: str) -> str | None:
    """Google's OpenAI->FunctionDeclaration translator merges ``anyOf`` branches into one node; a
    merged node carrying ``items`` under a non-array type is rejected (#109115 evidence)."""
    if isinstance(node, dict):
        branches = node.get("anyOf")
        if isinstance(branches, list) and len(branches) > 1:
            types = {b.get("type") for b in branches if isinstance(b, dict)}
            if len(types) > 1 and any(isinstance(b, dict) and "items" in b for b in branches):
                return (f"functionDeclaration `{where}` schema specified incorrect schema type field. "
                        "For schema with items, schema type should be ARRAY.")
        for key, child in node.items():
            found = _merged_anyof_error(child, f"{where}.{key}")
            if found:
                return found
    elif isinstance(node, list):
        for child in node:
            found = _merged_anyof_error(child, where)
            if found:
                return found
    return None


def _check_tools(tools: Any) -> str | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        return "Invalid value at 'tools': expected a list"
    seen: set[str] = set()
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else None
        if not isinstance(fn, dict):
            return "Invalid value at 'tools[]': only function tools are supported"
        name = str(fn.get("name", ""))
        if not _FUNC_NAME_RE.match(name):
            return f"Invalid function name `{name}`: must start with a letter or underscore, [a-zA-Z0-9_.:-], max 64"
        if name in seen:
            return f"Duplicate function declaration found: {name}"
        seen.add(name)
        params = fn.get("parameters")
        if params is not None and (not isinstance(params, dict) or params.get("type", "object") != "object"):
            return f"functionDeclaration `{name}` parameters must be an OBJECT schema"
        found = _merged_anyof_error(params, f"{name}.parameters")
        if found:
            return f"Unable to submit request because `{name}` {found}"
    return None


def validate_chat_body(body: dict[str, Any], issued_signatures: set[str]) -> str | None:
    """The first contract violation Vertex would 400 on, or None."""
    if not _MODEL_RE.match(str(body.get("model", ""))):
        return f"Invalid model name {body.get('model')!r}: the OpenAI-compatible endpoint expects 'google/<model>'"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "* GenerateContentRequest.contents: contents is not specified"
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") not in _ROLES:
            return f"Invalid value at 'messages[{idx}].role'"
        if msg.get("role") == "tool" and not msg.get("tool_call_id"):
            return f"messages[{idx}]: a tool message requires tool_call_id"
        for tc in msg.get("tool_calls") or []:
            bad = _check_tool_call_shape(tc)
            if bad:
                return f"messages[{idx}]: {bad}"
    return _check_tools(body.get("tools")) or _check_pairing(messages) or _check_signatures(messages, issued_signatures)


# Server ----------------------------------------------------------------------------------------------


@dataclass
class _Token:
    value: str
    expires_at: float


@dataclass
class TokenPolicy:
    expires_in: int = 3600
    error: tuple[int, str, str] | None = None  # (status, error, error_description) for every exchange
    reject_bearers: bool = False  # Vertex refuses every bearer (disabled SA / revoked grant)


class FakeVertex:
    """OAuth token endpoint + TLS-terminating CONNECT proxy serving Vertex. Context manager."""

    def __init__(self, root: Path, *, project: str, region: str, sa_project: str | None = None,
                 script: list[Response] | Responder | None = None, aux: Responder | None = None,
                 default_text: str = "ok", prompt_tokens_fn: Callable[[dict[str, Any]], int] | None = None) -> None:
        self.root, self.project, self.region = root, project, region
        self.host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        self._script: list[Response] = list(script) if isinstance(script, list) else []
        self._responder = script if callable(script) else None
        self._aux = aux or (lambda _rec: Say("Summary of the earlier conversation (fake)."))
        self.default_text = default_text
        self.prompt_tokens_fn = prompt_tokens_fn
        self.token_policy = TokenPolicy()
        self.requests: list[dict[str, Any]] = []
        self.token_requests: list[dict[str, Any]] = []
        self.connects: list[dict[str, Any]] = []
        self.issued_signatures: list[str] = []
        self.signature_by_call: dict[str, str | None] = {}
        self._tokens: dict[str, _Token] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._server: ThreadingHTTPServer | None = None
        self._sa_project = sa_project or project
        self.sa: ServiceAccount | None = None
        self.ca_pem: Path | None = None
        self._tls: ssl.SSLContext | None = None

    # lifecycle
    def __enter__(self) -> "FakeVertex":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        server.daemon_threads = True
        self._server = server
        self.ca_pem, cert, key = make_tls_material(self.root / "tls", [self.host])
        self._tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self._tls.load_cert_chain(cert, key)
        self.sa = make_service_account(self.root / "sa.json", self.token_uri, self._sa_project)
        threading.Thread(target=server.serve_forever, name="fake-vertex", daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    @property
    def token_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}/token"

    @property
    def chat_path(self) -> str:
        return f"/v1beta1/projects/{self.project}/locations/{self.region}/endpoints/openapi/chat/completions"

    def child_env(self) -> dict[str, str]:
        """Standard proxy + CA-trust env for the Hermes child (no Hermes-specific knobs)."""
        return {"HTTPS_PROXY": f"http://127.0.0.1:{self.port}", "NO_PROXY": "127.0.0.1,localhost",
                "SSL_CERT_FILE": str(self.ca_pem)}

    # scripting
    def push(self, *responses: Response) -> None:
        with self._lock:
            self._script.extend(responses)

    def _next_main(self, record: dict[str, Any]) -> Response:
        if self._responder is not None:
            return self._responder(record)
        with self._lock:
            return self._script.pop(0) if self._script else Say(self.default_text)

    def expire_all_tokens(self) -> None:
        with self._lock:
            for tok in self._tokens.values():
                tok.expires_at = 0.0

    # inspection
    def main_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["kind"] == "main"]

    def aux_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["kind"] == "aux"]

    def rejected(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r.get("rejected")]

    def minted_tokens(self) -> list[str]:
        return [t["access_token"] for t in self.token_requests if t.get("access_token")]

    # token endpoint
    def _mint(self, form: dict[str, str]) -> tuple[int, dict[str, Any]]:
        rec: dict[str, Any] = {"grant_type": form.get("grant_type"), "t": time.time()}
        with self._lock:
            self.token_requests.append(rec)
        policy = self.token_policy
        if form.get("grant_type") != JWT_BEARER_GRANT:
            rec["error"] = "unsupported_grant_type"
            return 400, {"error": "unsupported_grant_type", "error_description": "Invalid grant_type"}
        assert self.sa is not None
        claims, why = verify_jwt_assertion(form.get("assertion", ""), self.sa)
        rec["claims"] = claims
        if claims is None:
            rec["error"] = why
            return 400, {"error": "invalid_grant", "error_description": why}
        if policy.error:
            rec["error"] = policy.error[1]
            return policy.error[0], {"error": policy.error[1], "error_description": policy.error[2]}
        value = f"ya29.fake-{secrets.token_urlsafe(24)}"
        with self._lock:
            self._tokens[value] = _Token(value, time.time() + policy.expires_in)
        rec["access_token"] = value
        return 200, {"access_token": value, "expires_in": policy.expires_in, "token_type": "Bearer"}

    def _auth_ok(self, header: str) -> bool:
        if self.token_policy.reject_bearers:
            return False
        scheme, _, value = header.partition(" ")
        tok = self._tokens.get(value) if scheme == "Bearer" else None
        return bool(tok and tok.expires_at > time.time())

    def next_ids(self) -> tuple[str, str]:
        with self._lock:
            self._seq += 1
            sig = base64.b64encode(f"sig-{self._seq}-".encode() + secrets.token_bytes(24)).decode()
            return f"function-call-{self._seq}{secrets.randbelow(10**6):06d}", sig


def _vertex_error(status: int, message: str, list_body: bool = True) -> bytes:
    err = {"error": {"code": status, "message": message, "status": GRPC_STATUS.get(status, "UNKNOWN")}}
    return json.dumps([err] if list_body else err).encode()


def _chunk(model: str, delta: dict[str, Any], finish: str | None = None, usage: dict | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": "vertex-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                           "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        out["usage"] = usage
    return out


def _handler_for(fake: FakeVertex) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        tunnel_host: str | None = None

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=UTF-8")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        # proxy
        def do_CONNECT(self) -> None:  # noqa: N802
            host, _, port = self.path.partition(":")
            allowed = host == fake.host and port == "443"
            with fake._lock:
                fake.connects.append({"target": self.path, "allowed": allowed, "t": time.time()})
            if not allowed:
                self._send(403, b'{"error": "fake proxy: host not allowed"}')
                self.close_connection = True
                return
            self.send_response(200, "Connection Established")
            self.end_headers()
            self.wfile.flush()
            assert fake._tls is not None
            try:
                tls = fake._tls.wrap_socket(self.connection, server_side=True)
            except (ssl.SSLError, OSError):
                self.close_connection = True
                return
            self.connection = tls
            self.rfile = tls.makefile("rb")
            self.wfile = socketserver._SocketWriter(tls)  # type: ignore[attr-defined]
            self.tunnel_host = host
            self.close_connection = False

        # token endpoint (plain loopback) and Vertex (inside the TLS tunnel)
        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            if self.tunnel_host is None:
                if self.path != "/token":
                    self._send(404, b'{"error": "not found"}')
                    return
                form = dict(urllib.parse.parse_qsl(raw.decode()))
                status, payload = fake._mint(form)
                self._send(status, json.dumps(payload).encode())
                return
            self._vertex(raw)

        def do_GET(self) -> None:  # noqa: N802
            self._send(404, _vertex_error(404, f"The requested URL {self.path} was not found on this server."))

        def _vertex(self, raw: bytes) -> None:
            record: dict[str, Any] = {
                "host": self.tunnel_host, "path": self.path, "auth": self.headers.get("Authorization", ""),
                "headers": {k.lower(): v for k, v in self.headers.items()}, "t": time.time(), "rejected": None,
            }
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                body = None
            record["body"] = body
            record["kind"] = "main" if isinstance(body, dict) and body.get("tools") else "aux"
            with fake._lock:
                fake.requests.append(record)
            problem = self._precheck(record)
            if problem:
                record["rejected"], record["status"] = problem[1], problem[0]
                self._send(problem[0], _vertex_error(problem[0], problem[1]))
                return
            resp = fake._next_main(record) if record["kind"] == "main" else fake._aux(record)
            record["response"] = type(resp).__name__
            self._respond(resp, record)
            if getattr(resp, "expire_tokens_after", False):
                fake.expire_all_tokens()

        def _precheck(self, record: dict[str, Any]) -> tuple[int, str] | None:
            if record["host"] != fake.host or record["path"] != fake.chat_path:
                return 404, f"Resource not found: {record['host']}{record['path']} (expected {fake.host}{fake.chat_path})"
            if not fake._auth_ok(record["auth"]):
                return 401, UNAUTHENTICATED_MESSAGE
            if not isinstance(record["body"], dict):
                return 400, "Invalid JSON payload received."
            bad = validate_chat_body(record["body"], set(fake.issued_signatures))
            return (400, bad) if bad else None

        # rendering
        def _respond(self, resp: Response, record: dict[str, Any]) -> None:
            body = record["body"]
            model = body.get("model", "google/gemini")
            if isinstance(resp, Fail):
                record["status"] = resp.status
                headers = {"Retry-After": str(resp.retry_after)} if resp.retry_after is not None else {}
                self._send(resp.status, _vertex_error(resp.status, resp.message, resp.list_body), headers)
                return
            record["status"] = 200
            if isinstance(resp, Drop):
                self._start_sse()
                self._sse(_chunk(model, {"role": "assistant", "content": resp.text[: resp.after_chars]}))
                self.connection.close()  # no terminal chunk: the chunked body is left incomplete
                self.close_connection = True
                return
            pt = resp.prompt_tokens if resp.prompt_tokens is not None else (
                fake.prompt_tokens_fn(body) if fake.prompt_tokens_fn else 120)
            deltas, finish = self._deltas(resp, record)
            usage = {"prompt_tokens": pt, "completion_tokens": 20, "total_tokens": pt + 20}
            if not body.get("stream"):
                message: dict[str, Any] = {"role": "assistant", "content": "".join(d.get("content", "") for d in deltas) or None}
                calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
                if calls:
                    message["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in calls]
                self._send(200, json.dumps({"id": "vertex-fake", "object": "chat.completion", "created": int(time.time()),
                                            "model": model, "usage": usage,
                                            "choices": [{"index": 0, "message": message, "finish_reason": finish}]}).encode())
                return
            self._start_sse()
            for delta in deltas:
                self._sse(_chunk(model, {"role": "assistant", **delta}))
            self._sse(_chunk(model, {}, finish, usage))
            self._write_chunk(b"data: [DONE]\n\n")
            self._write_chunk(b"")
            self.close_connection = True

        def _deltas(self, resp: Say | Call, record: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
            if isinstance(resp, Say):
                size = max(1, resp.chunk_chars)
                return [{"content": resp.text[i:i + size]} for i in range(0, len(resp.text), size)] or [{"content": ""}], "stop"
            tool_calls = []
            for pos, (name, args) in enumerate(resp.calls):
                call_id, sig = fake.next_ids()
                tc: dict[str, Any] = {"index": pos, "id": call_id, "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(args)}}
                if resp.signed and pos == 0:
                    tc["extra_content"] = {"google": {"thought_signature": sig}}
                    with fake._lock:
                        fake.issued_signatures.append(sig)
                tool_calls.append(tc)
            record["tool_calls"] = tool_calls
            with fake._lock:
                fake.signature_by_call.update({tc["id"]: _first_signature(tc) for tc in tool_calls})
            deltas: list[dict[str, Any]] = [{"content": resp.text}] if resp.text else []
            deltas.append({"tool_calls": tool_calls})
            return deltas, "tool_calls"

        def _start_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()

        def _write_chunk(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _sse(self, payload: dict[str, Any]) -> None:
            self._write_chunk(f"data: {json.dumps(payload)}\n\n".encode())

    return Handler


MODEL = "google/gemini-3-flash-preview"
PROJECT = "hermes-e2e-proj"
SA_EMBEDDED_PROJECT = "hermes-sa-embedded-proj"  # differs from PROJECT: proves the config override wins
REGION = "us-central1"


def hermes_setup(fake: FakeVertex, *, model: str = MODEL, extra_config: dict[str, Any] | None = None,
                 context_length: int | None = None) -> dict[str, Any]:
    """``make_home`` kwargs for a Hermes home that selects ``provider: vertex`` against ``fake``: the SA
    key path in ``.env`` (VERTEX_CREDENTIALS_PATH) and project/region under ``vertex:`` in config.yaml."""
    assert fake.sa is not None
    block: dict[str, Any] = {"provider": "vertex", "default": model}
    if context_length:
        block["context_length"] = context_length
    cfg: dict[str, Any] = {"vertex": {"project_id": fake.project, "region": fake.region}}
    cfg.update(extra_config or {})
    return {"model": block, "env_file": {"VERTEX_CREDENTIALS_PATH": str(fake.sa.path)}, "extra_config": cfg}


def signatures_on_wire(body: dict[str, Any]) -> list[str]:
    """Every thought signature replayed in a request's assistant tool calls, in order."""
    return [sig for m in body.get("messages", []) if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or [] if (sig := _first_signature(tc))]


__all__ = [
    "MODEL", "PROJECT", "REGION", "SA_EMBEDDED_PROJECT", "hermes_setup",
    "Call", "Drop", "Fail", "FakeVertex", "GRPC_STATUS", "Say", "ServiceAccount", "TokenPolicy",
    "make_service_account", "make_tls_material", "signatures_on_wire", "validate_chat_body", "verify_jwt_assertion",
]
