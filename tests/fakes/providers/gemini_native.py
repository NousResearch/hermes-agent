"""Fake Google AI Studio ``generateContent`` / ``streamGenerateContent`` endpoint behind a real
TLS boundary.

Hermes routes to its native Gemini adapter only for the real Google host
(``generativelanguage.googleapis.com``), so the fake is reached the way any corporate egress
proxy would be: an HTTPS ``CONNECT`` proxy on loopback that terminates TLS for the Google host
with a leaf certificate signed by a throwaway CA. The child trusts that CA through the standard
``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` channel and reaches the proxy through
``HTTPS_PROXY`` (see :meth:`GeminiFake.child_env`). Every other host is refused and recorded,
so a test also proves the turn made no other egress.

Requests are validated against the published Gemini API reference (``google.ai.generativelanguage``
``v1beta`` / ``v1``: ``GenerateContentRequest``, ``Content``, ``Part``, ``Tool``,
``FunctionDeclaration``, ``Schema``) and rejected the way Google does: HTTP 400 with the
``{"error": {"code", "message", "status": "INVALID_ARGUMENT"}}`` body. That covers unknown proto
fields, role alternation, functionCall/functionResponse pairing (count, name, id), the Gemini 3
thought-signature rule for the current turn, and the OpenAPI ``Schema`` subset of
``FunctionDeclaration.parameters``. Scripted replies are built from the vendor's
``GenerateContentResponse`` shape, including fault injection (Google error bodies, blocked
candidates, a mid-stream connection drop).
"""

from __future__ import annotations

import base64
import datetime
import ipaddress
import json
import re
import secrets
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

GEMINI_HOST = "generativelanguage.googleapis.com"
MODEL_ID = "gemini-3-flash-preview"
API_KEY = "AIzaFakeGeminiKeyForHermesE2E0000000000"
# Documented dummy signatures that tell Gemini 3 to skip thought-signature validation.
SKIP_SIGNATURES = frozenset({"skip_thought_signature_validator", "context_engineering_is_the_way_to_go"})
# Hermes-side config for a home that talks to this fake: the user-facing provider id + model and the
# key in ``.env`` (``model.base_url`` may pin another Google API version, e.g. ``.../v1``).
HERMES_ENV = {"GEMINI_API_KEY": API_KEY}


def hermes_model(base_url: str | None = None, **extra: Any) -> dict[str, Any]:
    model: dict[str, Any] = {"provider": "gemini", "default": MODEL_ID, **extra}
    if base_url:
        model["base_url"] = base_url
    return model

# ── published proto field sets (JSON names; proto JSON parsing also accepts snake_case) ─────────
_REQUEST_FIELDS = {"contents", "tools", "toolConfig", "safetySettings", "systemInstruction",
                   "generationConfig", "cachedContent"}
_CONTENT_FIELDS = {"role", "parts"}
_PART_DATA_FIELDS = {"text", "inlineData", "functionCall", "functionResponse", "fileData",
                     "executableCode", "codeExecutionResult"}
_PART_FIELDS = _PART_DATA_FIELDS | {"thought", "thoughtSignature", "videoMetadata", "partMetadata"}
_FUNCTION_CALL_FIELDS = {"id", "name", "args"}
_FUNCTION_RESPONSE_FIELDS = {"id", "name", "response", "parts", "willContinue", "scheduling"}
_TOOL_FIELDS = {"functionDeclarations", "googleSearchRetrieval", "codeExecution", "googleSearch",
                "urlContext", "computerUse", "fileSearch", "googleMaps"}
_DECL_FIELDS_V1BETA = {"name", "description", "behavior", "parameters", "parametersJsonSchema",
                       "response", "responseJsonSchema"}
_DECL_FIELDS_V1 = {"name", "description", "behavior", "parameters", "response"}
_GENERATION_FIELDS = {"stopSequences", "responseMimeType", "responseSchema", "responseJsonSchema",
                      "responseModalities", "candidateCount", "maxOutputTokens", "temperature", "topP",
                      "topK", "seed", "presencePenalty", "frequencyPenalty", "responseLogprobs",
                      "logprobs", "enableEnhancedCivicAnswers", "speechConfig", "thinkingConfig",
                      "mediaResolution", "imageConfig"}
_THINKING_FIELDS = {"includeThoughts", "thinkingBudget", "thinkingLevel"}
# google.ai.generativelanguage.v1beta.Schema (the OpenAPI 3.0 subset behind ``parameters``).
_SCHEMA_FIELDS = {"type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
                  "properties", "required", "minProperties", "maxProperties", "minLength", "maxLength",
                  "pattern", "example", "anyOf", "propertyOrdering", "default", "items", "minimum",
                  "maximum"}
_SCHEMA_TYPES = {"TYPE_UNSPECIFIED", "STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT", "NULL"}
_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:\-]{0,127}$")
_ROLES = {"user", "model"}
_PATH_RE = re.compile(r"^/(?P<version>v1(?:beta|alpha)?)/models/(?P<model>[^/:]+):(?P<method>[A-Za-z]+)$")


class InvalidArgument(Exception):
    """A request Google would refuse with HTTP 400 INVALID_ARGUMENT."""


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _check_fields(obj: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise InvalidArgument(f"Invalid value at '{where}': expected an object")
    for key in obj:
        if _camel(key) not in allowed:
            raise InvalidArgument(f"Invalid JSON payload received. Unknown name \"{key}\" at '{where}': "
                                  "Cannot find field.")
    return {_camel(k): v for k, v in obj.items()}


# ── scripted replies ────────────────────────────────────────────────────────────────────────────
@dataclass
class Text:
    """A model text answer (optionally with a thought summary part and a trailing signature)."""
    text: str
    thought: str | None = None
    signed: bool = True
    prompt_tokens: int = 1200


@dataclass
class Call:
    """One ``functionCall`` part. The fake mints the id and (for the first call) the signature."""
    name: str
    args: dict[str, Any]


@dataclass
class Calls:
    """A model turn made of ``functionCall`` parts (Gemini 3: signature on the FIRST call only)."""
    calls: list[Call]
    thought: str | None = "Planning the tool call."
    prompt_tokens: int = 1200


@dataclass
class Blocked:
    """A candidate stopped by ``finishReason`` (SAFETY / RECITATION / ...) with no content, or, with
    ``prompt=True``, a prompt blocked via ``promptFeedback.blockReason`` and no candidates."""
    reason: str = "SAFETY"
    prompt: bool = False


@dataclass
class GoogleError:
    """A Google JSON error body (``google.rpc.Status``) with optional RetryInfo / Retry-After."""
    code: int
    status: str
    message: str
    retry_delay_s: float | None = None


@dataclass
class Drop:
    """Streaming: emit ``partial`` as one SSE chunk, then close the TLS connection mid-response."""
    partial: str = "Partial answer that never fin"


Reply = Text | Calls | Blocked | GoogleError | Drop
Responder = Callable[["Recorded"], "Reply | None"]


@dataclass
class Recorded:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: dict[str, Any] | None
    version: str = ""
    model: str = ""
    rpc: str = ""
    status: int = 0
    rejection: str | None = None
    reply: str = ""

    @property
    def stream(self) -> bool:
        return self.rpc == "streamGenerateContent"

    @property
    def contents(self) -> list[dict[str, Any]]:
        return list((self.body or {}).get("contents") or [])

    def parts(self, kind: str) -> list[dict[str, Any]]:
        """Every part carrying ``kind`` (e.g. ``functionCall``), in wire order."""
        return [p for c in self.contents for p in c.get("parts") or [] if kind in p]

    def declarations(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for tool in (self.body or {}).get("tools") or []:
            for decl in tool.get("functionDeclarations") or []:
                out[decl.get("name", "")] = decl
        return out

    def all_text(self) -> str:
        texts = [p.get("text") or "" for c in self.contents for p in c.get("parts") or []]
        system = ((self.body or {}).get("systemInstruction") or {}).get("parts") or []
        return "\n".join(texts + [p.get("text") or "" for p in system])

    def last_user_text(self) -> str:
        for content in reversed(self.contents):
            texts = [p["text"] for p in content.get("parts") or [] if isinstance(p.get("text"), str)]
            if content.get("role") == "user" and texts:
                return "\n".join(texts)
        return ""


# ── request validation (published contract) ────────────────────────────────────────────────────
def _validate_schema(node: Any, where: str) -> None:
    """``FunctionDeclaration.parameters`` is a proto ``Schema``: unknown keys, list-valued ``type``
    and non-string ``enum`` entries do not parse; ``required`` must name defined properties."""
    schema = _check_fields(node, _SCHEMA_FIELDS, where)
    type_ = schema.get("type")
    if type_ is not None and (not isinstance(type_, str) or type_.upper() not in _SCHEMA_TYPES):
        raise InvalidArgument(f"Invalid value at '{where}.type' (type.googleapis.com/"
                              f"google.ai.generativelanguage.v1beta.Type), {json.dumps(type_)}")
    for i, value in enumerate(schema.get("enum") or []):
        if not isinstance(value, str):
            raise InvalidArgument(f"Invalid value at '{where}.enum[{i}]' (TYPE_STRING), {json.dumps(value)}")
    props = schema.get("properties") or {}
    for name, sub in props.items():
        _validate_schema(sub, f"{where}.properties[{name}].value")
    for name in schema.get("required") or []:
        if name not in props:
            raise InvalidArgument(f"{where}.required[{name}]: property is not defined")
    if "items" in schema:
        _validate_schema(schema["items"], f"{where}.items")
    elif isinstance(type_, str) and type_.upper() == "ARRAY":
        raise InvalidArgument(f"{where}.items: missing field.")
    for i, sub in enumerate(schema.get("anyOf") or []):
        _validate_schema(sub, f"{where}.any_of[{i}]")


def _validate_json_schema_root(schema: Any, where: str) -> None:
    """``parametersJsonSchema`` must describe an object whose properties are the parameters."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise InvalidArgument(f"{where}: parameters_json_schema must describe an object (type: object)")


def _validate_tools(tools: Any, version: str) -> None:
    decl_fields = _DECL_FIELDS_V1BETA if version == "v1beta" else _DECL_FIELDS_V1
    for ti, tool in enumerate(tools if isinstance(tools, list) else []):
        tool = _check_fields(tool, _TOOL_FIELDS, f"tools[{ti}]")
        for di, decl in enumerate(tool.get("functionDeclarations") or []):
            where = f"tools[{ti}].function_declarations[{di}]"
            decl = _check_fields(decl, decl_fields, where)
            if not _FUNCTION_NAME_RE.match(str(decl.get("name") or "")):
                raise InvalidArgument(f"{where}.name: Invalid function name. Must start with a letter or an "
                                      "underscore. Must be alphameric (a-z, A-Z, 0-9), underscores (_), dots (.), "
                                      "colons (:), or dashes (-), with a maximum length of 128.")
            if "parameters" in decl and "parametersJsonSchema" in decl:
                raise InvalidArgument(f"{where}: parameters and parameters_json_schema are mutually exclusive")
            if "parameters" in decl:
                _validate_schema(decl["parameters"], f"{where}.parameters")
            if "parametersJsonSchema" in decl:
                _validate_json_schema_root(decl["parametersJsonSchema"], f"{where}.parameters_json_schema")


def _validate_part(part: Any, where: str) -> dict[str, Any]:
    part = _check_fields(part, _PART_FIELDS, where)
    data = [k for k in part if k in _PART_DATA_FIELDS]
    if len(data) != 1:
        raise InvalidArgument(f"{where}: a Part must set exactly one data field (oneof 'data'), got {data}")
    if "functionCall" in part:
        fc = _check_fields(part["functionCall"], _FUNCTION_CALL_FIELDS, f"{where}.function_call")
        if not fc.get("name") or not isinstance(fc.get("args", {}), dict):
            raise InvalidArgument(f"{where}.function_call: name is required and args must be an object")
    if "functionResponse" in part:
        fr = _check_fields(part["functionResponse"], _FUNCTION_RESPONSE_FIELDS, f"{where}.function_response")
        if not fr.get("name") or not isinstance(fr.get("response"), dict):
            raise InvalidArgument(f"{where}.function_response: name is required and response must be an object")
    sig = part.get("thoughtSignature")
    if sig is not None and not (isinstance(sig, str) and sig):
        raise InvalidArgument(f"{where}.thought_signature: invalid bytes value")
    return part


def _check_pairing(contents: list[dict[str, Any]], gemini3: bool) -> None:
    """Every functionCall turn is answered by the NEXT content with one functionResponse per call
    (same name, same id on Gemini 3); a functionResponse turn must follow a functionCall turn."""
    for i, content in enumerate(contents):
        calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        responses = [p["functionResponse"] for p in content["parts"] if "functionResponse" in p]
        if responses:
            prev_calls = [p["functionCall"] for p in contents[i - 1]["parts"] if "functionCall" in p] if i else []
            if not prev_calls:
                raise InvalidArgument("Please ensure that function response turn comes immediately after a "
                                      "function call turn.")
        if not calls:
            continue
        nxt = contents[i + 1]["parts"] if i + 1 < len(contents) else []
        answered = [p["functionResponse"] for p in nxt if "functionResponse" in p]
        if i + 1 < len(contents) and len(answered) != len(calls):
            raise InvalidArgument("Please ensure that the number of function response parts is equal to the "
                                  "number of function call parts of the function call turn.")
        for call, resp in zip(calls, answered):
            if call.get("name") != resp.get("name"):
                raise InvalidArgument(f"functionResponse name {resp.get('name')!r} does not match functionCall "
                                      f"name {call.get('name')!r} in contents[{i + 1}]")
        if gemini3 and answered:
            if sorted(str(c.get("id")) for c in calls) != sorted(str(r.get("id")) for r in answered):
                raise InvalidArgument(f"functionResponse ids in contents[{i + 1}] do not match the functionCall "
                                      "ids of the function call turn.")


def _current_turn_start(contents: list[dict[str, Any]]) -> int:
    """Index of the newest user content with standard (non-functionResponse) content."""
    for i in range(len(contents) - 1, -1, -1):
        c = contents[i]
        if c["role"] == "user" and any("functionResponse" not in p for p in c["parts"]):
            return i
    return 0


def _check_signatures(contents: list[dict[str, Any]], issued: set[str]) -> None:
    """Gemini 3: the FIRST functionCall part of each step of the current turn must carry a
    thoughtSignature Google issued (or a documented dummy); a forged one is corrupted."""
    for i in range(_current_turn_start(contents), len(contents)):
        c = contents[i]
        fc_parts = [p for p in c["parts"] if "functionCall" in p] if c["role"] == "model" else []
        if not fc_parts:
            continue
        sig = fc_parts[0].get("thoughtSignature")
        if not sig:
            raise InvalidArgument(f"Function call `{fc_parts[0]['functionCall']['name']}` in the `{i}.` content "
                                  "block is missing a `thought_signature`.")
        if sig not in issued and sig not in SKIP_SIGNATURES:
            raise InvalidArgument("Corrupted thought signature.")


def validate_generate_request(body: Any, version: str, model: str, issued: set[str]) -> None:
    body = _check_fields(body, _REQUEST_FIELDS, "")
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        raise InvalidArgument("* GenerateContentRequest.contents: contents is not specified")
    norm: list[dict[str, Any]] = []
    for ci, content in enumerate(contents):
        content = _check_fields(content, _CONTENT_FIELDS, f"contents[{ci}]")
        role = content.get("role", "user")
        if role not in _ROLES:
            raise InvalidArgument(f"Please use a valid role: user, model. (contents[{ci}].role={role!r})")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise InvalidArgument(f"* GenerateContentRequest.contents[{ci}].parts: contents.parts must not be empty.")
        norm.append({"role": role, "parts": [_validate_part(p, f"contents[{ci}].parts[{pi}]")
                                              for pi, p in enumerate(parts)]})
    for a, b in zip(norm, norm[1:]):
        if a["role"] == b["role"]:
            raise InvalidArgument("Please ensure that multiturn requests alternate between user and model.")
    if norm[-1]["role"] != "user":
        raise InvalidArgument("Please ensure that single turn requests end with a user role or the role field "
                              "is empty.")
    gemini3 = bool(re.match(r"gemini-([3-9]|\d\d)", model))
    _check_pairing(norm, gemini3)
    if gemini3:
        _check_signatures(norm, issued)
    if "systemInstruction" in body:
        system = _check_fields(body["systemInstruction"], _CONTENT_FIELDS, "system_instruction")
        for pi, p in enumerate(system.get("parts") or []):
            _validate_part(p, f"system_instruction.parts[{pi}]")
    _validate_tools(body.get("tools"), version)
    generation = _check_fields(body.get("generationConfig") or {}, _GENERATION_FIELDS, "generation_config")
    if "thinkingConfig" in generation:
        _check_fields(generation["thinkingConfig"], _THINKING_FIELDS, "generation_config.thinking_config")


# ── response building (GenerateContentResponse) ────────────────────────────────────────────────
def _usage(prompt_tokens: int, output_tokens: int = 24, thought_tokens: int = 16) -> dict[str, Any]:
    return {"promptTokenCount": prompt_tokens, "candidatesTokenCount": output_tokens,
            "thoughtsTokenCount": thought_tokens, "totalTokenCount": prompt_tokens + output_tokens + thought_tokens}


def _response(parts: list[dict[str, Any]] | None, finish: str | None, usage: dict[str, Any] | None,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cand: dict[str, Any] = {"index": 0}
    if parts is not None:
        cand["content"] = {"role": "model", "parts": parts}
    if finish:
        cand["finishReason"] = finish
    out: dict[str, Any] = {"candidates": [cand], "modelVersion": MODEL_ID,
                           "responseId": secrets.token_urlsafe(12)}
    if usage:
        out["usageMetadata"] = usage
    out.update(extra or {})
    return out


def _chunks(text: str, n: int = 3) -> list[str]:
    step = max(1, -(-len(text) // n))
    return [text[i:i + step] for i in range(0, len(text), step)] or [""]


# ── TLS material ────────────────────────────────────────────────────────────────────────────────
def _write_tls_material(directory: Path) -> tuple[Path, Path, Path]:
    """Throwaway CA + a leaf for the Google host. Returns (ca_pem, leaf_cert_pem, leaf_key_pem)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "hermes-e2e gemini fake CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                       content_commitment=False, key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, GEMINI_HOST)]))
            .issuer_name(ca_name).public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(GEMINI_HOST),
                                                        x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    directory.mkdir(parents=True, exist_ok=True)
    ca_pem, cert_pem, key_pem = directory / "ca.pem", directory / "leaf.pem", directory / "leaf.key"
    ca_pem.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_pem.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_pem.write_bytes(leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
    return ca_pem, cert_pem, key_pem


# ── the fake ────────────────────────────────────────────────────────────────────────────────────
class GeminiFake:
    """Loopback CONNECT proxy + TLS-terminated fake Gemini API.

    ``script`` replies are consumed in order by every valid generate call that ``route`` (optional
    responder, e.g. for compaction summaries) does not claim. An exhausted script answers with a
    Google 500 so an unexpected extra call is visible instead of hanging.
    """

    def __init__(self, workdir: Path, script: list[Reply] | None = None, *, route: Responder | None = None,
                 api_key: str = API_KEY) -> None:
        self.workdir = workdir
        self.api_key = api_key
        self.script: list[Reply] = list(script or [])
        self.route = route
        self.requests: list[Recorded] = []
        self.refused_hosts: list[str] = []
        self.issued_signatures: list[str] = []
        self.call_signatures: dict[str, str] = {}  # functionCall id -> signature minted on its part
        self._lock = threading.Lock()
        self._counter = 0
        self.ca_pem, cert, key = _write_tls_material(workdir / "tls")
        self._tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._tls.load_cert_chain(cert, key)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="gemini-fake", daemon=True)

    # lifecycle ---------------------------------------------------------------------------------
    def __enter__(self) -> "GeminiFake":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def child_env(self) -> dict[str, str]:
        """Env for the ``hermes`` child: route HTTPS through the proxy and trust the fake CA."""
        ca = str(self.ca_pem)
        return {"HTTPS_PROXY": self.proxy_url, "https_proxy": self.proxy_url,
                "HTTP_PROXY": self.proxy_url, "http_proxy": self.proxy_url,
                "NO_PROXY": "", "no_proxy": "",
                "SSL_CERT_FILE": ca, "REQUESTS_CA_BUNDLE": ca, "CURL_CA_BUNDLE": ca}

    # views ------------------------------------------------------------------------------------
    def generate_calls(self) -> list[Recorded]:
        with self._lock:
            return [r for r in self.requests if r.rpc in {"generateContent", "streamGenerateContent"}]

    def rejections(self) -> list[str]:
        with self._lock:
            return [f"{r.rpc} -> {r.status}: {r.rejection}" for r in self.requests if r.rejection]

    def main_calls(self) -> list[Recorded]:
        """Generate calls answered from ``script`` (not claimed by ``route``)."""
        return [r for r in self.generate_calls() if r.reply.startswith("script:")]

    # dispatch ---------------------------------------------------------------------------------
    def _mint_signature(self) -> str:
        sig = base64.b64encode(b"\x12\x34gemini-e2e-sig:" + secrets.token_bytes(24)).decode()
        with self._lock:
            self.issued_signatures.append(sig)
        return sig

    def _mint_call_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"fc-{self._counter:04d}-{secrets.token_hex(3)}"

    def _next_reply(self, rec: Recorded) -> Reply:
        if self.route is not None and (routed := self.route(rec)) is not None:
            rec.reply = f"route:{type(routed).__name__}"
            return routed
        with self._lock:
            reply = self.script.pop(0) if self.script else GoogleError(500, "INTERNAL", "fake script exhausted")
        rec.reply = f"script:{type(reply).__name__}"
        return reply

    def _build(self, reply: Reply) -> list[dict[str, Any]]:
        """Stream events for a reply (a unary response is their merge)."""
        builders: dict[type, Callable[[Any], list[dict[str, Any]]]] = {
            Text: self._text_events, Calls: self._call_events, Blocked: self._blocked_events,
        }
        return builders[type(reply)](reply)

    def _text_events(self, reply: Text) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if reply.thought:
            events.append(_response([{"text": reply.thought, "thought": True}], None, None))
        pieces = _chunks(reply.text)
        for i, piece in enumerate(pieces):
            part: dict[str, Any] = {"text": piece}
            last = i == len(pieces) - 1
            if last and reply.signed:
                part["thoughtSignature"] = self._mint_signature()
            events.append(_response([part], "STOP" if last else None, _usage(reply.prompt_tokens) if last else None))
        return events

    def _call_events(self, reply: Calls) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if reply.thought:
            events.append(_response([{"text": reply.thought, "thought": True}], None, None))
        parts = []
        for i, call in enumerate(reply.calls):
            call_id = self._mint_call_id()
            part: dict[str, Any] = {"functionCall": {"id": call_id, "name": call.name, "args": call.args}}
            if i == 0:
                part["thoughtSignature"] = self._mint_signature()
                with self._lock:
                    self.call_signatures[call_id] = part["thoughtSignature"]
            parts.append(part)
        events.append(_response(parts, "STOP", _usage(reply.prompt_tokens)))
        return events

    @staticmethod
    def _blocked_events(reply: Blocked) -> list[dict[str, Any]]:
        if reply.prompt:
            return [{"promptFeedback": {"blockReason": reply.reason, "safetyRatings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}]},
                "usageMetadata": {"promptTokenCount": 900, "totalTokenCount": 900}, "modelVersion": MODEL_ID}]
        ratings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}]
        extra = {"citationMetadata": {"citationSources": [{"startIndex": 0, "endIndex": 40,
                                                           "uri": "https://example.com/source"}]}}
        cand_extra = ratings if reply.reason == "SAFETY" else None
        resp = _response(None, reply.reason, _usage(900, 0, 0))
        if cand_extra:
            resp["candidates"][0]["safetyRatings"] = cand_extra
        if reply.reason == "RECITATION":
            resp["candidates"][0].update(extra)
        return [resp]

    @staticmethod
    def merge_events(events: list[dict[str, Any]]) -> dict[str, Any]:
        """Unary ``generateContent`` body = the stream's chunks folded into one candidate."""
        if not events or "candidates" not in events[-1]:
            return events[-1] if events else {}
        parts: list[dict[str, Any]] = []
        for ev in events:
            parts.extend(((ev.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        final = json.loads(json.dumps(events[-1]))
        if parts:
            final["candidates"][0]["content"] = {"role": "model", "parts": parts}
        return final

    # HTTP handler -----------------------------------------------------------------------------
    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            tunneled = False

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature
                return

            def do_CONNECT(self) -> None:  # noqa: N802 - http.server naming
                host = self.path.split(":", 1)[0].lower()
                if self.tunneled or host != GEMINI_HOST:
                    with fake._lock:
                        fake.refused_hosts.append(host)
                    self.send_response(403, "Forbidden by hermes e2e fake proxy")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    self.close_connection = True
                    return
                self.send_response(200, "Connection established")
                self.end_headers()
                self.wfile.flush()
                try:
                    tls = fake._tls.wrap_socket(self.connection, server_side=True)
                except (ssl.SSLError, OSError):
                    self.close_connection = True
                    return
                self.connection = self.request = tls
                self.rfile = tls.makefile("rb")
                self.wfile = tls.makefile("wb")
                self.tunneled = True
                self.close_connection = False

            def _refuse_plain(self) -> None:
                with fake._lock:
                    fake.refused_hosts.append(urlsplit(self.path).hostname or self.headers.get("Host", "?"))
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if not self.tunneled:
                    return self._refuse_plain()
                fake._handle(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                if not self.tunneled:
                    return self._refuse_plain()
                fake._handle(self, "POST")

        return Handler

    def _send_json(self, h: BaseHTTPRequestHandler, rec: Recorded, status: int, body: dict[str, Any],
                   headers: dict[str, str] | None = None) -> None:
        rec.status = status
        data = json.dumps(body).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json; charset=UTF-8")
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)
        h.wfile.flush()

    def _error(self, h: BaseHTTPRequestHandler, rec: Recorded, err: GoogleError) -> None:
        body: dict[str, Any] = {"error": {"code": err.code, "message": err.message, "status": err.status}}
        headers = {}
        if err.retry_delay_s is not None:
            body["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                         "retryDelay": f"{err.retry_delay_s:g}s"}]
            headers["Retry-After"] = f"{err.retry_delay_s:g}"
        if err.code == 400:
            rec.rejection = rec.rejection or err.message
        self._send_json(h, rec, err.code, body, headers)

    def _handle(self, h: BaseHTTPRequestHandler, method: str) -> None:
        url = urlsplit(h.path)
        length = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        rec = Recorded(method, url.path, parse_qs(url.query), {k.lower(): v for k, v in h.headers.items()}, body)
        with self._lock:
            self.requests.append(rec)
        match = _PATH_RE.match(url.path)
        if method == "GET" or not match:
            return self._error(h, rec, GoogleError(404, "NOT_FOUND", f"fake has no route for {method} {url.path}"))
        rec.version, rec.model, rec.rpc = match["version"], match["model"], match["method"]
        key = rec.headers.get("x-goog-api-key") or (rec.query.get("key") or [""])[0]
        if key != self.api_key:
            rec.rejection = "API key not valid"
            return self._error(h, rec, GoogleError(400, "INVALID_ARGUMENT",
                                                   "API key not valid. Please pass a valid API key."))
        if rec.stream and (rec.query.get("alt") or [""])[0] != "sse":
            rec.rejection = "streamGenerateContent without alt=sse"
        try:
            with self._lock:
                issued = set(self.issued_signatures)
            validate_generate_request(body, rec.version, rec.model, issued)
        except InvalidArgument as exc:
            rec.rejection = str(exc)
            return self._error(h, rec, GoogleError(400, "INVALID_ARGUMENT", str(exc)))
        reply = self._next_reply(rec)
        if isinstance(reply, GoogleError):
            return self._error(h, rec, reply)
        if isinstance(reply, Drop):
            return self._drop(h, rec, reply)
        events = self._build(reply)
        if not rec.stream:
            return self._send_json(h, rec, 200, self.merge_events(events))
        self._stream(h, rec, events)

    def _stream(self, h: BaseHTTPRequestHandler, rec: Recorded, events: list[dict[str, Any]]) -> None:
        rec.status = 200
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        for ev in events:
            self._write_chunk(h, f"data: {json.dumps(ev)}\r\n\r\n".encode())
        h.wfile.write(b"0\r\n\r\n")
        h.wfile.flush()

    @staticmethod
    def _write_chunk(h: BaseHTTPRequestHandler, data: bytes) -> None:
        h.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        h.wfile.flush()

    def _drop(self, h: BaseHTTPRequestHandler, rec: Recorded, reply: Drop) -> None:
        rec.status = 200
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        self._write_chunk(h, f"data: {json.dumps(_response([{'text': reply.partial}], None, None))}\r\n\r\n".encode())
        time.sleep(0.05)  # let the partial chunk reach the client before the reset
        h.close_connection = True
        try:
            h.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            h.connection.close()
        except OSError:
            pass
