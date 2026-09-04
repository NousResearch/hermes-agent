"""A2A HTTP/wire boundary — JSON-RPC dispatch, redaction helpers, and HTTP handler.

Extracted from ``adapter.py`` so the adapter module stays below the strict
physical-line limit.  This module owns only transport concerns; it never
imports ``adapter.py`` at runtime (``TYPE_CHECKING`` guard for typing).

Ownership:
  - method-name mapping
  - JSON-RPC redaction/bounding constants and helpers
  - ``_audit_safe``, ``_failure_outcome``, ``_send_result_from_outcome``
  - ``_A2AServer`` / ``A2ARequestHandler`` (HTTP server + handler)
  - transport-only imports/constants used solely by those symbols
"""
from __future__ import annotations

import collections.abc as _collections_abc
import json
import logging
import os
import select
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from . import protocol, security

if TYPE_CHECKING:
    from .adapter import A2AAdapter  # type: ignore[import]

logger = logging.getLogger(__name__)

# Keep _MAX_BODY in sync with adapter.py
_MAX_BODY = 1_048_576

_PORTABLE_NONBLOCK_RECV = getattr(socket, "MSG_DONTWAIT", 0)


def _method_info(method: str) -> tuple[str, bool]:
    mapping = {
        "SendMessage": ("send", True),
        "message/send": ("send", False),
        "SendStreamingMessage": ("stream", True),
        "message/stream": ("stream", False),
        "GetTask": ("get", True),
        "tasks/get": ("get", False),
        "ListTasks": ("list", True),
        "tasks/list": ("list", False),
        "CancelTask": ("cancel", True),
        "tasks/cancel": ("cancel", False),
        "SubscribeToTask": ("subscribe", True),
        "tasks/subscribe": ("subscribe", False),
        "CreateTaskPushNotificationConfig": ("push_create", True),
        "tasks/pushNotificationConfig/create": ("push_create", False),
        "tasks/pushNotificationConfig/set": ("push_create", False),
        "tasks/pushNotification/set": ("push_create", False),
        "GetTaskPushNotificationConfig": ("push_get", True),
        "tasks/pushNotificationConfig/get": ("push_get", False),
        "ListTaskPushNotificationConfigs": ("push_list", True),
        "tasks/pushNotificationConfig/list": ("push_list", False),
        "DeleteTaskPushNotificationConfig": ("push_delete", True),
        "tasks/pushNotificationConfig/delete": ("push_delete", False),
    }
    return mapping.get(method, ("", False))


_DETAIL_MAX_CODEPOINTS=300
_JSONRPC_KEY_MAX_CODEPOINTS=64
_JSONRPC_STRING_MAX_CODEPOINTS=300
_JSONRPC_MAX_DEPTH=4
_JSONRPC_MAX_WIDTH=16
_JSONRPC_MAX_BYTES=2048
_JSONRPC_CODE_MIN=-2147483648
_JSONRPC_CODE_MAX=2147483647
_TRUNCATION_MARKER="...[truncated]"
_REDACTED_MARKER="[redacted]"
_DATA_TRUNCATED_MARKER="[truncated]"
def _truncate_codepoints(v,c):
 m=_TRUNCATION_MARKER;return v if len(v)<=c else (m[:c] if c<len(m) else v[:c-len(m)]+m)
def _bounded_redacted_detail(v,c=300):
 try:s=v if isinstance(v,str) else str(v)
 except:s=_REDACTED_MARKER
 try:
  r=security.redact_outbound(s)
  if not isinstance(r,str):r=_REDACTED_MARKER
 except:r=_REDACTED_MARKER
 try:return _truncate_codepoints(r,c)
 except:return _REDACTED_MARKER[:c] if c<len(_REDACTED_MARKER) else _REDACTED_MARKER
def _sanitize_string_for_jsonrpc(v,m=300):
 try:
  r=security.redact_outbound(v)
  if not isinstance(r,str):r=_REDACTED_MARKER
 except:r=_REDACTED_MARKER
 return _truncate_codepoints(r,m)
def _redacted_reply_text(value: object) -> str:
 try:
  if not isinstance(value, str):
   return _REDACTED_MARKER
  s = value
 except:
  return _REDACTED_MARKER
 try:
  r = security.redact_outbound(s)
  if not isinstance(r, str):
   return _REDACTED_MARKER
  return r
 except:
  return _REDACTED_MARKER
def _sanitize_jsonrpc_value(v,d):
 try:
  if d>_JSONRPC_MAX_DEPTH:return _REDACTED_MARKER
  if v is None or isinstance(v,bool):return v
  if isinstance(v,int):return v if _JSONRPC_CODE_MIN<=v<=_JSONRPC_CODE_MAX else _REDACTED_MARKER
  if isinstance(v,float):return v if __import__("math").isfinite(v) else _REDACTED_MARKER
  if isinstance(v,str):return _sanitize_string_for_jsonrpc(v,_JSONRPC_STRING_MAX_CODEPOINTS)
  if isinstance(v,dict):
   out={}
   try:
    iterator = iter(dict.items(v))
   except:
    return _REDACTED_MARKER
   for _ in range(_JSONRPC_MAX_WIDTH):
    try:
     k,val = next(iterator)
    except StopIteration:
     break
    except:
     return _REDACTED_MARKER
    if isinstance(k,str):
     try:
      rk=security.redact_outbound(k)
      if not isinstance(rk,str):rk=_REDACTED_MARKER
     except:rk=_REDACTED_MARKER
     sk=_truncate_codepoints(rk,_JSONRPC_KEY_MAX_CODEPOINTS)
    else:sk=_REDACTED_MARKER
    if sk in out:
     continue
    out[sk]=_sanitize_jsonrpc_value(val,d+1)
   return out
  if isinstance(v,list):
   try:
    t=v[:_JSONRPC_MAX_WIDTH]
   except:return _REDACTED_MARKER
   return [_sanitize_jsonrpc_value(x,d+1) for x in t]
  if isinstance(v,_collections_abc.Mapping):
   return _REDACTED_MARKER
  return _REDACTED_MARKER
 except:return _REDACTED_MARKER
def _redacted_jsonrpc_detail(raw):
 payload={}
 try:
  if isinstance(raw,_collections_abc.Mapping) and not isinstance(raw,dict):
   payload={"message":_REDACTED_MARKER}
  elif isinstance(raw,dict):
   try:
    has_code = dict.__contains__(raw, "code")
   except:
    has_code = False
   if has_code:
    try:
     rc = dict.get(raw, "code")
    except:
     rc = None
    if isinstance(rc,int) and not isinstance(rc,bool) and _JSONRPC_CODE_MIN<=rc<=_JSONRPC_CODE_MAX:payload["code"]=rc
   try:
    has_message = dict.__contains__(raw, "message")
   except:
    has_message = False
   if has_message:
    try:
     rm = dict.get(raw, "message")
    except:
     rm = None
    if isinstance(rm,str):payload["message"]=_sanitize_string_for_jsonrpc(rm,_JSONRPC_STRING_MAX_CODEPOINTS)
    elif rm is not None:payload["message"]=_bounded_redacted_detail(rm,_JSONRPC_STRING_MAX_CODEPOINTS)
   try:
    has_data = dict.__contains__(raw, "data")
   except:
    has_data = False
   if has_data:
    try:
     _data = dict.get(raw, "data")
     payload["data"]=_sanitize_jsonrpc_value(_data,0)
    except:payload["data"]=_REDACTED_MARKER
   if not payload:payload["message"]=_REDACTED_MARKER
  else:
   payload["message"]=_bounded_redacted_detail(raw,_JSONRPC_STRING_MAX_CODEPOINTS)
   if not payload.get("message"):payload["message"]=_REDACTED_MARKER
  if "message" not in payload and "code" not in payload:payload["message"]=_REDACTED_MARKER
  try:
   ser=__import__("json").dumps(payload,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
   if len(ser)>_JSONRPC_MAX_BYTES:
    tp={}
    if "code" in payload:tp["code"]=payload["code"]
    if "message" in payload:tp["message"]=payload["message"]
    tp["data"]=_DATA_TRUNCATED_MARKER;payload=tp
    ser2=__import__("json").dumps(payload,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
    if len(ser2)>_JSONRPC_MAX_BYTES:raise ValueError
  except:payload={"message":_REDACTED_MARKER,"data":_DATA_TRUNCATED_MARKER}
 except:payload={"message":_REDACTED_MARKER,"data":_DATA_TRUNCATED_MARKER}
 try:
  parts=[]
  if "code" in payload:parts.append(str(payload["code"]))
  if "message" in payload and isinstance(payload["message"],str):parts.append(payload["message"])
  err=": ".join(parts) if len(parts)==2 else (parts[0] if parts else _REDACTED_MARKER)
  try:
   red=security.redact_outbound(err)
   if not isinstance(red,str):red=_REDACTED_MARKER
  except:red=_REDACTED_MARKER
  err=_truncate_codepoints(red,_DETAIL_MAX_CODEPOINTS)
 except:err=_REDACTED_MARKER
 try:
  fb=__import__("json").dumps(payload,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
  if len(fb)>_JSONRPC_MAX_BYTES:payload={"message":_REDACTED_MARKER,"data":_DATA_TRUNCATED_MARKER};err=_REDACTED_MARKER
 except:payload={"message":_REDACTED_MARKER,"data":_DATA_TRUNCATED_MARKER};err=_REDACTED_MARKER
 return err,payload
def _audit_safe(direction,peer,tid,detail,context_id=""):
 try:
  sp=_bounded_redacted_detail(peer,128) if peer else "";st=_bounded_redacted_detail(tid,128) if tid else "";sc=_bounded_redacted_detail(context_id,128) if context_id else "";sd=_bounded_redacted_detail(detail,_DETAIL_MAX_CODEPOINTS) if detail else ""
  security.audit(direction,sp,st,sd,context_id=sc)
 except Exception as exc:
  try:b=_bounded_redacted_detail(exc,_DETAIL_MAX_CODEPOINTS);__import__("logging").getLogger(__name__).warning("A2A: audit write failed (%s): %s",_bounded_redacted_detail(direction,64),b)
  except:pass
def _failure_outcome(category,detail,*,peer,task_id,context_id,payload=None):
 allowed={"routing","transport","jsonrpc","invalid_response","durability"}
 if category not in allowed:category="transport"
 if category=="jsonrpc":sd=_truncate_codepoints(detail if isinstance(detail,str) else _bounded_redacted_detail(detail,_DETAIL_MAX_CODEPOINTS),_DETAIL_MAX_CODEPOINTS);spayload=payload
 else:sd=_bounded_redacted_detail(detail,_DETAIL_MAX_CODEPOINTS);spayload=None
 direction="push_dropped" if category=="routing" else "push_failed"
 _audit_safe(direction,peer,task_id,sd,context_id=context_id)
 return __import__("plugins.platforms.a2a.protocol",fromlist=["PushOutcome"]).PushOutcome(success=False,category=category,error=sd,payload=spayload)
def _send_result_from_outcome(mid,outcome):
 if outcome.success:return __import__("gateway.platforms.base",fromlist=["SendResult"]).SendResult(success=True,message_id=mid)
 raw=f"{outcome.category}: {outcome.error}";safe=_bounded_redacted_detail(raw,_DETAIL_MAX_CODEPOINTS)
 return __import__("gateway.platforms.base",fromlist=["SendResult"]).SendResult(success=False,message_id=mid,error=safe)
class _A2AServer(ThreadingHTTPServer):

    daemon_threads = True

    def __init__(self, addr, handler_cls, adapter: "A2AAdapter"):
        super().__init__(addr, handler_cls)
        self.adapter = adapter


class A2ARequestHandler(BaseHTTPRequestHandler):

    @property
    def adapter(self) -> "A2AAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    # Silence the default stderr access log.
    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("A2A http: " + format, *args)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        # Flush HERE so a dead socket surfaces OSError AT THE CALL SITE —
        # a buffered write into a half-closed socket "succeeds" silently via
        # TCP buffering (R2: the reply vanished because only the buffer, not
        # the client, received it) and the exception would otherwise only
        # fire later in the base handler's finish(), outside our catch.
        self.wfile.flush()

    def _request_public_url(self) -> str:
        explicit = os.getenv("A2A_PUBLIC_URL", "").strip()
        if explicit:
            return explicit
        host = self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")
        if not host:
            return ""
        host = host.split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/"

    def do_GET(self):  # noqa: N802
        route = self.adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            public_url = self._request_public_url() or None
            self._json(200, self.adapter._build_card(public_url, agent=agent))
            return
        if subpath in ("/", "/health"):
            payload = {
                "status": "ok",
                "agent": agent.get("name") or self.adapter.agent_name,
            }
            # Do not leak profile/tenant topology on remote unauthenticated GETs.
            # Agent Cards are intentionally public; health topology is not.
            if self.adapter._security_context.localhost_only() or self.adapter._security_context.authenticate(
                self.headers.get("Authorization"),
                self.client_address[0] if self.client_address else "",
            ) is not None:
                payload["served_agents"] = self.adapter._served_agent_summary(
                    public_url=self._request_public_url() or None)
            self._json(200, payload)
            return
        if subpath == "/metrics":
            self._json(200, protocol.metrics.snapshot())
            return
        self._json(404, {"error": "not found"})

    def _a2a_client_alive(self) -> bool:
        sock = getattr(self, "connection", None)
        if sock is None:
            return True
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return True  # no EOF/data pending — assume alive
            # Data or EOF available. b"" (EOF) means the client closed.
            # MSG_DONTWAIT is not a Winsock receive flag; CPython does not
            # expose it on Windows.  MSG_PEEK already prevents consuming the
            # data, and select() has established readability, so the non-
            # blocking flag is a belt-and-suspenders guard on Unix.  On
            # Windows we fall back to MSG_PEEK alone.
            chunk = sock.recv(1, socket.MSG_PEEK | _PORTABLE_NONBLOCK_RECV)
            return bool(chunk)
        except (BlockingIOError, InterruptedError):
            return True
        except OSError:
            return False

    def _handle_send(self, req_id, params, identity, agent, is_v1):
        result = self.adapter._rpc_message_send(
            req_id, params, identity, agent=agent, v1_response=is_v1,
            client_alive=self._a2a_client_alive,
        )
        if result is None:
            # out_of_band_only with a completed reply: already pushed
            # directly — skip the socket write entirely (the client is gone).
            self.close_connection = True
            return
        # Bounded final pre-write liveness probe: the last keepalive probe
        # during _await_reply may have been seconds ago; a client that died
        # in that window would silently lose the reply via TCP buffering.
        # Probe once more before the write; route dead clients through the
        # existing rescue and skip the socket write entirely.
        #
        # RESIDUAL RACE (do NOT claim elimination): a client that dies
        # between THIS probe and the _json write is still lost here —
        # the broad OSError catch below is the final safety net.  A stable
        # delivery ID / application ACK protocol would close this gap but
        # is explicitly future work (see design decision 4).
        if not self._a2a_client_alive():
            self.adapter._push_reply_after_client_gone(req_id, result, is_v1=is_v1)
            self.close_connection = True
            return
        try:
            self._json(200, result)
        except OSError:
            self.adapter._push_reply_after_client_gone(req_id, result, is_v1=is_v1)

    def do_POST(self):  # noqa: N802
        adapter = self.adapter
        client_ip = self.client_address[0] if self.client_address else ""

        # Identity comes from the presented credential (or the socket in
        # localhost-only mode) — never from the request body.
        identity = adapter._security_context.authenticate(
            self.headers.get("Authorization"), client_ip
        )
        if identity is None:
            self._json(401, protocol.jsonrpc_error(None, protocol.ERR_UNAUTHORIZED, "unauthorized"))
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                self._json(413, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "payload too large"))
                return
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "parse error"))
            return

        if not isinstance(req, dict):
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object"))
            return

        req_id = req.get("id")
        method = str(req.get("method", ""))
        params = req.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, "params must be an object"))
            return

        version = (self.headers.get("A2A-Version") or "").strip()
        if version and version not in {"1.0", "1.0.0"}:
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"))
            return

        operation, is_v1 = _method_info(method)
        route = adapter._route_for_request(self.path, params)
        if route.get("error"):
            self._json(400, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, route["error"]))
            return
        agent = route["agent"]

        if not adapter._rate_limiter.allow(identity):
            protocol.metrics.rate_limit_triggers += 1
            self._json(429, protocol.jsonrpc_error(req_id, protocol.ERR_RATE_LIMITED, "rate limit exceeded"))
            return

        if not adapter._security_context.is_trusted_peer(identity):
            self._json(403, protocol.jsonrpc_error(
                req_id, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"))
            return

        if not operation:
            self._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"))
            return

        if operation == "send":
            self._handle_send(req_id, params, identity, agent=agent, is_v1=is_v1)
            return
        if operation == "stream":
            adapter._rpc_message_stream(self, req_id, params, identity, agent=agent)
            return
        if operation == "get":
            self._json(200, adapter._rpc_tasks_get(req_id, params, agent=agent))
            return
        if operation == "list":
            self._json(200, adapter._rpc_tasks_list(req_id, params, agent=agent))
            return
        if operation == "cancel":
            self._json(200, adapter._rpc_tasks_cancel(req_id, params, agent=agent))
            return
        if operation == "subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent)
            return
        if operation == "push_create":
            self._json(200, adapter._rpc_push_config_create(req_id, params, agent=agent))
            return
        if operation == "push_get":
            self._json(200, adapter._rpc_push_config_get(req_id, params, agent=agent))
            return
        if operation == "push_list":
            self._json(200, adapter._rpc_push_config_list(req_id, params, agent=agent))
            return
        if operation == "push_delete":
            self._json(200, adapter._rpc_push_config_delete(req_id, params, agent=agent))
            return
