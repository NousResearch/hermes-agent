"""Hermes External Linkage WebSocket bridge for AITuberKit v2 protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import websockets.sync.server as ws_server
except ImportError:  # pragma: no cover
    ws_server = None  # type: ignore[assignment]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _hermes_python() -> str:
    return os.environ.get("AITUBER_KIT_PYTHON") or sys.executable


def _run_hermes_oneshot(prompt: str, *, timeout: int = 120) -> dict[str, Any]:
    cmd = [_hermes_python(), "-m", "hermes_cli", "--oneshot", prompt]
    env = os.environ.copy()
    env.setdefault("HERMES_YOLO_MODE", "1")
    env.setdefault("HERMES_ACCEPT_HOOKS", "1")
    env.setdefault("HERMES_QUIET", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s", "command": cmd}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "command": cmd}
    reply = (proc.stdout or "").strip()
    ok = proc.returncode == 0 and bool(reply)
    return {
        "ok": ok,
        "reply": reply,
        "stderr": (proc.stderr or "")[-2000:],
        "exit_code": proc.returncode,
        "command": cmd,
    }


def _v2_envelope(
    msg_type: str,
    *,
    msg_id: str | None = None,
    session_id: str = "hermes-bridge",
    request_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "version": "2",
        "id": msg_id or f"msg_{uuid.uuid4().hex[:12]}",
        "type": msg_type,
        "sessionId": session_id,
        "timestamp": _now_iso(),
        "payload": payload or {},
        "metadata": {},
    }
    if request_id:
        body["requestId"] = request_id
    return body


def _handle_chat_message(
    event: dict[str, Any],
    *,
    system_prompt: str,
    session_id: str,
) -> list[dict[str, Any]]:
    request_id = str(event.get("id") or event.get("requestId") or f"req_{uuid.uuid4().hex[:12]}")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    user_text = str(payload.get("text") or event.get("content") or "").strip()
    if not user_text:
        return [
            _v2_envelope(
                "chat.error",
                request_id=request_id,
                session_id=session_id,
                payload={"requestId": request_id, "message": "empty user text"},
            )
        ]

    prompt = (
        f"{system_prompt}\n\n"
        "Return only the spoken reply for the AITuber character. Do not call tools.\n\n"
        f"User input:\n{user_text}"
    )
    result = _run_hermes_oneshot(prompt)
    if not result.get("ok"):
        return [
            _v2_envelope(
                "chat.error",
                request_id=request_id,
                session_id=session_id,
                payload={
                    "requestId": request_id,
                    "message": result.get("error") or result.get("stderr") or "Hermes oneshot failed",
                },
            )
        ]

    reply = str(result.get("reply") or "")
    return [
        _v2_envelope("chat.start", request_id=request_id, session_id=session_id, payload={"requestId": request_id}),
        _v2_envelope(
            "chat.delta",
            request_id=request_id,
            session_id=session_id,
            payload={"text": reply, "requestId": request_id},
        ),
        _v2_envelope(
            "chat.done",
            request_id=request_id,
            session_id=session_id,
            payload={"requestId": request_id, "text": reply},
        ),
    ]


def _connection_handler(
    websocket: Any,
    *,
    system_prompt: str,
    session_id: str,
) -> None:
    websocket.send(
        json.dumps(
            _v2_envelope(
                "session.ready",
                session_id=session_id,
                payload={
                    "protocol": "v2",
                    "server": "hermes-aituber-kit-bridge",
                    "capabilities": ["chat.message", "ping"],
                },
            ),
            ensure_ascii=False,
        )
    )
    for raw in websocket:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        msg_type = str(event.get("type") or "")
        if msg_type == "session.hello":
            continue
        if msg_type == "ping":
            websocket.send(json.dumps(_v2_envelope("pong", session_id=session_id), ensure_ascii=False))
            continue
        if msg_type == "chat.message":
            for outbound in _handle_chat_message(event, system_prompt=system_prompt, session_id=session_id):
                websocket.send(json.dumps(outbound, ensure_ascii=False))
            continue
        if msg_type == "control.cancel":
            websocket.send(
                json.dumps(
                    _v2_envelope(
                        "chat.done",
                        request_id=str(event.get("requestId") or ""),
                        session_id=session_id,
                        payload={"cancelled": True},
                    ),
                    ensure_ascii=False,
                )
            )


def serve_forever(*, host: str, port: int, system_prompt: str, session_id: str) -> None:
    if ws_server is None:
        raise RuntimeError("websockets package is not installed.")
    _emit({"event": "bridge_starting", "host": host, "port": port})
    with ws_server.serve(
        lambda ws: _connection_handler(ws, system_prompt=system_prompt, session_id=session_id),
        host,
        port,
    ) as server:
        _emit({"event": "bridge_ready", "host": host, "port": port, "url": f"ws://{host}:{port}/ws"})
        server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes AITuberKit External Linkage bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--session-id", default="hermes-aituber-kit")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI VTuber character speaking in Japanese unless asked otherwise.",
    )
    args = parser.parse_args(argv)
    try:
        serve_forever(
            host=args.host,
            port=int(args.port),
            system_prompt=str(args.system_prompt),
            session_id=str(args.session_id),
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        _emit({"event": "bridge_error", "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
