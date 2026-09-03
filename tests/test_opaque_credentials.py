from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def test_opaque_credential_e2e_no_plaintext_context_logs_auth_and_revoke(tmp_path, monkeypatch):
    secret = "dummy-secret-never-visible-1234567890"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))

    from hermes_logging import setup_logging
    from agent.credential_store import (
        request_credential,
        resolve_credential_value,
        revoke_credential,
        set_credential_value,
    )
    from agent.redact import redact_sensitive_text
    from model_tools import handle_function_call

    log_dir = setup_logging(hermes_home=Path(os.environ["HERMES_HOME"]), force=True)

    requested = request_credential("dummy-api", "api_key")
    ref = requested["ref"]
    assert secret not in json.dumps(requested)

    stored = set_credential_value("dummy-api", "api_key", secret)
    assert stored["ref"] == ref
    assert stored["has_secret"] is True
    assert secret not in json.dumps(stored)

    # Per-profile isolation: the same opaque ref is not resolvable from another
    # Hermes home/profile.
    original_home = os.environ["HERMES_HOME"]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
    try:
        resolve_credential_value(ref, expected_type="api_key")
        raise AssertionError("credential ref resolved across profiles")
    except Exception as exc:
        assert "not found" in str(exc)
    monkeypatch.setenv("HERMES_HOME", original_home)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.headers.get("Authorization") == f"Bearer {secret}":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"authenticated")
            else:
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"unauthorized")

        def log_message(self, format, *args):  # noqa: ANN001,A002
            logging.getLogger("tests.credential_e2e.server").info(format, *args)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # Authenticated action: a tool resolves plaintext only inside execution
        # code, injects it into the HTTP header, and never returns it to model
        # context.
        import urllib.request
        from tools.registry import registry

        def _authenticated_get(args, **kwargs):  # noqa: ANN001,ARG001
            token = resolve_credential_value(args["credential_ref"], expected_type="api_key")
            req = urllib.request.Request(f"http://127.0.0.1:{port}/")
            req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310
                return json.dumps({"success": True, "status": response.status, "body": response.read().decode("utf-8")})

        registry.register(
            name="credential_e2e_authenticated_get",
            toolset="credentials",
            schema={
                "name": "credential_e2e_authenticated_get",
                "description": "test-only authenticated request",
                "parameters": {
                    "type": "object",
                    "properties": {"credential_ref": {"type": "string"}},
                    "required": ["credential_ref"],
                },
            },
            handler=_authenticated_get,
            override=True,
        )
        auth_result = handle_function_call(
            "credential_e2e_authenticated_get",
            {"credential_ref": ref},
            enabled_toolsets=["credentials"],
        )
        assert secret not in auth_result
        auth_payload = json.loads(auth_result)
        assert auth_payload == {"success": True, "status": 200, "body": "authenticated"}

        # Model-facing credential surface sees only opaque refs and public metadata.
        tool_result = handle_function_call(
            "credential",
            {"operation": "status", "credential_ref": ref},
            enabled_toolsets=["credentials"],
        )
        assert ref in tool_result
        assert secret not in tool_result

        # Simulated model context: user prompt + assistant tool request + tool result.
        model_context = json.dumps(
            [
                {"role": "user", "content": "Use dummy-api"},
                {"role": "assistant", "tool_calls": [{"name": "credential", "arguments": {"operation": "status", "credential_ref": ref}}]},
                {"role": "tool", "name": "credential", "content": tool_result},
            ],
            ensure_ascii=False,
        )
        assert secret not in model_context

        # Logging redacts registered credentials even if a buggy caller logs one.
        logging.getLogger("tests.credential_e2e").warning("leaked? %s", secret)
        from hermes_logging import flush_log_queue
        flush_log_queue()
        log_text = "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in log_dir.glob("*.log")
        )
        assert secret not in log_text
        assert "«redacted-credential»" in log_text

        # Generic redactor also catches the exact stored value in tool-like output.
        assert secret not in redact_sensitive_text(f"token={secret}", force=True)

        revoked = revoke_credential(ref)
        assert revoked["status"] == "revoked"
        revoked_result = handle_function_call(
            "credential_e2e_authenticated_get",
            {"credential_ref": ref},
            enabled_toolsets=["credentials"],
        )
        assert secret not in revoked_result
        revoked_payload = json.loads(revoked_result)
        assert "error" in revoked_payload
        assert "not active" in revoked_payload["error"]
    finally:
        server.shutdown()
        server.server_close()
