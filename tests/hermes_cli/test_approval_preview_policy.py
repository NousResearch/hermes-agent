"""Approval previews conceal arbitrary values while retaining operational context."""

import json


def test_nested_payload_preview_masks_values_and_unsafe_url_parts(monkeypatch):
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **kw: [{"action": "approve"}])
    seen = []
    monkeypatch.setattr("tools.approval.request_tool_approval", lambda name, reason, **kw: seen.append(reason) or {"approved": False})
    _dispatch_pre_tool_call_hooks("write_file", {
        "path": "/safe/report.txt", "action": "write", "otp": 839512,
        "nested": [{"text": "private narrative", "misc": "unlabelled credential", "otp": "743912", "password": {"target": "sensitivevalue"}}],
        "url": "https://alice:password@example.invalid/callback?code=oauth-code-example#secret",
    })
    reason = seen[0]
    for secret in ("839512", "743912", "private narrative", "unlabelled credential", "alice", "oauth-code-example", "sensitivevalue"):
        assert secret not in reason
    preview = json.loads(reason.split("(secrets redacted): ", 1)[1])
    assert preview["path"] == "/safe/report.txt"
    assert preview["action"] == "write"
    assert preview["nested"][0]["text"] == "[REDACTED]"
