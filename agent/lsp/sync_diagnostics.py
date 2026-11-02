"""Synchronous diagnostics requests, for servers that answer "what is wrong with this file now" directly.

typescript-language-server publishes diagnostics asynchronously and stays silent when a file's
diagnostics are unchanged, so waiting for a push costs the push latency (hundreds of ms after an edit)
or the whole wait budget (a clean edit never publishes).  It also forwards raw tsserver requests through
``workspace/executeCommand typescript.tsserverRequest``: ``syntacticDiagnosticsSync`` +
``semanticDiagnosticsSync`` answer for the text last synced, in about 10ms on a warm project.  The edit
check (``agent.lsp.edit_check``) asks this module first and falls back to the push/pull wait when the
server has no synchronous request or the request fails.

Returned diagnostics use the shape typescript-language-server pushes (0-based range, numeric severity,
``source`` "typescript"), so a sync verdict and a pushed baseline compare equal by
``agent.lsp.client._diagnostic_key``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agent.lsp.protocol import LSPProtocolError, LSPRequestError

logger = logging.getLogger("agent.lsp.sync_diagnostics")

# tsserver category -> LSP DiagnosticSeverity, as typescript-language-server maps it.
_TS_SEVERITY = {"error": 1, "warning": 2, "message": 3, "suggestion": 4}
_TS_SERVERS = frozenset({"typescript"})
_TS_COMMANDS = ("syntacticDiagnosticsSync", "semanticDiagnosticsSync")
_UNSUPPORTED = "_hermes_sync_diagnostics_unsupported"


def supported(client: Any) -> bool:
    return client.server_id in _TS_SERVERS and not getattr(client, _UNSUPPORTED, False)


async def request(client: Any, abs_path: str, *, timeout: float) -> Optional[list[dict[str, Any]]]:
    """Diagnostics for the text last synced to ``client`` for ``abs_path``, or None when the server has
    no synchronous request (it failed): wait for a push then.  Raises ``TimeoutError`` when the server
    did not answer within ``timeout`` (a cold project still loading): there is no verdict in budget,
    and a push wait after it would only hold the next check of the file longer."""
    if not supported(client) or timeout <= 0:
        return None
    out: list[dict[str, Any]] = []
    for command in _TS_COMMANDS:
        try:
            reply = await client._send_request_with_retry("workspace/executeCommand", {
                "command": "typescript.tsserverRequest",
                "arguments": [command, {"file": abs_path, "includeLinePosition": True}, {"executionTarget": 0}],
            }, timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"{command} did not answer in {timeout:.1f}s") from None
        except LSPRequestError as e:  # the server rejects the command: use the push wait from now on
            logger.debug("[%s] %s rejected: %s", client.server_id, command, e)
            setattr(client, _UNSUPPORTED, True)
            return None
        except LSPProtocolError as e:  # connection trouble: the push wait reports it
            logger.debug("[%s] %s failed: %s", client.server_id, command, e)
            return None
        body = reply.get("body") if isinstance(reply, dict) and reply.get("success", True) else None
        if not isinstance(body, list):
            setattr(client, _UNSUPPORTED, True)
            return None
        out.extend(d for d in (_from_tsserver(item) for item in body) if d is not None)
    return out


def _from_tsserver(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    start, end = item.get("startLocation"), item.get("endLocation")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    diag: dict[str, Any] = {
        "range": {"start": {"line": start.get("line", 1) - 1, "character": start.get("offset", 1) - 1},
                  "end": {"line": end.get("line", 1) - 1, "character": end.get("offset", 1) - 1}},
        "message": str(item.get("message") or ""),
        "severity": _TS_SEVERITY.get(str(item.get("category") or "error"), 1),
        "source": item.get("source") or "typescript",
    }
    if item.get("code") is not None:
        diag["code"] = item["code"]
    return diag


__all__ = ["request", "supported"]
