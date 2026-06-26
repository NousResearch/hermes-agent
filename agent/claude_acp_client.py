"""Streaming ACP client for Claude Code (`claude-code-acp`).

This is an OpenAI-client-compatible facade (like ``copilot_acp_client``) but it
is STREAMING and render-oriented:

* It drives ``claude-code-acp`` over the Agent Client Protocol (JSON-RPC/stdio).
* Claude Code executes its OWN MCP tools (Gmail/Calendar/Trello, passed in
  ``session/new``) — reliable, no tool-stripping fight. We auto-approve its
  permission requests so it can act.
* It streams ACP ``session/update`` events live as OpenAI chunks:
    - ``agent_thought_chunk``                 -> ``delta.reasoning_content``
    - ``tool_call`` / ``tool_call_update``    -> ``delta.reasoning_content`` (live tool trace)
    - ``agent_message_chunk``                 -> ``delta.content`` (the answer)
  We deliberately do NOT emit ``delta.tool_calls`` — Claude already ran the
  tools, so the harness must render (not re-execute) them. Thinking + tool
  traces land in the separate reasoning channel; only the answer is content.

Wired in ``agent_runtime_helpers.create_openai_client`` for provider
``claude-acp`` (base_url ``acp://claude``). New file → no upstream merge
conflict; the only core edit is the 3-line dispatch branch.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

ACP_MARKER_BASE_URL = "acp://claude"
_DEFAULT_TIMEOUT_SECONDS = 1800.0

_MCP_LABELS = {
    "mcpc-gmail": "📨 Email",
    "mcpc-drive": "📂 Drive",
    "mcpc-sheets": "📊 Sheets",
    "mcpc-gcal": "📅 Calendar",
    "mcpc-trello": "📋 Trello",
}


def _resolve_command() -> str:
    return os.getenv("HERMES_CLAUDE_ACP_COMMAND", "").strip() or "claude-code-acp"


def _tool_label(title: str | None, kind: str | None) -> str:
    t = title or "tool"
    if t.startswith("mcp__"):
        parts = t.split("__")
        server = parts[1] if len(parts) > 1 else ""
        tool = parts[2] if len(parts) > 2 else t
        base = _MCP_LABELS.get(server)
        return f"{base}: {tool}" if base else f"🔧 {tool}"
    return f"🔧 {t}"


def _load_mcp_servers() -> list[dict[str, Any]]:
    """Read the proxy's MCP config (the mcpc servers) into ACP session/new shape."""
    path = os.getenv("CLAUDE_MCP_CONFIG", "").strip()
    if not path:
        # Fall back to the known mcpc server config shipped with the proxy.
        default = Path(os.path.expanduser("~/.hermes/claude-proxy-mcp.json"))
        path = str(default) if default.exists() else ""
    if not path or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for name, spec in (data.get("mcpServers") or {}).items():
        if not isinstance(spec, dict) or not spec.get("command"):
            continue
        out.append({
            "name": name,
            "command": spec["command"],
            "args": list(spec.get("args") or []),
            "env": [{"name": k, "value": str(v)} for k, v in (spec.get("env") or {}).items()],
        })
    return out


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten an OpenAI message list into a single ACP prompt string.

    Claude's tools come from the MCP servers (not the request `tools`), so the
    prompt is just the conversation — no tool-definition bloat.
    """
    parts: list[str] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            text = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") in ("text", "input_text")
            )
        else:
            text = "" if content is None else str(content)
        if role == "system":
            parts.append(f"<system>\n{text}\n</system>")
        elif role == "user":
            parts.append(text)
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                calls = "; ".join(
                    f"{tc.get('function', {}).get('name')}({tc.get('function', {}).get('arguments')})"
                    for tc in tcs
                )
                parts.append(f"<previous_action>{text} [called: {calls}]</previous_action>")
            else:
                parts.append(f"<previous_response>\n{text}\n</previous_response>")
        elif role == "tool":
            parts.append(f"<tool_result>\n{text}\n</tool_result>")
    return "\n\n".join(p for p in parts if p).strip()


def _chunk(model: str, *, content: str | None = None, reasoning: str | None = None,
           finish: str | None = None, usage: Any = None) -> SimpleNamespace:
    """Build an OpenAI-style streaming chunk the chat_completions transport reads."""
    if usage is not None:
        return SimpleNamespace(choices=[], model=model, usage=usage)
    delta = SimpleNamespace(content=content, reasoning_content=reasoning,
                            reasoning=reasoning, tool_calls=None, role="assistant")
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish)],
        model=model,
        usage=None,
    )


class _ACPChatCompletions:
    def __init__(self, client: "ClaudeACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ClaudeACPClient"):
        self.completions = _ACPChatCompletions(client)


class ClaudeACPClient:
    """Minimal OpenAI-client-compatible STREAMING facade for Claude Code ACP."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 acp_cwd: str | None = None, **_: Any):
        self.api_key = api_key or "claude-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._cmd = _resolve_command()
        self._mcp = _load_mcp_servers()
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    # ---- OpenAI-compatible entrypoint --------------------------------------
    def _create_chat_completion(self, *, model: str | None = None,
                                messages: list[dict[str, Any]] | None = None,
                                stream: bool = False, timeout: Any = None,
                                **_: Any) -> Any:
        mdl = model or "claude-acp"
        prompt = _flatten_messages(messages or [])
        gen = self._run(prompt, mdl, self._timeout(timeout))
        if stream:
            return gen
        # Non-streaming: drain the generator into a single response object.
        content, reasoning = [], []
        for ch in gen:
            if not ch.choices:
                continue
            d = ch.choices[0].delta
            if getattr(d, "content", None):
                content.append(d.content)
            if getattr(d, "reasoning_content", None):
                reasoning.append(d.reasoning_content)
        msg = SimpleNamespace(content="".join(content), tool_calls=None,
                              reasoning="".join(reasoning) or None,
                              reasoning_content="".join(reasoning) or None,
                              reasoning_details=None, role="assistant")
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(
            choices=[SimpleNamespace(index=0, message=msg, finish_reason="stop")],
            model=mdl, usage=usage)

    @staticmethod
    def _timeout(timeout: Any) -> float:
        if isinstance(timeout, (int, float)):
            return float(timeout)
        if timeout is not None:
            cands = [getattr(timeout, a, None) for a in ("read", "timeout", "pool", "connect")]
            nums = [float(v) for v in cands if isinstance(v, (int, float))]
            if nums:
                return max(nums)
        return _DEFAULT_TIMEOUT_SECONDS

    # ---- ACP session + live streaming --------------------------------------
    def _run(self, prompt: str, model: str, timeout_s: float) -> Iterator[SimpleNamespace]:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        # claude-code-acp (and the `claude` CLI it spawns) live in linuxbrew /
        # ~/.local/bin — make sure they're findable regardless of caller PATH.
        extra = ["/home/linuxbrew/.linuxbrew/bin", os.path.expanduser("~/.local/bin")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        try:
            proc = subprocess.Popen(
                [self._cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env, cwd=self._cwd,
            )
        except FileNotFoundError:
            yield _chunk(model, content=f"(claude-acp: command '{self._cmd}' not found)", finish="stop")
            yield _chunk(model, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0,
                         total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)))
            return
        handshake_timeout = min(timeout_s, 45.0)
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        resp: dict[int, queue.Queue] = {}
        nid = [0]
        lock = threading.Lock()

        def send(obj: dict) -> None:
            with lock:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.write(json.dumps(obj) + "\n")
                    proc.stdin.flush()

        def request(method: str, params: dict, t: float | None = None) -> dict:
            nid[0] += 1
            rid = nid[0]
            q: queue.Queue = queue.Queue()
            resp[rid] = q
            send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            return q.get(timeout=t if t is not None else timeout_s)

        def reader() -> None:
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except Exception:
                        continue
                    # response to one of our requests
                    if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
                        q = resp.pop(msg["id"], None)
                        if q:
                            q.put(msg)
                        continue
                    method = msg.get("method")
                    # agent -> client request (needs a reply)
                    if method and "id" in msg:
                        if method == "session/request_permission":
                            opts = (msg.get("params") or {}).get("options") or []
                            pick = next((o.get("optionId") for o in opts
                                         if "allow" in (o.get("optionId", "") + o.get("name", "")).lower()),
                                        opts[0].get("optionId") if opts else None)
                            send({"jsonrpc": "2.0", "id": msg["id"],
                                  "result": {"outcome": {"outcome": "selected", "optionId": pick}}})
                        elif method == "fs/read_text_file":
                            p = (msg.get("params") or {}).get("path")
                            try:
                                send({"jsonrpc": "2.0", "id": msg["id"],
                                      "result": {"content": Path(p).read_text()}})
                            except Exception as e:
                                send({"jsonrpc": "2.0", "id": msg["id"],
                                      "error": {"code": -32000, "message": str(e)}})
                        else:
                            send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
                        continue
                    # agent -> client notification
                    if method == "session/update":
                        events.put(("update", (msg.get("params") or {}).get("update") or {}))
            except Exception:
                pass
            finally:
                events.put(("eof", None))

        threading.Thread(target=reader, daemon=True).start()

        try:
            try:
                request("initialize", {"protocolVersion": 1,
                        "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": False}}},
                        t=handshake_timeout)
                sn = request("session/new", {"cwd": self._cwd, "mcpServers": self._mcp},
                             t=handshake_timeout)
            except queue.Empty:
                yield _chunk(model, content="(claude-acp: agent did not respond to handshake)", finish="stop")
                yield _chunk(model, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0,
                             total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)))
                return
            sid = (sn.get("result") or {}).get("sessionId")
            if not sid:
                yield _chunk(model, content="(claude-acp: failed to start session)", finish="stop")
                yield _chunk(model, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0,
                             total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)))
                return

            # Fire the prompt on a side thread; it signals completion by pushing
            # a ("done", result) sentinel onto the SAME event queue — which the
            # agent only sends after all session/update notifications, so the
            # loop drains every update first, then cleanly finishes.
            def run_prompt() -> None:
                try:
                    r = request("session/prompt", {"sessionId": sid,
                                "prompt": [{"type": "text", "text": prompt}]})
                    events.put(("done", r))
                except Exception as e:
                    events.put(("done", {"error": str(e)}))

            threading.Thread(target=run_prompt, daemon=True).start()

            # Buffer assistant message text. Text emitted BEFORE a tool call is
            # narration → reasoning channel; the final text (after the last tool)
            # is the answer → content. Real "thinking" (thought chunks) and tool
            # traces always go to the reasoning channel.
            seen_tool: set[str] = set()
            msg_buf = ""
            while True:
                kind, payload = events.get(timeout=timeout_s)
                if kind in ("eof", "done"):
                    break
                u = payload
                su = str(u.get("sessionUpdate") or "")
                if su == "agent_message_chunk":
                    msg_buf += _text_of(u.get("content"))
                elif su == "agent_thought_chunk":
                    txt = _text_of(u.get("content"))
                    if txt:
                        yield _chunk(model, reasoning=txt)
                elif su == "tool_call":
                    tcid = str(u.get("toolCallId") or "")
                    if tcid not in seen_tool:
                        seen_tool.add(tcid)
                        if msg_buf.strip():   # narration before this tool
                            yield _chunk(model, reasoning=msg_buf.strip() + "\n")
                            msg_buf = ""
                        yield _chunk(model, reasoning=f"› {_tool_label(u.get('title'), u.get('kind'))}…\n")
                elif su == "tool_call_update" and u.get("status") == "completed":
                    yield _chunk(model, reasoning="  ✓\n")
                # plan / other updates ignored for now

            # Whatever message text remains after the last tool is the answer.
            if msg_buf.strip():
                yield _chunk(model, content=msg_buf.strip())

            yield _chunk(model, finish="stop")
            yield _chunk(model, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0,
                         total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)))
        finally:
            try:
                proc.terminate()
            except Exception:
                pass


def _text_of(content: Any) -> str:
    if isinstance(content, dict):
        # ACP content block: {type:"text", text:"..."} or {type:"content", content:{...}}
        if content.get("type") == "text":
            return str(content.get("text") or "")
        inner = content.get("content")
        if isinstance(inner, dict):
            return str(inner.get("text") or "")
    if isinstance(content, str):
        return content
    return ""
