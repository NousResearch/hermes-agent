"""HTTP/2 Connect streaming client for the Cursor Agent API."""

from __future__ import annotations

import json
import ssl
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

from agent.cursor.connect_framing import (
    frame_connect_message,
    parse_connect_end_stream,
    parse_connect_frames,
)
from agent.cursor.constants import (
    CONNECT_END_STREAM_FLAG,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_API_URL,
    CURSOR_CLIENT_VERSION,
    normalize_cursor_model_id,
)
from agent.cursor.exec_handlers import CursorExecHandlers
from agent.cursor.proto import agent_pb2
from agent.cursor.request_builder import (
    build_mcp_tool_definitions,
    build_run_request_bytes,
    encode_tool_input_schema,
)


ToolHandler = Callable[[Any], Any]


def _make_tool_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments if arguments is not None else {}),
        ),
    )


def _make_usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _make_response(
    *,
    text: str,
    tool_calls: list[SimpleNamespace],
    finish_reason: str,
    usage: SimpleNamespace,
    model: str = "",
) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant",
        content=text or None,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _parse_mcp_arg_value(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.hex()


def _decode_mcp_args(mcp_args: agent_pb2.McpArgs) -> dict[str, Any]:
    return {key: _parse_mcp_arg_value(value) for key, value in dict(mcp_args.args).items()}


def _build_request_context_result(
    request_context_tools: list[agent_pb2.McpToolDefinition] | None = None,
) -> agent_pb2.RequestContextResult:
    request_context = agent_pb2.RequestContext(
        rules=[],
        repository_info=[],
        tools=request_context_tools or [],
        git_repos=[],
        project_layouts=[],
        mcp_instructions=[],
        file_contents={},
        custom_subagents=[],
    )
    return agent_pb2.RequestContextResult(
        success=agent_pb2.RequestContextSuccess(request_context=request_context)
    )


def _build_rejected_exec_message(
    exec_msg: agent_pb2.ExecServerMessage,
    request_context_tools: list[agent_pb2.McpToolDefinition] | None = None,
) -> agent_pb2.ExecClientMessage:
    message_case = exec_msg.WhichOneof("message")
    client = agent_pb2.ExecClientMessage(id=exec_msg.id, exec_id=exec_msg.exec_id)

    if message_case == "request_context_args":
        client.request_context_result.CopyFrom(
            _build_request_context_result(request_context_tools)
        )
        return client

    if message_case == "read_args":
        client.read_result.rejected.path = exec_msg.read_args.path
        client.read_result.rejected.reason = "Tool not available"
    elif message_case == "write_args":
        client.write_result.rejected.path = exec_msg.write_args.path
        client.write_result.rejected.reason = "Tool not available"
    elif message_case == "delete_args":
        client.delete_result.rejected.path = exec_msg.delete_args.path
        client.delete_result.rejected.reason = "Tool not available"
    elif message_case == "diagnostics_args":
        client.diagnostics_result.rejected.path = exec_msg.diagnostics_args.path
        client.diagnostics_result.rejected.reason = "Tool not available"
    elif message_case == "shell_args":
        client.shell_result.rejected.command = exec_msg.shell_args.command
        client.shell_result.rejected.working_directory = (
            exec_msg.shell_args.working_directory
        )
        client.shell_result.rejected.reason = "Tool not available"
        client.shell_result.rejected.is_readonly = False
    elif message_case == "background_shell_spawn_args":
        client.background_shell_spawn_result.rejected.command = (
            exec_msg.background_shell_spawn_args.command
        )
        client.background_shell_spawn_result.rejected.working_directory = (
            exec_msg.background_shell_spawn_args.working_directory
        )
        client.background_shell_spawn_result.rejected.reason = "Not implemented"
        client.background_shell_spawn_result.rejected.is_readonly = False
    elif message_case == "write_shell_stdin_args":
        client.write_shell_stdin_result.error.error = "Not implemented"
    elif message_case == "fetch_args":
        client.fetch_result.error.url = exec_msg.fetch_args.url
        client.fetch_result.error.error = "Not implemented"
    elif message_case == "mcp_args":
        client.mcp_result.rejected.reason = "Tool not available"
        client.mcp_result.rejected.is_readonly = False
    elif message_case == "ls_args":
        client.ls_result.rejected.path = exec_msg.ls_args.path
        client.ls_result.rejected.reason = "Tool not available"
    elif message_case == "grep_args":
        client.grep_result.rejected.reason = "Tool not available"
    elif message_case == "list_mcp_resources_exec_args":
        client.list_mcp_resources_exec_result.success.CopyFrom(
            agent_pb2.ListMcpResourcesSuccess(resources=[])
        )
    elif message_case == "read_mcp_resource_exec_args":
        client.read_mcp_resource_exec_result.not_found.uri = (
            exec_msg.read_mcp_resource_exec_args.uri
        )
    elif message_case == "record_screen_args":
        client.record_screen_result.failure.error = "Not implemented"
    elif message_case == "computer_use_args":
        client.computer_use_result.error.error = "Not implemented"
    else:
        return client

    return client


def _build_kv_response(
    kv_msg: agent_pb2.KvServerMessage,
    blob_store: dict[str, bytes],
) -> agent_pb2.KvClientMessage:
    client = agent_pb2.KvClientMessage(id=kv_msg.id)
    message_case = kv_msg.WhichOneof("message")

    if message_case == "get_blob_args":
        blob_id = kv_msg.get_blob_args.blob_id
        blob_data = blob_store.get(blob_id.hex())
        if blob_data is not None:
            client.get_blob_result.blob_data = blob_data
        else:
            client.get_blob_result.CopyFrom(agent_pb2.GetBlobResult())
        return client

    if message_case == "set_blob_args":
        blob_store[kv_msg.set_blob_args.blob_id.hex()] = kv_msg.set_blob_args.blob_data
        client.set_blob_result.CopyFrom(agent_pb2.SetBlobResult())
        return client

    return client


def _default_exec_handler(
    exec_msg: agent_pb2.ExecServerMessage,
    request_context_tools: list[agent_pb2.McpToolDefinition],
) -> agent_pb2.ExecClientMessage:
    client = _build_rejected_exec_message(exec_msg, request_context_tools)
    if exec_msg.WhichOneof("message") == "request_context_args":
        client.request_context_result.CopyFrom(
            _build_request_context_result(request_context_tools)
        )
    return client


def _send_shell_stream_event(
    send_agent_client_message: Callable[[agent_pb2.AgentClientMessage], None],
    exec_msg: agent_pb2.ExecServerMessage,
    *,
    stdout: str | None = None,
    stderr: str | None = None,
    exit_code: int | None = None,
    cwd: str | None = None,
    start: bool = False,
) -> None:
    event = {}
    if start:
        event["start"] = agent_pb2.ShellStreamStart()
    elif stdout is not None:
        event["stdout"] = agent_pb2.ShellStreamStdout(data=stdout)
    elif stderr is not None:
        event["stderr"] = agent_pb2.ShellStreamStderr(data=stderr)
    elif exit_code is not None:
        event["exit"] = agent_pb2.ShellStreamExit(code=exit_code, cwd=cwd or "", aborted=False)
    else:
        return
    send_agent_client_message(
        agent_pb2.AgentClientMessage(
            exec_client_message=agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                shell_stream=agent_pb2.ShellStream(**event),
            )
        )
    )


def _resolve_exec_handler(exec_handlers: Any, exec_case: str | None) -> Callable[[Any], Any] | None:
    if exec_case is None or exec_handlers is None:
        return None
    if isinstance(exec_handlers, dict):
        return exec_handlers.get(exec_case)
    method_map = {
        "read_args": "read",
        "ls_args": "ls",
        "grep_args": "grep",
        "write_args": "write",
        "delete_args": "delete",
        "shell_args": "shell",
        "shell_stream_args": "shell_stream",
        "diagnostics_args": "diagnostics",
        "mcp_args": "mcp",
    }
    method_name = method_map.get(exec_case)
    return getattr(exec_handlers, method_name, None) if method_name else None


def _encode_client_message(message: agent_pb2.AgentClientMessage) -> bytes:
    return message.SerializeToString()


def _build_exec_client_message(
    exec_msg: agent_pb2.ExecServerMessage,
    exec_case: str | None,
    exec_handlers: dict[str, ToolHandler] | CursorExecHandlers | None,
    request_context_tools: list[agent_pb2.McpToolDefinition],
    send_agent_client_message: Callable[[agent_pb2.AgentClientMessage], None],
) -> agent_pb2.ExecClientMessage:
    handler = _resolve_exec_handler(exec_handlers, exec_case)
    if handler is not None:
        if exec_case == "shell_stream_args" and not isinstance(exec_handlers, dict):
            _send_shell_stream_event(
                send_agent_client_message,
                exec_msg,
                start=True,
            )
            streamed_stdout = ""
            streamed_stderr = ""

            class _StreamCallbacks:
                def onStdout(self, data: str) -> None:
                    nonlocal streamed_stdout
                    streamed_stdout += data
                    _send_shell_stream_event(
                        send_agent_client_message,
                        exec_msg,
                        stdout=data,
                    )

                def onStderr(self, data: str) -> None:
                    nonlocal streamed_stderr
                    streamed_stderr += data
                    _send_shell_stream_event(
                        send_agent_client_message,
                        exec_msg,
                        stderr=data,
                    )

            shell_result = handler(exec_msg.shell_stream_args, _StreamCallbacks())
            stdout = None
            stderr = None
            exit_code = 0
            result_case = shell_result.WhichOneof("result")
            if result_case == "success":
                full_stdout = shell_result.success.stdout or ""
                full_stderr = shell_result.success.stderr or ""
                stdout = (
                    full_stdout[len(streamed_stdout):]
                    if full_stdout.startswith(streamed_stdout)
                    else full_stdout
                ) or None
                stderr = (
                    full_stderr[len(streamed_stderr):]
                    if full_stderr.startswith(streamed_stderr)
                    else full_stderr
                ) or None
                exit_code = shell_result.success.exit_code
                cwd = shell_result.success.working_directory
            elif result_case == "failure":
                full_stdout = shell_result.failure.stdout or ""
                full_stderr = shell_result.failure.stderr or ""
                stdout = (
                    full_stdout[len(streamed_stdout):]
                    if full_stdout.startswith(streamed_stdout)
                    else full_stdout
                ) or None
                stderr = (
                    full_stderr[len(streamed_stderr):]
                    if full_stderr.startswith(streamed_stderr)
                    else full_stderr
                ) or None
                exit_code = shell_result.failure.exit_code
                cwd = shell_result.failure.working_directory
            else:
                cwd = exec_msg.shell_stream_args.working_directory
            if stdout:
                _send_shell_stream_event(
                    send_agent_client_message,
                    exec_msg,
                    stdout=stdout,
                )
            if stderr:
                _send_shell_stream_event(
                    send_agent_client_message,
                    exec_msg,
                    stderr=stderr,
                )
            _send_shell_stream_event(
                send_agent_client_message,
                exec_msg,
                exit_code=exit_code,
                cwd=cwd,
            )
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                shell_result=shell_result,
            )

        if isinstance(exec_handlers, dict):
            response = handler(exec_msg)
        else:
            response = handler(getattr(exec_msg, exec_case))
        if isinstance(response, agent_pb2.ExecClientMessage):
            return response
        if isinstance(response, agent_pb2.ReadResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                read_result=response,
            )
        if isinstance(response, agent_pb2.WriteResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                write_result=response,
            )
        if isinstance(response, agent_pb2.DeleteResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                delete_result=response,
            )
        if isinstance(response, agent_pb2.LsResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                ls_result=response,
            )
        if isinstance(response, agent_pb2.GrepResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                grep_result=response,
            )
        if isinstance(response, agent_pb2.ShellResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                shell_result=response,
            )
        if isinstance(response, agent_pb2.DiagnosticsResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                diagnostics_result=response,
            )
        if isinstance(response, agent_pb2.McpResult):
            return agent_pb2.ExecClientMessage(
                id=exec_msg.id,
                exec_id=exec_msg.exec_id,
                mcp_result=response,
            )
    return _default_exec_handler(exec_msg, request_context_tools)


def _process_connect_payload(
    payload: bytes,
    *,
    blob_store: dict[str, bytes],
    request_context_tools: list[agent_pb2.McpToolDefinition],
    exec_handlers: dict[str, ToolHandler] | CursorExecHandlers | None,
    on_text_delta: Callable[[str], None] | None,
    send_agent_client_message: Callable[[agent_pb2.AgentClientMessage], None],
    text_parts: list[str],
    tool_calls: list[SimpleNamespace],
    current_tool_call: dict[str, Any] | None,
    saw_token_delta: bool,
    completion_tokens: int,
    finish_reason: str,
) -> tuple[dict[str, Any] | None, bool, int, str, bool]:
    server_message = agent_pb2.AgentServerMessage()
    server_message.ParseFromString(payload)
    message_case = server_message.WhichOneof("message")
    turn_ended = False

    if message_case == "interaction_update":
        update = server_message.interaction_update
        update_case = update.WhichOneof("message")

        if update_case == "text_delta":
            delta = update.text_delta.text
            text_parts.append(delta)
            if on_text_delta:
                on_text_delta(delta)
        elif update_case == "thinking_delta":
            delta = update.thinking_delta.text
            text_parts.append(delta)
            if on_text_delta:
                on_text_delta(delta)
        elif update_case == "tool_call_started":
            started = update.tool_call_started
            tool_call = started.tool_call
            tool_case = tool_call.WhichOneof("tool")
            if tool_case == "mcp_tool_call":
                args = tool_call.mcp_tool_call.args
                current_tool_call = {
                    "id": args.tool_call_id or started.call_id or str(uuid.uuid4()),
                    "name": args.tool_name or args.name,
                    "partial_json": "",
                    "arguments": {},
                }
            elif tool_case == "update_todos_tool_call":
                todos = []
                for todo in tool_call.update_todos_tool_call.args.todos:
                    todos.append(
                        {
                            "id": todo.id or None,
                            "content": todo.content,
                            "activeForm": todo.content,
                            "status": (
                                "in_progress"
                                if todo.status == 2
                                else "completed"
                                if todo.status == 3
                                else "pending"
                            ),
                        }
                    )
                tool_calls.append(
                    _make_tool_call(
                        started.call_id or str(uuid.uuid4()),
                        "todo",
                        {"todos": todos},
                    )
                )
        elif update_case in {"partial_tool_call", "tool_call_delta"}:
            if current_tool_call is not None:
                if update_case == "partial_tool_call":
                    delta = update.partial_tool_call.args_text_delta
                else:
                    delta = (
                        update.tool_call_delta.tool_call_delta.edit_tool_call_delta.stream_content_delta
                    )
                current_tool_call["partial_json"] += delta
                try:
                    current_tool_call["arguments"] = json.loads(current_tool_call["partial_json"])
                except Exception:
                    pass
        elif update_case == "tool_call_completed":
            completed = update.tool_call_completed
            tool_call = completed.tool_call
            if current_tool_call is not None and tool_call.WhichOneof("tool") == "mcp_tool_call":
                args = _decode_mcp_args(tool_call.mcp_tool_call.args)
                if args:
                    current_tool_call["arguments"] = args
                tool_calls.append(
                    _make_tool_call(
                        current_tool_call["id"],
                        current_tool_call["name"],
                        current_tool_call["arguments"],
                    )
                )
                current_tool_call = None
        elif update_case == "token_delta":
            saw_token_delta = True
            completion_tokens += update.token_delta.tokens
        elif update_case in {"turn_ended", "step_completed"}:
            finish_reason = "tool_calls" if tool_calls else "stop"
            turn_ended = True
    elif message_case == "conversation_checkpoint_update":
        if not saw_token_delta:
            completion_tokens = (
                server_message.conversation_checkpoint_update.token_details.used_tokens
            )
    elif message_case == "kv_server_message":
        kv_client = _build_kv_response(server_message.kv_server_message, blob_store)
        send_agent_client_message(agent_pb2.AgentClientMessage(kv_client_message=kv_client))
    elif message_case == "exec_server_message":
        exec_msg = server_message.exec_server_message
        exec_case = exec_msg.WhichOneof("message")
        exec_client = _build_exec_client_message(
            exec_msg,
            exec_case,
            exec_handlers,
            request_context_tools,
            send_agent_client_message,
        )
        send_agent_client_message(
            agent_pb2.AgentClientMessage(exec_client_message=exec_client)
        )

    return current_tool_call, saw_token_delta, completion_tokens, finish_reason, turn_ended


def _build_stream_state() -> dict[str, Any]:
    return {
        "text_parts": [],
        "tool_calls": [],
        "current_tool_call": None,
        "saw_token_delta": False,
        "completion_tokens": 0,
        "finish_reason": "stop",
        "turn_completed": False,
    }


def _apply_connect_payload(
    payload: bytes,
    *,
    state: dict[str, Any],
    blob_store: dict[str, bytes],
    request_context_tools: list[agent_pb2.McpToolDefinition],
    exec_handlers: dict[str, ToolHandler] | CursorExecHandlers | None,
    on_text_delta: Callable[[str], None] | None,
    send_agent_client_message: Callable[[agent_pb2.AgentClientMessage], None],
) -> bool:
    (
        state["current_tool_call"],
        state["saw_token_delta"],
        state["completion_tokens"],
        state["finish_reason"],
        turn_ended,
    ) = _process_connect_payload(
        payload,
        blob_store=blob_store,
        request_context_tools=request_context_tools,
        exec_handlers=exec_handlers,
        on_text_delta=on_text_delta,
        send_agent_client_message=send_agent_client_message,
        text_parts=state["text_parts"],
        tool_calls=state["tool_calls"],
        current_tool_call=state["current_tool_call"],
        saw_token_delta=state["saw_token_delta"],
        completion_tokens=state["completion_tokens"],
        finish_reason=state["finish_reason"],
    )
    if turn_ended:
        state["turn_completed"] = True
    return turn_ended


def _response_from_stream_state(state: dict[str, Any]) -> SimpleNamespace:
    return _make_response(
        text="".join(state["text_parts"]),
        tool_calls=state["tool_calls"],
        finish_reason=state["finish_reason"],
        usage=_make_usage(completion_tokens=state["completion_tokens"]),
    )


def _consume_connect_stream(
    frames: Iterable[bytes],
    *,
    blob_store: dict[str, bytes],
    request_context_tools: list[agent_pb2.McpToolDefinition],
    exec_handlers: dict[str, ToolHandler] | CursorExecHandlers | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    send_client_frame: Callable[[bytes], None] | None = None,
    interrupt_event: threading.Event | None = None,
) -> SimpleNamespace:
    state = _build_stream_state()

    def send_agent_client_message(client_message: agent_pb2.AgentClientMessage) -> None:
        if send_client_frame is None:
            return
        send_client_frame(frame_connect_message(_encode_client_message(client_message)))

    for frame_bytes in frames:
        if interrupt_event is not None and interrupt_event.is_set():
            raise InterruptedError("Request was aborted")

        for flags, payload in parse_connect_frames(bytearray(frame_bytes)):
            if flags & CONNECT_END_STREAM_FLAG:
                error = parse_connect_end_stream(payload)
                if error and not state.get("turn_completed"):
                    raise error
                continue
            turn_ended = _apply_connect_payload(
                payload,
                state=state,
                blob_store=blob_store,
                request_context_tools=request_context_tools,
                exec_handlers=exec_handlers,
                on_text_delta=on_text_delta,
                send_agent_client_message=send_agent_client_message,
            )
            if turn_ended:
                break

    return _response_from_stream_state(state)


def run_cursor_agent_turn(
    *,
    api_key: str,
    model_id: str,
    messages: list[dict],
    system_prompt: list[str] | str | None = None,
    tools: list[dict] | None = None,
    conversation_id: str,
    blob_store: dict[str, bytes] | None = None,
    conversation_state: agent_pb2.ConversationStateStructure | None = None,
    exec_handlers: dict[str, ToolHandler] | CursorExecHandlers | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    signal: threading.Event | None = None,
    interrupt_event: threading.Event | None = None,
    base_url: str = CURSOR_API_URL,
    custom_system_prompt: str | None = None,
) -> SimpleNamespace:
    """Run one Cursor Agent API turn over HTTP/2 Connect."""

    if not api_key:
        raise ValueError("Cursor API key (access token) is required")

    if blob_store is None:
        blob_store = {}

    model_id = normalize_cursor_model_id(model_id)

    request_bytes, next_conversation_state, blob_store = build_run_request_bytes(
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        model_id=model_id,
        conversation_id=conversation_id,
        blob_store=blob_store,
        conversation_state=conversation_state,
        custom_system_prompt=custom_system_prompt,
    )

    request_context_tools = []
    for tool_def in build_mcp_tool_definitions(tools):
        request_context_tools.append(
            agent_pb2.McpToolDefinition(
                name=tool_def["name"],
                description=tool_def["description"],
                provider_identifier=tool_def["providerIdentifier"],
                tool_name=tool_def["toolName"],
                input_schema=encode_tool_input_schema(tool_def["inputSchema"]),
            )
        )

    parsed_url = __import__("urllib.parse").parse.urlsplit(base_url)
    try:
        import h2.connection
        import h2.events
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The 'h2' package is required for the Cursor streaming transport. "
            "Install it with: pip install -e '.[cursor]'"
        ) from exc

    host = parsed_url.hostname or "api2.cursor.sh"
    port = parsed_url.port or 443

    connection = h2.connection.H2Connection()
    stream_ended = False
    response_status: str | None = None
    grpc_status: str | None = None
    grpc_message: str | None = None
    send_lock = threading.Lock()

    sock = None
    ssl_sock = None
    heartbeat_stop = threading.Event()

    def send_data(data: bytes) -> None:
        nonlocal ssl_sock
        if ssl_sock is None:
            raise RuntimeError("HTTP/2 stream is not connected")
        ssl_sock.sendall(data)

    def send_client_frame(frame: bytes) -> None:
        with send_lock:
            connection.send_data(stream_id, frame)
            pending = connection.data_to_send()
        if pending:
            send_data(pending)

    def heartbeat_loop() -> None:
        while not heartbeat_stop.wait(5.0):
            if signal is not None and signal.is_set():
                return
            heartbeat = agent_pb2.AgentClientMessage(
                client_heartbeat=agent_pb2.ClientHeartbeat()
            )
            try:
                send_client_frame(frame_connect_message(_encode_client_message(heartbeat)))
            except Exception:
                return

    try:
        sock = __import__("socket").create_connection((host, port))
        context = ssl.create_default_context()
        context.set_alpn_protocols(["h2"])
        ssl_sock = context.wrap_socket(sock, server_hostname=host)
        if ssl_sock.selected_alpn_protocol() != "h2":
            raise RuntimeError(
                "Cursor Agent API requires HTTP/2, but TLS ALPN did not negotiate h2"
            )

        connection.initiate_connection()
        with send_lock:
            pending = connection.data_to_send()
        if pending:
            send_data(pending)

        stream_id = connection.get_next_available_stream_id()
        headers = [
            (":method", "POST"),
            (":path", CURSOR_AGENT_RUN_PATH),
            (":scheme", "https"),
            (":authority", host),
            ("authorization", f"Bearer {api_key}"),
            ("content-type", "application/connect+proto"),
            ("connect-protocol-version", "1"),
            ("te", "trailers"),
            ("x-ghost-mode", "true"),
            ("x-cursor-client-version", CURSOR_CLIENT_VERSION),
            ("x-cursor-client-type", "cli"),
            ("x-request-id", str(uuid.uuid4())),
        ]
        with send_lock:
            connection.send_headers(stream_id, headers, end_stream=False)
            connection.send_data(stream_id, frame_connect_message(request_bytes), end_stream=False)
            pending = connection.data_to_send()
        if pending:
            send_data(pending)

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        read_buffer = bytearray()
        stream_state = _build_stream_state()

        def live_send_agent_client_message(client_message: agent_pb2.AgentClientMessage) -> None:
            send_client_frame(frame_connect_message(_encode_client_message(client_message)))

        while not stream_ended:
            if signal is not None and signal.is_set():
                raise InterruptedError("Request was aborted")
            if interrupt_event is not None and interrupt_event.is_set():
                raise InterruptedError("Request was aborted")

            data = ssl_sock.recv(65535)
            if not data:
                break

            events = connection.receive_data(data)
            with send_lock:
                pending = connection.data_to_send()
            if pending:
                send_data(pending)

            for event in events:
                if isinstance(event, h2.events.ResponseReceived):
                    response_status = str(dict(event.headers).get(":status", ""))
                elif isinstance(event, h2.events.TrailersReceived):
                    trailers = dict(event.headers)
                    grpc_status = trailers.get("grpc-status")
                    grpc_message = trailers.get("grpc-message")
                elif isinstance(event, h2.events.DataReceived):
                    with send_lock:
                        connection.acknowledge_received_data(
                            event.flow_controlled_length, event.stream_id
                        )
                        pending = connection.data_to_send()
                    if pending:
                        send_data(pending)
                    read_buffer.extend(event.data)
                    turn_ended = False
                    for flags, payload in parse_connect_frames(read_buffer):
                        if flags & CONNECT_END_STREAM_FLAG:
                            error = parse_connect_end_stream(payload)
                            if error and not (
                                stream_ended or stream_state.get("turn_completed")
                            ):
                                raise error
                            continue
                        turn_ended = _apply_connect_payload(
                            payload,
                            state=stream_state,
                            blob_store=blob_store,
                            request_context_tools=request_context_tools,
                            exec_handlers=exec_handlers,
                            on_text_delta=on_text_delta,
                            send_agent_client_message=live_send_agent_client_message,
                        )
                        if turn_ended:
                            stream_ended = True
                            break
                elif isinstance(event, h2.events.StreamEnded):
                    stream_ended = True

        if response_status and response_status != "200":
            raise RuntimeError(f"Cursor request failed with HTTP {response_status}")
        if grpc_status and grpc_status != "0":
            detail = __import__("urllib.parse").parse.unquote(grpc_message or "")
            raise RuntimeError(f"gRPC error {grpc_status}: {detail}")

        response = _response_from_stream_state(stream_state)
        response.conversation_state = next_conversation_state
        response.blob_store = blob_store
        return response
    finally:
        heartbeat_stop.set()
        try:
            if ssl_sock is not None:
                ssl_sock.close()
        finally:
            if sock is not None:
                sock.close()


class CursorStreamClient:
    """Small OO wrapper around ``run_cursor_agent_turn``."""

    def __init__(self, api_key: str, base_url: str = CURSOR_API_URL):
        self.api_key = api_key
        self.base_url = base_url

    def run_turn(self, **kwargs) -> SimpleNamespace:
        return run_cursor_agent_turn(api_key=self.api_key, base_url=self.base_url, **kwargs)

