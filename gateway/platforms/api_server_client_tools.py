"""Single-turn inference for client-owned Chat Completions functions.

This deliberately never constructs an AIAgent: a function named ``terminal`` in
an API request belongs to the caller, not Hermes's local tool registry. History
is supplied by the client, including assistant tool_calls and role=tool results.
"""
import asyncio
import time
import uuid

from aiohttp import web


def _codex_completion(client, model, kwargs):
    from agent.auxiliary_client import _CodexCompletionsAdapter

    class ClientFunctionsAdapter(_CodexCompletionsAdapter):
        def _build_responses_kwargs(self, kwargs):
            payload, model_name, timeout = super()._build_responses_kwargs(kwargs)
            choice = kwargs.get("tool_choice", "auto")
            if isinstance(choice, dict):
                name = choice["function"]["name"]
                aliases = payload.get("_wire_aliases", {})
                name = next((alias for alias, original in aliases.items() if original == name), name)
                choice = {"type": "function", "name": name}
            payload["tool_choice"] = choice
            if "parallel_tool_calls" in kwargs:
                payload["parallel_tool_calls"] = kwargs["parallel_tool_calls"]
            return payload, model_name, timeout

    return ClientFunctionsAdapter(client, model).create(**kwargs)


def _complete(adapter, body, route, overrides):
    from agent.auxiliary_client import _create_openai_client
    from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs

    runtime = dict(_resolve_runtime_agent_kwargs())
    model = runtime.pop("model", None) or _resolve_gateway_model()
    model, _, _, _ = adapter._select_agent_runtime(
        runtime, model, requested_model=overrides.get("requested_model"),
        requested_provider=overrides.get("requested_provider"), route=route,
        session_model=None, confirmed_runtime_lock=False,
        gateway_session_key=None, session_id=None)
    if runtime.get("api_mode") not in (None, "chat_completions", "codex_responses"):
        raise ValueError("Client function calling is not supported by this provider transport")
    client = _create_openai_client(api_key=runtime["api_key"], base_url=runtime["base_url"])
    try:
        kwargs = dict(model=model, messages=body["messages"], tools=body.get("tools", []),
                      tool_choice=body.get("tool_choice", "auto"), stream=False)
        if "parallel_tool_calls" in body:
            kwargs["parallel_tool_calls"] = body["parallel_tool_calls"]
        if runtime.get("api_mode") == "codex_responses":
            response = _codex_completion(client, model, kwargs)
        else:
            response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = {"role": "assistant", "content": choice.message.content}
        if choice.message.tool_calls:
            message["tool_calls"] = [{"id": tc.id, "type": "function", "function": {
                "name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.message.tool_calls]
        usage = response.usage
        return {"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else choice.finish_reason}, {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0}
    finally:
        client.close()


def _validate(body):
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        return "'tools' must be an array of function definitions"
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return "Only client function tools are supported"
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            return "Each function must have a nonempty name"
        names.add(function["name"])
    choice = body.get("tool_choice", "auto")
    if isinstance(choice, dict):
        function = choice.get("function")
        if (choice.get("type") != "function" or not isinstance(function, dict)
                or not isinstance(function.get("name"), str) or function["name"] not in names):
            return "'tool_choice' must select a declared function"
    elif choice not in ("auto", "none", "required"):
        return "Invalid 'tool_choice'"
    if choice == "required" and not names:
        return "'tool_choice=required' needs at least one function"
    if any(not isinstance(message, dict) for message in body["messages"]):
        return "Each message must be an object"
    if "parallel_tool_calls" in body and not isinstance(body["parallel_tool_calls"], bool):
        return "'parallel_tool_calls' must be a boolean"
    if body.get("n", 1) != 1:
        return "Client function calling currently supports only n=1"
    if body.get("stream_options") is not None and not isinstance(body["stream_options"], dict):
        return "'stream_options' must be an object"
    return None


async def handle_client_tools(adapter, request, body):
    from gateway.platforms.api_server import _invalid_request
    validation_error = _validate(body)
    if validation_error:
        return _invalid_request(validation_error)
    route, overrides, error = adapter._select_request_route(
        body, session_id=None, gateway_session_key=None,
        model_alias=body.get("model", adapter._model_name))
    if error is not None:
        return error

    async def compute():
        adapter._inflight_agent_runs += 1
        adapter._activate_admitted_request()
        task = asyncio.create_task(asyncio.to_thread(_complete, adapter, body, route, overrides))
        adapter._background_tasks.add(task)

        def finished(future):
            adapter._inflight_agent_runs -= 1
            adapter._background_tasks.discard(future)
            if not future.cancelled():
                future.exception()  # Consume errors if the HTTP caller disconnected.

        task.add_done_callback(finished)
        # Cancellation of the HTTP request cannot stop a synchronous SDK worker.
        # Keep its concurrency slot until it really exits and closes the client.
        return await asyncio.shield(task)

    outcome, error = await adapter._run_idempotent(
        request, body, compute, log_label="client function completion",
        fingerprint_keys=list(body), route="client_function_completion")
    if error is not None:
        return error
    choice, usage = outcome
    envelope = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "object": "chat.completion",
        "created": int(time.time()), "model": body.get("model", adapter._model_name)}
    from gateway.platforms.api_server import _coerce_request_bool, _sse_frame
    if not _coerce_request_bool(body.get("stream"), default=False):
        return web.json_response({**envelope, "choices": [{"index": 0, **choice}], "usage": usage})

    # The shared Responses adapter assembles the complete function arguments before
    # returning. Emit one complete delta per call (valid SSE), never run the tools.
    response = await adapter._prepare_sse_response(request, None, None)
    envelope["object"] = "chat.completion.chunk"

    async def emit(delta, finish_reason=None):
        await response.write(_sse_frame({**envelope, "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}]}))

    try:
        await emit({"role": "assistant"})
        message = choice["message"]
        if message.get("content"):
            await emit({"content": message["content"]})
        for index, call in enumerate(message.get("tool_calls", [])):
            await emit({"tool_calls": [{"index": index, **call}]})
        await emit({}, choice["finish_reason"])
        if (body.get("stream_options") or {}).get("include_usage"):
            await response.write(_sse_frame({**envelope, "choices": [], "usage": usage}))
        await response.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, BrokenPipeError):
        pass  # Provider call has already completed and its client is closed.
    return response
