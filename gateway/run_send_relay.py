"""Local control-socket delivery through the profile's live gateway adapter."""
from pathlib import Path
import json


def send_message_verb(runner):
    """Socket permissions provide local-user authorization; homes must be served."""
    def handler(params: dict) -> dict:
        from gateway.config import Platform
        from gateway.run import _profile_runtime_scope
        from tools.send_message_tool import send_message_tool

        target, message, home = params.get("target"), params.get("message"), params.get("home")
        if (not isinstance(target, str) or not target.lower().startswith("matrix:")
                or not isinstance(message, str) or not message.strip() or not isinstance(home, str)):
            return {"error": "send-message requires a Matrix target, message and profile home"}
        requested = Path(home)
        profile = runner._served_profile_for_home(requested)
        if profile is None:
            return {"error": "Profile home is not served by this gateway"}
        with _profile_runtime_scope(requested):
            if runner._authorization_adapter(Platform.MATRIX, profile) is None:
                return {"error": "Gateway Matrix adapter is not connected"}
            args = {"action": "send", "target": target, "message": message}
            if "mentions" in params:
                args["mentions"] = params["mentions"]
            result = send_message_tool(args)
            return json.loads(result) if isinstance(result, str) else result
    return handler


def matrix_send_verb(runner):
    """Deliver lower-level text/media calls without reparsing their payload as tool syntax."""
    def handler(params: dict) -> dict:
        import asyncio
        from concurrent.futures import TimeoutError
        from gateway.config import Platform
        from gateway.run import _profile_runtime_scope
        from tools.send_message_senders import _matrix_send_core

        home, chat, message = params.get("home"), params.get("chat_id"), params.get("message")
        media, thread = params.get("media_files", []), params.get("thread_id")
        force_document = params.get("force_document", False)
        if (not isinstance(home, str) or not isinstance(chat, str) or not chat.strip()
                or not isinstance(message, str) or not isinstance(media, list)
                or any(not isinstance(item, (list, tuple)) or len(item) != 2
                       or not isinstance(item[0], str) or not item[0]
                       or not isinstance(item[1], bool) for item in media)
                or (thread is not None and not isinstance(thread, str))
                or not isinstance(force_document, bool) or not (message.strip() or media)):
            return {"error": "Invalid Matrix relay payload"}
        requested = Path(home)
        profile = runner._served_profile_for_home(requested)
        if profile is None:
            return {"error": "Profile home is not served by this gateway"}
        with _profile_runtime_scope(requested):
            adapter = runner._authorization_adapter(Platform.MATRIX, profile)
            if adapter is None:
                return {"error": "Gateway Matrix adapter is not connected"}
            metadata = {"thread_id": thread} if thread else None
            future = asyncio.run_coroutine_threadsafe(
                _matrix_send_core(adapter, chat, message, media, metadata, force_document=force_document),
                runner._gateway_loop,
            )
            try:
                return future.result(timeout=55)
            except TimeoutError:
                return {"error": "Gateway delivery outcome unknown; not retrying to avoid duplicates"}
    return handler
