import re

with open("acp_adapter/server.py", "r") as f:
    content = f.read()

new_methods = """
    async def _send_history_update(self, session_id: str, update: Any) -> bool:
        \"\"\"Send an ACP history update to the client.\"\"\"
        if not self._conn:
            return False
        try:
            await self._conn.session_update(session_id=session_id, update=update)
            return True
        except Exception:
            logger.warning(
                "Failed to replay ACP history for session %s",
                session_id,
                exc_info=True,
            )
            return False

    async def _replay_message_text(self, message: dict[str, Any], role: str, session_id: str) -> bool:
        \"\"\"Replay text chunk for a user or assistant history message.\"\"\"
        text = self._history_message_text(message)
        if text:
            update = self._history_message_update(role=role, text=text)
            if update is not None and not await self._send_history_update(session_id, update):
                return False
        return True

    async def _replay_assistant_tools(
        self,
        message: dict[str, Any],
        session_id: str,
        active_tool_calls: dict[str, tuple[str, dict[str, Any]]],
    ) -> bool:
        \"\"\"Replay tool start updates from an assistant history message.\"\"\"
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return True

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = self._history_tool_call_id(tool_call)
            if not tool_call_id:
                continue
            tool_name, args = self._history_tool_call_name_args(tool_call)
            active_tool_calls[tool_call_id] = (tool_name, args)
            if not await self._send_history_update(
                session_id, build_tool_start(tool_call_id, tool_name, args)
            ):
                return False
        return True

    async def _replay_tool_result(
        self,
        message: dict[str, Any],
        session_id: str,
        active_tool_calls: dict[str, tuple[str, dict[str, Any]]],
    ) -> bool:
        \"\"\"Replay tool complete updates from a tool history message.\"\"\"
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        tool_name = str(message.get("tool_name") or "").strip()
        function_args: dict[str, Any] | None = None

        if tool_call_id in active_tool_calls:
            tool_name, function_args = active_tool_calls.pop(tool_call_id)

        if not tool_call_id or not tool_name:
            return True

        result = message.get("content")
        update = build_tool_complete(
            tool_call_id,
            tool_name,
            result=result if isinstance(result, str) else None,
            function_args=function_args,
        )
        return await self._send_history_update(session_id, update)

    async def _replay_session_history(self, state: SessionState) -> None:
        \"\"\"Send persisted user/assistant history to clients during session/load.

        Zed's ACP history UI calls ``session/load`` after the user picks an item
        from the Agents sidebar. The agent must then replay the full conversation
        as user/assistant chunks plus reconstructed tool-call start/completion
        notifications; merely restoring server-side state makes Hermes remember
        context, but leaves the editor looking like a clean thread.
        \"\"\"
        if not self._conn or not state.history:
            return

        active_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

        for message in state.history:
            role = str(message.get("role") or "")

            if role in {"user", "assistant"}:
                if not await self._replay_message_text(message, role, state.session_id):
                    return

            if role == "assistant":
                if not await self._replay_assistant_tools(message, state.session_id, active_tool_calls):
                    return
                continue

            if role == "tool":
                if not await self._replay_tool_result(message, state.session_id, active_tool_calls):
                    return
"""

search_pattern = r'    async def _replay_session_history\(self, state: SessionState\) -> None:\n.*?(?=\n    async def new_session\()'

content_new = re.sub(search_pattern, new_methods.strip('\n'), content, flags=re.DOTALL)

with open("acp_adapter/server.py", "w") as f:
    f.write(content_new)
