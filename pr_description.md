🧹 [code health: Reduce complexity of `_replay_session_history`]

🎯 **What:**
The `_replay_session_history` method in `acp_adapter/server.py` had high cognitive complexity due to deep nesting and handling multiple message roles (user/assistant text chunks, assistant tool starts, and tool results) in a single loop. I refactored it by extracting the logic for each role into smaller, private helper methods (`_replay_message_text`, `_replay_assistant_tools`, `_replay_tool_result`), as well as a centralized `_send_history_update` method for the ACP updates.

💡 **Why:**
This makes `_replay_session_history` much easier to read at a glance, adhering to the Single Responsibility Principle. Future modifications to how specific role updates are replayed can now be isolated to their respective helper methods without touching the main orchestration loop.

✅ **Verification:**
I ran `uv run ruff check acp_adapter/server.py` and `uv run ruff format acp_adapter/server.py` to ensure it still conforms to our styling. I then ran the full ACP test suite with `uv run pytest -o addopts="" tests/acp/` to verify that existing behavior remains intact. (The failing test on slash commands is pre-existing).

✨ **Result:**
The method is now linear and its complexity is significantly reduced. Maintainability and readability are highly improved.
