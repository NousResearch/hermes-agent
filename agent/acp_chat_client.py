"""Generic ACP chat client facade.

Currently reuses the Copilot ACP transport implementation for any local
ACP-compatible CLI that speaks the same stdio protocol.
"""

from agent.copilot_acp_client import CopilotACPClient as ACPChatClient
