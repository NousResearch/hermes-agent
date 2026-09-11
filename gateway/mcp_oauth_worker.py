"""Reuse Hermes' SDK probe; isolate uncommitted credentials from live profiles."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory


def probe_and_commit(flow, cfg):
    from gateway.run import _profile_runtime_scope
    from hermes_cli.mcp_config import _probe_single_server, _get_mcp_servers, _save_mcp_server, _resolve_mcp_server_config
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.mcp_dashboard_oauth import dashboard_oauth_flow
    from tools.mcp_oauth import HermesTokenStorage, force_interactive_oauth, suppress_interactive_oauth
    from tools.mcp_oauth_manager import get_manager
    from tools.mcp_tool_discovery import discover_mcp_tools, _clear_connect_failure
    from tools.mcp_tool_loop import reconnect_mcp_server, _wait_for_server_session_ready
    from tools import mcp_tool

    home = Path(flow.owner[0])
    try:
        with _profile_runtime_scope(home):
            original_cfg = _get_mcp_servers().get(flow.server_name)
            if original_cfg is not None and original_cfg != cfg:
                raise RuntimeError("MCP configuration changed before login started")
            target = HermesTokenStorage(flow.server_name, hermes_home=home)
            original_tokens = target.snapshot()
            resolved = _resolve_mcp_server_config(cfg)
            with TemporaryDirectory(prefix="mcp-oauth-") as staging:
                staged = HermesTokenStorage(flow.server_name, hermes_home=staging)
                client = asyncio.run(target.get_client_info())
                if client is not None:
                    asyncio.run(staged.set_client_info(client))
                token = set_hermes_home_override(staging)
                try:
                    with force_interactive_oauth(), dashboard_oauth_flow(flow):
                        _probe_single_server(flow.server_name, resolved, connect_timeout=300)
                    tokens = asyncio.run(staged.get_tokens())
                    client = asyncio.run(staged.get_client_info())
                    metadata = staged.load_oauth_metadata()
                    if tokens is None or client is None:
                        raise RuntimeError("OAuth did not produce credentials")
                finally:
                    get_manager().remove(flow.server_name, hermes_home=staging)
                    reset_hermes_home_override(token)
                with flow._lock:
                    flow.check_live()
                    with mcp_tool._lock:
                        scope = mcp_tool._mcp_registry_scope()
                        if (flow.server_name in mcp_tool._servers and
                                (mcp_tool._server_scope_keys.get(flow.server_name) != scope or
                                 mcp_tool._server_tool_scopes.get(flow.server_name, set()) - {scope})):
                            raise RuntimeError("MCP server name is already used by another profile")
                    if (_get_mcp_servers().get(flow.server_name) != original_cfg or
                            target.snapshot() != original_tokens):
                        raise RuntimeError("MCP configuration or credentials changed during login")
                    # Token is the commit marker. Cancellation after this commit
                    # does not revoke an already completed authorization.
                    try:
                        asyncio.run(target.set_client_info(client))
                        if metadata is not None:
                            target.save_oauth_metadata(metadata)
                        asyncio.run(target.set_tokens(tokens))
                        if original_cfg != cfg and not _save_mcp_server(flow.server_name, cfg):
                            raise RuntimeError("Could not save MCP configuration")
                    except Exception:
                        target.restore(original_tokens)
                        raise
                    flow.credentials_committed = True
            # Discovery updates the registry; never mutate a cached agent's tools
            # or system prompt. Existing sessions adopt them at a fresh session.
            with suppress_interactive_oauth():
                with mcp_tool._lock:
                    _clear_connect_failure(flow.server_name)
                    old_server = mcp_tool._servers.get(flow.server_name)
                    old_session = old_server.session if old_server else None
                reconnect_mcp_server(flow.server_name)
                discover_mcp_tools(allowed_mcp_names=[flow.server_name])
                with mcp_tool._lock:
                    server = mcp_tool._servers.get(flow.server_name)
                if server is None or not _wait_for_server_session_ready(server, old_session=old_session):
                    raise RuntimeError("Credentials saved but MCP discovery failed")
            flow.mark_approved()
    except Exception:
        flow.mark_error("OAuth failed")
    finally:
        flow.mark_worker_done()
