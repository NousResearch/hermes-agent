"""Workspace ownership for out-of-process Slack delivery."""


async def tokens_for_scope(tokens, scope_id):
    import aiohttp
    from plugins.platforms.slack import adapter as slack

    session_kw, request_kw = slack._standalone_proxy_kwargs()
    scoped = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), **session_kw) as session:
        for token in tokens:
            identity = await slack._slack_json_post(session, token, "auth.test", {}, request_kw)
            if identity.get("ok") and identity.get("team_id") == scope_id:
                scoped.append(token)
    return scoped


def scope_id_for_chat(adapter, chat_id: str):
    """Return the workspace id owning ``chat_id``.
    ``None`` for unknown channels and for channels claimed by several workspaces (dropped from
    the map) — no scope beats a wrong one. A channel unseen since boot (the map only fills from
    inbound events) still resolves when exactly one workspace is authenticated: every inbound
    reply will carry that team_id, so a caller keying a session on it must match (#111896)."""
    team_id = chat_id and (getattr(adapter, "_channel_team", None) or {}).get(str(chat_id))
    if not team_id and chat_id and str(chat_id) not in (getattr(adapter, "_channel_teams", None) or {}):
        team_clients = getattr(adapter, "_team_clients", None) or {}
        if len(team_clients) == 1:
            team_id = next(iter(team_clients))
    return str(team_id) if team_id else None
