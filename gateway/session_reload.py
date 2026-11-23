"""``reload.all`` on the shared owner: ``/reload`` (#159205) for the session the connection controls.

The owner runs the session's agent, MCP servers, plugins and skills, so the classic CLI (rewired)
and Ink send ``/reload`` here. It composes the owner's own verbs in the ``/reload`` order:
``reload.env`` → plugins (a forced rediscovery in the owner's profile scope,
``plugins_activation.rediscover_plugins``) → ``reload.mcp`` (reconcile this profile's servers,
refresh its cached agents' tools[] with ``preserve_prefix=False``, which is what publishes the
rediscovered plugin tools into the chat, and commit the session's MCP generation) →
``skills.reload``. Clients ask the ``approvals.mcp_reload_confirm`` consent first (``confirm``) and
refuse while their own turn runs; a turn running here anyway is answered ``busy`` before anything
reloads. The model's note waits in the runner's one-shot per-session slot ``/reload-skills`` uses,
prepended to the next user message.
"""
import asyncio
import logging
from pathlib import Path

from gateway.session_busy_controls import authorize
from persistence.core.errors import AuthorityRefusal

logger = logging.getLogger(__name__)


def handlers(connection):
    from functools import partial
    return {'reload.all': partial(reload_all, connection)}


async def reload_all(connection, ref, params):
    if (set(params) - {'session_id', 'profile', 'confirm', 'always'} or not isinstance(ref.session_id, str)
            or not ref.session_id):
        raise AuthorityRefusal('invalid_params')
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise AuthorityRefusal('not_found')
    authorize(connection, ref, params, 'session:control')
    from agent.i18n import t
    home = Path(authority.profile_id)
    if params.get('confirm') is not True and await asyncio.to_thread(_confirm_required, home):
        return {'status': 'confirm_required', 'message': t('cli.reload_all.confirm_required')}
    if _running(authority, ref):
        return {'status': 'busy', 'message': t('cli.reload_all.busy')}
    from hermes_cli import chat_reload
    report = chat_reload.ReloadReport()
    before = chat_reload.tool_names(authority.agent(ref))
    message = await _run(connection, ref, report)
    if (agent := authority.agent(ref)) is not None:
        chat_reload.record_tools(report, agent, before)
    if note := chat_reload.model_note(report):
        runner = authority.runner
        notes = getattr(runner, '_pending_skills_reload_notes', None)
        if notes is None:
            notes = runner._pending_skills_reload_notes = {}
        notes[authority.sessions[ref.session_id].route] = note
    if params.get('always') is True:
        await asyncio.to_thread(_persist_opt_out, home)
    return {'status': 'reloaded', 'output': '\n'.join(chat_reload.summary_lines(report)),
            'report': report.to_dict(), **({'message': message} if message else {})}


async def _run(connection, ref, report):
    """The four steps into *report*; a failed one is recorded and the rest still run. Returns
    ``reload.mcp``'s note when the session's MCP generation could not be committed (busy)."""
    from gateway.session_ancillary import reload_mcp, reload_skills
    from gateway.session_authorities import owner_scope
    from gateway.session_tui_ops import reload_env
    from hermes_cli.plugins_activation import rediscover_plugins
    from tools.mcp_tool_discovery import mcp_servers_ready
    session = {'session_id': ref.session_id}
    mcp = {}

    async def env():
        report.env_updated = int((await reload_env(connection, ref, session))['updated'])

    def rediscover():
        with owner_scope(connection.authority):  # the owning profile's plugin manager and config
            return rediscover_plugins()

    async def plugins():
        report.plugins = await asyncio.to_thread(rediscover)

    async def servers():
        ready = set(await asyncio.to_thread(mcp_servers_ready))
        try:
            mcp.update(await reload_mcp(connection, ref, {**session, 'confirm': True}))
        finally:
            now = set(await asyncio.to_thread(mcp_servers_ready))
            report.mcp = {'added': sorted(now - ready), 'removed': sorted(ready - now), 'connected': sorted(now),
                          'failed': list(mcp.get('unavailable') or ())}

    async def skills():
        report.skills = (await reload_skills(connection, ref, session))['result']
    for name, step in (('env', env), ('plugins', plugins), ('mcp', servers), ('skills', skills)):
        try:
            await step()
        except Exception as exc:  # health: allow BLE001 -- one failed step is reported; the rest still reload
            logger.warning('/reload: %s step failed', name, exc_info=True)
            report.errors[name] = getattr(exc, 'detail', None) or str(exc) or type(exc).__name__
    return mcp.get('message') if mcp.get('session') == 'busy' else None


def _running(authority, ref):
    from gateway.session_detached import running_key
    running = getattr(authority.runner, '_running_agents', None) or {}
    return running_key(authority.runner, authority.sessions[ref.session_id].route, ref.session_id) in running


def _confirm_required(home):
    """``approvals.mcp_reload_confirm`` of the owning profile (default on; on when unreadable)."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import load_config
    try:
        with _profile_runtime_scope(home):
            approvals = load_config().get('approvals')
    except Exception:  # health: allow BLE001 -- an unreadable config keeps the consent question
        return True
    return bool(approvals.get('mcp_reload_confirm', True)) if isinstance(approvals, dict) else True


def _persist_opt_out(home):
    from hermes_cli.config_backend import write_config_key
    write_config_key(home / 'config.yaml', 'approvals.mcp_reload_confirm', False)
