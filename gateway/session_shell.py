"""Ink ``!cmd`` / ``{!cmd}`` on the shared owner: one captured shell command for a local session.

The legacy sidecar ran ``subprocess.run(cwd=os.getcwd())`` in its own process: on the owner that is
the daemon's directory (the install root for an auto-started one), not the session's launch cwd or
``-w`` worktree, and the sidecar's contract refused the ``session_id`` Ink sends (dokterdok N22).
Here the command runs for exactly one local session, in its frozen launch cwd, behind the same
hardline/dangerous-command refusal, with Hermes-managed secrets scrubbed from the child and the
output redacted before it crosses the wire. Nothing enters the transcript or the admission FIFO.
"""
import asyncio
from functools import partial
import json
import os
import subprocess

from hermes_state_runtime import RuntimeStoreError

TIMEOUT_SECONDS = 30
_LOCAL_BACKENDS = frozenset({'', 'local'})


def handlers(connection):
    return {'shell.exec': partial(shell_exec, connection)}


async def shell_exec(connection, ref, params):
    command = params.get('command')
    if (set(params) - {'session_id', 'profile', 'command'} or not isinstance(ref.session_id, str)
            or not ref.session_id or not isinstance(command, str) or not command.strip()):
        raise RuntimeStoreError('invalid_params')
    from gateway.session_busy_controls import authorize
    authorize(connection, ref, params, 'session:control')
    authority = connection.authority
    authority._require_admission_open()
    from gateway.session_local import authorize_local_source
    from gateway.session_policy import policy_for_source
    live = authority.sessions[ref.session_id]
    # Local terminal sessions only: an API, messaging or A2A session has no composer, and running
    # commands for it would be a remote-execution surface with no human at the keyboard.
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    policy = policy_for_source(authority.runner, live.source)
    if policy is None:
        raise RuntimeStoreError('permission_denied')
    # The frozen terminal policy chose a sandbox (docker, ssh, ...): the owner's host shell is not
    # that backend, so refuse rather than run on the host behind the session's back.
    if str(json.loads(policy.terminal_json).get('TERMINAL_ENV') or '').strip().lower() not in _LOCAL_BACKENDS:
        raise RuntimeStoreError('unsupported_terminal_backend')
    from tools.approval_detection import detect_dangerous_command, detect_hardline_command
    if detect_hardline_command(command)[0] or detect_dangerous_command(command)[0]:
        raise RuntimeStoreError('dangerous_command')
    return await asyncio.to_thread(_run, command, policy.cwd)


def _run(command, cwd):
    from agent.redact import redact_sensitive_text
    from hermes_cli._subprocess_compat import windows_hide_flags
    from tools.environments.local import build_subprocess_env
    # shell=True keeps the user-facing !cmd grammar (pipes, redirects); the child never inherits
    # credentials the long-lived owner holds, nor a launch profile's .env residue in a routed one.
    if not os.path.isdir(cwd):  # a deleted checkout / worktree: never fall back to the owner's cwd
        raise RuntimeStoreError('cwd_unavailable')
    env = build_subprocess_env(strip_launch_profile=True)
    try:
        result = subprocess.run(command, cwd=cwd, shell=True, env=env, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=TIMEOUT_SECONDS,
                                stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(), check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeStoreError('command_timeout') from None
    except OSError:
        raise RuntimeStoreError('command_failed') from None
    # Redact before tailing so a credential crossing the slice boundary cannot survive as fragments.
    stdout = redact_sensitive_text(result.stdout or '', force=True, redact_url_credentials=True)[-4000:]
    stderr = redact_sensitive_text(result.stderr or '', force=True, redact_url_credentials=True)[-2000:]
    return {'stdout': stdout, 'stderr': stderr, 'code': result.returncode}
