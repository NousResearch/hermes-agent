"""Claude Code subprocess transport that launches from a sanitized environment.

``ClaudeAgentOptions.env`` cannot express "do not pass this variable through".
The SDK builds the child environment as::

    process_env = {**inherited_env, "CLAUDE_CODE_ENTRYPOINT": ..., **options.env, ...}

(``claude_agent_sdk/_internal/transport/subprocess_cli.py``) — ``options.env``
is merged *over* a copy of ``os.environ``, so it can override a key but never
delete one.  Setting ``ANTHROPIC_API_KEY=""`` leaves the name present, and
whether the CLI then treats it as absent is a per-variable, per-version
accident rather than a contract.

Mutating ``os.environ`` around the spawn is not an option either: Hermes is
multi-threaded and other providers run concurrently, so a process-global
mutation would leak into them.

So the removal happens where it actually can: this transport subclasses the
SDK's ``SubprocessCLITransport`` and overrides ``connect()`` — the one method
that reads ``os.environ`` — to spawn from
:func:`agent.claude_billing.sanitized_child_env` instead.  Everything else
(command construction, CLI discovery, stdout framing, stdin, stderr pumping,
teardown) is inherited unchanged.

The class is built lazily inside :func:`build_sanitized_transport` because
``claude-agent-sdk`` is an optional extra and this module must import without
it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from agent.claude_billing import BLOCKED_CHILD_ENV_VARS, sanitized_child_env

logger = logging.getLogger(__name__)

_TRANSPORT_CLASS: Any = None


async def _empty_stream() -> AsyncIterator[Dict[str, Any]]:
    """A prompt stream that never yields, matching ``ClaudeSDKClient``.

    The client writes turns with ``transport.write()``; the transport only
    needs *an* async iterable to hold the connection open.
    """
    return
    yield {}  # pragma: no cover - unreachable, marks this an async generator


def build_child_env(options: Any, *, cwd: Optional[str] = None) -> Dict[str, str]:
    """Build the child environment for a subscription-mode Claude spawn.

    Mirrors the SDK's own merge, minus the credentials that outrank the user's
    subscription.  Split out from the transport so the launcher's output can be
    asserted on directly, without spawning anything.
    """
    try:
        from claude_agent_sdk._version import __version__ as sdk_version
    except Exception:  # pragma: no cover - SDK absent or restructured
        sdk_version = ""

    env = sanitized_child_env()
    # CLAUDECODE would make the child believe it is nested inside a Claude Code
    # parent; the SDK drops it for the same reason.
    env.pop("CLAUDECODE", None)
    env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
    env.update(getattr(options, "env", None) or {})
    if sdk_version:
        env["CLAUDE_AGENT_SDK_VERSION"] = sdk_version

    # Re-applied after options.env: a caller-supplied override must not be able
    # to put a higher-precedence credential back into the child.
    for key in BLOCKED_CHILD_ENV_VARS:
        env.pop(key, None)

    if getattr(options, "enable_file_checkpointing", False):
        env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
    resolved_cwd = cwd if cwd is not None else getattr(options, "cwd", None)
    if resolved_cwd:
        env["PWD"] = str(resolved_cwd)
    # HOME is deliberately absent from every write above. Overriding it
    # relocates the macOS login-keychain lookup ($HOME/Library/Keychains), so
    # the CLI cannot find its stored OAuth credentials and reports
    # "Not logged in". CLAUDE_CONFIG_DIR is the supported way to isolate config
    # and passes through untouched.
    return env


def _build_transport_class() -> Any:
    """Define the transport subclass against the installed SDK."""
    import anyio
    from anyio.streams.text import TextReceiveStream, TextSendStream
    from subprocess import PIPE

    from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        _ACTIVE_CHILDREN,
        SubprocessCLITransport,
    )
    from claude_agent_sdk._internal._task_compat import spawn_detached

    class SanitizedSubprocessCLITransport(SubprocessCLITransport):
        """``SubprocessCLITransport`` that spawns from a sanitized environment.

        Only ``connect()`` differs from the base class, and only in the
        ``env=`` it hands to ``anyio.open_process``.
        """

        def build_child_env(self) -> Dict[str, str]:
            return build_child_env(self._options, cwd=self._cwd)

        async def connect(self) -> None:
            if self._process:
                return

            if self._cli_path is None:
                self._cli_path = await anyio.to_thread.run_sync(self._find_cli)
            self._reject_windows_batch_cli(self._cli_path)
            if not os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
                await self._check_claude_version()

            cmd = self._build_command()
            try:
                process_env = self.build_child_env()

                # The SDK pipes stderr only when a callback is registered;
                # otherwise the child inherits the parent's stderr, which under
                # Electron can be a closed handle. Hermes always registers one
                # (see agent.claude_runtime).
                stderr_dest = PIPE if self._options.stderr is not None else None

                self._process = await anyio.open_process(
                    cmd,
                    # stdin is always a pipe: Claude Code probes stdin and
                    # blocks forever on an unusable inherited one.
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=stderr_dest,
                    cwd=self._cwd,
                    env=process_env,
                    user=self._options.user,
                )
                _ACTIVE_CHILDREN.add(self._process)

                if self._process.stdout:
                    self._stdout_stream = TextReceiveStream(self._process.stdout)
                if stderr_dest is PIPE and self._process.stderr:
                    self._stderr_stream = TextReceiveStream(self._process.stderr)
                    self._stderr_task = spawn_detached(self._handle_stderr())
                if self._process.stdin:
                    self._stdin_stream = TextSendStream(self._process.stdin)

                self._ready = True
            except FileNotFoundError as exc:
                if self._cwd and not os.path.exists(self._cwd):
                    error: Exception = CLIConnectionError(
                        f"Working directory does not exist: {self._cwd}"
                    )
                else:
                    error = CLINotFoundError(
                        f"Claude Code not found at: {self._cli_path}"
                    )
                self._exit_error = error
                raise error from exc
            except Exception as exc:
                error = CLIConnectionError(f"Failed to start Claude Code: {exc}")
                self._exit_error = error
                raise error from exc

    return SanitizedSubprocessCLITransport


def sanitized_transport_class() -> Any:
    """The transport subclass, built once per process."""
    global _TRANSPORT_CLASS
    if _TRANSPORT_CLASS is None:
        _TRANSPORT_CLASS = _build_transport_class()
    return _TRANSPORT_CLASS


def build_sanitized_transport(options: Any, *, prompt: Any = None) -> Any:
    """Construct the transport ``ClaudeSDKClient`` should use for *options*.

    Supplying a custom transport makes ``ClaudeSDKClient.connect()`` skip
    ``materialize_resume_session()`` — the materialized options would never
    reach a transport that is already constructed, so the SDK does not bother.
    :meth:`agent.transports.claude_agent_session.ClaudeAgentSession._materialize`
    therefore runs it one step earlier, *before* calling this factory, and
    passes the repointed options here.  Nothing in this module needs to change
    for that: the materialized ``options.env["CLAUDE_CONFIG_DIR"]`` flows
    through :func:`build_child_env` like any other caller-supplied variable
    (it is a pass-through, not a blocked one), and ``options.resume`` becomes
    ``--resume=<id>`` in the inherited ``_build_command()``.  The blocklist is
    still re-applied after ``options.env``, so materialization cannot
    reintroduce a credential that outranks the user's subscription.
    """
    transport_class = sanitized_transport_class()
    return transport_class(
        prompt=prompt if prompt is not None else _empty_stream(),
        options=options,
    )


__all__ = [
    "build_child_env",
    "build_sanitized_transport",
    "sanitized_transport_class",
]
