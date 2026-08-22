"""CLI entry point for the hermes-agent ACP adapter.

Loads environment variables from ``~/.hermes/.env``, configures logging
to write to stderr (so stdout is reserved for ACP JSON-RPC transport),
and starts the ACP agent server.

Usage::

    python -m acp_adapter.entry
    # or
    hermes acp
    # or
    hermes-acp
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass
else:
    # Stop a ``utils/``/``proxy/``/``ui/`` package in the launch directory from
    # shadowing Hermes's own modules — ``hermes acp`` can be started from any
    # cwd, including a project that has same-named packages on its path.
    hermes_bootstrap.harden_import_path()

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from hermes_constants import get_hermes_home


# Methods clients send as periodic liveness probes. They are not part of the
# ACP schema, so the acp router correctly returns JSON-RPC -32601 to the
# caller — but the supervisor task that dispatches the request then surfaces
# the raised RequestError via ``logging.exception("Background task failed")``,
# which dumps a traceback to stderr every probe interval. Clients like
# acp-bridge already treat the -32601 response as "agent alive", so the
# traceback is pure noise. We keep the protocol response intact and only
# silence the stderr noise for this specific benign case.
_BENIGN_PROBE_METHODS = frozenset({"ping", "health", "healthcheck"})


class _BenignProbeMethodFilter(logging.Filter):
    """Suppress acp 'Background task failed' tracebacks caused by unknown
    liveness-probe methods (e.g. ``ping``) while leaving every other
    background-task error — including method_not_found for any non-probe
    method — visible in stderr.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "Background task failed":
            return True
        exc_info = record.exc_info
        if not exc_info:
            return True
        exc = exc_info[1]
        # Imported lazily so this module stays importable when the optional
        # ``agent-client-protocol`` dependency is not installed.
        try:
            from acp.exceptions import RequestError
        except ImportError:
            return True
        if not isinstance(exc, RequestError):
            return True
        if getattr(exc, "code", None) != -32601:
            return True
        data = getattr(exc, "data", None)
        method = data.get("method") if isinstance(data, dict) else None
        return method not in _BENIGN_PROBE_METHODS


def _setup_logging() -> None:
    """Route all logging to stderr so stdout stays clean for ACP stdio."""
    from agent.redact import RedactingFormatter

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_BenignProbeMethodFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _load_env() -> None:
    """Load .env from HERMES_HOME (default ``~/.hermes``)."""
    from hermes_cli.env_loader import load_hermes_dotenv

    hermes_home = get_hermes_home()
    loaded = load_hermes_dotenv(hermes_home=hermes_home)
    if loaded:
        for env_file in loaded:
            logging.getLogger(__name__).info("Loaded env from %s", env_file)
    else:
        logging.getLogger(__name__).info(
            "No .env found at %s, using system env", hermes_home / ".env"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-acp",
        description="Run Hermes Agent as an ACP stdio server.",
    )
    parser.add_argument("--version", action="store_true", help="Print Hermes version and exit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify ACP dependencies and adapter imports, then exit",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive Hermes provider/model setup for ACP terminal auth",
    )
    parser.add_argument(
        "--setup-browser",
        action="store_true",
        help="Install agent-browser + Playwright Chromium into ~/.hermes/node/ "
             "for browser tool support. Idempotent.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        dest="assume_yes",
        help="Accept all prompts (currently used by --setup-browser to skip the "
             "~400 MB Chromium download confirmation).",
    )
    return parser.parse_args(argv)


def _print_version() -> None:
    from hermes_cli import __version__ as hermes_version

    print(hermes_version)


def _run_check() -> None:
    import acp  # noqa: F401
    from acp_adapter.server import HermesACPAgent  # noqa: F401

    print("Hermes ACP check OK")


def _run_setup() -> None:
    from hermes_cli.main import main as hermes_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0] if old_argv else "hermes", "model"]
        hermes_main()
    finally:
        sys.argv = old_argv

    # Offer browser-tools install as a follow-up. The terminal auth method
    # is the one supported first-run UX for registry installs, so this is
    # the natural moment to ask. Skip silently if stdin isn't a TTY (the
    # answer can't be collected anyway).
    if not sys.stdin.isatty():
        return
    try:
        reply = input(
            "\nInstall browser tools? Downloads agent-browser (npm) and "
            "optionally Playwright Chromium (~400 MB). [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if reply in {"y", "yes"}:
        _run_setup_browser(assume_yes=False)


def _run_setup_browser(assume_yes: bool = False) -> int:
    """Bootstrap agent-browser + Chromium.

    Routes through dep_ensure -> install.{sh,ps1} --ensure, sharing code
    with the runtime lazy installer.

    Returns 0 on success, 1 on failure.
    """
    from hermes_cli.dep_ensure import ensure_dependency

    try:
        node_ok = ensure_dependency("node", interactive=not assume_yes)
        if not node_ok:
            print("Node.js installation failed — cannot proceed with browser tools.",
                  file=sys.stderr)
            return 1

        browser_ok = ensure_dependency("browser", interactive=not assume_yes)
        if not browser_ok:
            print("Browser tools installation failed.", file=sys.stderr)
            return 1

        return 0
    except OSError as exc:
        print(f"Browser bootstrap failed: {exc}", file=sys.stderr)
        return 1


def _point_win32_stdin_at(fd: int) -> bool:
    """Repoint the process-wide Win32 ``STD_INPUT_HANDLE`` at *fd*.

    This is the handle a child launched with ``stdin=None`` actually inherits
    on Windows — fd 0 alone does not shield anything there — so its success is
    load-bearing rather than advisory.

    Returns True when there is nothing to do (non-Windows) or the redirect
    succeeded, False when the Win32 call reported failure.
    """
    if sys.platform != "win32":
        return True

    import ctypes
    import msvcrt
    from ctypes import wintypes

    STD_INPUT_HANDLE = -10

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Without explicit argtypes ctypes passes the HANDLE as a C int, which
    # truncates on 64-bit Windows.
    kernel32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
    kernel32.SetStdHandle.restype = wintypes.BOOL

    if not kernel32.SetStdHandle(STD_INPUT_HANDLE, msvcrt.get_osfhandle(fd)):
        logging.getLogger(__name__).warning(
            "SetStdHandle(STD_INPUT_HANDLE) failed (WinError %d); "
            "child processes would still inherit the ACP stdin pipe",
            ctypes.get_last_error(),
        )
        return False
    return True


def _shield_stdin_from_children() -> None:
    """Keep the ACP JSON-RPC stdin out of every child process.

    The host IDE hands this process a pipe as stdin. Any subprocess spawned
    with ``stdin=None`` inherits that handle — on Windows via the process
    STD_INPUT_HANDLE, not fd 0 — and MSYS/mingw programs (git, bash) can
    block on it at startup. When they do, ``subprocess.run(timeout=...)``
    kills only the ``Git\\cmd``/``Git\\bin`` wrapper; the surviving mingw
    grandchild keeps the pipe open and the post-kill ``communicate()`` drain
    (which has no timeout) wedges the calling thread — and with it the whole
    agent turn. Re-home the transport's stdin onto a private duplicate and
    point fd 0 plus the Win32 std input handle at the null device so every
    child inherits NUL instead.
    """
    import io

    logger = logging.getLogger(__name__)
    try:
        original_fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return  # no real stdin (embedded/test harness) — nothing to shield
    try:
        private_fd = os.dup(original_fd)
        os.set_inheritable(private_fd, False)
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull_fd, original_fd)
        os.close(devnull_fd)
        if not _point_win32_stdin_at(original_fd):
            # Half-shielded is the worst outcome: fd 0 reads NUL, but children
            # still inherit the real pipe through STD_INPUT_HANDLE, and the
            # log would claim the stdin was shielded. Put fd 0 back so the
            # process is left in the state it started in, unshielded and
            # consistent, and let the caller proceed without the shield.
            os.dup2(private_fd, original_fd)
            os.close(private_fd)
            return
        # The ACP transport reads sys.stdin.buffer; hand it the private copy.
        sys.stdin = io.TextIOWrapper(
            io.BufferedReader(io.FileIO(private_fd, "rb")),
            encoding="utf-8",
            errors="replace",
        )
        logger.info("ACP stdin shielded from child processes (children inherit NUL)")
    except Exception:
        logger.warning("Could not shield ACP stdin from child processes", exc_info=True)


def main(argv: list[str] | None = None) -> None:
    """Entry point: load env, configure logging, run the ACP agent."""
    args = _parse_args(argv)
    if args.version:
        _print_version()
        return
    if args.check:
        _run_check()
        return
    if args.setup:
        _run_setup()
        return
    if args.setup_browser:
        rc = _run_setup_browser(assume_yes=args.assume_yes)
        if rc != 0:
            sys.exit(rc)
        return

    _setup_logging()
    _load_env()
    _shield_stdin_from_children()

    logger = logging.getLogger(__name__)
    logger.info("Starting hermes-agent ACP adapter")

    # Ensure the project root is on sys.path so ``from run_agent import AIAgent`` works
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import acp
    from .server import HermesACPAgent

    # MCP tool discovery from config.yaml — fire-and-forget in a
    # background daemon thread so the ACP server becomes responsive
    # immediately while MCP servers connect.  Previously this blocked
    # asyncio.run() for 2-5 s.  (ACP also registers per-session MCP
    # servers dynamically via asyncio.to_thread inside the event loop;
    # that path is unaffected.)  Moved from model_tools.py module scope
    # to avoid freezing the gateway's loop on lazy import (#16856).
    # Metadata-only hosts can opt out of unrelated global MCP startup.
    if os.environ.get("HERMES_ACP_SKIP_CONFIGURED_MCP", "").strip() != "1":
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(
                logger=logger,
                thread_name="acp-mcp-discovery",
            )
        except Exception:
            logger.debug("MCP tool discovery failed at ACP startup", exc_info=True)

    agent = HermesACPAgent()
    try:
        asyncio.run(acp.run_agent(agent, use_unstable_protocol=True))
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
    except Exception:
        logger.exception("ACP agent crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
