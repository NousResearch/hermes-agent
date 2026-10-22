"""CLI entry point for the Hermes AG-UI adapter (HTTP/SSE server).

Usage::

    hermes agui                     # or: hermes-agui / python -m agui_adapter
    hermes agui --host 0.0.0.0 --port 8000 --token "$(openssl rand -hex 32)"

Configuration:
    Behavioral settings live in the ``agui`` section of ``config.yaml``
    (``host``, ``port``, ``toolsets`` + optional ``provider`` / ``model`` /
    ``api_mode`` / ``base_url`` model overrides). See ``DEFAULT_CONFIG`` in
    ``hermes_cli.config`` and ``session.AgentConfig``. The ``hermes agui`` flags
    (``--host`` / ``--port`` / ``--token``) override those per invocation.

Environment (secrets only):
    HERMES_AGUI_SESSION_TOKEN  required off-loopback; optional loopback defense-in-depth
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
else:
    # Stop a ``utils/``/``proxy/``/``ui/`` package in the launch directory from
    # shadowing Hermes's own modules — ``hermes agui`` can be started from any
    # cwd, including a project that has same-named packages on its path.
    hermes_bootstrap.harden_import_path()

# No ``from __future__ import annotations`` here: a future statement must
# precede every other import, which would displace hermes_bootstrap and break
# the entry-point contract (tests/test_hermes_bootstrap.py). It is unnecessary
# anyway — requires-python is >=3.11, so PEP 604/585 annotations are native.
# None of the other Hermes entry points carry one either.
import logging
import sys


def _setup_logging() -> None:
    from agent.redact import RedactingFormatter

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _listener_value(key, value):
    if key == "host":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("agui.host must be a nonempty string.")
        return value
    message = "agui.port must be an integer from 0 to 65535."
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise ValueError(message) from None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535:
        raise ValueError(message)
    return value


def _load_listener_config() -> tuple[str, int]:
    """Resolve (host, port) from the ``agui`` section of ``config.yaml``.

    Behavioral settings, so they come from config — not the environment. A
    missing/invalid config falls back to the loopback defaults.
    """
    listener = {"host": "127.0.0.1", "port": 8000}
    try:
        from hermes_cli.config import load_config

        section = load_config().get("agui")
        if isinstance(section, dict):
            for key, default in listener.items():
                try:
                    listener[key] = _listener_value(key, section.get(key, default))
                except ValueError as exc:
                    logging.getLogger(__name__).warning("%s Using %r.", exc, default)
    except Exception:
        logging.getLogger(__name__).debug(
            "load_config() failed; using listener defaults", exc_info=True
        )
    return listener["host"], listener["port"]


def main(
    argv: list[str] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    token: str | None = None,
) -> None:
    """Start the AG-UI server.

    ``host`` / ``port`` / ``token`` are the ``hermes agui`` CLI overrides; when
    omitted, host/port come from ``config.yaml`` (``agui`` section) and the
    token from the ``HERMES_AGUI_SESSION_TOKEN`` environment variable (a secret).
    """
    _setup_logging()
    from hermes_cli.env_loader import load_hermes_dotenv

    # Unlike `hermes agui`, the script/module launchers do not pass through
    # main.py's environment bootstrap. Use the same profile-aware loader.
    load_hermes_dotenv()
    import uvicorn

    from agui_adapter.auth import require_token_or_refuse
    from agui_adapter.server import create_app
    from agent.secret_scope import get_secret

    cfg_host, cfg_port = _load_listener_config()
    try:
        host = _listener_value("host", cfg_host if host is None else host)
        port = _listener_value("port", cfg_port if port is None else port)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    token = token or get_secret("HERMES_AGUI_SESSION_TOKEN") or None
    # main() is the authoritative fail-closed guard: it passes the SAME host to both
    # require_token_or_refuse and uvicorn.run below, so a network-accessible bind
    # without a usable token refuses to start. create_app() also re-checks against
    # the bound_host it is GIVEN, but that only protects an embedder that passes a
    # bound_host matching its real serve interface (see create_app's docstring).
    require_token_or_refuse(host, token)
    logging.getLogger(__name__).info(
        "Starting Hermes AG-UI adapter on %s:%d", host, port
    )
    uvicorn.run(
        create_app(session_token=token, bound_host=host),
        host=host,
        port=port,
        workers=1,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
