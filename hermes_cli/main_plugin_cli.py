"""Deferred-platform CLI registration for the ``hermes`` parser.

Split out of ``hermes_cli/main.py``.
"""

import logging


def _resolve_deferred_platform_cli_command(command_name: str | None) -> None:
    """Materialize the deferred platform whose top-level CLI command matches.

    Bundled platforms are *deferred* entries (no gateway SDK imports at
    startup), so a platform's ``register_cli_command`` side effect only runs
    on import; ``discover_plugins()`` alone leaves ``hermes photon`` failing
    with ``invalid choice``. Importing just the matching platform keeps
    startup cheap.

    On the unknown-top-level-command slow path, ``discover_plugins()`` records the deferred loader but does
    not import it, so the CLI registration never happens and ``hermes photon`` fails with argparse ``invalid
    choice`` (issue #54678).
    """
    if not command_name:
        return
    try:
        from gateway.platform_registry import platform_registry

        platform_registry.get(command_name)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Deferred platform CLI resolution failed for %s: %s",
            command_name,
            exc,
        )


def _resolve_all_deferred_platform_cli_commands() -> None:
    """Materialize every deferred platform so each one's CLI command is registered.

    Only ``hermes completion`` pays for this: it is a one-shot request for the
    full command set, unlike the single-command resolve above.
    """
    try:
        from gateway.platform_registry import platform_registry

        platform_registry.all_entries()
    except Exception as exc:
        logging.getLogger(__name__).debug("Deferred platform CLI resolution failed: %s", exc, exc_info=True)
