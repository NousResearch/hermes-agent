"""Resolve gateway ``terminal.cwd`` placeholder values to ``TERMINAL_CWD``.

When ``terminal.cwd`` is unset or a placeholder (``.``, ``auto``, ``cwd``),
the gateway must not blindly map host ``Path.home()`` into container backends.
Docker with workspace mounting still needs an explicit host path signal
(``MESSAGING_CWD`` or an absolute config path) for ``terminal_tool`` to map
``/host/project`` → ``/workspace``.
"""

from __future__ import annotations

CWD_PLACEHOLDERS = frozenset({".", "auto", "cwd"})


def resolve_placeholder_terminal_cwd(
    *,
    configured_cwd: str,
    terminal_backend: str,
    messaging_cwd: str | None,
    docker_mount_cwd_to_workspace: bool,
    home_fallback: str,
) -> str | None:
    """Return the ``TERMINAL_CWD`` value to set, or ``None`` to leave it unset.

    local + placeholder → ``MESSAGING_CWD`` or ``home_fallback``; docker +
    placeholder + mount on + host ``MESSAGING_CWD`` → that host path (for the
    ``/workspace`` mapping); any other non-local backend → ``None`` (sandbox default).
    """
    if configured_cwd and configured_cwd not in CWD_PLACEHOLDERS:
        return configured_cwd
    backend = (terminal_backend or "local").strip().lower()
    messaging = (messaging_cwd or "").strip()
    if backend == "local":
        return messaging or home_fallback
    if backend == "docker" and docker_mount_cwd_to_workspace and messaging and messaging not in CWD_PLACEHOLDERS:
        return messaging
    return None
