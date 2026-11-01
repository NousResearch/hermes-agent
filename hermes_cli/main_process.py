"""Process-level startup helpers for the Hermes CLI entry point."""

import os


def _advertise_agent_env() -> None:
    """Advertise the agent harness to child processes.

    ``AI_AGENT`` is the cross-agent standard (huggingface_hub reads it); the
    value must be our id in the public agent-harness registry
    (``hermes-agent``) — matching is exact. ``HERMES_AGENT`` is the
    Hermes-specific marker. setdefault: never clobber an outer harness.

    ``AI_AGENT`` is the emerging cross-agent standard (huggingface_hub's agent
    detection reads it; pi and other agents set it) so generic tooling can
    attribute subprocesses to the harness that spawned them.
    """
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")
