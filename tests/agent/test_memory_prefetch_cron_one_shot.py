"""Cron runs are one turn: queue no post-turn memory prefetch (#152517).

``AIAgent._sync_external_memory_for_turn`` mirrors a finished turn into the external
memory providers with ``sync_all`` and then queues a background ``queue_prefetch_all``
warm-up for the *next* turn. A cron job runs ``run_conversation`` once and tears its
agent down, so nothing ever consumes that warm-up — while a local recall backend pays a
full recall for it (~60 s on a CPU reranker) and those runs compete with the user's own
interactive recalls.

The gate is the agent's ``platform`` (``"cron"`` for ``cron/scheduler.py``'s agent),
which is the same input ``_memory_provider_init_kwargs`` turns into ``agent_context``.
``MemoryManager`` itself keeps queueing for every context — `tests/agent/
test_memory_manager_fork_scope.py` pins that cron keeps the provider-side contract.
"""

from unittest.mock import MagicMock


def _bare_agent(platform=None):
    """An ``AIAgent`` with only the attributes ``_sync_external_memory_for_turn`` reads."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "cron_session_001"
    agent.platform = platform
    return agent


def _run_turn(agent):
    agent._sync_external_memory_for_turn(
        original_user_message="Summarize today's headlines.",
        final_response="Three headlines.",
        interrupted=False,
    )


def test_cron_turn_syncs_but_queues_no_prefetch():
    agent = _bare_agent("cron")

    _run_turn(agent)

    # The turn itself is still mirrored: cron writes stay the provider's call.
    agent._memory_manager.sync_all.assert_called_once()
    agent._memory_manager.queue_prefetch_all.assert_not_called()


def test_interactive_platforms_still_queue_the_next_turn_prefetch():
    """Control: the same turn on a conversational surface keeps the warm-up (#6672)."""
    for platform in (None, "cli", "telegram", "desktop"):
        agent = _bare_agent(platform)

        _run_turn(agent)

        agent._memory_manager.sync_all.assert_called_once()
        agent._memory_manager.queue_prefetch_all.assert_called_once()


def test_cron_turn_with_a_trivial_prompt_still_syncs():
    """The existing trivial-prompt gate is orthogonal to the platform gate."""
    agent = _bare_agent("cron")

    agent._sync_external_memory_for_turn(
        original_user_message="hi", final_response="Hello.", interrupted=False,
    )

    agent._memory_manager.sync_all.assert_called_once()
    agent._memory_manager.queue_prefetch_all.assert_not_called()
