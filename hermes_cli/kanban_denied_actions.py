"""The Kanban CLI verbs a ``delegate_task`` child may not run — the single definition.

Data only: no imports, no logic, so BOTH the CLI's own check (``hermes_cli/kanban.py``) and the
pre-spawn floor guard (``tools/kanban_cli_mutation_guard.py``) can import it without either pulling
in the other's machinery. The guard previously duplicated these sets behind a "keep these two lists
in sync" comment because importing ``hermes_cli.kanban`` would drag the full CLI arg-parser in; that
comment is not a mechanism, and it failed exactly as you would expect — ``set-model`` was added to
the parser and to neither list, leaving a mutating verb unguarded on both paths until a review
caught it. A leaf module removes the class of bug instead of restating the rule.

``tests/tools/test_kanban_denied_actions.py`` asserts these sets against the verbs
``hermes_cli/kanban_parser.py`` actually exposes, so a newly-added mutating verb fails the suite
rather than silently widening what a delegated child can reach.
"""
from __future__ import annotations

#: Task-level verbs that mutate board state.
DENIED_KANBAN_ACTIONS: frozenset[str] = frozenset({
    "init", "create", "swarm", "assign", "reclaim", "reassign", "link", "unlink",
    "claim", "comment", "attach", "attach-rm", "complete", "edit", "block",
    "schedule", "unblock", "promote", "archive", "dispatch", "daemon", "repair",
    "heartbeat", "notify-subscribe", "notify-unsubscribe", "specify", "decompose",
    "request-review", "request-changes", "reopen-review", "set-model", "gc",
})

#: ``kanban boards <verb>`` forms that mutate the board registry. ``hold``/``resume`` are the
#: board-automation verbs #110196 adds; denying a verb the parser does not yet expose is inert, so
#: listing them ahead of that merge closes the gap whichever PR lands first.
DENIED_BOARD_ACTIONS: frozenset[str] = frozenset({
    "create", "new", "rm", "remove", "delete", "switch", "use", "rename",
    "hold", "resume", "set-default-workdir", "import",
})
