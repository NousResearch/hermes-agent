"""``hermes memory`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_memory_parser(subparsers, *, cmd_memory: Callable) -> None:
    """Attach the ``memory`` subcommand to ``subparsers``."""
    memory_parser = subparsers.add_parser(
        "memory",
        help="Configure external memory provider",
        description=(
            "Set up and manage external memory provider plugins.\n\n"
            "Available providers: honcho, openviking, mem0, hindsight,\n"
            "holographic, retaindb, byterover.\n\n"
            "Only one external provider can be active at a time.\n"
            "Built-in memory (MEMORY.md/USER.md) is always active."
        ),
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    _setup_parser = memory_sub.add_parser(
        "setup", help="Interactive provider selection and configuration"
    )
    _setup_parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider to configure directly (e.g. honcho), skipping the picker",
    )
    memory_sub.add_parser("status", help="Show current memory provider config")
    memory_sub.add_parser("off", help="Disable external provider (built-in only)")
    _reset_parser = memory_sub.add_parser(
        "reset",
        help="Erase all built-in memory (MEMORY.md and USER.md)",
    )
    _reset_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    _reset_parser.add_argument(
        "--target",
        choices=["all", "memory", "user"],
        default="all",
        help="Which store to reset: 'all' (default), 'memory', or 'user'",
    )
    # Phase 1: `hermes memory search <query>` — cross-memory FTS5 search.
    # Purely additive; existing setup/status/off/reset are untouched.
    _search_parser = memory_sub.add_parser(
        "search",
        help="Full-text search over indexed memory (Layer 5 index + identity)",
    )
    _search_parser.add_argument(
        "query",
        nargs="*",
        default=[],
        help="Search query (free text). Keywords route the intent automatically.",
    )
    _search_parser.add_argument(
        "--query",
        dest="query_opt",
        default=None,
        help="Search query as a named flag (alternative to positional).",
    )
    _search_parser.add_argument(
        "--intent",
        dest="intent_hint",
        default=None,
        choices=[
            "identity", "project", "decision", "historical",
            "relationship", "recent", "context",
        ],
        help="Explicit intent override (skips keyword classification).",
    )
    _search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10).",
    )
    _search_parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit results as JSON instead of human-readable text.",
    )
    # Phase 5/6 bridge: `hermes memory decision` — read accepted ADRs (Layer 4)
    # and the human-gated accept/draft lifecycle. Additive; reuses MemoryAPI.
    _decision_parser = memory_sub.add_parser(
        "decision",
        help="Architectural decisions / ADRs (Layer 4)",
    )
    _decision_sub = _decision_parser.add_subparsers(dest="decision_command")
    _decision_list = _decision_sub.add_parser("list", help="List accepted decisions (newest first)")
    _decision_list.add_argument("--project", default=None, help="Restrict to a project key")
    _decision_list.add_argument("--limit", type=int, default=20)
    _decision_list.add_argument("--json", dest="json_output", action="store_true")
    _decision_get = _decision_sub.add_parser("get", help="Get one decision by id")
    _decision_get.add_argument("decision_id", help="Global id, e.g. hermes-aios/001")
    _decision_get.add_argument("--json", dest="json_output", action="store_true")
    _decision_search = _decision_sub.add_parser("search", help="Search decisions by topic")
    _decision_search.add_argument("topic", nargs="*", default=[], help="Topic query (free text)")
    _decision_search.add_argument("--topic", dest="topic_opt", default=None)
    _decision_search.add_argument("--project", default=None)
    _decision_search.add_argument("--json", dest="json_output", action="store_true")
    _decision_project = _decision_sub.add_parser("project", help="List decisions for a project")
    _decision_project.add_argument("project", help="Project key, e.g. hermes-aios")
    _decision_project.add_argument("--json", dest="json_output", action="store_true")
    _decision_accept = _decision_sub.add_parser(
        "accept", help="Approve a PROPOSED ADR — requires --by (human authority)"
    )
    _decision_accept.add_argument("decision_id")
    _decision_accept.add_argument("--by", required=True, help="Who approved (human authority)")
    _decision_accept.add_argument("--supersedes", nargs="*", default=[], help="Ids this replaces")
    _decision_accept.add_argument("--status", default="accepted")
    _decision_draft = _decision_sub.add_parser(
        "draft", help="Create a PROPOSED draft (non-authoritative suggestion)"
    )
    _decision_draft.add_argument("title")
    _decision_draft.add_argument("--context", required=True)
    _decision_draft.add_argument("--decision", required=True)
    _decision_draft.add_argument("--alternatives", default="")
    _decision_draft.add_argument("--reasoning", default="")
    _decision_draft.add_argument("--consequences", default="")
    _decision_draft.add_argument("--project", default="_system")
    _decision_draft.add_argument("--proposed-by", dest="proposed_by", default="hermes")
    _decision_draft.add_argument("--tags", default="")
    # Phase 6: `hermes memory project` — read curated L2 project state
    # (present, not history) and the human-gated set/status/next. Additive;
    # reuses MemoryAPI. Hermes NEVER writes L2 without explicit human action.
    _project_parser = memory_sub.add_parser(
        "project",
        help="Project state (Layer 2) — where we are / what to do next",
    )
    _project_sub = _project_parser.add_subparsers(dest="project_command")
    _project_show = _project_sub.add_parser("show", help="Show current L2 project state")
    _project_show.add_argument("project", nargs="?", default=None, help="Project key (else resolved)")
    _project_show.add_argument("--json", dest="json_output", action="store_true")
    _project_status = _project_sub.add_parser("status", help="Show just the lifecycle status")
    _project_status.add_argument("project", nargs="?", default=None, help="Project key (else resolved)")
    _project_next = _project_sub.add_parser("next", help="List next actions with owners")
    _project_next.add_argument("project", nargs="?", default=None, help="Project key (else resolved)")
    _project_set = _project_sub.add_parser("set", help="Human-gated write of L2 STATUS.md")
    _project_set.add_argument("project", help="Project key, e.g. hermes-aios")
    _project_set.add_argument("--title", default=None)
    _project_set.add_argument("--status", default=None, choices=["active", "paused", "blocked", "done", "archived"])
    _project_set.add_argument("--owners", default="", help="Comma-separated owners")
    _project_set.add_argument("--goals", default="", help="Comma-separated goals")
    _project_set.add_argument("--blockers", default="", help="Comma-separated blockers")
    _project_set.add_argument("--next", default="", help="Comma-separated 'what|owner' next actions")
    _project_set.add_argument("--links-adr", dest="links_adr", default="", help="Comma-separated ADR ids")
    _project_set.add_argument("--narrative", default="")
    _project_set.add_argument("--by", required=True, dest="updated_by", help="Who curated (human authority)")
    _project_set.add_argument("--last-verified", dest="last_verified", default="", help="Informational only")
    _project_set.add_argument("--verified-by", dest="verified_by", default="", help="Informational only")
    memory_parser.set_defaults(func=cmd_memory)
