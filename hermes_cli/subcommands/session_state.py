"""Project SESSION_STATE CLI commands."""

from __future__ import annotations

from typing import Callable


def build_session_state_parsers(
    subparsers,
    *,
    cmd_resume: Callable,
    cmd_next: Callable,
    cmd_checkpoint: Callable,
    cmd_gc: Callable,
) -> None:
    """Attach first-class SESSION_STATE commands."""
    resume = subparsers.add_parser(
        "resume",
        help="Validate and load canonical SESSION_STATE.json without chat history",
        description="Locate the latest SESSION_STATE.json and validate manifest, hashes, and validation report.",
    )
    resume.add_argument("--path", help="Directory to search from (default: current directory)")
    resume.add_argument("--json", action="store_true", help="Print canonical state JSON after validation")
    resume.set_defaults(func=cmd_resume)

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="Generate SESSION_STATE.json from the latest passed checkpoint",
        description="Validate latest manifest/hash/report artifacts and write SESSION_STATE.json to the project root.",
    )
    checkpoint.add_argument("--checkpoint-dir", help="Specific checkpoint directory to seal into SESSION_STATE")
    checkpoint.add_argument("--project-root", help="Project root to search/write from (default: current directory)")
    checkpoint.add_argument("--output", help="Output SESSION_STATE.json path (default: <project-root>/SESSION_STATE.json)")
    checkpoint.set_defaults(func=cmd_checkpoint)

    next_cmd = subparsers.add_parser(
        "next",
        help="checkpoint → status → resume → execute next recommended prompt",
        description="Advance using SESSION_STATE.next_recommended_prompt without copying prompts between chats.",
    )
    next_cmd.add_argument("--checkpoint-dir", help="Specific checkpoint directory to seal first")
    next_cmd.add_argument("--project-root", help="Project root to search/write from (default: current directory)")
    next_cmd.add_argument("--output", help="Output SESSION_STATE.json path")
    next_cmd.add_argument("--dry-run", action="store_true", help="Validate and print the next prompt without invoking chat")
    next_cmd.set_defaults(func=cmd_next)

    gc = subparsers.add_parser(
        "gc",
        help="Archive obsolete temporary conversation metadata while preserving canonical state",
        description="Never deletes design/checkpoint directories, manifests, verified hashes, validation reports, canonical artifacts, or SESSION_STATE.json.",
    )
    gc.add_argument("--hermes-home", help="Hermes home to clean (default: active profile/home)")
    gc.add_argument("--dry-run", action="store_true", help="Show files that would be archived")
    gc.set_defaults(func=cmd_gc)
