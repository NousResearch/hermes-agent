"""``hermes skills`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_skills_parser(subparsers, *, cmd_skills: Callable) -> None:
    """Attach the ``skills`` subcommand to ``subparsers``."""
    skills_parser = subparsers.add_parser(
        "skills",
        help="Search, install, configure, and manage skills",
        description="Search, install, inspect, audit, configure, and manage skills from skills.sh, well-known agent skill endpoints, GitHub, ClawHub, and other registries.",
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_browse = skills_subparsers.add_parser(
        "browse", help="Browse all available skills (paginated)"
    )
    skills_browse.add_argument(
        "--page", type=int, default=1, help="Page number (default: 1)"
    )
    skills_browse.add_argument(
        "--size", type=int, default=20, help="Results per page (default: 20)"
    )
    skills_browse.add_argument(
        "--source",
        default="all",
        choices=[
            "all",
            "official",
            "skills-sh",
            "well-known",
            "github",
            "clawhub",
            "lobehub",
            "browse-sh",
        ],
        help="Filter by source (default: all)",
    )

    skills_search = skills_subparsers.add_parser(
        "search", help="Search skill registries"
    )
    skills_search.add_argument("query", help="Search query")
    skills_search.add_argument(
        "--source",
        default="all",
        choices=[
            "all",
            "official",
            "skills-sh",
            "well-known",
            "github",
            "clawhub",
            "lobehub",
            "browse-sh",
        ],
    )
    skills_search.add_argument("--limit", type=int, default=10, help="Max results")
    skills_search.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of a table (full identifiers, scripting-friendly)",
    )

    skills_install = skills_subparsers.add_parser("install", help="Install a skill")
    skills_install.add_argument(
        "identifier",
        help="Skill identifier (e.g. openai/skills/skill-creator) or a direct HTTP(S) URL to a SKILL.md file",
    )
    skills_install.add_argument(
        "--category", default="", help="Category folder to install into"
    )
    skills_install.add_argument(
        "--name",
        default="",
        help="Override the skill name (useful when installing from a URL whose SKILL.md has no `name:` frontmatter)",
    )
    skills_install.add_argument(
        "--force", action="store_true", help="Install despite blocked scan verdict"
    )
    skills_install.add_argument(
        "--require-vetting",
        action="store_true",
        help=(
            "Refuse to install a skill whose SKILL.md does not already "
            "carry a vetted_by stamp (i.e. has not been through "
            "`hermes skills vet`). Use in CI / fleet-rollout pipelines "
            "to enforce a review gate on every new install."
        ),
    )
    skills_install.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt (needed in TUI mode)",
    )

    skills_inspect = skills_subparsers.add_parser(
        "inspect", help="Preview a skill without installing"
    )
    skills_inspect.add_argument("identifier", help="Skill identifier")

    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.add_argument(
        "--source",
        default="all",
        choices=["all", "hub", "builtin", "local-edit", "local"],
        help="Filter by Source / provenance category",
    )
    skills_list.add_argument(
        "--enabled-only",
        action="store_true",
        help="Hide disabled skills. Use with -p <profile> to see exactly "
        "which skills will load for that profile.",
    )
    skills_list.add_argument(
        "--provenance",
        action="store_true",
        help="Show the Provenance column — the install-origin path each "
        "skill was seeded from (e.g. ~/.hermes/skills/apple/macos-computer-use "
        "for a builtin, the upstream registry path for a hub install).",
    )
    skills_list.add_argument(
        "--vetted",
        action="store_true",
        help="Show only vetted skills (those with a non-'unvetted' "
        "vetted_by frontmatter field, set by `hermes skills vet`).",
    )
    skills_list.add_argument(
        "--unvetted",
        action="store_true",
        help="Show only unvetted skills. Useful as a TODO list for "
        "`hermes skills vet <name> --by <reviewer>` runs.",
    )

    skills_check = skills_subparsers.add_parser(
        "check", help="Check installed hub skills for updates"
    )
    skills_check.add_argument(
        "name", nargs="?", help="Specific skill to check (default: all)"
    )

    skills_update = skills_subparsers.add_parser(
        "update", help="Update installed hub skills"
    )
    skills_update.add_argument(
        "name",
        nargs="?",
        help="Specific skill to update (default: all outdated skills)",
    )

    skills_audit = skills_subparsers.add_parser(
        "audit", help="Re-scan installed hub skills"
    )
    skills_audit.add_argument(
        "name", nargs="?", help="Specific skill to audit (default: all)"
    )
    skills_audit.add_argument(
        "--deep",
        action="store_true",
        help="Run AST-level analysis on Python files (opt-in diagnostic)",
    )

    skills_vet = skills_subparsers.add_parser(
        "vet",
        help="Run the four vetting validators and stamp vetted skills (t_8a86fc9c)",
        description=(
            "Run the four SKILL.md vetting validators (frontmatter, security, "
            "content-quality, link/file-existence) and, on success, stamp "
            "vetted_at + vetted_by in the SKILL.md frontmatter. "
            "Use --dry-run to validate without writing."
        ),
    )
    skills_vet.add_argument(
        "name",
        nargs="?",
        help="Specific installed skill to vet. Required unless --all is set.",
    )
    skills_vet.add_argument(
        "--by",
        default="",
        help=(
            "Reviewer identity to record in vetted_by (e.g. 'wags-reviewer', "
            "'human:charlie'). Required to stamp; ignored with --dry-run."
        ),
    )
    skills_vet.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing vetted_at / vetted_by.",
    )
    skills_vet.add_argument(
        "--all",
        action="store_true",
        dest="all_skills",
        help="Vet every local installed skill.",
    )
    skills_vet.add_argument(
        "--when",
        default=None,
        help=(
            "Override the vetted_at timestamp (ISO-8601, e.g. "
            "'2026-06-23T18:30:00Z'). Defaults to the current UTC time. "
            "Useful for tests, audits, and backfilling stamps."
        ),
    )

    skills_uninstall = skills_subparsers.add_parser(
        "uninstall", help="Remove a hub-installed skill"
    )
    skills_uninstall.add_argument("name", help="Skill name to remove")

    skills_reset = skills_subparsers.add_parser(
        "reset",
        help="Reset a bundled skill — clears 'user-modified' tracking so updates work again",
        description=(
            "Clear a bundled skill's entry from the sync manifest (~/.hermes/skills/.bundled_manifest) "
            "so future 'hermes update' runs stop marking it as user-modified. Pass --restore to also "
            "replace the current copy with the bundled version."
        ),
    )
    skills_reset.add_argument(
        "name", help="Skill name to reset (e.g. google-workspace)"
    )
    skills_reset.add_argument(
        "--restore",
        action="store_true",
        help="Also delete the current copy and re-copy the bundled version",
    )
    skills_reset.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when using --restore",
    )

    skills_list_modified = skills_subparsers.add_parser(
        "list-modified",
        help="List bundled skills you've edited (which `hermes update` keeps)",
        description=(
            "Show the bundled skills whose local copy differs from the version last "
            "synced, i.e. the ones `hermes update` reports as user-modified and skips. "
            "Use `hermes skills diff <name>` to see changes and `hermes skills reset "
            "<name>` to resume updates."
        ),
    )
    skills_list_modified.add_argument(
        "--json",
        action="store_true",
        help="Output the list as JSON",
    )

    skills_diff = skills_subparsers.add_parser(
        "diff",
        help="Show how your copy of a bundled skill differs from the stock version",
        description=(
            "Print a unified diff between your local copy of a bundled skill and the "
            "current bundled (stock) version, so you can confirm what changed before "
            "running `hermes skills reset`."
        ),
    )
    skills_diff.add_argument(
        "name", help="Skill name to diff (e.g. google-workspace)"
    )

    skills_opt_out = skills_subparsers.add_parser(
        "opt-out",
        help="Stop bundled skills from being seeded into this profile",
        description=(
            "Write the .no-bundled-skills marker so the installer, "
            "`hermes update`, and any direct sync stop seeding bundled skills "
            "into the active profile. By default nothing already on disk is "
            "touched. Pass --remove to ALSO delete bundled skills that are "
            "unmodified (user-edited and hub/local skills are never removed)."
        ),
    )
    skills_opt_out.add_argument(
        "--remove",
        action="store_true",
        help="Also delete already-present unmodified bundled skills",
    )
    skills_opt_out.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when using --remove",
    )

    skills_opt_in = skills_subparsers.add_parser(
        "opt-in",
        help="Re-enable bundled-skill seeding (undo opt-out)",
        description=(
            "Remove the .no-bundled-skills marker so bundled skills are seeded "
            "again on the next `hermes update`. Pass --sync to re-seed now."
        ),
    )
    skills_opt_in.add_argument(
        "--sync",
        action="store_true",
        help="Re-seed bundled skills immediately instead of waiting for update",
    )

    skills_repair_official = skills_subparsers.add_parser(
        "repair-official",
        help="Backfill or restore official optional skills from repo source",
        description=(
            "Repair official optional skill provenance. By default, only backfills "
            "hub metadata for exact matches. Pass --restore to replace missing or "
            "mutated active copies from optional-skills/, moving existing copies to "
            "a restore backup first. Use name 'all' to repair every optional skill."
        ),
    )
    skills_repair_official.add_argument(
        "name", help="Official optional skill folder/frontmatter name, or 'all'"
    )
    skills_repair_official.add_argument(
        "--restore",
        action="store_true",
        help="Restore from official optional source, backing up existing matching copies",
    )
    skills_repair_official.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when using --restore",
    )

    skills_publish = skills_subparsers.add_parser(
        "publish", help="Publish a skill to a registry"
    )
    skills_publish.add_argument("skill_path", help="Path to skill directory")
    skills_publish.add_argument(
        "--to", default="github", choices=["github", "clawhub"], help="Target registry"
    )
    skills_publish.add_argument(
        "--repo", default="", help="Target GitHub repo (e.g. openai/skills)"
    )

    skills_snapshot = skills_subparsers.add_parser(
        "snapshot", help="Export/import skill configurations"
    )
    snapshot_subparsers = skills_snapshot.add_subparsers(dest="snapshot_action")
    snap_export = snapshot_subparsers.add_parser(
        "export", help="Export installed skills to a file"
    )
    snap_export.add_argument("output", help="Output JSON file path (use - for stdout)")
    snap_import = snapshot_subparsers.add_parser(
        "import", help="Import and install skills from a file"
    )
    snap_import.add_argument("input", help="Input JSON file path")
    snap_import.add_argument(
        "--force", action="store_true", help="Force install despite caution verdict"
    )

    skills_tap = skills_subparsers.add_parser("tap", help="Manage skill sources")
    tap_subparsers = skills_tap.add_subparsers(dest="tap_action")
    tap_subparsers.add_parser("list", help="List configured taps")
    tap_add = tap_subparsers.add_parser("add", help="Add a GitHub repo as skill source")
    tap_add.add_argument("repo", help="GitHub repo (e.g. owner/repo)")
    tap_rm = tap_subparsers.add_parser("remove", help="Remove a tap")
    tap_rm.add_argument("name", help="Tap name to remove")

    # config sub-action: interactive enable/disable
    skills_subparsers.add_parser(
        "config",
        help="Interactive skill configuration — enable/disable individual skills",
    )
    skills_parser.set_defaults(func=cmd_skills)
