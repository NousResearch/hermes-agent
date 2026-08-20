"""Vertical Agent Kit — scaffold a domain-bound Hermes profile from blueprints.

This command surfaces the Hermes-native pattern documented in
``website/docs/guides/vertical-agents.md`` as a reusable CLI helper.  It does
not add new runtime primitives; it composes the existing ones (profiles,
SOUL.md, USER.md, skills, platform_toolsets) from a set of bundled blueprints.

Subcommands:
  hermes vertical-agent init          Interactive wizard to scaffold a profile
  hermes vertical-agent list          List bundled blueprints
  hermes vertical-agent verify PATH   Validate the generated scaffold shape
  hermes vertical-agent smoke PATH      Best-effort runnability check
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional

try:
    from hermes_constants import display_hermes_home, get_hermes_home
except Exception:  # pragma: no cover - fallback when running outside Hermes package
    def display_hermes_home() -> str:
        return "~/.hermes"

    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"


logger = None  # lazily loaded hermes_logging

_KIT_DATA = Path(__file__).parent / "vertical_agent_kit_data"
_BLUEPRINTS_DIR = _KIT_DATA / "blueprints"
_TEMPLATES_DIR = _KIT_DATA / "templates"

_DEFAULT_BLUEPRINTS = ["support", "research"]


def _get_logger():
    """Return a logger, lazily initializing from hermes_logging if available."""
    global logger
    if logger is None:
        try:
            import hermes_logging as _hl

            logger = _hl.get_logger(__name__)
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
    return logger


# -----------------------------------------------------------------------------
# Blueprint discovery
# -----------------------------------------------------------------------------


def list_blueprints() -> List[str]:
    """Return the names of bundled blueprints."""
    if not _BLUEPRINTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in _BLUEPRINTS_DIR.iterdir()
        if d.is_dir() and (d / "SOUL.md").exists()
    )


def blueprint_path(name: str) -> Optional[Path]:
    """Return the path to a bundled blueprint, or None if it does not exist."""
    candidate = _BLUEPRINTS_DIR / name
    if candidate.is_dir() and (candidate / "SOUL.md").exists():
        return candidate
    return None


# -----------------------------------------------------------------------------
# Templating
# -----------------------------------------------------------------------------


_SIMPLE_VARS = [
    "PROFILE_NAME",
    "ROLE",
    "OBJECTIVE",
    "USERS",
    "TONE",
    "SCOPE",
    "REFUSALS",
    "SOURCES",
    "SYSTEMS",
    "DECISION_STYLE",
]


def _sluggify(value: str) -> str:
    """Make a filesystem-friendly slug from free text."""
    value = re.sub(r"[^\w\s-]", "", value.lower().strip())
    value = re.sub(r"[\s_]+", "-", value).strip("-")
    return value or "agent"


def _render_template(text: str, variables: Dict[str, str]) -> str:
    """Substitute ``{{VAR}}`` placeholders in a template string."""
    return Template(re.sub(r"\{\{(\w+)\}\}", r"${\1}", text)).safe_substitute(variables)


def _validate_profile_name(profile_name: str) -> str:
    """Return a safe directory name or raise ValueError for invalid input."""
    profile_name = profile_name.strip()
    if not profile_name or profile_name in {".", ".."}:
        raise ValueError("Profile name cannot be empty, '.', or '..'")
    # Reject path separators, parent references, and absolute paths.
    if any(c in profile_name for c in "\\/:") or ".." in profile_name:
        raise ValueError(f"Invalid profile name: {profile_name}")
    return profile_name


def render_blueprint(
    blueprint_name: str,
    output_dir: Path,
    variables: Dict[str, str],
    *,
    overwrite: bool = False,
) -> List[Path]:
    """Copy a bundled blueprint into ``output_dir``, rendering placeholders.

    Returns the list of files written.
    """
    src = blueprint_path(blueprint_name)
    if src is None:
        raise ValueError(f"Unknown blueprint: {blueprint_name}")

    profile_name = _validate_profile_name(
        variables.get("PROFILE_NAME", blueprint_name)
    )
    output_dir = output_dir.expanduser().resolve()
    dest = output_dir / profile_name

    # Ensure dest is strictly contained within output_dir.
    try:
        dest.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError(f"Profile name escapes output directory: {profile_name}") from exc

    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dest}")
    if dest.exists():
        # Only overwrite a directory that looks like a previously generated
        # scaffold. Wholesale deletion of arbitrary directories is dangerous.
        existing = _find_scaffold_files(dest)
        if not any(existing.values()):
            raise FileExistsError(
                f"Destination exists and does not look like a vertical-agent scaffold: {dest}"
            )
        shutil.rmtree(dest)

    written: List[Path] = []
    for src_file in sorted(src.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        content = src_file.read_text(encoding="utf-8")
        rendered = _render_template(content, variables)
        dst_file.write_text(rendered, encoding="utf-8")
        written.append(dst_file)

    return written


# -----------------------------------------------------------------------------
# User interaction
# -----------------------------------------------------------------------------


def _input_default(prompt: str, default: str = "") -> str:
    """Read a line of input, returning ``default`` when the user presses enter.

    ``EOFError`` is re-raised so that callers can treat Ctrl+D/EOF as an abort
    signal rather than silently accepting the default.
    """
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ")
    return answer.strip() or default


def _prompt_for_variables(blueprint_name: str) -> Dict[str, str]:
    """Run the interactive wizard and return the render variables."""
    print("\nVertical Agent Kit — init wizard")
    print("Press Enter to accept the default shown in brackets.\n")

    profile = _input_default("Profile / agent name", "my-agent")
    available = list_blueprints()
    if blueprint_name not in available:
        print(f"\nAvailable blueprints: {', '.join(available)}")
        blueprint = _input_default("Blueprint", "support")
    else:
        blueprint = blueprint_name

    if blueprint not in list_blueprints():
        raise ValueError(f"Unknown blueprint: {blueprint}")

    out_dir = _input_default("Output directory", "./out")

    print("\nProfile questions:")
    role = _input_default("Role", f"{blueprint.replace('-', ' ').title()} specialist")
    objective = _input_default("Objective", f"Help the team with {blueprint} work")
    users = _input_default("Primary users", "internal team")
    tone = _input_default("Tone", "concise, calm, direct")
    scope = _input_default("Scope boundary", "stay inside the support domain")
    refusals = _input_default("Refusal edges", "redirect requests outside scope")
    sources = _input_default("Evidence sources", "knowledge base, tickets, account data")
    systems = _input_default("Systems of record", "help desk / CRM")
    decision_style = _input_default("Decision style", "evidence first, escalate when uncertain")

    return {
        "PROFILE_NAME": profile,
        "ROLE": role,
        "OBJECTIVE": objective,
        "USERS": users,
        "TONE": tone,
        "SCOPE": scope,
        "REFUSALS": refusals,
        "SOURCES": sources,
        "SYSTEMS": systems,
        "DECISION_STYLE": decision_style,
        "BLUEPRINT": blueprint,
        "OUTPUT_DIR": out_dir,
    }


# -----------------------------------------------------------------------------
# Verification / smoke
# -----------------------------------------------------------------------------


def _find_scaffold_files(path: Path) -> Dict[str, Optional[Path]]:
    """Locate the expected files directly inside a scaffold directory."""
    return {
        "SOUL.md": path / "SOUL.md" if (path / "SOUL.md").exists() else None,
        "USER.template.md": path / "USER.template.md" if (path / "USER.template.md").exists() else None,
        "OPERATIONS.md": path / "OPERATIONS.md" if (path / "OPERATIONS.md").exists() else None,
    }


def verify_scaffold(path: Path) -> List[str]:
    """Return a list of verification errors; empty means valid."""
    errors: List[str] = []
    if not path.exists():
        errors.append(f"Path does not exist: {path}")
        return errors
    if not path.is_dir():
        errors.append(f"Not a directory: {path}")
        return errors

    expected = _find_scaffold_files(path)
    for name, found in expected.items():
        if found is None:
            errors.append(f"Missing {name}")
        elif not found.read_text(encoding="utf-8").strip():
            errors.append(f"{name} is empty")

    return errors


def smoke_scaffold(path: Path) -> tuple[List[str], List[str]]:
    """Best-effort smoke test: verify files and try a lightweight Hermes probe.

    Returns ``(errors, warnings)``. The missing-CLI case is a warning, not an
    error, because the kit is often run before Hermes is fully configured.
    """
    errors = verify_scaffold(path)
    warnings: List[str] = []
    if errors:
        return errors, warnings

    soul = _find_scaffold_files(path).get("SOUL.md")
    assert soul is not None
    if "voice" not in soul.read_text(encoding="utf-8").lower():
        errors.append("SOUL.md does not mention voice/identity")

    # Optional: if Hermes CLI is on PATH, try a dry-run prompt.
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        try:
            subprocess.run(
                [hermes_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or "").strip()[:200]
            errors.append(f"Hermes --version probe failed: {err or exc}")
        except subprocess.TimeoutExpired as exc:
            errors.append(f"Hermes --version probe timed out: {exc}")
        except Exception as exc:
            errors.append(f"Hermes --version probe failed: {exc}")
    else:
        warnings.append(
            "Hermes CLI not found on PATH; smoke test limited to file checks"
        )

    return errors, warnings


# -----------------------------------------------------------------------------
# argparse builders / handlers
# -----------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Run the init wizard and render the chosen blueprint."""
    try:
        variables = _prompt_for_variables(args.blueprint or "")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(variables["OUTPUT_DIR"]).expanduser().resolve()
    try:
        profile_name = _validate_profile_name(variables["PROFILE_NAME"])
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        written = render_blueprint(
            variables["BLUEPRINT"],
            output_dir,
            variables,
            overwrite=args.force,
        )
    except FileExistsError as exc:
        print(f"Error: {exc}. Use --force to overwrite.", file=sys.stderr)
        return 1
    except Exception as exc:
        _get_logger().exception("Failed to render blueprint")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\n✓ Scaffolded {variables['BLUEPRINT']} agent into {output_dir / profile_name}")
    print("Files generated:")
    for f in written:
        print(f"  - {f.relative_to(output_dir)}")

    user_target = get_hermes_home() / "memories" / "USER.md"
    print(
        f"\nNext steps:\n"
        f"  1. Review SOUL.md, OPERATIONS.md, and USER.template.md.\n"
        f"  2. Copy USER.template.md to {display_hermes_home()}/memories/USER.md if this is the default profile.\n"
        f"  3. Create a Hermes profile: hermes profile create {profile_name} --clone\n"
        f"  4. Verify: hermes vertical-agent verify {output_dir / profile_name}"
    )
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    """Print the bundled blueprints."""
    blueprints = list_blueprints()
    if not blueprints:
        print("No bundled blueprints found.")
        return 0
    print("Bundled vertical-agent blueprints:")
    for name in blueprints:
        readme = blueprint_path(name)
        desc = ""
        if readme:
            readme_file = readme / "README.md"
            if readme_file.exists():
                first = readme_file.read_text(encoding="utf-8").splitlines()[0]
                desc = f" — {first.lstrip('# ').strip()}"
        print(f"  - {name}{desc}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Verify a generated scaffold."""
    errors = verify_scaffold(Path(args.path).expanduser().resolve())
    if errors:
        print("Verification failed:")
        for err in errors:
            print(f"  ✗ {err}")
        return 1
    print(f"✓ {args.path} looks like a valid vertical-agent scaffold.")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Smoke-test a generated scaffold."""
    errors, warnings = smoke_scaffold(Path(args.path).expanduser().resolve())
    if errors:
        print("Smoke test failed:")
        for err in errors:
            print(f"  ✗ {err}")
        return 1
    for warn in warnings:
        print(f"  ⚠ {warn}")
    print(f"✓ {args.path} passed the smoke test.")
    return 0


def build_vertical_agent_parser(
    subparsers: "argparse._SubParsersAction[Any]",
) -> argparse.ArgumentParser:
    """Register ``hermes vertical-agent`` and its subcommands."""
    parser = subparsers.add_parser(
        "vertical-agent",
        aliases=["vak"],
        help="Scaffold a constrained vertical Hermes agent from blueprints",
        description=(
            "Generate Hermes profiles, SOUL.md, USER.md, and operating rules "
            "for a domain-specific agent. Companion to the 'Building Constrained "
            "Vertical Agents' guide."
        ),
    )
    sub = parser.add_subparsers(dest="vertical_agent_command", required=True)

    init = sub.add_parser("init", help="Interactive wizard to scaffold a new agent")
    init.add_argument(
        "--blueprint",
        default=None,
        help="Blueprint to use (default: prompt from list)",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing scaffold directory",
    )
    init.set_defaults(func=_cmd_init)

    sub.add_parser("list", aliases=["ls"], help="List bundled blueprints").set_defaults(
        func=_cmd_list
    )

    verify = sub.add_parser("verify", help="Validate a generated scaffold")
    verify.add_argument("path", help="Path to the scaffold directory")
    verify.set_defaults(func=_cmd_verify)

    smoke = sub.add_parser("smoke", help="Best-effort smoke test of a scaffold")
    smoke.add_argument("path", help="Path to the scaffold directory")
    smoke.set_defaults(func=_cmd_smoke)

    return parser


def cmd_vertical_agent(args: argparse.Namespace) -> int:
    """Dispatch ``hermes vertical-agent`` to the right subcommand handler."""
    return args.func(args)
