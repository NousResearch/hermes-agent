"""
Skill Factory Plugin for Hermes
================================
Watches your workflows and generates reusable skills automatically.

This plugin provides the command interface for the Skill Factory meta-skill.
The core behavior is defined in skills/skill-factory/SKILL.md — install both
for the full experience.

Install:
    cp skill_factory.py ~/.hermes/plugins/
    cp -r skills/skill-factory ~/.hermes/skills/meta/

Usage:
    /skill-factory propose          Propose a skill from current session
    /skill-factory list             List skills generated this session
    /skill-factory status           Show tracked patterns
    /skill-factory queue            Show all pending pattern proposals
    /skill-factory save <name>      Save last proposed skill with a custom name
    /skill-factory clear            Clear current session tracking log

Note: The Hermes plugin API (hermes.command, hermes.tool, etc.) is based on
the v2026.3.x plugin architecture. Adjust registration calls if you are on a
different version.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

PLUGIN_NAME = "skill-factory"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Meta-skill that watches workflows and generates reusable Hermes skills"

def get_hermes_home() -> Path:
    """Resolve the Hermes home directory dynamically."""
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
PLUGINS_DIR = HERMES_HOME / "plugins"

# ---------------------------------------------------------------------------
# Session state (in-memory, per Hermes session)
# ---------------------------------------------------------------------------

class SessionTracker:
    """Tracks workflow patterns within the current Hermes session."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.generated_skills: list[dict[str, Any]] = []
        self.proposal_queue: list[dict[str, Any]] = []
        self.last_proposal: dict[str, Any] | None = None
        self.session_start = datetime.now()

    def record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record a workflow event for pattern analysis."""
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })

    def add_to_queue(self, proposal: dict[str, Any]) -> None:
        self.proposal_queue.append(proposal)
        self.last_proposal = proposal

    def mark_generated(self, skill_name: str, files: list[str]) -> None:
        self.generated_skills.append({
            "name": skill_name,
            "files": files,
            "generated_at": datetime.now().isoformat(),
        })

    def clear(self) -> None:
        self.events.clear()
        self.proposal_queue.clear()
        self.last_proposal = None


# Global tracker (one per plugin lifetime / session)
_tracker = SessionTracker()


# ---------------------------------------------------------------------------
# Skill file generation
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Convert any string to a valid kebab-case skill name."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def _skill_dir(category: str, skill_name: str) -> Path:
    return SKILLS_DIR / category / skill_name


def generate_skill_md(
    skill_name: str,
    category: str,
    description: str,
    workflow_steps: list[str],
    examples: list[str],
    tags: list[str],
) -> tuple[Path, str]:
    """Generate a SKILL.md file and return (path, content)."""

    steps_md = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(workflow_steps))
    examples_md = ""
    for i, example in enumerate(examples, 1):
        examples_md += f"\n### Example {i}\n\n{example}\n"

    tags_str = ", ".join(tags)
    display_name = skill_name.replace("-", " ").title()

    content = textwrap.dedent(f"""\
        ---
        name: {display_name}
        version: 1.0.0
        category: {category}
        description: {description}
        tags: [{tags_str}]
        generated_by: skill-factory
        generated_at: {datetime.now().strftime("%Y-%m-%d")}
        ---

        # {display_name}

        {description}

        ## When to Activate

        Activate this skill when you need to perform the {display_name} workflow.
        This skill was auto-generated from a live session by Skill Factory.

        ## Workflow

        ### Steps

        {steps_md}

        ## Quality Checklist

        Before completing this workflow:
        - [ ] All steps completed in order
        - [ ] Output verified against expected result
        - [ ] No side effects left behind

        ## Examples
        {examples_md}

        ## Integration

        This skill was generated by the [Skill Factory](https://github.com/your-username/hermes-skill-factory)
        meta-skill. Edit this file to refine the workflow steps and examples.
    """)

    target_dir = _skill_dir(category, skill_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "SKILL.md"
    target_path.write_text(content, encoding="utf-8")
    return target_path, content


def generate_plugin_py(
    skill_name: str,
    description: str,
    workflow_steps: list[str],
) -> tuple[Path, str]:
    """Generate a plugin .py file and return (path, content)."""

    steps_comments = "\n    ".join(f"# Step {i+1}: {s}" for i, s in enumerate(workflow_steps))
    display_name = skill_name.replace("-", " ").title()

    content = textwrap.dedent(f'''\
        """
        {display_name} Plugin — Auto-generated by Skill Factory
        {description}

        Install: cp {skill_name}.py ~/.hermes/plugins/
        Usage:   /{skill_name}
        """

        from __future__ import annotations

        PLUGIN_NAME = "{skill_name}"
        PLUGIN_VERSION = "1.0.0"
        PLUGIN_DESCRIPTION = "{description}"


        def register(hermes):
            """Register the {display_name} skill as a Hermes command."""

            @hermes.command(
                name="{skill_name}",
                description="{description}",
                usage="/{skill_name} [args]",
            )
            async def run_{skill_name.replace("-", "_")}(ctx, args: str = ""):
                """Execute the {display_name} workflow.

                Auto-generated from a live session. Edit the steps below to
                refine the implementation.
                """
                {steps_comments}

                await ctx.reply(
                    "Running **{display_name}** workflow...\\n"
                    "Edit `~/.hermes/plugins/{skill_name}.py` to implement the steps."
                )
    ''')

    target_path = PLUGINS_DIR / f"{skill_name}.py"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return target_path, content


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(hermes):
    """Register Skill Factory commands with the Hermes agent."""

    # ------------------------------------------------------------------
    # /skill-factory propose
    # ------------------------------------------------------------------
    @hermes.command(
        name="skill-factory propose",
        description="Analyze the current session and propose the top detected workflow as a skill",
        usage="/skill-factory propose",
    )
    async def cmd_propose(ctx, args: str = ""):
        """Trigger an immediate skill proposal from the Skill Factory skill."""
        await ctx.reply(
            "🏭 **Skill Factory** — analyzing current session...\n\n"
            "The Skill Factory SKILL.md is now active. I'll review the workflows "
            "from this session and propose the most reusable one.\n\n"
            "_Tip: Make sure the `skill-factory` skill is installed in "
            "`~/.hermes/skills/meta/skill-factory/SKILL.md` for full AI-driven proposals._"
        )
        # The actual pattern analysis is handled by the Hermes AI via SKILL.md.
        # This command surfaces the intent to the agent loop.
        await ctx.inject_system_message(
            "The user has triggered /skill-factory propose. "
            "Analyze the full conversation history of this session. "
            "Identify the most valuable repeatable workflow. "
            "Present a Skill Factory proposal following the format defined in your active SKILL.md."
        )

    # ------------------------------------------------------------------
    # /skill-factory list
    # ------------------------------------------------------------------
    @hermes.command(
        name="skill-factory list",
        description="List all skills generated by Skill Factory in this session",
        usage="/skill-factory list",
    )
    async def cmd_list(ctx, args: str = ""):
        """Show skills generated during the current session."""
        if not _tracker.generated_skills:
            await ctx.reply("No skills generated yet this session. Try `/skill-factory propose`.")
            return

        lines = ["🏭 **Skills generated this session:**\n"]
        for skill in _tracker.generated_skills:
            lines.append(f"- **{skill['name']}** — {', '.join(skill['files'])}")
            lines.append(f"  Generated at {skill['generated_at'][:16]}")
        await ctx.reply("\n".join(lines))

    # ------------------------------------------------------------------
    # /skill-factory status
    # ------------------------------------------------------------------
    @hermes.command(
        name="skill-factory status",
        description="Show what patterns Skill Factory is currently tracking",
        usage="/skill-factory status",
    )
    async def cmd_status(ctx, args: str = ""):
        """Display the current session tracking status."""
        session_age = datetime.now() - _tracker.session_start
        minutes = int(session_age.total_seconds() / 60)

        status = (
            f"🏭 **Skill Factory Status**\n\n"
            f"- Session duration: {minutes} min\n"
            f"- Events tracked: {len(_tracker.events)}\n"
            f"- Skills in queue: {len(_tracker.proposal_queue)}\n"
            f"- Skills generated: {len(_tracker.generated_skills)}\n\n"
            f"_Skill Factory is watching silently. "
            f"Run `/skill-factory propose` to surface a proposal now._"
        )
        await ctx.reply(status)

    # ------------------------------------------------------------------
    # /skill-factory queue
    # ------------------------------------------------------------------
    @hermes.command(
        name="skill-factory queue",
        description="Show all detected patterns queued for skill proposals",
        usage="/skill-factory queue",
    )
    async def cmd_queue(ctx, args: str = ""):
        """Show the pending proposal queue."""
        if not _tracker.proposal_queue:
            await ctx.reply(
                "🏭 No patterns queued yet. Keep working — "
                "Skill Factory will detect repeatable workflows automatically."
            )
            return

        lines = [f"🏭 **Proposal Queue** ({len(_tracker.proposal_queue)} pending)\n"]
        for i, proposal in enumerate(_tracker.proposal_queue, 1):
            lines.append(f"{i}. **{proposal.get('name', 'unnamed')}** — {proposal.get('description', '')}")
        lines.append("\nRun `/skill-factory propose` to step through them.")
        await ctx.reply("\n".join(lines))

    # ------------------------------------------------------------------
    # /skill-factory save <name>
    # ------------------------------------------------------------------
    @hermes.command(
        name="skill-factory save",
        description="Save the last proposed skill with a custom name",
        usage="/skill-factory save <skill-name>",
    )
    async def cmd_save(ctx, args: str = ""):
        """Immediately generate and save the last proposed skill."""
        skill_name = _sanitize_name(args.strip()) if args.strip() else None

        if not skill_name:
            await ctx.reply(
                "Please provide a skill name. Example:\n"
                "`/skill-factory save my-workflow-name`"
            )
            return

        if not _tracker.last_proposal:
            await ctx.reply(
                "No skill proposal active. Run `/skill-factory propose` first, "
                "then use `/skill-factory save <name>` to save it."
            )
            return

        proposal = _tracker.last_proposal
        category = proposal.get("category", "custom")
        description = proposal.get("description", f"Auto-generated skill: {skill_name}")
        steps = proposal.get("steps", ["Step 1: implement me"])
        examples = proposal.get("examples", [])
        tags = proposal.get("tags", ["generated", "skill-factory"])

        generated_files = []

        try:
            md_path, _ = generate_skill_md(skill_name, category, description, steps, examples, tags)
            generated_files.append(str(md_path))

            py_path, _ = generate_plugin_py(skill_name, description, steps)
            generated_files.append(str(py_path))

            _tracker.mark_generated(skill_name, generated_files)

            await ctx.reply(
                f"✅ **Skill '{skill_name}' saved!**\n\n"
                f"Files written:\n"
                + "\n".join(f"- `{f}`" for f in generated_files)
                + f"\n\nRun `hermes skills reload` or restart Hermes to activate."
            )
        except Exception as e:
            await ctx.reply(f"❌ Failed to save skill: {e}")

    # ------------------------------------------------------------------
    # /skill-factory clear
    # ------------------------------------------------------------------
    @hermes.command(
        name="skill-factory clear",
        description="Clear the current session tracking log",
        usage="/skill-factory clear",
    )
    async def cmd_clear(ctx, args: str = ""):
        """Reset the session tracker."""
        _tracker.clear()
        await ctx.reply("🏭 Session log cleared. Skill Factory is watching fresh.")

    # ------------------------------------------------------------------
    # Hook: record tool calls for pattern detection
    # ------------------------------------------------------------------
    @hermes.on("tool_call")
    async def on_tool_call(ctx, tool_name: str, tool_args: dict, tool_result: Any):
        """Passively record tool calls for pattern analysis."""
        _tracker.record_event("tool_call", {
            "tool": tool_name,
            "args_keys": list(tool_args.keys()) if isinstance(tool_args, dict) else [],
        })

    @hermes.on("command")
    async def on_command(ctx, command: str, args: str):
        """Passively record commands for pattern analysis."""
        _tracker.record_event("command", {"command": command, "has_args": bool(args)})
