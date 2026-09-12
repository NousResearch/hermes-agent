"""Generated Discord slash-command groups backed by live Hermes state."""

from __future__ import annotations

from typing import Optional

from .. import adapter as _adapter

discord = _adapter.discord
logger = _adapter.logger


class GeneratedCommandMixin:
    """Mixin containing catalog-backed slash command registration."""

    def _register_skill_group(self, tree) -> None:
        """Register one flat ``/skill`` command with autocomplete on ``name``.
        A nested ``/skill <category> <name>`` layout blew Discord's ~8000-byte payload cap and broke
        ``tree.sync()``; autocomplete options are fetched dynamically. Entries live on ``self``.

        The older nested layout (``/skill <category> <name>``) registered one giant command whose serialized
        payload grew linearly with the skill catalog — with the default ~75 skills the payload was ~14 KB
        and ``tree.sync()`` rejected the entire slash-command batch (issues 11321, #10259, #11385, #10261,
        #10214).
        """
        try:
            existing_names = set()
            try:
                existing_names = {cmd.name for cmd in tree.get_commands()}
            except Exception:
                pass
            # Instance-level state so the callbacks always read the freshest entries.
            self._skill_entries: list[tuple[str, str, str]] = []
            self._skill_lookup: dict[str, tuple[str, str]] = {}
            self._skill_group_reserved_names: set[str] = set(existing_names)
            self._refresh_skill_catalog_state()
            if not self._skill_entries:
                return

            async def _autocomplete_name(interaction: "discord.Interaction", current: str) -> list:
                """Filter skills by typed prefix against name and description (Discord caps at 25).
                Unauthorized users get ``[]``: no catalog leak, no per-keystroke ephemeral rejections."""
                try:
                    allowed, _reason = self._evaluate_slash_authorization(interaction)
                except Exception:
                    # Never raise from autocomplete; fail closed.
                    return []
                if not allowed:
                    return []
                q = (current or "").strip().lower()
                choices: list = []
                for name, desc, _key in self._skill_entries:
                    if not q or q in name.lower() or (desc and q in desc.lower()):
                        label = f"{name} — {desc}" if desc else name
                        # Discord's Choice.name is capped at 100 chars.
                        if len(label) > 100:
                            label = label[:97] + "..."
                        choices.append(discord.app_commands.Choice(name=label, value=name))
                        if len(choices) >= 25:
                            break
                return choices

            @discord.app_commands.describe(
                name="Which skill to run", args="Optional arguments for the skill",
            )
            @discord.app_commands.autocomplete(name=_autocomplete_name)
            async def _skill_handler(interaction: "discord.Interaction", name: str, args: str = ""):
                # Authorize BEFORE lookup so unknown/known names reject identically (no catalog probing).
                if not await self._check_slash_authorization(interaction, "/skill"):
                    return
                entry = self._skill_lookup.get(name)
                if not entry:
                    await interaction.response.send_message(
                        f"Unknown skill: `{name}`. Start typing for "
                        f"autocomplete suggestions.",
                        ephemeral=True,
                    )
                    return
                _desc, cmd_key = entry
                await self._run_simple_slash(interaction, f"{cmd_key} {args}".strip())
            cmd = discord.app_commands.Command(
                name="skill", description="Run a Hermes skill", callback=_skill_handler,
            )
            tree.add_command(cmd)
            logger.info(
                "[%s] Registered /skill command with %d skill(s) via autocomplete",
                self.name, len(self._skill_entries),
            )
            if self._skill_group_hidden_count:
                logger.info(
                    "[%s] %d skill(s) filtered out of /skill (name clamp / reserved)",
                    self.name, self._skill_group_hidden_count,
                )
        except Exception as exc:
            logger.warning("[%s] Failed to register /skill command: %s", self.name, exc)

    def _refresh_skill_catalog_state(self) -> None:
        """Re-scan disk and repopulate ``self._skill_entries``/``_skill_lookup`` in place.
        No Discord API calls: autocomplete and handler read these attributes directly."""
        from hermes_cli.commands_platforms import discord_skill_commands_by_category
        reserved = getattr(self, "_skill_group_reserved_names", set())
        categories, uncategorized, hidden = discord_skill_commands_by_category(
            reserved_names=set(reserved),
        )
        entries: list[tuple[str, str, str]] = list(uncategorized)
        for cat_skills in categories.values():
            entries.extend(cat_skills)
        # Stable alphabetical order so autocomplete is predictable across restarts.
        entries.sort(key=lambda t: t[0])
        self._skill_entries = entries
        self._skill_lookup = {n: (d, k) for n, d, k in entries}
        self._skill_group_hidden_count = hidden

    def refresh_skill_group(self) -> tuple[int, int]:
        """Rescan skills and refresh live ``/skill`` autocomplete; returns ``(new_count, hidden_count)``.
        Called after ``reload_skills``; no ``tree.sync()`` since autocomplete options are dynamic."""
        try:
            self._refresh_skill_catalog_state()
        except Exception as exc:
            logger.warning(
                "[%s] Failed to refresh /skill autocomplete after reload: %s", self.name, exc,
            )
            return (len(getattr(self, "_skill_entries", [])), 0)
        logger.info(
            "[%s] Refreshed /skill autocomplete: %d skill(s) available (%d filtered)", self.name,
            len(self._skill_entries), self._skill_group_hidden_count,
        )
        return (len(self._skill_entries), self._skill_group_hidden_count)
