"""``PluginContext`` registrars for plugin text that core renders into a prompt: bounded system
prompt sections (frozen into each new session) and background-review destinations (appended to the
review fork's prompt when the user enables them, with the plugin's own tools admitted for it)."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from hermes_cli.plugins_dispatch import (
    DEFAULT_REVIEW_DESTINATION_MAX_CHARS, DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    MAX_REVIEW_DESTINATION_CHARS, MAX_REVIEW_DESTINATION_TOOLS, MAX_SYSTEM_PROMPT_SECTION_CHARS,
    REVIEW_DESTINATION_SCOPES, SYSTEM_PROMPT_SECTION_POSITIONS, PluginSystemPromptSection,
    ReviewDestination, is_valid_system_prompt_section_id,
)
from hermes_cli.plugins_ledger import PluginRegistration


def _names(value: Iterable[str] | str) -> tuple:
    """One name for a bare string, else the iterable's items deduplicated in order."""
    return (value,) if isinstance(value, str) else tuple(dict.fromkeys(value))


class PluginPromptCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``_manager``, ``plugin_id``,
    ``_register_entry``)."""

    def register_system_prompt_section(
        self, id: str, content: str | Callable[[Mapping[str, Any]], str], *,
        position: str = "after_memory", max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded context frozen into each new session prompt. Callables receive a
        read-only session-info mapping; the rendered prompt is persisted by core verbatim."""
        if not is_valid_system_prompt_section_id(id):
            raise ValueError("system prompt section id must be 1-128 lowercase characters "
                             "using letters, numbers, '.', '_', or '-'")
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError("system prompt section position must be one of: "
                             + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS)))
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS):
            raise ValueError(f"system prompt section max_chars must be between 1 and {MAX_SYSTEM_PROMPT_SECTION_CHARS}")
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(f"system prompt section {id!r} is already registered by plugin {existing.plugin!r}")
        section = PluginSystemPromptSection(
            id=id, content=content, position=position, max_chars=max_chars, plugin=self.plugin_id,
        )
        return self._register_entry("system_prompt_section", id, self._manager._system_prompt_sections,
                                    section, "Plugin %s registered system prompt section: %s", id,
                                    previous=existing)

    def register_review_destination(
        self, id: str, instructions: str | Callable[[Mapping[str, Any]], str], *,
        tools: Iterable[str] | str, on: Iterable[str] | str = ("memory",),
        max_chars: int = DEFAULT_REVIEW_DESTINATION_MAX_CHARS,
    ) -> PluginRegistration:
        """Offer the background review a place to keep things beside memory and skills: bounded
        ``instructions`` plus up to four ``tools`` this plugin registered (``register_tool``
        first). Inert until the user lists ``id`` in ``auxiliary.background_review.destinations``;
        then each review whose trigger is in ``on`` (``"memory"``, ``"skills"``) appends the
        instructions and may call those tools. Callables receive a read-only session-info mapping
        and run in the review thread, once per review."""
        if not is_valid_system_prompt_section_id(id):
            raise ValueError("review destination id must be 1-128 lowercase characters "
                             "using letters, numbers, '.', '_', or '-'")
        if not isinstance(instructions, str) and not callable(instructions):
            raise TypeError("review destination instructions must be a string or callable")
        tool_names, scopes = _names(tools), frozenset(_names(on))
        if (not 0 < len(tool_names) <= MAX_REVIEW_DESTINATION_TOOLS
                or not all(isinstance(name, str) and name for name in tool_names)):
            raise ValueError(f"review destination tools must name 1 to {MAX_REVIEW_DESTINATION_TOOLS} tools")
        foreign = [name for name in tool_names
                   if not self._manager._owns_registration(self.plugin_id, "tool", name)]
        if foreign:
            raise ValueError(f"review destination {id!r} names tools plugin {self.plugin_id!r} has not "
                             f"registered: {', '.join(foreign)} (call ctx.register_tool first)")
        if not scopes or not scopes <= REVIEW_DESTINATION_SCOPES:
            raise ValueError("review destination on= must name review triggers from: "
                             + ", ".join(sorted(REVIEW_DESTINATION_SCOPES)))
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 0 < max_chars <= MAX_REVIEW_DESTINATION_CHARS):
            raise ValueError(f"review destination max_chars must be between 1 and {MAX_REVIEW_DESTINATION_CHARS}")
        existing = self._manager._review_destinations.get(id)
        if existing is not None:
            raise ValueError(f"review destination {id!r} is already registered by plugin {existing.plugin!r}")
        destination = ReviewDestination(
            id=id, instructions=instructions, tools=tool_names, scopes=scopes, max_chars=max_chars,
            plugin=self.plugin_id,
        )
        return self._register_entry("review_destination", id, self._manager._review_destinations,
                                    destination, "Plugin %s registered review destination: %s", id)
