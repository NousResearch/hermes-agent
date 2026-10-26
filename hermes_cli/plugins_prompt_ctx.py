"""``PluginContext`` registrars for plugin text that core renders into a prompt: bounded system
prompt sections (frozen into each new session)."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from hermes_cli.plugins_dispatch import (
    DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS, MAX_SYSTEM_PROMPT_SECTION_CHARS,
    SYSTEM_PROMPT_SECTION_POSITIONS, PluginSystemPromptSection, is_valid_system_prompt_section_id,
)
from hermes_cli.plugins_ledger import PluginRegistration


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
