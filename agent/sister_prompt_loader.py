#!/usr/bin/env python3
"""
SISTER PROMPT LOADER
-------------------
Ensures byte-stable system prompt generation for the Hermes Sister system.
This module handles the assembly of role, domain, and identity into a
consistent format to maximize provider cache efficiency.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from agent import sister_registry

@dataclass(frozen=True)
class SisterPrompt:
    """A compiled prompt package for a specific sister."""
    sister_id: str
    system_prompt: str
    metadata: Dict[str, Any]

    def to_system_prompt(self) -> str:
        """Return the rendered prompt text.

        Kept as a compatibility helper for CLI/tooling code that expects a
        prompt package object to expose a conversion method.
        """
        return self.system_prompt

class SisterPromptLoader:
    """Handles the construction and caching of sister system prompts."""

    def __init__(self):
        self._registry = sister_registry.get_registry()
        self._cache: Dict[str, SisterPrompt] = {}

    def get_prompt_loader(self) -> SisterPromptLoader:
        """Singleton accessor."""
        return self

    def build_sister_prompt(self, sister: sister_registry.Sister) -> SisterPrompt:
        """
        Constructs a consistent system prompt for the given sister.

        The format is strictly structured to avoid whitespace variations
        that would invalidate LLM prompt caches.
        """
        if sister.id in self._cache:
            return self._cache[sister.id]

        # 1. Header: Core Identity
        # Format: [SISTER_ID] [NAME] - [ROLE]
        header = f"[{sister.id.upper()}] {sister.name} - {sister.role}"

        # 2. Domain and Archetype
        # Format: Domain: [DOMAIN] | Archetype: [ARCHETYPE]
        context = f"Domain: {sister.domain} | Archetype: {sister.archetype}"

        # 3. Core Directive (The 'Soul' of the persona)
        directive = sister.core_directive

        # 4. Constraints and Style
        # We join constraints and style into a stable list
        style_lines = [
            f"Style: {sister.personality_style}",
            f"Delegation Scope: {', '.join(sister.delegation_scope)}",
            f"Risk Level: {sister.risk_level}"
        ]
        constraints = "\n".join([f"- {line}" for line in style_lines])

        # Combine all parts with double newlines for clear separation
        full_prompt = (
            f"{header}\n"
            f"{'=' * len(header)}\n"
            f"{context}\n\n"
            f"CORE DIRECTIVE:\n{directive}\n\n"
            f"OPERATIONAL CONSTRAINTS:\n{constraints}"
        )

        prompt_pkg = SisterPrompt(
            sister_id=sister.id,
            system_prompt=full_prompt,
            metadata={
                "role": sister.role,
                "domain": sister.domain,
                "risk": sister.risk_level,
            }
        )

        self._cache[sister.id] = prompt_pkg
        return prompt_pkg

    def get_fallback_sister_id(self) -> str:
        """Returns the default system orchestrator ID."""
        return "astra"

    def get_sister_prompt(self, sister_id: str) -> Optional[SisterPrompt]:
        """High-level accessor to get a prompt by ID."""
        sister = self._registry.get(sister_id)
        if not sister:
            # Fallback to Astra
            sister = self._registry.get("astra")
            if not sister:
                return None

        return self.build_sister_prompt(sister)

# Global singleton for easy access
_loader_instance = SisterPromptLoader()

def get_prompt_loader() -> SisterPromptLoader:
    """Access the global prompt loader instance."""
    return _loader_instance


def get_system_prompt(sister_id: str) -> str:
    """Return the rendered system prompt for ``sister_id``.

    Invalid IDs fall back to Astra via ``SisterPromptLoader.get_sister_prompt``.
    An empty string is returned only if no registry/default prompt is available.
    """
    prompt = _loader_instance.get_sister_prompt(sister_id)
    return prompt.system_prompt if prompt else ""


def build_sister_system_prompt(sister_id: str) -> str:
    """Compatibility wrapper used by system-prompt assembly."""
    return get_system_prompt(sister_id)
