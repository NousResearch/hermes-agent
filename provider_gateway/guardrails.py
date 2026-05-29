"""PII Sanitizer & Guardrails Cerdas.

Detects, redacts, and restores sensitive information (PII) from prompts and
response streams to prevent leaks to third-party LLMs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class PIISanitizer:
    """Scans and redacts PII/secrets in prompts, and restores them in responses."""

    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}
        self._counter = 1

        # Regex patterns for sensitive information
        self.patterns = {
            "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "PRIVATE_KEY": re.compile(
                r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"
            ),
            "API_KEY": re.compile(
                r"\b(?:sk-[a-zA-Z0-9]{32,}|AIzaSy[a-zA-Z0-9_-]{33}|key-[a-zA-Z0-9]{16,})\b"
            ),
        }

    def sanitize_prompt(self, prompt: str) -> str:
        """Scan and redact PII in the prompt, bypassing markdown code blocks."""
        if not prompt:
            return prompt

        # Split prompt into markdown code blocks and normal text
        # This regex matches code blocks like ```python ... ```
        code_block_regex = re.compile(r"(```[\s\S]*?```)")
        parts = code_block_regex.split(prompt)

        sanitized_parts = []
        for i, part in enumerate(parts):
            # Odd indexes are code blocks, leave them untouched to avoid false positives
            if i % 2 == 1:
                sanitized_parts.append(part)
            else:
                sanitized_parts.append(self._sanitize_text(part))

        return "".join(sanitized_parts)

    def _sanitize_text(self, text: str) -> str:
        """Apply PII sanitization to a plain text string."""
        current_text = text

        for pii_type, regex in self.patterns.items():
            def replace_match(match: re.Match[str]) -> str:
                val = match.group(0)
                # Avoid re-sanitizing already sanitized parts or duplicate replacements
                for ph, orig in self._mapping.items():
                    if orig == val:
                        return ph

                placeholder = f"[REDACTED_{pii_type}_{self._counter}]"
                self._mapping[placeholder] = val
                self._counter += 1
                return placeholder

            current_text = regex.sub(replace_match, current_text)

        return current_text

    def restore_response(self, response_text: str) -> str:
        """Restore all redacted placeholders in the response back to original values."""
        if not response_text or not self._mapping:
            return response_text

        restored_text = response_text
        # Sort placeholders by length descending to avoid partial replacement issues
        for placeholder, original in sorted(self._mapping.items(), key=lambda x: len(x[0]), reverse=True):
            restored_text = restored_text.replace(placeholder, original)

        return restored_text

    def get_deanonimizer(self) -> StreamingDeanonimizer:
        """Create a streaming de-anonimizer bound to this sanitizer instance."""
        return StreamingDeanonimizer(self)


class StreamingDeanonimizer:
    """Manages real-time de-anonymization of text streaming chunks using a sliding buffer."""

    def __init__(self, sanitizer: PIISanitizer) -> None:
        self.sanitizer = sanitizer
        self.buffer = ""
        self.emitted_restored_len = 0

    def process_chunk(self, chunk: str) -> str:
        """Process an incoming stream chunk, buffering potential partial placeholders."""
        self.buffer += chunk

        # Look for a potential opening bracket of a placeholder near the end of the buffer
        # Placeholder format is like: [REDACTED_EMAIL_1]
        # We search up to 40 characters from the end for a '[' that has no matching ']'
        last_bracket_idx = self.buffer.rfind("[")
        
        if last_bracket_idx != -1:
            # Check if there is a closing bracket after it
            close_bracket_idx = self.buffer.find("]", last_bracket_idx)
            if close_bracket_idx == -1:
                # Bracket is open and unfinished! Hold this part back.
                safe_part = self.buffer[:last_bracket_idx]
            else:
                # Bracket is closed, it's safe to process everything
                safe_part = self.buffer
        else:
            safe_part = self.buffer

        # Restore PII in the safe part
        restored_safe = self.sanitizer.restore_response(safe_part)
        
        # Calculate delta to emit
        delta = restored_safe[self.emitted_restored_len:]
        self.emitted_restored_len = len(restored_safe)
        
        return delta

    def flush(self) -> str:
        """Flush the remaining buffer at the end of the stream."""
        restored_all = self.sanitizer.restore_response(self.buffer)
        delta = restored_all[self.emitted_restored_len:]
        self.emitted_restored_len = len(restored_all)
        return delta
