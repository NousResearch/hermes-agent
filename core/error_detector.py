#!/usr/bin/env python3
"""
Error Pattern Detector — inspired by ClaudeCodeFramework's ReflectionEngine.

Detects repeated error patterns from a list of error strings and returns
structured pattern information including root cause hints.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional


class ErrorPatternDetector:
    """Detects repeated error patterns from a list of error strings."""

    def __init__(self, threshold: int = 2):
        """Initialize the detector.

        Args:
            threshold: Minimum number of occurrences to report as a pattern.
        """
        self.threshold = threshold

    def detect(self, errors: List[str]) -> List[Dict]:
        """Detect repeated error patterns from a list of errors.

        Args:
            errors: List of error strings to analyze.

        Returns:
            List of dicts, each containing:
                - pattern: The normalized error pattern string
                - count: Number of occurrences
                - examples: Up to 3 example error strings
                - root_cause_hint: Human-readable root cause description
        """
        if not errors:
            return []

        # Normalize all errors
        normalized = [self._normalize_error(e) for e in errors]

        # Group similar patterns
        groups: List[List[str]] = []
        used = set()

        for i, norm in enumerate(normalized):
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j in range(i + 1, len(normalized)):
                if j in used:
                    continue
                if self._similarity(norm, normalized[j]) > 0.7:
                    group.append(j)
                    used.add(j)
            groups.append(group)

        # Build result patterns
        results = []
        for group in groups:
            count = len(group)
            if count < self.threshold:
                continue

            # Use the most common normalized form as the pattern
            pattern = normalized[group[0]]

            # Collect examples (original errors, not normalized)
            example_indices = group[:3]
            examples = [errors[idx] for idx in example_indices]

            results.append({
                "pattern": pattern,
                "count": count,
                "examples": examples,
                "root_cause_hint": self._infer_root_cause(pattern),
            })

        # Sort by count descending
        results.sort(key=lambda x: x["count"], reverse=True)
        return results

    def _normalize_error(self, error: str) -> str:
        """Normalize an error string by removing specific values.

        Removes file paths, line numbers, numeric values, and UUIDs
        to extract the error TYPE and MESSAGE STRUCTURE.

        Args:
            error: The raw error string.

        Returns:
            Normalized error string with specific values replaced.
        """
        if not error:
            return ""

        result = error

        # Remove full file paths with line numbers: /path/to/file.py:45 or /path/to/file.py:line:45
        result = re.sub(r'[/\\][\w\-. ]+\.py:\d+', '<file>', result)
        result = re.sub(r'[/\\][\w\-. ]+\.py', '<file>', result)

        # Remove UUIDs
        result = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '<uuid>',
            result,
            flags=re.IGNORECASE,
        )

        # Remove hex addresses (0x7f...)
        result = re.sub(r'0x[0-9a-fA-F]+', '<addr>', result)

        # Remove numeric values (but keep version numbers like v1, v2)
        # Replace standalone numbers
        result = re.sub(r'(?<![w\w])\b\d+\b(?![w\d.])', '<num>', result)

        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings.

        Uses difflib.SequenceMatcher.ratio() for fuzzy matching.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Similarity ratio between 0.0 and 1.0.
        """
        return SequenceMatcher(None, s1, s2).ratio()

    def _infer_root_cause(self, pattern: str) -> str:
        """Infer the root cause from an error pattern.

        Args:
            pattern: The normalized error pattern string.

        Returns:
            Human-readable root cause hint.
        """
        pattern_lower = pattern.lower()

        # Import errors
        if "import" in pattern_lower or "modulenotfounderror" in pattern_lower \
                or "importerror" in pattern_lower:
            return "ImportError - check module path and dependencies"

        # Network errors
        if "connection" in pattern_lower or "timeout" in pattern_lower \
                or "network" in pattern_lower or "httperror" in pattern_lower:
            return "NetworkError - check network and timeout"

        # Permission errors
        if "permission" in pattern_lower or "access denied" in pattern_lower \
                or "eacces" in pattern_lower:
            return "PermissionError - check file permissions"

        # Memory errors
        if "memory" in pattern_lower or "out of memory" in pattern_lower \
                or "oom" in pattern_lower:
            return "MemoryError - check memory usage"

        # Syntax errors
        if "syntax" in pattern_lower or "syntaxerror" in pattern_lower \
                or "parse error" in pattern_lower:
            return "SyntaxError - check code syntax"

        # Type errors
        if "typeerror" in pattern_lower or "type error" in pattern_lower:
            return "TypeError - check data types"

        # Value errors
        if "valueerror" in pattern_lower or "value error" in pattern_lower:
            return "ValueError - check invalid values"

        # File errors
        if "file" in pattern_lower or "not found" in pattern_lower \
                or "enoent" in pattern_lower or "no such file" in pattern_lower:
            return "FileError - check file paths"

        # Default
        return "Unknown - further analysis needed"


# Module-level convenience function
def detect_error_patterns(errors: List[str], threshold: int = 2) -> List[Dict]:
    """Convenience function to detect error patterns.

    Args:
        errors: List of error strings to analyze.
        threshold: Minimum occurrences to report as pattern.

    Returns:
        List of detected patterns with metadata.
    """
    detector = ErrorPatternDetector(threshold=threshold)
    return detector.detect(errors)
