"""
Safe Truncation Module — preserves important identifiers when truncating context.

When context is truncated via head+tail, important symbols (file paths, URLs,
variable names, etc.) are extracted from the middle and preserved.
"""

import re
from typing import List, Tuple

# ── Identifier Patterns ──────────────────────────────────────────────────────

_IDENTIFIER_PATTERNS: List[Tuple[str, str]] = [
    # File paths: /home/user/project/file.py, ./config.yml, ~/dotfiles/.zshrc
    (r'(?:/[a-zA-Z0-9_.~-]+)+/[a-zA-Z0-9_.~-]+', 'file_path'),
    # URLs: https://example.com/path, http://api.service:8080/v2
    (r'https?://[a-zA-Z0-9._~:/\-?#\[\]@!$&\'()*+,;=%]+', 'url'),
    # Variable names: $VAR, ${VAR}, $HOME, $USER_ID
    (r'\$[a-zA-Z_][a-zA-Z0-9_]*|\$\{[a-zA-Z_][a-zA-Z0-9_]*\}', 'env_var'),
    # Python/Filesystem identifiers with dots: module.submodule, obj.attr
    (r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+\b', 'qualified_name'),
    # Function definitions: def function_name(
    (r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', 'function_def'),
    # Class definitions: class ClassName
    (r'class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[\(:]', 'class_def'),
    # Import statements: from module import name, import module
    (r'(?:from\s+[a-zA-Z_][a-zA-Z0-9_.]+\s+)?import\s+[a-zA-Z_][a-zA-Z0-9_,\s]+', 'import_statement'),
    # API endpoints: /api/v2/users, /v1/resource/{id}
    (r'/[a-zA-Z0-9_/{}~-]+(?:/[a-zA-Z0-9_~./{}-]*[a-zA-Z0-9_}~-]?)?', 'api_path'),
    # Hash/Commit refs: abc1234, 0x1a2b3c
    (r'\b[0-9a-f]{6,40}\b', 'hash_ref'),
]

# Compile patterns for performance
_COMPILED_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), label) for pattern, label in _IDENTIFIER_PATTERNS
]


# ── Core Functions ───────────────────────────────────────────────────────────

def _extract_identifiers(text: str) -> List[str]:
    """
    Extract all important identifiers from text.

    Returns a deduplicated list of identifiers found, preserving order of first
    appearance. Filters out very short matches that are likely noise.
    """
    seen = set()
    identifiers = []

    for pattern, label in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            identifier = match.group()
            # Filter: skip very short matches (likely false positives)
            if len(identifier) < 3:
                continue
            # Deduplicate by value
            if identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)

    return identifiers


def safe_truncate(
    text: str,
    max_length: int,
    head: int = None,
    tail: int = None,
    head_ratio: float = 0.4,
    tail_ratio: float = 0.4,
) -> Tuple[str, List[str]]:
    """
    Truncate text while preserving important identifiers from the middle.

    Strategy:
    - Keep `head_ratio` of available space from the start (default 40%)
    - Keep `tail_ratio` of available space from the end (default 40%)
    - If middle content is removed, extract and append identifiers from middle

    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        head: Explicit head size (overrides head_ratio if set)
        tail: Explicit tail size (overrides tail_ratio if set)
        head_ratio: Fraction of available space for head when head not explicit
        tail_ratio: Fraction of available space for tail when tail not explicit

    Returns:
        Tuple of (truncated_text, list_of_preserved_identifiers)
    """
    if len(text) <= max_length:
        return text, []

    marker = "[Identifiers preserved from middle]"

    # Extract identifiers from the middle BEFORE we decide head/tail sizes
    # Middle = everything between head and tail regions
    middle = text  # will recompute after we know head/tail
    identifiers = []

    head_size = head if head is not None else int(max_length * head_ratio)
    tail_size = tail if tail is not None else int(max_length * tail_ratio)

    # Enforce invariants
    head_size = min(head_size, max_length // 2)
    tail_size = min(tail_size, max_length // 2)

    # Middle = text between head and tail (what would be removed)
    middle = text[head_size:-tail_size] if tail_size > 0 else text[head_size:]

    identifiers = _extract_identifiers(middle)
    preserved_line = ""

    if identifiers:
        # Strategy: head and tail go at the edges; marker+ids fill the gap between them.
        # Compute how much space is left after reserving head and tail.
        # We want:  head_part + marker_line + tail_part  <= max_length

        head_part = text[:head_size]
        tail_part = text[-tail_size:] if tail_size > 0 else ""

        marker = "[Identifiers preserved from middle]"
        marker_label = f"\n{marker}: "

        # Start with all identifiers and shrink until everything fits
        id_list = list(identifiers)
        while id_list:
            preserved_line = marker_label + ", ".join(id_list)
            if len(head_part) + len(preserved_line) + len(tail_part) <= max_length:
                break
            id_list.pop()

        if not id_list:
            # Even a single identifier is too long — truncate it
            max_chars = max(0, max_length - len(head_part) - len(tail_part) - len(marker_label) - 4)
            if max_chars > 0:
                id_str = ", ".join(identifiers)
                preserved_line = marker_label + id_str[:max_chars] + "..."
            else:
                # No room for any identifiers at all
                preserved_line = ""

        # If still over budget, shrink head and tail to fit
        while len(head_part) + len(preserved_line) + len(tail_part) > max_length:
            if len(tail_part) > len(head_part) and len(tail_part) > 0:
                tail_part = tail_part[:-1]
            elif len(head_part) > 0:
                head_part = head_part[:-1]
            else:
                # Nothing left to shrink — truncate preserved_line
                max_avail = max(0, max_length - 1)
                preserved_line = preserved_line[:max_avail]
                break

        result = head_part + preserved_line + tail_part
    else:
        # No identifiers — simple head + tail, both must fit in max_length
        head_part = text[:head_size]
        tail_part = text[-tail_size:] if tail_size > 0 else ""

        while len(head_part) + len(tail_part) > max_length and (len(head_part) > 0 or len(tail_part) > 0):
            if len(head_part) > len(tail_part):
                head_part = head_part[:-1]
            else:
                tail_part = tail_part[1:] if tail_part else ""

        result = head_part + tail_part

    return result, identifiers


def truncate_context(
    text: str,
    max_length: int,
    head: int = None,
    tail: int = None,
) -> str:
    """
    Convenience wrapper — returns just the truncated text.
    Drop-in replacement for simple head+tail truncation.
    """
    result, _ = safe_truncate(text, max_length, head=head, tail=tail)
    return result


# ── Tests ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    def test_file_paths():
        text = (
            "File: /home/user/project/src/main.py contains the application entry point. "
            "Configuration loaded from /etc/app/config.yaml. "
            "Logs written to /var/log/app/application.log. "
            "User data stored in /home/user/.config/app/data.json"
        )
        result, ids = safe_truncate(text, max_length=120)
        print("=== File Paths Test ===")
        print(f"Original length: {len(text)}")
        print(f"Truncated:\n{result}")
        print(f"Preserved identifiers: {ids}")
        assert len(result) <= 120, f"Result too long: {len(result)}"
        # main.py and /home/user/project are in the head — check result contains them
        assert "main.py" in result, "main.py (head) should be in result"
        # Middle identifiers from the preserved middle section
        assert any("/etc/app/config" in id or "config.yaml" in id for id in ids), \
            "Middle identifiers (config.yaml, /etc/app/config) should be preserved"
        print("PASS\n")

    def test_urls():
        text = (
            "API available at https://api.example.com/v2/users endpoint. "
            "Documentation at http://docs.example.com/guide/quickstart.html. "
            "Callback URL: https://example.com/callback?token=abc123&redirect=/home. "
            "Health check: http://localhost:8080/health"
        )
        result, ids = safe_truncate(text, max_length=150)
        print("=== URLs Test ===")
        print(f"Original length: {len(text)}")
        print(f"Truncated:\n{result}")
        print(f"Preserved identifiers: {ids}")
        assert len(result) <= 150, f"Result too long: {len(result)}"
        assert any("https://" in id or "http://" in id for id in ids), "URLs should be preserved"
        print("PASS\n")

    def test_variable_names():
        text = (
            "Environment variables: $HOME=/home/user, $PROJECT_ROOT=/home/user/project. "
            "Access via os.environ.get('HOME'), config.get('PROJECT_ROOT'). "
            "User ID: $USER_ID, Session: $SESSION_TOKEN. "
            "Python vars: sys.path, os.environ, json.dumps()"
        )
        result, ids = safe_truncate(text, max_length=140)
        print("=== Variable Names Test ===")
        print(f"Original length: {len(text)}")
        print(f"Truncated:\n{result}")
        print(f"Preserved identifiers: {ids}")
        assert len(result) <= 140, f"Result too long: {len(result)}"
        assert any("$HOME" in id or "$USER" in id for id in ids), "Env vars should be preserved"
        print("PASS\n")

    def test_no_truncation_needed():
        short = "This is a short text."
        result, ids = safe_truncate(short, max_length=1000)
        assert result == short, "Short text should be unchanged"
        assert ids == [], "No identifiers for short text"
        print("=== No Truncation Test ===")
        print("PASS\n")

    def test_mixed_content():
        text = (
            "def process_user_data(user_id: int, config_path: str) -> dict:\n"
            "    import os\n"
            "    from src.utils.helpers import parse_config\n"
            "    # Load config from /etc/app/config.yaml\n"
            "    url = 'https://api.example.com/users/' + str(user_id)\n"
            "    return {'status': 'ok', 'user_id': user_id}\n"
            "    # Log file: /var/log/app/users.log\n"
            "    # Commit: a1b2c3d4e5f6"
        )
        result, ids = safe_truncate(text, max_length=200)
        print("=== Mixed Content Test ===")
        print(f"Original length: {len(text)}")
        print(f"Truncated:\n{result}")
        print(f"Preserved identifiers: {ids}")
        assert len(result) <= 200, f"Result too long: {len(result)}"
        print("PASS\n")

    def test_truncate_context_convenience():
        text = "A" * 500 + " important_value " + "B" * 500
        result = truncate_context(text, max_length=100)
        print("=== truncate_context() wrapper test ===")
        print(f"Original length: {len(text)}")
        print(f"Truncated length: {len(result)}")
        assert len(result) <= 100, f"Result too long: {len(result)}"
        print("PASS\n")

    tests = [
        test_file_paths,
        test_urls,
        test_variable_names,
        test_no_truncation_needed,
        test_mixed_content,
        test_truncate_context_convenience,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}\n")
            failed += 1

    print(f"Ran {passed + failed} tests, {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
