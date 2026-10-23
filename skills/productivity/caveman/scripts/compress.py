#!/usr/bin/env python3
"""
Caveman Compress — Compress natural language files into caveman format.
Usage: python3 compress.py <filepath>
"""

import sys
import re
import shutil
from pathlib import Path


# Patterns to preserve EXACTLY
PRESERVE_PATTERNS = [
    (r'```[\s\S]*?```', 'CODE_BLOCK'),           # Fenced code blocks
    (r'`[^`\n]+`', 'INLINE_CODE'),               # Inline code
    (r'https?://[^\s\)]+', 'URL'),               # URLs
    (r'`[^`]+`', 'BACKTICK'),
    (r'\b[A-Z]{2,}\b', 'ACRONYM'),               # Acronyms (DB, API, etc.)
    (r'\$\w+', 'ENV_VAR'),                       # $HOME, $NODE_ENV
    (r'\b\d+\.\d+\.\d+\b', 'VERSION'),           # Semver
    (r'\b\d{4}-\d{2}-\d{2}\b', 'DATE'),          # ISO dates
]

# Words to remove
REMOVE_WORDS = {
    'a', 'an', 'the',
    'just', 'really', 'basically', 'actually', 'simply', 'essentially', 'generally',
    'sure', 'certainly', 'of course', 'happy to',
    'however', 'furthermore', 'additionally', 'in addition',
    'perhaps', 'maybe', 'I think', 'I believe', 'it seems',
}

# Hedging phrases to remove
HEDGING_PATTERNS = [
    r'\bit might be worth\b',
    r'\byou could consider\b',
    r'\bit would be good to\b',
    r'\byou should\b',
    r'\bmake sure to\b',
    r'\bremember to\b',
    r'\bensure that\b',
    r'\bin order to\b',
    r'\bthe reason is because\b',
]

SHORT_SYNONYMS = {
    'extensive': 'big', 'solution': 'fix', 'utilize': 'use', 'assist': 'help',
    'obtain': 'get', 'display': 'show', 'locate': 'find', 'construct': 'build',
    'verify': 'check', 'resolve': 'fix', 'implement': 'add', 'delete': 'remove',
    'modify': 'change', 'upgrade': 'update', 'enhance': 'improve', 'performance': 'opt',
    'application': 'app', 'configuration': 'config', 'environment': 'env',
    'database': 'DB', 'authentication': 'auth', 'authorization': 'authz',
    'request': 'req', 'response': 'res', 'function': 'fn', 'implementation': 'impl',
    'middleware': 'mw', 'parameter': 'param', 'argument': 'arg', 'variable': 'var',
    'component': 'comp', 'repository': 'repo', 'directory': 'dir', 'production': 'prod',
    'development': 'dev', 'staging': 'stage', 'integration': 'int', 'e2e': 'e2e',
}


def compress_text(text: str) -> str:
    """Compress natural language text while preserving code/technical content."""

    # Step 1: Extract and preserve protected regions
    protected = []
    placeholder = '___CAVEMAN_PROTECTED_{}___'

    def protect(match):
        idx = len(protected)
        protected.append(match.group(0))
        return placeholder.format(idx)

    # Apply protection patterns
    working = text
    for pattern, _ in PRESERVE_PATTERNS:
        working = re.sub(pattern, protect, working)

    # Step 2: Compress the unprotected text
    # Remove hedging patterns
    for pattern in HEDGING_PATTERNS:
        working = re.sub(pattern, '', working, flags=re.IGNORECASE)

    # Remove filler words (whole words only)
    words = working.split()
    filtered = []
    for word in words:
        clean = word.strip('.,;:!?()[]{}"\'')
        if clean.lower() not in REMOVE_WORDS:
            filtered.append(word)
    working = ' '.join(filtered)

    # Apply short synonyms
    for long, short in SHORT_SYNONYMS.items():
        working = re.sub(rf'\b{long}\b', short, working, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    working = re.sub(r'\s+([.,;:!?)}\]])', r'\1', working)
    working = re.sub(r'([({[])\s+', r'\1', working)
    working = re.sub(r'\s{2,}', ' ', working)

    # Step 3: Restore protected regions
    for idx, original in enumerate(protected):
        working = working.replace(placeholder.format(idx), original)

    return working.strip()


def process_file(filepath: Path) -> tuple[bool, str]:
    """Process a single file. Returns (success, message)."""

    # Check file type
    if filepath.suffix in {'.py', '.js', '.ts', '.json', '.yaml', '.yml',
                            '.toml', '.env', '.lock', '.css', '.html', '.xml',
                            '.sql', '.sh', '.rs', '.go', '.java', '.cpp', '.c',
                            '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt',
                            '.scala', '.clj', '.ex', '.exs', '.elm', '.ml', '.mli',
                            '.fs', '.fsi', '.v', '.sv', '.vhd', '.vhdl'}:
        return False, f"Skipped: {filepath.suffix} files are not compressed"

    # Read original
    try:
        original = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"Read error: {e}"

    # Backup
    backup = filepath.with_name(filepath.name + '.original.md')
    if not backup.exists():
        shutil.copy2(filepath, backup)

    # Compress
    compressed = compress_text(original)

    if compressed == original:
        return True, "No changes (already compressed or no compressible content)"

    # Write
    try:
        filepath.write_text(compressed, encoding='utf-8')
        return True, f"Compressed ({len(original)} → {len(compressed)} chars, {100*(1-len(compressed)/len(original)):.0f}% reduction)"
    except Exception as e:
        return False, f"Write error: {e}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compress.py <filepath>")
        sys.exit(1)

    filepath = Path(sys.argv[1]).expanduser().resolve()

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    success, msg = process_file(filepath)
    print(msg)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()