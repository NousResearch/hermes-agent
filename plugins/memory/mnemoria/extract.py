"""Rule-based fact extraction from conversation messages and text.

Extracts high-signal facts (errors, URLs near errors, file paths near errors,
user directives) as MEMORY_SPEC-formatted strings that MnemoriaStore.store()
can parse directly.
"""

import re
from dataclasses import dataclass
from typing import List

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "or", "but", "not", "no", "so",
    "project", "dark", "mode", "setup",
})


def content_slug(text: str, max_words: int = 3) -> str:
    """Generate a short slug from content for target discrimination."""
    if not text:
        return "general"
    words = re.findall(r"[\w.]+", text.lower())
    meaningful = [w for w in words if w not in _STOP_WORDS]
    if not meaningful:
        return "general"
    return "-".join(meaningful[:max_words])


@dataclass
class ExtractedFact:
    content: str        # MEMORY_SPEC notation, e.g. "V[url]: https://..."
    source: str         # "tool_result" or "user_statement"
    confidence: float   # 0.0-1.0

_ERROR_RE = re.compile(
    r"(?:error|failed|exception|traceback|assert(?:ion)?error|FAILED)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+")
_FILE_PATH_RE = re.compile(r"(/[\w./_+-]+\.\w+)")
_USER_DIRECTIVE_RE = re.compile(
    r"((?:always use|never use|prefer to use|don\'t use|don\'t ever|always make sure)\s+.+)",
    re.IGNORECASE,
)

def _extract_errors(lines: List[str], text: str) -> List[ExtractedFact]:
    facts = []
    for line in lines:
        if _ERROR_RE.search(line):
            summary = line.strip()[:200]
            facts.append(ExtractedFact(content=f"?[error]: {summary}", source="tool_result", confidence=0.9))
            break
    return facts

def _extract_urls_near_errors(lines: List[str]) -> List[ExtractedFact]:
    facts = []
    error_lines = {i for i, line in enumerate(lines) if _ERROR_RE.search(line)}
    if not error_lines:
        return facts
    seen = set()
    for i, line in enumerate(lines):
        if not any(abs(i - e) <= 3 for e in error_lines):
            continue
        for match in _URL_RE.finditer(line):
            url = match.group(0).rstrip(".,;:)\"'")
            if url not in seen:
                seen.add(url)
                facts.append(ExtractedFact(content=f"V[url]: {url}", source="tool_result", confidence=0.85))
    return facts

def _extract_file_paths_near_errors(lines: List[str]) -> List[ExtractedFact]:
    facts = []
    error_lines = {i for i, line in enumerate(lines) if _ERROR_RE.search(line)}
    if not error_lines:
        return facts
    seen = set()
    for i, line in enumerate(lines):
        if not any(abs(i - e) <= 3 for e in error_lines):
            continue
        for match in _FILE_PATH_RE.finditer(line):
            path = match.group(1)
            if len(path) < 5 or path.count("/") < 1:
                continue
            if path not in seen:
                seen.add(path)
                facts.append(ExtractedFact(content=f"V[file]: {path}", source="tool_result", confidence=0.85))
    return facts

def _extract_user_directives(text: str) -> List[ExtractedFact]:
    facts = []
    for match in _USER_DIRECTIVE_RE.finditer(text):
        directive = match.group(1).strip()[:200]
        facts.append(ExtractedFact(content=f"C[user.pref]: {directive}", source="user_statement", confidence=0.6))
    return facts

def _extract_user_urls(text: str) -> List[ExtractedFact]:
    facts = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:)\"'")
        facts.append(ExtractedFact(content=f"V[url]: {url}", source="user_statement", confidence=0.8))
    return facts

def _extract_user_file_paths(text: str) -> List[ExtractedFact]:
    facts = []
    for match in _FILE_PATH_RE.finditer(text):
        path = match.group(1)
        if len(path) < 5 or path.count("/") < 1:
            continue
        facts.append(ExtractedFact(content=f"V[file]: {path}", source="user_statement", confidence=0.8))
    return facts

def extract_from_text(text: str, source: str = "tool_result") -> List[ExtractedFact]:
    """Extract facts from a plain text string."""
    if not text or not text.strip():
        return []
    lines = text.splitlines()
    if source == "user_statement":
        facts = []
        facts.extend(_extract_user_directives(text))
        facts.extend(_extract_user_urls(text))
        facts.extend(_extract_user_file_paths(text))
        return facts
    facts = []
    facts.extend(_extract_errors(lines, text))
    facts.extend(_extract_urls_near_errors(lines))
    facts.extend(_extract_file_paths_near_errors(lines))
    return facts

def extract_from_messages(messages: List[dict], start_index: int = 0) -> tuple:
    """Extract facts from conversation messages. Returns (facts, new_last_index)."""
    all_facts: List[ExtractedFact] = []
    for i in range(start_index, len(messages)):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue
        if role == "tool":
            all_facts.extend(extract_from_text(content, source="tool_result"))
        elif role == "user":
            all_facts.extend(extract_from_text(content, source="user_statement"))
    return all_facts, len(messages)
