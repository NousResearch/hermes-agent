"""
skillify.py - Auto-generate SKILL.md files from solved problems

This module takes a problem description and successful solution, then generates
a reusable SKILL.md file that can be loaded by the Hermes skill system.
"""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def skillify_solution(
    problem: str,
    solution: str,
    context: Dict[str, Any],
    skills_dir: str = "~/.hermes/skills/auto-generated",
) -> str:
    """
    Generate a SKILL.md file from a problem description and solution.

    Args:
        problem: Description of the problem that was solved.
        solution: The solution that successfully resolved the problem.
        context: Additional context (may include 'category', 'keywords', etc.).
        skills_dir: Directory to store generated skill files.

    Returns:
        The name of the created skill (e.g., 'skill-a1b2c3d4').
    """
    skill_name = _generate_skill_name(problem)
    frontmatter = _generate_frontmatter(skill_name, problem, context)
    body = _generate_body(solution, problem)
    content = frontmatter + "\n" + body

    # Resolve and create skills directory
    skills_path = Path(os.path.expanduser(skills_dir))
    skills_path.mkdir(parents=True, exist_ok=True)

    # Write the SKILL.md file
    skill_file = skills_path / skill_name / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(content)

    return skill_name


def _generate_skill_name(problem: str) -> str:
    """
    Generate a skill name by hashing the problem string.

    Args:
        problem: The problem description.

    Returns:
        A skill name like 'skill-<hash>' where hash is first 8 chars of md5.
    """
    hash_digest = hashlib.md5(problem.encode()).hexdigest()[:8]
    return f"skill-{hash_digest}"


def _generate_frontmatter(
    skill_name: str,
    problem: str,
    context: Dict[str, Any],
) -> str:
    """
    Generate YAML frontmatter for a SKILL.md file.

    Args:
        skill_name: The generated skill name.
        problem: The problem description.
        context: Additional context dict.

    Returns:
        YAML-formatted frontmatter string.
    """
    description = problem[:100].strip()
    if len(problem) > 100:
        description += "..."

    # Extract keywords from context or generate from problem
    keywords = context.get("keywords", [])
    if not keywords:
        keywords = _extract_keywords(problem)

    category = context.get("category", _infer_category(problem))
    created_date = datetime.now().strftime("%Y-%m-%d")

    # Build YAML frontmatter
    frontmatter = f"""---
name: {skill_name}
description: {description}
trigger keywords:
{_format_list_field(keywords)}
category: {category}
created: {created_date}
---"""

    return frontmatter


def _generate_body(solution: str, problem: str) -> str:
    """
    Generate the body content for a SKILL.md file.

    Args:
        solution: The solution that was applied.
        problem: The original problem description.

    Returns:
        Formatted skill body with Problem, Solution, and Usage sections.
    """
    body = f"""## Problem

{problem}

## Solution

{solution}

## Usage

Apply this skill when you encounter a similar problem.
Review the solution section and adapt the approach to your specific context.
"""
    return body


def _extract_keywords(problem: str) -> list:
    """
    Extract trigger keywords from problem text.

    Args:
        problem: The problem description.

    Returns:
        List of meaningful keywords.
    """
    # Remove common stop words and extract significant terms
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "i", "you", "we", "they",
        "what", "which", "who", "whom", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "any", "no", "not", "only", "same", "so", "than", "too", "very",
    }

    # Extract words ( alphanumeric + underscores )
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", problem.lower())

    # Filter stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Return unique keywords (max 8)
    unique = list(dict.fromkeys(keywords))[:8]
    return unique


def _infer_category(problem: str) -> str:
    """
    Infer a category from the problem description.

    Args:
        problem: The problem description.

    Returns:
        A category string.
    """
    problem_lower = problem.lower()

    category_keywords = {
        "code": ["code", "function", "method", "class", "algorithm", "implementation", "bug", "error", "exception"],
        "data": ["data", "database", "query", "sql", "table", "record", "schema", "migration"],
        "file": ["file", "directory", "path", "folder", "read", "write", "parse", "format"],
        "system": ["system", "process", "server", "service", "config", "setup", "install", "deploy"],
        "api": ["api", "request", "response", "endpoint", "http", "json", "rest", "graphql"],
        "git": ["git", "commit", "branch", "merge", "rebase", "diff", "stash"],
        "docker": ["docker", "container", "image", "dockerfile", "compose", "kubernetes", "k8s"],
        "test": ["test", "testing", "unit", "integration", "coverage", "pytest", "unittest"],
    }

    for category, keywords in category_keywords.items():
        if any(kw in problem_lower for kw in keywords):
            return category

    return "general"


def _format_list_field(items: list) -> str:
    """
    Format a list as YAML list items.

    Args:
        items: List of items to format.

    Returns:
        YAML-formatted list string.
    """
    if not items:
        return "  - (none)"
    return "\n".join(f"  - {item}" for item in items)
