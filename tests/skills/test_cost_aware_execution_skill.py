import os
import pytest

def get_skill_path():
    """Locate the cost-aware-execution skill file directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "skills", "productivity", "cost-aware-execution", "SKILL.md")

def test_skill_markdown_file_exists():
    """Ensure the cost-aware-execution SKILL.md file exists in the repository layout."""
    skill_file = get_skill_path()
    assert os.path.exists(skill_file), f"SKILL.md file not found at {skill_file}"

def test_skill_frontmatter_constraints():
    """Verify that frontmatter properties comply with Hermes agent specifications."""
    skill_file = get_skill_path()
    
    with open(skill_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check for markdown YAML block isolation bounds
    assert content.count("---") >= 2, "SKILL.md lacks a properly formatted YAML frontmatter section"
    
    frontmatter_block = content.split("---")[1]
    lines = [line.strip() for line in frontmatter_block.split("\n") if line.strip()]
    
    # Store clean key-value lookups
    frontmatter = {}
    for line in lines:
        if ":" in line and not line.startswith("-"):
            key, val = line.split(":", 1)
            frontmatter[key.strip()] = val.strip()

    # Constraint Check 1: Verify correct name identification
    assert frontmatter.get("name") == "cost-aware-execution", "Skill name property missing or mismatch"
    
    # Constraint Check 2: Verify descriptions do not exceed strict 60 character threshold
    description = frontmatter.get("description", "")
    assert description, "Skill description is completely missing"
    assert len(description) <= 60, f"Description contains {len(description)} characters. Max limit is 60."

def test_skill_slash_command_activation_path():
    """Ensure the slash command syntax is properly documented for system parsing registration."""
    skill_file = get_skill_path()
    
    with open(skill_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Framework validation: Make sure the normalized command trigger is declared explicitly
    assert "/cost-aware-execution" in content, "Markdown fails to declare the framework supported /cost-aware-execution command"
