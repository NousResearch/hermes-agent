# Test Documentation: skillify_solution()

## Overview

This document provides test cases for the `skillify_solution()` function. This function transforms a problem-solution pair into a skill documentation file with proper frontmatter formatting, skill name generation, and structured body content.

---

## Test Case Table

| ID | Description | Input | Expected Output |
|----|-------------|-------|-----------------|
| SF-001 | Generate skill with valid problem and solution | skillify_solution("Fixing Python import errors", "Use sys.path.insert(0, directory)") | Skill file with generated name, frontmatter, and body |
| SF-002 | Skill name generation from problem | skillify_solution("How to configure SSH keys?") | Skill name auto-generated from problem keywords |
| SF-003 | Frontmatter format - required fields | skillify_solution with standard problem/solution | Frontmatter contains: skill_name, description, created_at, tags |
| SF-004 | Frontmatter format - tags array | skillify_solution with problem containing "Python", "SSH" | Tags array includes ["python", "ssh"] (lowercase) |
| SF-005 | Body format - problem section | skillify_solution with problem | Body contains "## Problem" section with problem text |
| SF-006 | Body format - solution section | skillify_solution with solution | Body contains "## Solution" section with solution text |
| SF-007 | Empty solution handling | skillify_solution("Any problem", "") | Error returned or empty solution handled gracefully |
| SF-008 | Empty problem handling | skillify_solution("", "Some solution") | Error returned or problem required validation |
| SF-009 | Long problem text | skillify_solution("Long " + "x" * 5000, "Solution") | Skill file generated, truncation if configured |
| SF-010 | Long solution text | skillify_solution("Problem", "Long " + "x" * 10000) | Skill file generated, content preserved or truncated per config |
| SF-011 | Special characters in problem | skillify_solution("Fix <error> with 'quotes' and \"double quotes\"", "Solution") | Special chars preserved in output |
| SF-012 | Special characters in solution | skillify_solution("Problem", "Solution with <html> & \"entities\"") | Special chars preserved in output |
| SF-013 | Unicode in problem | skillify_solution("How to handle 日本語 error?", "Solution") | Unicode preserved in output |
| SF-014 | Unicode in solution | skillify_solution("Problem", "Solution: 使用中文") | Unicode preserved in output |
| SF-015 | Markdown in problem | skillify_solution("Use `code` and **bold**", "Solution") | Markdown syntax preserved |
| SF-016 | Markdown in solution | skillify_solution("Problem", "Use ```python\nprint('hello')\n```") | Markdown code blocks preserved |
| SF-017 | Skill name collision handling | skillify_solution("Fix import errors", "Solution") twice | Unique skill names or append suffix (_1, _2) |
| SF-018 | Missing optional parameters | skillify_solution("Problem", "Solution") with no tags | Frontmatter has empty or default tags |
| SF-019 | Custom tags provided | skillify_solution("Problem", "Solution", tags=["custom", "test"]) | Tags array contains provided custom tags |
| SF-020 | Generated skill file path | skillify_solution returns file path | Valid path string returned for file creation |

---

## Test Execution Steps

### SF-001: Basic Skill Generation
1. Call skillify_solution("Fixing Python import errors", "Use sys.path.insert(0, directory)")
2. Verify return contains valid skill_name
3. Verify frontmatter section exists (--- delimited)
4. Verify body contains Problem and Solution sections

### SF-002: Skill Name Generation
1. Call skillify_solution with descriptive problem text
2. Verify generated skill_name is non-empty
3. Verify skill_name derives from problem keywords
4. Verify skill_name format is valid (alphanumeric, hyphens, underscores)

### SF-003 & SF-004: Frontmatter Format
1. Call skillify_solution with problem containing keywords
2. Extract frontmatter section (between --- markers)
3. Parse as YAML
4. Verify required fields present: skill_name, description, created_at, tags
5. For SF-004: Verify tags are lowercase and extracted from problem

### SF-005 & SF-006: Body Format
1. Call skillify_solution with known problem and solution
2. Extract body (after second --- marker)
3. Verify "## Problem" header present with problem text
4. Verify "## Solution" header present with solution text

### SF-007 & SF-008: Empty Input Handling
1. For SF-007: Call with empty solution string
2. Verify error returned OR empty solution handled gracefully
3. For SF-008: Call with empty problem string
4. Verify error returned OR validation message about required problem

### SF-009 & SF-010: Long Input Handling
1. For SF-009: Create problem with 5000+ characters
2. Call skillify_solution and verify successful generation
3. Check if truncation occurs per configuration
4. For SF-010: Create solution with 10000+ characters
5. Verify generation succeeds
6. Verify content preservation or truncation per spec

### SF-011 & SF-012: Special Characters
1. For SF-011: Include <, >, ', " characters in problem
2. For SF-012: Include same in solution
3. Call skillify_solution
4. Verify special characters preserved (or properly escaped in YAML)

### SF-013 & SF-014: Unicode Handling
1. For SF-013: Include non-ASCII characters (Japanese, Chinese) in problem
2. For SF-014: Include same in solution
3. Call skillify_solution
4. Verify unicode characters preserved correctly

### SF-015 & SF-016: Markdown Preservation
1. For SF-015: Include inline code and bold in problem
2. For SF-016: Include fenced code block in solution
3. Call skillify_solution
4. Verify markdown syntax preserved in body

### SF-017: Skill Name Collision
1. Call skillify_solution twice with same problem
2. Verify both return successfully
3. Verify skill names are unique (suffix added if needed)

### SF-018: Missing Optional Parameters
1. Call skillify_solution without tags parameter
2. Verify frontmatter has tags (empty list or default value)

### SF-019: Custom Tags
1. Call skillify_solution with tags=["custom", "test"]
2. Verify frontmatter tags array contains exactly the provided tags

### SF-020: Return Value - File Path
1. Call skillify_solution
2. Verify return value includes a valid file path
3. Verify path format is appropriate for skill file storage

---

## Pass/Fail Criteria

- **All tests pass** if:
  - skillify_solution() returns valid skill document structure
  - Frontmatter contains all required fields
  - Body format includes Problem and Solution sections
  - Skill names are auto-generated when not provided
  - Tags are correctly extracted or accept custom values
  - Special characters and unicode are preserved
  - Markdown syntax is preserved
  - Long inputs are handled gracefully (generated or truncated)
  - Empty inputs are validated and rejected with clear error
  - Skill name collisions result in unique names
  - File path is returned for downstream file creation

- **Test fails** if:
  - Output lacks required frontmatter fields
  - Body sections are missing or incorrectly formatted
  - Special characters are corrupted or cause parsing errors
  - Unicode is mangled
  - Markdown is stripped or incorrectly rendered
  - Empty inputs cause crashes instead of validation errors
  - Skill name collision results in duplicate names
  - No file path returned when file creation is needed
