---
name: writing-plans
description: Use when you have a spec or requirements for a multi-step task. Creates comprehensive implementation plans with bite-sized tasks, exact file paths, and complete code examples.
version: 1.0.0
author: Hermes Agent (adapted from Superpowers)
license: MIT
metadata:
  hermes:
    tags: [planning, design, implementation, workflow, documentation]
    related_skills: [subagent-driven-development, test-driven-development, requesting-code-review]
---

# Writing Implementation Plans

## Overview

Transform specifications into actionable implementation plans. Write comprehensive plans that any developer can follow - even with zero context about your codebase.

**Core principle:** Document everything: exact file paths, complete code, test commands, verification steps.

**Assume the implementer:**
- Is a skilled developer
- Knows almost nothing about your codebase
- Has questionable taste in code style
- Needs explicit guidance

## When to Use

**Always use this skill:**
- Before implementing multi-step features
- After design approval (from brainstorming)
- When breaking down complex requirements
- Before delegating to subagents

**Don't skip when:**
- Feature seems simple (assumptions cause bugs)
- You plan to implement yourself (future you needs guidance)
- Working alone (documentation matters)

## Plan Document Structure

### Header (Required)

Every plan MUST start with:

```markdown
# [Feature Name] Implementation Plan

> **For Hermes:** Use subagent-driven-development or executing-plans skill to implement this plan.

**Goal:** One sentence describing what this builds

**Architecture:** 2-3 sentences about approach

**Tech Stack:** Key technologies/libraries

---
```

### Task Structure

Each task follows this format:

```markdown
### Task N: [Descriptive Name]

**Objective:** What this task accomplishes (one sentence)

**Files:**
- Create: `exact/path/to/new_file.py`
- Modify: `exact/path/to/existing.py:45-67` (line numbers if known)
- Test: `tests/path/to/test_file.py`

**Implementation Steps:**

**Step 1: [Action description]**
```python
# Complete code to write
class NewClass:
    def __init__(self):
        self.value = None
    
    def process(self, input):
        return input.upper()
```

**Step 2: [Action description]**
```bash
# Command to run
pytest tests/test_new.py -v
```
Expected: Tests pass with 3 green dots

**Step 3: [Action description]**
```python
# More code if needed
```

**Verification:**
- [ ] Test passes
- [ ] Function returns expected output
- [ ] No syntax errors

**Commit:**
```bash
git add src/new_file.py tests/test_new.py
git commit -m "feat: add new feature component"
```
```

## Task Granularity

**Each task = 2-5 minutes of work**

**Break down:**
- "Write failing test" - one task
- "Run test to verify it fails" - one task  
- "Implement minimal code" - one task
- "Run test to verify pass" - one task
- "Commit" - one task

**Examples:**

**Too big:**
```markdown
### Task 1: Build authentication system
[50 lines of code across 5 files]
```

**Right size:**
```markdown
### Task 1: Create User model with email field
[10 lines, 1 file]

### Task 2: Add password hash field to User
[8 lines, 1 file]

### Task 3: Create password hashing utility
[15 lines, 1 file]
```

## Principles

### DRY (Don't Repeat Yourself)

**Bad:** Copy-paste validation in 3 places
**Good:** Extract validation function, use everywhere

```python
# Good - DRY
def validate_email(email):
    if not re.match(r'^[^@]+@[^@]+$', email):
        raise ValueError("Invalid email")
    return email

# Use everywhere
validate_email(user_input)
validate_email(config_email)
validate_email(imported_data)
```

### YAGNI (You Aren't Gonna Need It)

**Bad:** Add "flexibility" for future requirements
**Good:** Implement only what's needed now

```python
# Bad - YAGNI violation
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.preferences = {}  # Not needed yet!
        self.metadata = {}     # Not needed yet!
        self.settings = {}     # Not needed yet!

# Good - YAGNI
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### TDD (Test-Driven Development)

Every task that produces code should include:
1. Write failing test
2. Run to verify failure
3. Write minimal code
4. Run to verify pass

See `test-driven-development` skill for details.

### Frequent Commits

Commit after every task:
```bash
git add [files]
git commit -m "type: description"
```

## Writing Process

### Step 1: Understand Requirements

Read and understand:
- Feature requirements
- Design documents
- Acceptance criteria
- Constraints

### Step 2: Explore Codebase

```bash
# Understand project structure
find src -type f -name "*.py" | head -20

# Look at similar features
grep -r "similar_pattern" src/

# Check existing tests
ls tests/
```

### Step 3: Design Approach

Decide:
- Architecture pattern
- File organization
- Dependencies needed
- Testing strategy

### Step 4: Write Tasks

Create tasks in order:
1. Setup/infrastructure
2. Core functionality
3. Edge cases
4. Integration
5. Cleanup

### Step 5: Add Details

For each task, add:
- Exact file paths
- Complete code examples
- Exact commands
- Expected outputs
- Verification steps

### Step 6: Review Plan

Check:
- [ ] Tasks are sequential and logical
- [ ] Each task is bite-sized (2-5 min)
- [ ] File paths are exact
- [ ] Code examples are complete
- [ ] Commands are exact with expected output
- [ ] No missing context

### Step 7: Save Plan

```bash
# Create plans directory
mkdir -p docs/plans

# Save plan
cat > docs/plans/YYYY-MM-DD-feature-name.md << 'EOF'
[plan content]
EOF

# Commit plan
git add docs/plans/YYYY-MM-DD-feature-name.md
git commit -m "docs: add implementation plan for feature"
```

## Example Plan

```markdown
# User Authentication Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement this plan.

**Goal:** Add JWT-based user authentication to the Flask API

**Architecture:** Use PyJWT for tokens, bcrypt for hashing. Middleware validates tokens on protected routes.

**Tech Stack:** Python, Flask, PyJWT, bcrypt

---

### Task 1: Create User model

**Objective:** Define User model with email and hashed password

**Files:**
- Create: `src/models/user.py`
- Test: `tests/models/test_user.py`

**Step 1: Write failing test**
```python
def test_user_creation():
    user = User(email="test@example.com", password="secret123")
    assert user.email == "test@example.com"
    assert user.password_hash is not None
    assert user.password_hash != "secret123"  # Should be hashed
```

**Step 2: Run to verify failure**
```bash
pytest tests/models/test_user.py -v
```
Expected: FAIL - User class not defined

**Step 3: Implement User model**
```python
import bcrypt

class User:
    def __init__(self, email, password):
        self.email = email
        self.password_hash = bcrypt.hashpw(
            password.encode(), 
            bcrypt.gensalt()
        )
```

**Step 4: Run to verify pass**
```bash
pytest tests/models/test_user.py -v
```
Expected: PASS

**Commit:**
```bash
git add src/models/user.py tests/models/test_user.py
git commit -m "feat: add User model with password hashing"
```

### Task 2: Create login endpoint

**Objective:** Add POST /login endpoint that returns JWT

**Files:**
- Modify: `src/app.py`
- Test: `tests/test_login.py`

[Continue...]
```

## Common Mistakes

### Vague Tasks

**Bad:**
```markdown
### Task 1: Add authentication
```

**Good:**
```markdown
### Task 1: Create User model with email and password_hash fields
```

### Incomplete Code

**Bad:**
```markdown
Step 1: Add validation function
```

**Good:**
```markdown
Step 1: Add validation function
```python
def validate_email(email):
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError(f"Invalid email: {email}")
    return email
```

### Missing Verification

**Bad:**
```markdown
Step 3: Test it works
```

**Good:**
```markdown
Step 3: Verify authentication
```bash
curl -X POST http://localhost:5000/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"secret123"}'
```
Expected: Returns 200 with JWT token in response
```

## Integration with Other Skills

### With brainstorming

**Sequence:**
1. brainstorming → Explore and refine design
2. writing-plans → Create implementation plan
3. subagent-driven-development → Execute plan

### With subagent-driven-development

Plans feed into subagent-driven-development:
- Subagents implement each task
- Two-stage review ensures quality
- Plan provides context and requirements

### With test-driven-development

Every code-producing task should follow TDD:
1. Write failing test
2. Verify failure
3. Write minimal code
4. Verify pass

## Success Checklist

Before considering a plan complete:

- [ ] Header with goal, architecture, tech stack
- [ ] All tasks are bite-sized (2-5 min each)
- [ ] Exact file paths for every file
- [ ] Complete code examples (not partial)
- [ ] Exact commands with expected output
- [ ] Verification steps for each task
- [ ] Commit commands included
- [ ] DRY, YAGNI, TDD principles applied
- [ ] Tasks are sequential and logical
- [ ] Plan saved to docs/plans/

## Remember

```
Bite-sized tasks
Exact file paths
Complete code
Exact commands
Verification steps
DRY, YAGNI, TDD
```

**A good plan makes implementation obvious.**
