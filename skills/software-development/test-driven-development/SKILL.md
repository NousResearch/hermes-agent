---
name: test-driven-development
description: Use when implementing any feature or bugfix, before writing implementation code. Enforces RED-GREEN-REFACTOR cycle with test-first approach.
version: 1.0.0
author: Hermes Agent (adapted from Superpowers)
license: MIT
metadata:
  hermes:
    tags: [testing, tdd, development, quality, red-green-refactor]
    related_skills: [systematic-debugging, writing-plans, subagent-driven-development]
---

# Test-Driven Development (TDD)

## Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

**Violating the letter of the rules is violating the spirit of the rules.**

## When to Use

**Always:**
- New features
- Bug fixes
- Refactoring
- Behavior changes

**Exceptions (ask your human partner):**
- Throwaway prototypes
- Generated code
- Configuration files

Thinking "skip TDD just this once"? Stop. That's rationalization.

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over.

**No exceptions:**
- Don't keep it as "reference"
- Don't "adapt" it while writing tests
- Don't look at it
- Delete means delete

Implement fresh from tests. Period.

## Red-Green-Refactor Cycle

### RED - Write Failing Test

Write one minimal test showing what should happen.

**Good Example:**
```python
def test_retries_failed_operations_3_times():
    attempts = 0
    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception('fail')
        return 'success'
    
    result = retry_operation(operation)
    
    assert result == 'success'
    assert attempts == 3
```
- Clear name, tests real behavior, one thing

**Bad Example:**
```python
def test_retry_works():
    mock = MagicMock()
    mock.side_effect = [Exception(), Exception(), 'success']
    
    result = retry_operation(mock)
    
    assert result == 'success'  # What about retry count? Timing?
```
- Vague name, mocks behavior not reality, unclear what it tests

### Verify RED

Run the test. It MUST fail.

**If it passes:**
- Test is wrong (not testing what you think)
- Code already exists (delete it, start over)
- Wrong test file running

**What to check:**
- Error message makes sense
- Fails for expected reason
- Stack trace points to right place

### GREEN - Minimal Code

Write just enough code to pass. Nothing more.

**Good:**
```python
def add(a, b):
    return a + b  # Nothing extra
```

**Bad:**
```python
def add(a, b):
    result = a + b
    logging.info(f"Adding {a} + {b} = {result}")  # Extra!
    return result
```

Cheating is OK in GREEN:
- Hardcode return values
- Copy-paste
- Duplicate code
- Skip edge cases

**We'll fix it in refactor.**

### Verify GREEN

Run tests. All must pass.

**If fails:**
- Fix minimal code
- Don't expand scope
- Stay in GREEN

### REFACTOR - Clean Up

Now improve the code while keeping tests green.

**Safe refactorings:**
- Rename variables/functions
- Extract helper functions
- Remove duplication
- Simplify expressions
- Improve readability

**Golden rule:** Tests stay green throughout.

**If tests fail during refactor:**
- Undo immediately
- Smaller refactoring steps
- Check you didn't change behavior

## Implementation Workflow

### 1. Create Test File

```bash
# Python
touch tests/test_feature.py

# JavaScript
touch tests/feature.test.js

# Rust
touch tests/feature_tests.rs
```

### 2. Write First Failing Test

```python
# tests/test_calculator.py
def test_adds_two_numbers():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5
```

### 3. Run and Verify Failure

```bash
pytest tests/test_calculator.py -v
# Expected: FAIL - Calculator not defined
```

### 4. Write Minimal Implementation

```python
# src/calculator.py
class Calculator:
    def add(self, a, b):
        return a + b  # Minimal!
```

### 5. Run and Verify Pass

```bash
pytest tests/test_calculator.py -v
# Expected: PASS
```

### 6. Commit

```bash
git add tests/test_calculator.py src/calculator.py
git commit -m "feat: add calculator with add method"
```

### 7. Next Test

```python
def test_adds_negative_numbers():
    calc = Calculator()
    result = calc.add(-2, -3)
    assert result == -5
```

Repeat cycle.

## Testing Anti-Patterns

### Mocking What You Don't Own

**Bad:** Mock database, HTTP client, file system
**Good:** Abstract behind interface, test interface

### Testing Implementation Details

**Bad:** Test that function was called
**Good:** Test the result/behavior

### Happy Path Only

**Bad:** Only test expected inputs
**Good:** Test edge cases, errors, boundaries

### Brittle Tests

**Bad:** Test breaks when refactoring
**Good:** Tests verify behavior, not structure

## Common Pitfalls

### "I'll Write Tests After"

No, you won't. Write them first.

### "This is Too Simple to Test"

Simple bugs cause complex problems. Test everything.

### "I Need to See It Work First"

Temporary code becomes permanent. Test first.

### "Tests Take Too Long"

Untested code takes longer. Invest in tests.

## Language-Specific Commands

### Python (pytest)

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_feature.py::test_name -v

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Watch mode
pytest-watch
```

### JavaScript (Jest)

```bash
# Run all tests
npm test

# Run specific test
npm test -- test_name

# Watch mode
npm test -- --watch

# Coverage
npm test -- --coverage
```

### TypeScript (Jest with ts-jest)

```bash
# Run tests
npx jest

# Run specific file
npx jest tests/feature.test.ts
```

### Go

```bash
# Run all tests
go test ./...

# Run specific test
go test -run TestName

# Verbose
go test -v

# Coverage
go test -cover
```

### Rust

```bash
# Run tests
cargo test

# Run specific test
cargo test test_name

# Show output
cargo test -- --nocapture
```

## Integration with Other Skills

### With writing-plans

Every plan task should specify:
- What test to write
- Expected test failure
- Minimal implementation
- Refactoring opportunities

### With systematic-debugging

When fixing bugs:
1. Write test that reproduces bug
2. Verify test fails (RED)
3. Fix bug (GREEN)
4. Refactor if needed

### With subagent-driven-development

Subagent implements one test at a time:
1. Write failing test
2. Minimal code to pass
3. Commit
4. Next test

## Success Indicators

**You're doing TDD right when:**
- Tests fail before code exists
- You write <10 lines between test runs
- Refactoring feels safe
- Bugs are caught immediately
- Code is simpler than expected

**Red flags:**
- Writing 50+ lines without running tests
- Tests always pass
- Fear of refactoring
- "I'll test later"

## Remember

1. **RED** - Write failing test
2. **GREEN** - Minimal code to pass  
3. **REFACTOR** - Clean up safely
4. **REPEAT** - Next behavior

**If you didn't see it fail, it doesn't test what you think.**
