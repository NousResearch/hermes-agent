# Review Checklist

Quick-reference checklist for code review. Load this file with `skill_view`
when you need the full checklist during a review.

## Pre-review

- [ ] Identified the review target (staged, unstaged, file, PR)
- [ ] Got the diff or file content
- [ ] Ran common-problem scans (debug artifacts, secrets, conflict markers)

## Correctness

- [ ] Code does what its name/docstring/commit message claims
- [ ] Edge cases handled: empty, null, zero, negative, max-size inputs
- [ ] Loop boundaries correct (no off-by-one)
- [ ] Error paths return/throw/propagate correctly
- [ ] Type conversions are safe (no silent truncation or overflow)

## Security

- [ ] No hardcoded secrets, keys, or passwords in the diff
- [ ] User input validated/sanitized before use
- [ ] Database queries use parameterized statements
- [ ] File paths validated against traversal (no unsanitized user input in paths)
- [ ] Shell commands use array form, not string interpolation
- [ ] Auth checks present on protected endpoints

## Logic and data flow

- [ ] Conditionals test the intended condition
- [ ] No unreachable code after early returns/throws
- [ ] State mutations sequenced correctly
- [ ] Async/await used correctly (no missing await, no unhandled promise)
- [ ] Race conditions considered for concurrent access

## Code quality

- [ ] Names are clear, consistent with codebase conventions
- [ ] No unnecessary abstractions or premature generalization
- [ ] Duplicated logic extracted if repeated 3+ times
- [ ] Functions have a single clear responsibility
- [ ] No commented-out code left behind

## Testing

- [ ] New behavior has tests
- [ ] Both happy path and error cases covered
- [ ] Tests are deterministic (no flaky timing, no external dependencies)
- [ ] Test names describe the behavior being verified

## Performance (flag only when real)

- [ ] No N+1 queries
- [ ] No blocking I/O in async code paths
- [ ] Large collection operations are bounded or paginated
- [ ] No unnecessary re-computation inside loops
