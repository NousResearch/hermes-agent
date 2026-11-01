# Commit Categorization Reference

`scripts/changelog.py` maps each commit subject to a Keep a Changelog
category. Conventional Commit prefixes are matched first; unprefixed
repos fall back to keyword scanning of the subject.

## Prefix mapping

| Category | Prefixes / patterns |
|---|---|
| Breaking Changes | `!` after the type (`feat!`, `fix(scope)!`), or `BREAKING CHANGE:` in the body |
| Added | `feat:`, `add:`, `new:` |
| Fixed | `fix:`, `bugfix:`, `patch:` |
| Changed | `refactor:`, `change:`, `update:`, `improve:`, `perf:` |
| Removed | `remove:`, `delete:`, `drop:` |
| Security | `security:`, `sec:` |
| Deprecated | `deprecate:`, `deprecated:` |
| Documentation | `docs:`, `doc:` |
| Tests | `test:`, `tests:` (omitted by default) |
| Chores | `chore:`, `ci:`, `build:`, `release:` (omitted by default) |
| Other | Anything unmatched |

## Keyword fallback

When a subject has no recognized prefix, the first matching keyword wins,
in this order: breaking, fixed, added, removed, changed, documentation,
otherwise Other.

| Keywords (word-boundary match) | Category |
|---|---|
| break, breaking, incompatible | Breaking Changes |
| fix, bug, error, crash, patch | Fixed |
| add, new, feature, implement, introduce | Added |
| remove, delete, drop | Removed |
| update, refactor, improve, perf, optimize | Changed |
| doc, docs, readme, comment | Documentation |

## Breaking changes

A commit is breaking when the type carries a `!` (`feat!:`,
`feat(scope)!:`) or the commit body/footer contains `BREAKING CHANGE:`.
The body is not on the subject line, so read it with `git log
--format=%B <hash>` when you need to confirm a footer flag. Breaking
changes render first so they cannot be missed.

## Detectable metadata

- PR / issue number: the first `#N` found in the subject is appended as
  `(#N)`.
- Prefixes are stripped from the rendered line (`feat: add dark mode`
  renders as `Add dark mode`).
