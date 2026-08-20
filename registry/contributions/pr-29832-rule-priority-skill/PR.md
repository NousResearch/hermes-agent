# PR #29832 — feat: add rule-priority optional skill for L0-L3 governance

**URL:** https://github.com/NousResearch/hermes-agent/pull/29832
**Branch:** `feat/rule-priority-skill`
**Date:** 2026-05-21
**Author:** Minksgo (via Commerce Bureau)

## Summary

Added `optional-skills/governance/rule-priority/` — a pure-Python L0-L3 rule priority governance skill with conflict resolution, system prompt injection, and L3 tool blocking. Disabled by default.

## Files

- `optional-skills/governance/rule-priority/SKILL.md` (37 lines)
- `optional-skills/governance/rule-priority/scripts/rule_priority.py` (167 lines)
- `optional-skills/governance/rule-priority/scripts/__init__.py` (empty)
- `optional-skills/governance/rule-priority/tests/test_rule_priority.py` (169 lines, 16 tests)

## Design

- Priority: L0 (Universal) > L3 (Global) > L1 (Project) > L2 (User)
- Same level: last-write-wins
- L3 only for tool_block constraints
- Pure stdlib, zero deps
- Disabled by default

## Verification

- [x] No existing PR found for "rule priority" in upstream
- [x] Branch created from clean `upstream/main` (48be2e0e4)
- [x] Diff contains only rule-priority files (4 files, +373 lines)
- [x] Privacy check: no 蔷薇/玫瑰/百合/研究院/director content
- [x] PR submitted successfully via GitHub API (proxy: localhost:7890)
- [x] One PR, one feature
