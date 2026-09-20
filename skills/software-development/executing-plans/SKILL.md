---
name: executing-plans
description: 'Execute a written implementation plan task-by-task with review checkpoints. Use when you have a plan and are ready to implement.'
version: 1.0.0
author: Hermes Agent (adapted from obra/superpowers)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [execution, implementation, plans, tasks]
    related_skills: [writing-plans, test-driven-development, systematic-debugging]
---

# Executing Plans

## Overview

Load plan, review critically, execute all tasks, report when complete.

## The Process

### Step 1: Load and Review Plan

1. Read plan file with `read_file`
2. Review critically — identify any questions or concerns about the plan
3. If concerns: Raise them with your human partner before starting
4. If no concerns: Create todos for the plan items and proceed

### Step 2: Execute Tasks

For each task:

1. Mark as in_progress
2. Follow each step exactly (plan has bite-sized steps)
3. Run verifications as specified
4. Mark as completed

### Step 3: Complete Development

After all tasks complete and verified:

- Run the full test suite
- Review the diff for unintended changes
- Present completion summary

## When to Stop and Ask for Help

**STOP executing immediately when:**

- Hit a blocker (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## When to Revisit Earlier Steps

**Return to Review (Step 1) when:**

- Partner updates the plan based on your feedback
- Fundamental approach needs rethinking

**Don't force through blockers** — stop and ask.

## Hermes Integration

- Use `read_file` to load the plan
- Use `terminal` for build/test/lint commands
- Use `write_file` / `replace_string_in_file` for code changes
- Use `delegate_task` for parallel task execution when tasks are independent
- Use `search_files` to find patterns across the codebase
- For Hermes Agent development, always use `scripts/run_tests.sh`

## Remember

- Review plan critically first
- Follow plan steps exactly
- Don't skip verifications
- Commit after each completed task
