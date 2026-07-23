---
name: goal-planner
description: Break big goals into trackable milestones with progress tracking.
version: "1.0"
author: Hermes / adapted from Ruflo goals (ruvnet/ruflo, MIT)
license: MIT
metadata:
  hermes:
    tags: ["planning", "productivity", "goal-tracking"]
    category: productivity
---

# Goal Planner Skill

Use when the user wants to plan, track, or break down a large task or project.

## When to Use
- User says "I want to accomplish X" or "plan this project"
- User needs milestones for a big goal
- User wants to track progress across sub-tasks

## How to Run
1. Call `goal_create(title, description, milestones)` to define the goal
2. Call `goal_track(goal_id, milestone_id, status)` as milestones complete
3. Call `goal_list(status)` to review progress

## Verification
- Every goal has clear, measurable milestones
- Progress percentage updates automatically as milestones complete
- Completed goals are archived but never deleted