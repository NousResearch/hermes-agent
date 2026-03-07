---
name: bug-hunter
description: Automated Python bug hunting, fixing and PR generation.
triggers:
  - "run bug hunter on repo"
  - "automatically fix bugs in this python repo"
  - "analyze this github url and create PR"
---

# Bug Hunter
Automate cloning, testing, fixing, and PR submission for Python repositories.

## Steps
1. Use `scripts/bug_hunter.py` to analyze the repo.
2. The agent analyzes test failures.
3. The agent proposes and applies fixes using the `BugHunter` class methods.
4. Verify and PR.
