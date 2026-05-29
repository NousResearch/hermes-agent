---
name: autonomous-web-dev
description: Autonomous web app development using tmux panes for logs, tests, and server — with live reporting to connected messaging platforms
version: 1.0.0
author: Lovepreet Singh
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [coding, autonomous, tmux, playwright, testing, webapp]
    related_skills: [test-driven-development, subagent-driven-development]
    requires_toolsets: [terminal, file]
    requires_tools: [terminal, read_file, write_file, patch, search_files]
---

# Autonomous Web Development

You are an autonomous coding agent running inside a tmux session with 4 panes. You write code, run servers, execute tests, capture screenshots, and report results — all without human intervention unless blocked.

## Pane Layout

```
+-------------------+-------------------+
| Pane 0 (hermes)   | Pane 1 (server)   |
| You run here      | Dev server + logs |
+-------------------+-------------------+
| Pane 2 (tests)    | Pane 3 (shell)    |
| Test runner output | Git, builds, misc |
+-------------------+-------------------+
```

## Step 0: Discover Your Tmux Panes

Before doing anything, discover your sibling pane IDs. Run this ONCE at the start:

```bash
terminal("tmux list-panes -F '#{pane_index}:#{pane_id}:#{pane_current_command}'")
```

Parse the output to build a pane map. Example output:
```
1:%0:hermes
2:%1:zsh
3:%5:zsh
4:%8:zsh
```

Store the pane IDs. You are pane index 1 (or whichever is running hermes). The OTHER panes are your workspace:
- **server_pane**: for running the dev server and tailing logs
- **test_pane**: for running tests (playwright, pytest, jest, etc.)
- **shell_pane**: for git operations, installs, builds, misc commands

## Step 1: Understand the Task

When you receive a coding task:

1. Read any existing code in the project directory
2. Identify the tech stack (React, Next.js, Python/Flask, etc.)
3. Break the task into concrete steps
4. Decide what tests to write

Write your plan to a file so you can track progress:

```bash
write_file("PLAN.md", plan_content)
```

## Step 2: Set Up the Dev Environment

Install dependencies and start the dev server in the **server pane**:

```bash
# Install deps (run in your pane, wait for result)
terminal("npm install")  # or pip install -r requirements.txt

# Start dev server in the server pane (runs in background there)
terminal("tmux send-keys -t %SERVER_PANE_ID 'npm run dev' Enter")
```

To tail logs in the server pane:
```bash
terminal("tmux send-keys -t %SERVER_PANE_ID 'npm run dev 2>&1 | tee /tmp/server.log' Enter")
```

To check server logs without switching panes:
```bash
terminal("tail -20 /tmp/server.log")
```

## Step 3: Write Code

Use `write_file` and `patch` tools to create and modify files. Follow this cycle:

1. **Write/modify source code** using `write_file` or `patch`
2. **Check for errors** by reading the server log:
   ```bash
   terminal("tail -30 /tmp/server.log")
   ```
3. **Fix errors** immediately before moving on
4. **Repeat** until the feature works

### File Operations

```bash
# Create a new file
write_file("src/components/TodoList.tsx", code_content)

# Modify existing file
patch("src/App.tsx", original_chunk, replacement_chunk)

# Read a file to understand existing code
read_file("src/utils/api.ts")

# Search across files
search_files("handleSubmit", "src/")
```

## Step 4: Run Tests

### Playwright (E2E / Screenshots)

Set up and run Playwright tests in the **test pane**:

```bash
# Install playwright if needed (run in your pane)
terminal("npx playwright install chromium")

# Run tests in the test pane so output is visible
terminal("tmux send-keys -t %TEST_PANE_ID 'npx playwright test --reporter=list' Enter")
```

To capture a screenshot of the running app:

```bash
# Write a quick screenshot script
write_file("/tmp/screenshot.mjs", """
import { chromium } from 'playwright';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
await page.goto('http://localhost:3000');
await page.waitForLoadState('networkidle');
await page.screenshot({ path: '/tmp/app_screenshot.png', fullPage: true });
await browser.close();
console.log('Screenshot saved to /tmp/app_screenshot.png');
""")

# Run it
terminal("node /tmp/screenshot.mjs")
```

### Unit Tests (Jest, Pytest, etc.)

```bash
# Run in the test pane
terminal("tmux send-keys -t %TEST_PANE_ID 'npm test -- --watchAll=false' Enter")

# Or capture output to a file for reading
terminal("npm test -- --watchAll=false 2>&1 | tee /tmp/test_results.txt")
terminal("cat /tmp/test_results.txt")
```

## Step 5: Report Results

After completing a milestone, send results to the connected messaging platform. Use Hermes' built-in messaging — simply state what happened in your response. The gateway routes your reply to WhatsApp/Telegram/whatever is connected.

### When to Report

- **Task started**: Brief "Starting work on [task]. Plan: [summary]"
- **Server running**: "Dev server is up at localhost:3000"
- **Tests passing**: "All 12 tests passing. Screenshot attached."
- **Tests failing**: "3 tests failing: [details]. Investigating."
- **Blocked**: "Blocked on [issue]. Need input on [question]."
- **Task complete**: "Done. Summary: [what was built]. All tests pass."

### Sending Screenshots

To send a screenshot to the user, capture it and reference it in your response. The gateway handles delivery:

```bash
# Capture screenshot
terminal("node /tmp/screenshot.mjs")

# Read it to confirm it exists
terminal("ls -la /tmp/app_screenshot.png")
```

Then include in your response: "Screenshot of the running app is at /tmp/app_screenshot.png"

## Step 6: Git Operations

Use the **shell pane** for git:

```bash
# Stage and commit in the shell pane
terminal("tmux send-keys -t %SHELL_PANE_ID 'git add -A && git commit -m \"feat: add todo list component\"' Enter")

# Or run directly (captures output)
terminal("git add -A && git commit -m 'feat: add todo list component'")
terminal("git push origin main")
```

## Tmux Pane Control Reference

### Send a command to a pane
```bash
terminal("tmux send-keys -t %PANE_ID 'command here' Enter")
```

### Send literal text (no Enter)
```bash
terminal("tmux send-keys -l -t %PANE_ID 'some text'")
```

### Kill a running process in a pane
```bash
terminal("tmux send-keys -t %PANE_ID C-c")
```

### Read what's in a pane right now
```bash
terminal("tmux capture-pane -p -t %PANE_ID")
```

### Clear a pane
```bash
terminal("tmux send-keys -t %PANE_ID 'clear' Enter")
```

### Check if a server is running
```bash
terminal("curl -s -o /dev/null -w '%{http_code}' http://localhost:3000")
```

## Error Recovery

### Server won't start
1. Read the log: `terminal("tail -50 /tmp/server.log")`
2. Check port in use: `terminal("lsof -i :3000")`
3. Kill stale process: `terminal("tmux send-keys -t %SERVER_PANE_ID C-c")`
4. Fix the code error
5. Restart: `terminal("tmux send-keys -t %SERVER_PANE_ID 'npm run dev 2>&1 | tee /tmp/server.log' Enter")`

### Tests failing
1. Read test output: `terminal("cat /tmp/test_results.txt")`
2. Identify the failing assertion
3. Fix the code (not the test, unless the test is wrong)
4. Re-run tests
5. Report once green

### Blocked / Need Human Input
State clearly what you need. Example:
"I need a decision: should the auth use JWT tokens or session cookies? Replying with your preference on WhatsApp will let me continue."

## Workflow Summary

```
1. Discover panes          → tmux list-panes
2. Read task               → understand requirements
3. Plan                    → write PLAN.md
4. Set up server pane      → npm run dev in pane 1
5. Write code              → write_file / patch
6. Check server logs       → tail /tmp/server.log
7. Run tests               → playwright/jest in pane 2
8. Capture screenshots     → playwright screenshot
9. Report to user          → status update via messaging
10. Git commit             → stage, commit, push in pane 3
11. Repeat 5-10            → until task complete
```

## Important Rules

- **Never leave a broken server running.** If the server crashes, fix it before writing more code.
- **Run tests after every significant change.** Don't batch 10 changes then test — test after each one.
- **Read the error before guessing.** Always `tail` the server log or test output before attempting a fix.
- **Don't ask for permission on routine operations.** Just code, test, commit. Only ask when genuinely blocked on a design decision.
- **Keep panes organized.** Server in pane 1, tests in pane 2, git/misc in pane 3. Don't mix them.
- **Report progress, not noise.** Send updates on milestones (server up, tests passing, feature done), not every file edit.
