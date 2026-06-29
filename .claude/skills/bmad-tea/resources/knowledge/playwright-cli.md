# Playwright CLI — Browser Automation for Coding Agents

## Principle

When an AI agent needs to look at a webpage — take a snapshot, grab selectors, capture a screenshot — it shouldn't have to load thousands of tokens of DOM trees and tool schemas into its context window just to do that. Playwright CLI gives the agent a lightweight way to talk to a browser through simple shell commands, keeping the context window free for reasoning and code generation.

## Rationale

Playwright MCP is powerful, but it's heavy. Every interaction loads full accessibility trees and tool definitions into the LLM context. That's fine for complex, stateful flows where you need rich introspection. But for the common case — "open this page, tell me what's on it, take a screenshot" — it's overkill.

Playwright CLI solves this by returning concise **element references** (`e15`, `e21`) instead of full DOM dumps. The result: ~93% fewer tokens per interaction, which means the agent can run longer sessions, reason more deeply, and still have context left for your actual code.

**The trade-off is simple:**

- **CLI** = fast, lightweight, stateless — great for quick looks at pages
- **MCP** = rich, stateful, full-featured — great for complex multi-step automation

TEA uses both where each shines (see `tea_browser_automation: "auto"`).

## Prerequisites

```bash
npm install -g @playwright/cli@latest    # Install globally (Node.js 18+)
playwright-cli install --skills          # Register as an agent skill
```

The global npm install is one-time. Run `playwright-cli install --skills` from your project root to register skills in `.claude/skills/` (works with Claude Code, GitHub Copilot, and other coding agents). Agents without skills support can use the CLI directly via `playwright-cli --help`. TEA documents this during installation but does not run it for you.

## How It Works

The agent interacts with the browser through shell commands. Each command is a single, focused action:

```bash
# 1. Open a page
playwright-cli -s=tea-explore open https://app.com/login

# 2. Take a snapshot — returns element references, not DOM trees
playwright-cli -s=tea-explore snapshot
# Output: [{ref: "e15", role: "textbox", name: "Email"},
#          {ref: "e21", role: "textbox", name: "Password"},
#          {ref: "e33", role: "button", name: "Sign In"}]

# 3. Interact using those references
playwright-cli -s=tea-explore fill e15 "user@example.com"
playwright-cli -s=tea-explore fill e21 "password123"
playwright-cli -s=tea-explore click e33

# 4. Capture evidence
playwright-cli -s=tea-explore screenshot --filename=login-flow.png

# 5. Clean up
playwright-cli -s=tea-explore close
```

The `-s=tea-explore` flag scopes everything to a named session, preventing state leakage between workflows.

## What TEA Uses It For

**Selector verification** — Before generating test code, TEA can snapshot a page to see the actual labels, roles, and names of elements. Instead of guessing that a button says "Login", it knows it says "Sign In":

```
snapshot ref {role: "button", name: "Sign In"}
  → generates: page.getByRole('button', { name: 'Sign In' })
```

**Page discovery** — During `test-design` exploratory mode, TEA snapshots pages to understand what's actually there, rather than relying only on documentation.

**Evidence collection** — During `test-review`, TEA can capture screenshots, traces, and network logs as evidence without the overhead of a full MCP session.

**Agent-side test debugging** — For existing failing Playwright tests, TEA should prefer Playwright's agent-facing debug loop over ad hoc manual reproduction: `npx playwright test --debug=cli` to step through the test in CLI mode (no GUI Inspector — designed for coding agents), then `npx playwright trace ...` to inspect the resulting trace artifact from the command line. The `--debug=cli` flag (Playwright 1.59+) lets agents attach, step through execution, and inspect page state without ever opening a browser window.

## How CLI Relates to Playwright Utils and API Testing

CLI and playwright-utils are **complementary tools that work at different layers**:

|              | Playwright CLI                               | Playwright Utils                                 |
| ------------ | -------------------------------------------- | ------------------------------------------------ |
| **When**     | During test _generation_ (the agent uses it) | During test _execution_ (your test code uses it) |
| **What**     | Shell commands to observe your app           | Fixtures and helpers imported in test files      |
| **Examples** | `snapshot`, `screenshot`, `network`          | `apiRequest`, `auth-session`, `network-recorder` |

They work together naturally. The agent uses CLI to _understand_ your app, then generates test code that _imports_ playwright-utils:

```bash
# Agent uses CLI to observe network traffic on the dashboard page
playwright-cli -s=tea-discover open https://app.com/dashboard
playwright-cli -s=tea-discover network
# Output: GET /api/users → 200, POST /api/audit → 201, GET /api/settings → 200
playwright-cli -s=tea-discover close
```

```typescript
// Agent generates API tests using what it discovered, with playwright-utils
import { test } from '@seontechnologies/playwright-utils/api-request/fixtures';

test('GET /api/users returns user list', async ({ apiRequest }) => {
  const { status, body } = await apiRequest<User[]>({
    method: 'GET',
    path: '/api/users',
  });
  expect(status).toBe(200);
  expect(body.length).toBeGreaterThan(0);
});
```

**For pure API testing** (no UI involved), `playwright-cli` browser commands (snapshot, screenshot, click) don't apply — there's no page. But **trace analysis is highly valuable**. Playwright captures full network traces for API tests (requests, responses, headers, timing), and the trace CLI lets the agent inspect them programmatically:

```bash
# API test fails in CI → open the trace artifact
npx playwright trace open test-results/api-users/trace.zip

# What HTTP call failed?
npx playwright trace requests --failed
# Output: #3  POST /api/users  → 422  12ms

# Full request/response details (headers, body, timing)
npx playwright trace request 3

# What assertion failed and why?
npx playwright trace errors

# Done
npx playwright trace close
```

This gives the agent the full HTTP conversation — wrong payload, expired auth token, schema mismatch, upstream 5xx — without a human opening UI mode. The agent generates API tests directly from documentation, specs, or code analysis using `apiRequest` and `recurse` from playwright-utils, and uses trace analysis to diagnose failures.

**For E2E testing**, CLI shines at both ends — browser commands (snapshot, screenshot) during test generation, and trace analysis (actions, snapshots, requests) during debugging.

**Bottom line:** CLI helps the agent _write better tests_. Playwright-utils helps those tests _run reliably_. Trace analysis helps the agent _fix them when they break_.

## Session Isolation

Every CLI command targets a named session. This prevents workflows from interfering with each other:

```bash
# Workflow A uses one session
playwright-cli -s=tea-explore open https://app.com

# Workflow B uses a different session (can run in parallel)
playwright-cli -s=tea-verify open https://app.com/admin
```

For parallel safety (multiple agents on the same machine), append a unique suffix:

```bash
playwright-cli -s=tea-explore-<timestamp> open https://app.com
```

## Autonomous Trace Investigation (Playwright 1.59+)

For generated tests that already exist and are failing, Playwright 1.59 introduced CLI-native debugging and trace analysis designed specifically for AI agents. Instead of downloading traces and opening the GUI Trace Viewer, agents can now consume the entire trace context directly from the command line.

### Debug a Failing Test (CLI Mode)

```bash
# Start the test in CLI debug mode — no GUI Inspector, agent-friendly output
npx playwright test --debug=cli
playwright-cli attach <session-id>
playwright-cli --session <session-id> step-over
```

With `--debug=cli`, the agent can:

- Step through test execution in real-time
- Inspect the page's HTML source at each step
- Review network calls and console logs at the moment of failure
- Capture before/after snapshots without opening a browser

### Investigate a Trace Artifact

```bash
# Open a trace from CI or local runs — this starts a session
npx playwright trace open test-results/<run>/trace.zip

# List all actions as a numbered tree (# column = 1-based ordinal)
npx playwright trace actions
# Output: #  Time     Action                Duration
#         1  0:00.00  navigate(...)         120ms
#         2  0:00.12  fill(#email, ...)     45ms
#         ...
#         9  0:01.50  expect(toBeVisible)   ✗ 30s

# Filter to failing assertions
npx playwright trace actions --grep="expect"

# Drill into action #9 (the ordinal from the list above)
npx playwright trace action 9

# See the page snapshot after that action (valid: before | input | after)
npx playwright trace snapshot 9 --name after

# Other useful subcommands
npx playwright trace errors                  # errors with stack traces
npx playwright trace requests --failed       # failed network requests
npx playwright trace console --errors-only   # console errors

# Close when done (removes extracted data)
npx playwright trace close
```

### Autonomous Diagnostic Loop

When TEA encounters a failing test in healing/review mode, the recommended investigation flow is:

1. **Run with `--debug=cli`** to step through the failure and identify the failing action
2. **Get a trace artifact** — configure `trace: 'retain-on-failure'` in `playwright.config.ts` (recommended), add `--trace=retain-on-failure` to the test run, or use an existing CI trace artifact. For `playwright-cli` sessions (not `--debug=cli`), use `tracing-start` / `tracing-stop` instead.
3. **Filter to assertions** (`trace actions --grep="expect"`) to find the failure point
4. **Inspect the snapshot** (`trace snapshot <n> --name after`) to see exact page state at failure
5. **Analyze network/console** to rule out backend issues or timing problems
6. **Propose a fix** — updated locator, added wait, or flagged flake for human review

This reduces Mean Time to Repair (MTTR) by giving the agent full failure context rather than just an error message.

### When to Use Each Tool

- `playwright-cli` session commands remain the best lightweight tool for page exploration and selector verification.
- `npx playwright test --debug=cli` is better for stepping through an already-written failing test (agent-native, no GUI).
- `npx playwright trace ...` is better for understanding flakes and assertion failures from saved artifacts.

If your environment exposes the Playwright dashboard or bound-browser flow, it can help humans inspect what an agent is doing in the background, but TEA should treat that as optional observability rather than a hard dependency.

### Binding a Browser for Agent Inspection (`browser.bind()`)

Playwright 1.59 added `browser.bind()` — a programmatic API that makes a running browser instance available to `playwright-cli` and MCP clients. This is the bridge between "a test is running" and "an agent can see what the test sees."

```typescript
// In a test or fixture: bind the browser so playwright-cli can attach
const { endpoint } = await browser.bind('my-debug-session', {
  workspaceDir: process.cwd(),
});
// Now: playwright-cli attach my-debug-session
```

**When TEA uses this:**

- **Debugging a complex E2E failure** — A test fixture calls `browser.bind()` before the failing scenario, then TEA runs `playwright-cli attach` to inspect live page state, network, and console without re-running the test from scratch.
- **Bridging CLI and MCP** — A bound browser is accessible to both `playwright-cli` and `@playwright/mcp`. TEA's `auto` mode can start with lightweight CLI inspection and escalate to MCP if richer introspection is needed, all against the same browser instance.
- **CI artifact enhancement** — A CI helper can bind the browser during test runs, letting a post-failure agent attach and investigate before the process exits.

Call `await browser.unbind()` when done to release the session (async — must be awaited).

## Command Quick Reference

| What you want to do       | Command                                          |
| ------------------------- | ------------------------------------------------ |
| Open a page               | `open <url>`                                     |
| See what's on the page    | `snapshot`                                       |
| Take a screenshot         | `screenshot [--filename=path]`                   |
| Click something           | `click <ref>`                                    |
| Type into a field         | `fill <ref> <text>`                              |
| Navigate                  | `goto <url>`, `go-back`, `reload`                |
| Mock a network request    | `route <pattern> --status=200 --body='...'`      |
| Start recording a trace   | `tracing-start`                                  |
| Stop and save the trace   | `tracing-stop`                                   |
| Save auth state for reuse | `state-save auth.json`                           |
| Load saved auth state     | `state-load auth.json`                           |
| See network requests      | `network`                                        |
| Manage tabs               | `tab-list`, `tab-new`, `tab-close`, `tab-select` |
| Close the session         | `close`                                          |

## When CLI vs MCP (Auto Mode Decision)

| Situation                             | Tool | Why                                |
| ------------------------------------- | ---- | ---------------------------------- |
| "What's on this page?"                | CLI  | One-shot snapshot, no state needed |
| "Verify this selector exists"         | CLI  | Single check, minimal tokens       |
| "Capture a screenshot for evidence"   | CLI  | Stateless capture                  |
| "Walk through a multi-step wizard"    | MCP  | State carries across steps         |
| "Debug why this test fails" (healing) | CLI  | `--debug=cli` + trace analysis     |
| "Record a drag-and-drop flow"         | MCP  | Complex interaction semantics      |

## Related Fragments

- `overview.md` — Playwright Utils installation and fixture patterns (the test code layer that CLI complements)
- `api-request.md` — Typed HTTP client for API tests (CLI discovers endpoints, apiRequest tests them)
- `api-testing-patterns.md` — Pure API test patterns (when CLI isn't needed)
- `auth-session.md` — Token management (CLI `state-save` informs auth-session usage)
- `selector-resilience.md` — Robust selector strategies (CLI verifies them against real DOM)
- `visual-debugging.md` — Trace viewer usage (CLI captures traces)
