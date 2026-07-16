---
name: kanban-browserbase-worker
description: >
  Use when the browser-agent Kanban profile picks up a task (assignee=browser).
  Executes the task via the Browserbase Agents REST API — POST /v1/agents/runs,
  poll until terminal state, return structured result to the Kanban board.
version: 2.0.0
author: Kyle Jeong / Hermes
license: MIT
metadata:
  hermes:
    tags: [kanban, browserbase, browser, worker, automation, agents-api]
    related_skills: [kanban-codex-lane, browserbase-agent]
---

# Kanban Browserbase Worker

Executes Kanban browser tasks via the **Browserbase Agents REST API**.
No local browser. No CDP. Just curl → poll → report.

## Credentials (always available in this profile's .env)

```bash
BB_KEY="bb_live_Tsm1wObH8ZippKfZa-h9jw82q3Y"
BB_AGENT="1387922d-aec4-4225-a93e-a602e19f1f48"
BASE="https://api.browserbase.com"
```

## Execution Loop

### Step 1 — Parse the Kanban task

Read title + body. Extract:
- **URL(s)** to visit
- **Action** (scrape, click, extract, verify, fill)
- **Output format** (JSON schema, plain text, table)
- **Success criteria**

### Step 2 — Create a Browserbase agent run

For plain extraction (free-form result):
```bash
RUN=$(curl -s -X POST "$BASE/v1/agents/runs" \
  -H "X-BB-API-Key: $BB_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"agentId\": \"$BB_AGENT\", \"task\": \"YOUR TASK HERE\"}")
RUN_ID=$(echo "$RUN" | python3 -c "import sys,json; print(json.load(sys.stdin)['runId'])")
echo "Started run: $RUN_ID"
```

For structured JSON output (preferred when task asks for data):
```bash
RUN=$(curl -s -X POST "$BASE/v1/agents/runs" \
  -H "X-BB-API-Key: $BB_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agentId": "'"$BB_AGENT"'",
    "task": "YOUR TASK HERE",
    "resultSchema": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {"type": "object"}
        },
        "summary": {"type": "string"}
      }
    }
  }')
RUN_ID=$(echo "$RUN" | python3 -c "import sys,json; print(json.load(sys.stdin)['runId'])")
```

### Step 3 — Poll until terminal state

```bash
while true; do
  RESP=$(curl -s "$BASE/v1/agents/runs/$RUN_ID" -H "X-BB-API-Key: $BB_KEY")
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATUS"
  case "$STATUS" in
    COMPLETED|FAILED|STOPPED|TIMED_OUT) echo "$RESP"; break ;;
  esac
  sleep 5
done
```

### Step 4 — Handle the result

**On COMPLETED:**
```bash
SUMMARY=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',{}).get('summary',''))")
```
Write `SUMMARY` + any structured data fields to the Kanban task comment, then mark complete.

**On FAILED:**
```bash
CAUSE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cause',{}).get('message','unknown'))")
```
Block the Kanban task with: `"Browserbase agent failed: $CAUSE"`

**On TIMED_OUT / STOPPED:**
Block with reason + run ID for debugging.

## Task body format (what creators should write)

```
URL: https://example.com/page
Action: [scrape | extract | verify | fill | click]
Output: [JSON with fields X,Y,Z | plain text | table]
Success: [what done looks like]
```

## Pitfalls

- **Always poll** — 201 just means queued
- **Terminal states only**: COMPLETED, FAILED, STOPPED, TIMED_OUT
- `result` key is only populated on COMPLETED
- Don't chain runs for a single task — one task = one run
- Use `resultSchema` to get clean structured output, not just `.summary`
- Runs typically finish in 5–30s; budget 3 min max poll time
