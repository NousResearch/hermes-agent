# Hermes Agent Learning Ramp — Your First Month

A practical 4-week guide to go from "first chat" to "autonomous triage" with
Hermes Agent. Each week builds on the previous one. Do the daily practice
before moving on — the agent's power comes from how you use it, not just
knowing the commands.

**Who this is for:** Developers, DevOps engineers, and power users who want to
use Hermes Agent as a daily driver for real work.

**What you'll learn by the end:**

- Run the plan → execute → verify loop without hand-holding
- Delegate confidently to subagents with clear acceptance criteria
- Teach the agent your preferences through skills, memory, and config
- Automate recurring tasks with cron, gateway, and metrics

---

## Week 1: The Manual Loop (Plan → Execute → Verify)

**Theme:** You are the pilot. The agent is the co-pilot. Every interaction is a
loop: you plan, the agent executes, you verify the result.

### Day 1–2: Your First Conversation

**Goal:** Have a real conversation with the agent and understand the interaction
model.

**Morning:**

1. **Install and launch.** If you haven't already:
   ```bash
   pip install hermes-agent
   hermes
   ```
   You'll see the CLI prompt. Type your first message.

2. **Ask a simple question.** Try something like:
   ```
   What time is it in Tokyo?
   ```
   Notice the agent picks a tool (`current_time` with location) and returns
   the answer. This is your first plan → execute → verify cycle, even if you
   didn't plan it explicitly.

3. **Ask the agent to explain itself.** Type:
   ```
   What tools do you have?
   ```
   Review the output. You'll see a list of tools like `execute_command`,
   `read`, `write`, `search_files`, `web_search`, `delegate_task`, and
   others. Each tool is a capability the agent can use.

**Afternoon:**

4. **Run a small file operation.** Ask:
   ```
   Create a file called hello.txt with the text "Hello, Hermes!" in my home
   directory.
   ```
   Watch the agent plan (it may show you a tool call or just do it). Verify
   the file exists:
   ```bash
   cat ~/hello.txt
   ```
   This is your first explicit plan → execute → verify loop. You asked for a
   result, the agent acted, you checked. **Always verify.**

5. **Read the conversation loop.** Open a second terminal and tail the agent
   log to see what happens behind the scenes:
   ```bash
   hermes logs --follow
   ```
   Send another message and watch the log entries appear. Each tool call,
   each model response, each error is recorded here.

**Key insight:** The agent is a loop — user message → model chooses tools →
tools run → results fed back → model responds. Your job in Week 1 is to
supervise every step of this loop manually.

### Day 3–4: Planning Before Coding

**Goal:** Stop treating the agent like a magic box. Start treating it like a
junior engineer who needs a clear plan.

**Morning:**

1. **Write a plan first.** Before asking the agent to build something, type
   your intent in natural language. For example, instead of:
   ```
   Build a script to check disk space
   ```
   Write:
   ```
   I need a Python script that checks disk usage for the root partition.
   First, let me plan: (1) check what tools we have, (2) decide on output
   format, (3) create the script, (4) test it. Ready to start with step 1.
   ```

2. **Use the `plan` skill.** If you have it loaded, the agent can structure
   its work:
   ```
   Load the planning skill and help me create a backup script.
   ```
   The skill forces the agent to enumerate steps before executing. When it
   presents a plan, review it carefully. Approve or refine.

**Afternoon:**

3. **Ask for a plan before execution.** For every non-trivial task this week,
   start your message with:
   ```
   Plan first, then execute.
   ```
   Or use a single-message pattern:
   ```
   Plan: <your brief>
   Steps: [list them explicitly]
   Execute step 1.
   ```

4. **Anti-pattern to avoid:**
   - ❌ "Fix my server" — no context, no plan, no acceptance criteria.
   - ✓ "My server is returning 502 on /api/health. Plan: (1) check nginx
     config, (2) check the upstream service is running, (3) check logs for
     errors. Start with step 1."

5. **Practice with a real task.** Take something you'd normally do by hand
   and walk the agent through it step by step:
   ```
   I want to clean up old Docker images. Plan: (1) list dangling images,
   (2) review them, (3) remove them if I approve. Go.
   ```

### Day 5–7: Verification — Don't Trust, Verify

**Goal:** Build the habit of never accepting an agent's output at face value.

**Morning:**

1. **Verify every file.** Whenever the agent writes code or a config file,
   read it yourself before using it:
   ```
   Show me the full file.
   ```
   Then check it with tools:
   ```bash
   python3 -c "import ast; ast.parse(open('file.py').read()); print('Syntax OK')"
   ```

2. **Use `hermes logs` to audit.** After a session, review what actually
   happened:
   ```bash
   hermes logs --session <session-id> --level DEBUG
   ```
   Look for:
   - Tool calls that failed and were silently retried.
   - Tools that ran on the wrong files or directories.
   - Commands that produced errors the agent didn't mention.

3. **Test the output.** For any script the agent creates, run it yourself:
   ```bash
   ./the_script.py --dry-run
   ```
   Or construct a minimal test:
   ```
   Create a pytest test for the script you just wrote.
   ```

**Afternoon:**

4. **Practice the full loop end-to-end:**
   - **Plan:** "I want to find the 10 largest files in /var/log."
   - **Execute:** The agent runs `du` or uses a tool.
   - **Verify:** You run the same command yourself and compare outputs.
   
   Do this 3–5 times with different tasks. You should start to feel a rhythm.

5. **Track your verification rate.** For every task this week, tick one of:
   - [ ] Verified: I checked the output myself.
   - [ ] Trusted: I took the agent's word for it.
   
   Aim for 100% verified by Day 7.

**Weekly checkpoint (end of Day 7):** You should be able to:
- Start a conversation and steer it toward a goal.
- Break work into plan → execute → verify steps.
- Audit the agent's tool calls in the logs.
- Catch at least one mistake the agent made (it will make them).

---

## Week 2: Delegation (Subagents, Atomic Tasks, Acceptance Tests)

**Theme:** The agent can spawn child agents. Your job is to write task
descriptions so clear that the subagent succeeds without you.

### Day 8–10: Writing Atomic Task Descriptions

**Goal:** Master the art of writing a task spec that produces the right result
every time.

**Morning:**

1. **Understand `delegate_task`.** Read the tool's docstring:
   ```
   What does delegate_task do and what parameters does it take?
   ```
   Or inspect the source at `tools/delegate_tool.py`. Key parameters:
   - `goal` — what the subagent must achieve.
   - `context` — background info the subagent needs.
   - `acceptance_criteria` — how you'll know it succeeded.
   - `toolset` — which tools the subagent can use.
   - `files` — files or directories to scope the work to.

2. **Write your first atomic task.** A task is atomic if it:
   - Has a single, measurable outcome.
   - Takes the subagent < 5 minutes.
   - Has an acceptance test you can run in one command.
   
   Example:
   ```
   Delegate: Create a Python script that calculates fibonacci numbers.
   Context: The script should be saved at ~/fib.py.
   Acceptance criteria: Running `python3 ~/fib.py 10` prints 55.
   ```

**Afternoon:**

3. **Practice splitting work.** Take a multi-step problem (e.g. "set up a
   Flask app with a health endpoint and a Dockerfile") and split it into
   atomic tasks:
   - Task 1: Create `app.py` with a `/health` endpoint that returns `{"status": "ok"}`.
   - Task 2: Create `Dockerfile` that exposes port 5000 and runs `app.py`.
   - Task 3: Create `docker-compose.yml` that maps port 5000.

   Each task has one output, one acceptance criterion. Delegate them
   sequentially.

4. **Anti-pattern to avoid:**
   - ❌ "Fix the project" — too vague.
   - ❌ "Make the tests pass" — what tests? What's the expected state?
   - ✓ "Update the test suite so that `pytest tests/ -x` passes. The three
     failing tests are ..."

### Day 11–12: Subagent Delegation Patterns

**Goal:** Learn the three main delegation patterns and when to use each.

**Morning:**

1. **Sequential delegation.** One subagent hands off to the next. Use when
   tasks depend on each other:
   ```
   Delegate task 1: Parse the CSV and create a JSON schema.
   Wait for completion.
   Delegate task 2: Use the schema to validate the data.
   ```

2. **Parallel delegation.** Run multiple subagents simultaneously. Use when
   tasks are independent:
   ```
   Delegate task A: Audit frontend dependencies for security issues.
   Delegation task B: Audit backend dependencies.
   (Both run at the same time.)
   ```
   The agent handles parallelisation automatically when you send multiple
   delegation calls in the same message.

3. **Fan-out delegation.** One parent delegates the same task to multiple
   subagents with different inputs. Use for batch processing:
   ```
   For each CSV file in data/raw/, delegate a subagent to:
   - Normalise the column names
   - Write to data/processed/{filename}
   - Report row counts
   ```

**Afternoon:**

4. **Practice: Build a mini ETL pipeline.**
   - Create 3 CSV files with different schemas.
   - Delegate one subagent per file to normalise them.
   - Delegate a final subagent to merge them.
   - Verify the merged output.

5. **Watch the isolation boundaries.** Each subagent gets:
   - A fresh conversation with no parent history.
   - Its own task_id (own terminal session, file ops cache).
   - No access to `delegate_task` (no recursive delegation), `clarify`
     (no user interaction), or `memory` (no writes to shared memory).
   
   This means you cannot rely on subagents asking for help. Your task
   description must be complete.

### Day 13–14: Acceptance Tests in Task Context

**Goal:** Embed verification directly into the task so the subagent
self-verifies.

**Morning:**

1. **Add acceptance criteria to every delegation.** Your criteria should be
   executable, not aspirational:
   ```
   Acceptance criteria (must all pass):
   - `python3 validate.py --input data.csv` exits with code 0.
   - `wc -l data_clean.csv` shows at least 1000 rows.
   - No line in data_clean.csv contains empty cells.
   ```

2. **Ask the subagent to self-test.** Add to your task context:
   ```
   After creating the script, run these commands and report the output:
   1. python3 -m pytest tests/
   2. python3 my_script.py --test
   ```

**Afternoon:**

3. **Build a delegation template.** Create a reusable pattern:
   ```
   Task: <one-line summary>
   Context:
     - Repository: /path/to/project
     - Files you may modify: <list>
     - Files you must NOT modify: <list>
   Acceptance criteria (MUST verify each):
     1. `command to check` == expected output
     2. `other command to check` == expected output
   ```

4. **Practice with a failing task.** Deliberately give a subagent incomplete
   context and see what happens. Then improve your task description and
   re-delegate. The goal is to internalise how much context a subagent
   actually needs.

**Weekly checkpoint (end of Day 14):** You should be able to:
- Decompose a project into atomic tasks.
- Delegate tasks sequentially and in parallel.
- Write acceptance criteria that a subagent can self-verify.
- Predict when a delegation will succeed or fail based on task quality.

---

## Week 3: Teaching the Agent (Skills, Memory, USER.md)

**Theme:** Stop repeating yourself. Write things down once so the agent learns
your preferences.

### Day 15–17: Writing Effective Skills

**Goal:** Create skills that encode your team's patterns, conventions, and
standard operating procedures.

**Morning:**

1. **Understand the skills system.** Skills live in two places:
   - Bundled: `hermes-agent/skills/<name>/` (shipped with the repo).
   - User: `~/.hermes/skills/<name>/` (yours to create).

   Each skill is a directory with a `skill.yaml` manifest and a prompt file.
   Inspect an existing skill:
   ```
   Show me the contents of a skill - for example, the TDD skill.
   ```
   Look at `hermes-agent/skills/software-development/test-driven-development`
   to see the structure.

2. **Create your first skill from scratch.**
   ```bash
   mkdir -p ~/.hermes/skills/my-project-conventions
   ```
   Create `skill.yaml`:
   ```yaml
   name: my-project-conventions
   description: Coding conventions for the Acme project
   version: 1.0.0
   ```
   Create the prompt file (e.g. `main.md`):
   ```markdown
   When working on the Acme project, follow these conventions:
   - Python: use type hints on all public functions.
   - Tests: put tests in tests/unit/ for unit tests.
   - Naming: snake_case for functions, PascalCase for classes.
   - Error handling: use custom exception classes in exceptions.py.
   ```

3. **Load your skill and test it.**
   ```
   Load the my-project-conventions skill.
   I need to add a function to the Acme project. What conventions apply?
   ```
   The agent should reference your written-down rules.

**Afternoon:**

4. **Learn the skill loading commands.**
   - `hermes skill list` — see what's available.
   - `hermes skill load <name>` — activate a skill.
   - `hermes skill unload <name>` — deactivate.
   
   You can also set skills to auto-load in `~/.hermes/config.yaml`:
   ```yaml
   skills:
     auto_load:
       - my-project-conventions
       - systematic-debugging
   ```

5. **Identify 3 things you repeat.** Over the next few days, every time you
   find yourself correcting the agent about the same thing, turn it into a
   skill. Good candidates:
   - Code style preferences.
   - Project-specific directory layout.
   - Standard response formats (e.g. "always show a table of files changed").

6. **Anti-pattern to avoid:**
   - ❌ Writing skills that are too long — the agent will ignore them.
   - ❌ Writing skills that are too vague — "write good code" helps no one.
   - ✓ Writing specific, actionable rules with examples.

### Day 18–19: Memory Hygiene

**Goal:** Use the memory system intentionally. Know what goes in, what stays,
and what gets cleaned up.

**Morning:**

1. **Understand how memory works.** Hermes uses Mnemosyne (SQLite-backed) or
   alternative memory providers (honcho, mem0, supermemory, etc.). The default
   is the built-in mnemosyne provider.

   Key concepts:
   - **Facts:** Individual pieces of information (e.g. "User prefers tabs over
     spaces").
   - **Conversations:** The agent may remember things you said in previous
     sessions.
   - **Memory commands:** `hermes memory` CLI subcommands for inspection.

2. **Query the memory.**
   ```
   What do you remember about me?
   ```
   Or from the CLI:
   ```bash
   hermes memory search "preferences"
   ```

3. **Ask the agent to store something intentionally.**
   ```
   Remember that I prefer pytest over unittest for all Python testing.
   ```
   Then verify it was stored:
   ```
   What Python testing framework do I prefer?
   ```

**Afternoon:**

4. **Practice memory hygiene.**
   - **Store deliberately:** Only commit facts you want the agent to act on
     consistently. Don't store throwaway preferences.
   - **Review weekly:** Run `hermes memory list` and prune outdated entries.
   - **Correct mistakes:** If the agent misremembers, correct it explicitly.
   
   ```bash
   hermes memory list --limit 50
   hermes memory delete <fact-id>
   ```

5. **Configure memory in `config.yaml`.** Set retention limits:
   ```yaml
   memory:
     provider: mnemosyne
     max_facts: 200
     auto_prune: true
   ```

6. **Anti-pattern to avoid:**
   - ❌ Letting the agent auto-store every conversation — your memory will
     fill with noise.
   - ❌ Never reviewing memory — stale facts accumulate and cause wrong
     behaviour.
   - ✓ Reviewing and curating memory weekly.

### Day 20–21: USER.md and AGENTS.md

**Goal:** Create your identity file so every new agent session starts with
your preferences already loaded.

**Morning:**

1. **Create `~/.hermes/USER.md`.** This file is loaded at the start of every
   conversation. It's your permanent identity document. Start with:
   ```markdown
   # User Profile
   
   ## Contact
   - Name: Jane Doe
   - Timezone: Europe/London
   - Preferred editor: VS Code
   
   ## Preferences
   - Shell: zsh
   - Python: 3.12+
   - Testing: pytest
   - Git: conventional commits
   
   ## Recurring tasks
   - Weekly log review: every Friday at 4pm
   - Monthly dependency audit: 1st of each month
   ```
   
   The agent will read this at session start and adapt its behaviour.

2. **Test USER.md.** Start a new session:
   ```
   What do you know about my preferences?
   ```
   It should reference your USER.md content. If it doesn't, check that the
   file is in the right location (`~/.hermes/USER.md`).

**Afternoon:**

3. **Create `AGENTS.md` for projects.** For each project you work on, create
   an `AGENTS.md` in the project root. This is like USER.md but per-project:
   ```markdown
   # Project: Acme API
   
   ## Architecture
   - Language: Python 3.12
   - Framework: FastAPI
   - Database: PostgreSQL 16
   - Queue: Redis + Celery
   
   ## Conventions
   - Routes: /api/v1/<resource>
   - Auth: JWT in Authorization header
   - Errors: RFC 7807 problem+json format
   
   ## Critical files
   - src/app.py — FastAPI app factory
   - src/config.py — settings from environment
   - alembic/ — database migrations
   ```

4. **Understand the priority chain.** When the agent resolves a question
   about your preferences, it checks (in order):
   1. Current conversation context.
   2. Loaded skills.
   3. `AGENTS.md` in the project root.
   4. `USER.md` in `~/.hermes/`.
   5. Memory.
   
   This means a project-level `AGENTS.md` can override user-level preferences
   without changing your global config.

**Weekly checkpoint (end of Day 21):** You should be able to:
- Write a skill that changes how the agent behaves.
- Query and curate the agent's memory.
- Set up USER.md and AGENTS.md so new sessions start personalised.
- Predict which source (skill, USER.md, AGENTS.md, or memory) will win in a
  conflict.

---

## Week 4: Automation (Cron, Gateway, Metrics)

**Theme:** The agent should work when you're not watching. Set it up,
monitor it, and measure its effectiveness.

### Day 22–24: Cron Jobs

**Goal:** Schedule the agent to run tasks on a recurring basis without your
involvement.

**Morning:**

1. **Understand the cron system.** Hermes cron is built into the gateway. It
   uses:
   - `cron/jobs.py` — job storage and management (`~/.hermes/cron/jobs.json`).
   - `cron/scheduler.py` — tick-based scheduler that checks due jobs every
     60 seconds.
   - `croniter` for cron expression parsing.

   Jobs are stored as JSON and output goes to
   `~/.hermes/cron/output/{job_id}/{timestamp}.md`.

2. **Create your first cron job via the CLI.**
   ```bash
   hermes cron add --name "daily-summary" --schedule "0 9 * * *" \
     --task "Check the project boards and send a summary of what changed yesterday"
   ```
   Verify it was added:
   ```bash
   hermes cron list
   ```

3. **Also try creating a job from within a conversation:**
   ```
   Create a cron job that runs every weekday at 6pm and checks if my
   staging server is healthy.
   ```

**Afternoon:**

4. **Monitor cron output.** After a job runs, inspect its output:
   ```bash
   hermes cron log daily-summary --tail
   ```
   Or browse the output directory:
   ```bash
   ls ~/.hermes/cron/output/daily-summary/
   ```

5. **Handle failures gracefully.** Add error handling to your cron tasks:
   - If the agent can't complete the task, it will report in the output log.
   - Configure the gateway to send alerts on cron failures.
   - Review failed jobs:
     ```bash
     hermes cron list --status failed
     ```

6. **Useful cron schedule patterns:**
   ```
   * * * * *        every minute
   0 * * * *        every hour
   0 9 * * 1-5      weekdays at 9am
   0 0 * * 0        Sunday midnight
   30 8 1 * *       8:30am on the 1st of every month
   ```

7. **Anti-pattern to avoid:**
   - ❌ Creating a job that runs every minute without understanding the
     resource cost.
   - ❌ Never checking cron output — a failing job is worse than no job.
   - ✓ Starting with once-daily jobs and adding frequency only when needed.

### Day 25–26: Gateway Setup

**Goal:** Connect Hermes to the platforms you use (Telegram, Discord, Slack)
so you can interact with it from anywhere.

**Morning:**

1. **Understand the gateway.** The gateway (`gateway/`) is a multi-platform
   message relay. It supports Telegram, Discord, Slack, WhatsApp,
   Home Assistant, Signal, Matrix, Mattermost, email, SMS, DingTalk, WeCom,
   Feishu, QQ Bot, and more. Each platform has an adapter in
   `gateway/platforms/`.

2. **Set up one platform.** The most common first gateway is Telegram (it has
   the simplest setup):
   ```bash
   # Set up your bot token
   hermes gateway setup --platform telegram
   ```
   Follow the prompts. You'll need a bot token from @BotFather on Telegram.

3. **Verify the gateway is running.**
   ```bash
   hermes gateway status
   hermes logs --gateway --follow
   ```
   Send a message to your bot on Telegram. You should see it appear in the
   logs and get a response.

**Afternoon:**

4. **Configure multiple platforms.** Add a second platform (e.g. Discord):
   ```bash
   hermes gateway setup --platform discord
   ```
   The gateway can run multiple platforms simultaneously. All messages go
   through the same agent loop.

5. **Understand gateway config.** Look at `~/.hermes/config.yaml` under the
   `gateway:` section. Key settings:
   ```yaml
   gateway:
     enabled_platforms: [telegram, discord]
     auto_start: true
     session_reset: new_topic  # how sessions are handled
     delivery:
       progress: true  # show tool progress in platform messages
   ```

6. **Test with a real-world scenario.** From Telegram, ask:
   ```
   Check if the production server is healthy.
   ```
   The agent should use tools, get the answer, and reply via Telegram. You
   just interacted with the agent without being at your terminal.

7. **Anti-pattern to avoid:**
   - ❌ Connecting every platform at once — start with one, get it working,
     then add more.
   - ❌ Exposing the agent on a public channel without rate limits or
     permission controls.
   - ✓ Test in private chat or a dedicated channel first.

### Day 27–28: Metrics and Kickback Rate Tracking

**Goal:** Measure how well the agent is serving you and identify where it
needs improvement.

**Morning:**

1. **Understand the metrics system.** Hermes tracks:
   - Tool call success/failure rates.
   - Conversation length (turns per session).
   - Token usage (input + output per session).
   - Kickback rate (how often you reject or correct the agent's output).

2. **Track your kickback rate manually.** For each session this week, record:
   ```
   Session: <id>
   Total turns: 8
   Corrections: 2
   Kickback rate: 25%
   ```
   A kickback is any time you say "no, that's not right" or "try again" or
   "that's not what I meant". Track this for every session.

3. **Use `hermes logs` for metrics.**
   ```bash
   hermes logs --session <id> --level INFO | grep "tool_call\|error"
   ```
   Count the errors and compare to total tool calls.

**Afternoon:**

4. **Set up the observability plugin.** If your Hermes installation includes
   it, enable tracing:
   ```yaml
   plugins:
     observability:
       enabled: true
       metrics: true
       traces: true
   ```
   This gives you structured data on every tool call, model response, and
   error.

5. **Define your personal metric targets.** Example:
   - **Kickback rate:** < 20% (by end of Month 1), < 10% (by Month 3).
   - **First-attempt success:** > 70% for tasks you've done before.
   - **Cron job success rate:** > 95%.
   - **Average turns per task:** Decreasing over time as delegation improves.

6. **Review weekly.** Every Friday, run:
   ```bash
   hermes logs --since "7 days ago" --level ERROR | wc -l
   ```
   Check your kickback log. Look for patterns:
   - Do you correct the same kind of mistake repeatedly? → Write a skill.
   - Does the agent misunderstand your project structure? → Update AGENTS.md.
   - Does delegation fail on complex tasks? → Improve your task descriptions.

**Weekly checkpoint (end of Day 28):** You should be able to:
- Schedule recurring tasks with cron and review their output.
- Interact with the agent from at least one remote platform.
- Track your kickback rate and identify improvement areas.
- Plan your next month of agent usage based on real metrics.

---

## Anti-Patterns

These patterns show up repeatedly in new users. Avoid them.

| Anti-pattern | Why it fails | What to do instead |
|---|---|---|
| **Vague requests** ("fix the project") | The agent doesn't know what success looks like. | Specify the desired outcome with measurable criteria. |
| **Not verifying output** | The agent can produce plausible-looking wrong answers. | Always verify. Run the code. Check the logs. |
| **One giant task** | The agent runs out of context or iterations. | Break work into atomic tasks of < 5 minutes each. |
| **Ignoring subagent isolation** | Expecting a subagent to know what happened in a previous delegation. | Put all necessary context in the task description. |
| **Memory as dumping ground** | Filling memory with noise buries the signal. | Store only stable, reusable facts. Prune regularly. |
| **Inconsistent delegation style** | Each delegation is structured differently, leading to variable results. | Use a consistent task template for all delegations. |
| **Skills that are novels** | Long skills get ignored by the model. | Keep skills under 500 words with clear, numbered rules. |
| **Cron-and-forget** | Failing jobs run silently and waste resources. | Check cron output daily during Week 4. Set up alerts. |
| **No USER.md** | Every session starts from zero. | Create USER.md on Day 20. It's the single biggest quality-of-life improvement. |
| **Not reviewing metrics** | You don't know if you're improving. | Track kickback rate. Review logs weekly. Adjust. |

---

## Quick Reference

### CLI Commands

| Command | Purpose |
|---|---|
| `hermes` | Start the interactive CLI |
| `hermes --tui` | Start the terminal UI (Ink-based) |
| `hermes logs [--follow]` | View agent logs |
| `hermes logs --session <id>` | View logs for a specific session |
| `hermes skill list` | List available skills |
| `hermes skill load <name>` | Activate a skill |
| `hermes memory list` | List stored facts |
| `hermes memory search <query>` | Search memory |
| `hermes memory delete <id>` | Remove a fact |
| `hermes cron add` | Create a scheduled job |
| `hermes cron list` | List scheduled jobs |
| `hermes cron log <name>` | View job output |
| `hermes gateway status` | Check gateway connectivity |
| `hermes gateway setup --platform <p>` | Configure a platform |

### Config Paths

| Path | Purpose |
|---|---|
| `~/.hermes/config.yaml` | Main configuration |
| `~/.hermes/.env` | API keys only |
| `~/.hermes/USER.md` | User identity (loaded every session) |
| `~/.hermes/skills/<name>/` | User-installed skills |
| `~/.hermes/cron/jobs.json` | Cron job definitions |
| `~/.hermes/cron/output/<id>/` | Cron job output |
| `~/.hermes/logs/` | Log files |
| `/path/to/project/AGENTS.md` | Project-level instructions |

### Key Codebase Files

| File | Purpose |
|---|---|
| `tools/delegate_tool.py` | Subagent delegation implementation |
| `cron/scheduler.py` | Cron job scheduler engine |
| `cron/jobs.py` | Cron job storage and management |
| `gateway/config.py` | Multi-platform gateway configuration |
| `gateway/run.py` | Gateway runner |
| `plugins/memory/` | Memory provider plugins (mnemosyne, honcho, etc.) |
| `skills/` | Built-in skills repository |
| `optional-skills/` | Heavier skills not active by default |

---

## When to Move On from This Ramp

This guide is structured as a 4-week plan, but you should adjust the pace to
your context. You're ready to move on when:

- **Week 1 complete:** You instinctively think "plan, execute, verify" before
  sending any message to the agent.
- **Week 2 complete:** You can decompose a complex problem into atomic tasks
  and the subagents succeed more often than they fail.
- **Week 3 complete:** The agent adapts to your project conventions without
  you having to re-explain them.
- **Week 4 complete:** The agent handles recurring tasks autonomously and you
  review metrics weekly to drive improvement.

Beyond Week 4, the path diverges based on your use case:
- **Software engineering:** Focus on the ACP adapter (VS Code / Zed /
  JetBrains integration) and batch runner.
- **DevOps / SRE:** Deepen cron usage, gateway alerting, and multi-platform
  incident response.
- **Research / data science:** Explore the kanban multi-agent orchestrator
  and parallel subagent fan-out patterns.
- **General productivity:** Build a personal knowledge base through skills
  and memory, and connect the gateway to your daily tools.

---

*Remember: Hermes Agent is a tool, not a replacement. The goal is not to
automate yourself out of the loop — it's to make the loop faster, more
reliable, and less tedious. Your judgment is the most important component.*
