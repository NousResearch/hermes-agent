# /wrap — Session Wrap Skill

**Invocation:** `/wrap` or `/wrap [target-agent]`
**Works for:** any agent (saga, freya, mimir)

With no argument: wraps own session and saves state.
With `[target-agent]`: wraps own session AND writes a handoff to `progress/[target_agent].md`. Use when work is complete and changes character — e.g. strategy done, mimir should build.

---

<wrap-steps>

  <constraints>
    - Derive everything from the conversation. Do NOT ask the user any questions.
    - Your agent_id is your WDS base name: saga, freya, or mimir. Never a project name.
    - Show substance to user BEFORE spawning subagent — user must see what is being saved.
    - The subagent handles all mechanical execution. You only compile and show.
    - If `[target-agent]` was given: after saving state, also write a handoff to `progress/[target_agent].md` (step 4).
  </constraints>

  <step id="0-milestone-check">
    Before writing anything: assess whether this is a natural milestone boundary.

    A milestone boundary is when a discrete unit of work is complete — a feature shipped,
    a spec finalized, a phase closed. NOT mid-task, mid-investigation, or mid-dialog.

    **If NOT at a milestone:** note this as "mid-session" in Context. The Next task should
    be the immediate continuation of interrupted work.

    **If at a milestone:** proceed normally.

    **Call threshold:** If this session has had 15+ tool calls, surface once as part of step 2:
    `Note: session at [N] calls — good time to wrap for fresh context.`
  </step>

  <step id="1-compile">
    Compile the session substance internally. Do NOT write to disk. Do NOT output anything.

    Compose these four fields:

    **learned:** What will benefit future sessions: decisions with reasons, patterns,
    non-obvious constraints. "None" if nothing was learned.

    **context:** What was done. State of artifacts. Open threads. Be specific.
    If mid-session: "Wrapped mid-task: [what was in progress]"

    **plan:** The overarching plan and end goal. Where we are. What remains.
    If multi-session: list numbered milestones with status:
      - [DONE] Milestone 1 — description
      - [CURRENT] Milestone 2 — description (~1 session)
      - [ ] Milestone 3 — description (~2 sessions)
    Omit milestone list if single-session work.

    **next:** Single immediately-actionable next task.
    Prefix with model: MODEL:[Haiku|Sonnet|Opus] — task description.
    Model selection = task type × complexity × stakes:
      - Haiku: simple, low-stakes, short — lookups, summaries
      - Sonnet: moderate complexity — strategy, spec, dialog, UX, config, analysis
      - Opus: any code; OR high-stakes/production work; OR long or complex tasks
    Default to lightest model that can handle the task.

    **spec_sync:** Did anything change that diverges from a written spec/brief/doc?
    "None" if nothing changed.
  </step>

  <step id="2-show">
    Print EXACTLY this block to the user — nothing before, nothing after:

    ── Handover ──────────────────────────────────
    Next:    [next — including MODEL prefix]
    Plan:    [plan — one line summary or current milestone]
    Open:    [blocking issues or "None"]
    Learned: [learned — one line or "None"]
    ──────────────────────────────────────────────

    [If call threshold reached: print "Note: session at [N] calls — good time to wrap."]

    Wait for no input. Proceed immediately to step 3.
  </step>

  <step id="3-subagent">
    Spawn a subagent using the Agent tool with this exact prompt —
    substitute the bracketed values from step 1:

    ---
    You are a wrap executor. Your only job is to save a session wrap file.
    Follow these steps exactly. No interpretation. No additions.

    **Session data:**
    - agent_id: [saga|freya|mimir]
    - learned: [learned]
    - context: [context]
    - plan: [plan]
    - next: [next]
    - spec_sync: [spec_sync]

    **Step A — Save state via memory tool:**
    Read `~/.claude/wds/tools/memory/SKILL.md` and follow the `save` operation:
    - agent_id: [agent_id]
    - data:
    ```
    ## Wrapped
    [current date and time]

    ## Context
    [context]

    ## Plan
    [plan]

    ## Next
    [next]

    ## Learned
    [learned]

    ## Spec Sync
    [spec_sync]
    ```

    **Step B — Update project index:**
    1. Run `git rev-parse HEAD` → `current_head`
    2. Read `progress/project-index.md` if it exists → extract HEAD hash from `## Updated` line as `last_head`
    3. Get changed files:
       - If `last_head` exists: `git diff --name-only [last_head] [current_head]`
       - If first time (no index): `git ls-files -- '*.md'` excluding `progress/`, `node_modules/`, `.git/`
    4. For each changed file that exists: read its first H1 heading and first non-heading paragraph → one-line description. If deleted: mark for removal.
    5. Read current `progress/project-index.md` (if exists), update changed entries, add new ones, remove deleted ones.
    6. Write `progress/project-index.md`:

    ```
    ## Project Index
    Updated: [agent_id] [current date] [current_head]

    ## Phase Status
    [preserve existing phase lines, update if plan indicates phase change]

    ## Artifacts
    [absolute path] — [type: brief|scenario|spec|design|code|config] — [one-line description]
    [one entry per relevant file, sorted by path]
    ```

    **Step D — Confirm:**
    Return ONLY: `Saved to progress/[agent_id].md — index updated ([N] files)`
    ---

    Print whatever the subagent returns.

    **If the subagent fails at any step:** complete the remaining steps manually.
    Failure does not excuse skipping the final output.
  </step>

  <step id="4-handoff" condition="only if target-agent argument was given">
    Spawn a second sub-agent with this exact prompt — substitute the bracketed values:

    ---
    You are a handoff writer. Your only job is to save a handoff file via the memory tool.

    **Step A — Save handoff via memory tool:**
    Read `~/.claude/wds/tools/memory/SKILL.md` and follow the `save` operation:
    - agent_id: [target_agent]
    - data:
    ```
    ## Wrapped
    [current date and time]

    ## Context
    [context]

    ## Next
    [next]

    ## Learned
    [learned]

    ## Spec Sync
    [spec_sync]
    ```

    **Step B — Confirm:**
    Return ONLY: `done`
    ---

    Wait for the sub-agent to return. Then print EXACTLY these two lines — the label, then the command in a code block:
    → Open a new chat and run:
    ```
    /[target_agent] progress/[target_agent].md
    ```

    **If the sub-agent fails:** write the handoff file manually, then still output the command block above.

    Session complete. Do not respond to further input.

    **The command block above is always the last thing output. Nothing is printed after it —
    no summary, no explanation, no confirmation. The block is the signal that the wrap is complete.**
  </step>

</wrap-steps>
