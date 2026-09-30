# UI/UX Research Report — OpenAI Codex Desktop App

**Scope:** Codex app for macOS/Windows (as shipped in the ChatGPT desktop app, build 26.x), plus Codex Cloud web surfaces where they differ.
**Audience:** designers/engineers rebuilding a comparable multi-agent coding UI.
**Method note:** built from OpenAI primary documentation (developers.openai.com/codex/*.md — the docs' own markdown renditions), the official Codex changelog, the official launch blog, and direct pixel analysis of official product screenshots. Every claim is tagged:
- `[doc]` — stated in official OpenAI documentation/changelog/blog.
- `[observed]` — measured by me directly from official OpenAI product screenshots (pixel/color/geometry analysis, plus visual inspection).
- `[secondary]` — third-party description, not verified against primary source.

**Important framing correction:** as of the 2026-07-09 release (`26.707`), Codex was absorbed into the **ChatGPT desktop app** and is now a "surface" (view) within it, alongside Chat and Work. Much of the current documentation is branded "ChatGPT desktop app" with per-surface conditional content. Any design reference should target the Codex surface, which is the one with panes, diff, and terminal. `[doc]`

---

## 1. Layout and Information Architecture

### 1.1 The three-region shell

The Codex app window is a **three-column shell**, not a tab-per-task or single-column chat app. From the official screenshot analysis of `codex-windows-dark.webp` (1919×1152 capture of the app window, 1639px of usable window width at 1× capture scale): `[observed]`

| Region | Measured extent (dark Windows capture) | Notes |
|---|---|---|
| Left sidebar | x ≈ 0–556 (**33.9%** of window width) | Project + thread navigation, app-level nav, Settings pinned at bottom |
| Main column | x ≈ 558–1638 (66%) | Conversation/thread transcript + composer |
| Right auxiliary column | Toggleable | Review pane, file tree, terminal, browser tab, artifact preview |

The app's own alt-text for its launch screenshot is the cleanest official statement of the IA: **"Codex app showing a project sidebar, thread list, and review pane."** `[doc]`

This three-pane structure is the app's defining architectural decision and is stated in the launch blog: agents "run in separate threads organized by projects," and "you can review the agent's changes in the thread, comment on the diff, and even open it in your editor." `[doc]`

### 1.2 Sidebar anatomy

Measured and visually confirmed from the official dark-theme capture. `[observed]`

**Top — app identity + global nav:**
- Codex logo mark + "Codex" wordmark (the ChatGPT build adds a Chat / Work / **Codex** surface switcher, `Ctrl+1/2/3` or `Alt+1/2/3`). `[doc]`
- Four global action rows: a window/home icon, **New thread**, **Automations**, **Skills**. `[observed]`
- **Activity bell** — a dedicated sidebar entry filtering to "chats that are unread, running, or waiting for your response." `Cmd/Ctrl+Opt+U` toggles Activity view. `[doc]` `[observed]`

**Middle — Recents (un-headed, highest priority):** `[observed]`
Rows carry: a leading status glyph, the thread title (truncated with `…`), an optional **status pill**, an optional expand chevron, and a dimmed relative timestamp right-aligned (`5m`, `7m`, `8m`).
- Green pill = **"Awaiting approval"** `[observed]`
- Solid blue filled circle = active/in-progress thread `[observed]`
- Sparkle glyph = agent-generated/derived thread `[observed]`

**Lower — section header + Projects:** `[observed]`
A muted grey `Threads`/`Projects` header carries two right-aligned affordances (import/folder-add, and a `⋯` overflow menu). Below it, one row per project, each with a folder icon; the **active project row** gets a subtly lighter rounded fill (`#1F1F2D` over a `#1A1F39` sidebar base in the dark capture — about 5% lightness lift, i.e. selection is expressed by a *very* low-contrast fill, not a border or accent bar). `[observed]` The last row is **"Add project"** with a `+` folder icon. `[observed]`
- Project rows can carry **inline diff stats** — `+1 -9` with green additions / red deletions — plus a timestamp. `[observed]`
- Permanent worktrees appear **as their own projects**, created from the project `⋯` menu. `[doc]`
- Settings is a fixed row pinned to the sidebar bottom. `[observed]`

**Ordering logic:** Recents sits *above* the Projects section and takes priority, with no explicit "Recents" header — a deliberate choice that privileges running work over the project tree. `[observed]`

### 1.3 The composer

**Measured:** the composer is a **floating, centered card, not a full-width bar.** `[observed]`
- Bounding box: x 729–1453 → **724px wide**, y 840–910 → **70px tall** (collapsed, one-line state).
- Width relative to main column (1080px): **0.67** — leaving ~171px of gutter on each side.
- Corner radius measured at the top-left by insetting scan: inset goes 21 → 5 → 4 → 3 → 2 → 1 → 0 across dy=0…6, i.e. **radius ≈ 6–8px**. A restrained radius, not the 12–16px pill shape common in consumer chat UIs.
- Fill `#242424` on a `#181818` main-pane background — a **1-step elevation** expressed as a flat fill, **no visible drop shadow**.
- Internal sub-structure: content at ~y856 (input text line), a control row at ~y888–896 (icon cluster), and the send affordance on the right of that row. `[observed]`

**The composer is the app's primary control cluster.** Everything an operator needs to change per-run sits beneath it. This is an unusually important IA decision and is corroborated by the docs describing the composer as the home of:
- the **permissions control** ("use the permissions control beneath the composer") `[doc]`
- the **model + reasoning control** ("use the model and reasoning control beneath the composer") `[doc]`
- the **Worktree / Local / Cloud** selector (in the new-chat view, "select **Worktree** under the composer") `[doc]`
- the **starting-branch selector** for worktrees ("Below the composer, choose the Git branch") `[doc]`
- the **IDE context toggle** `[doc]`
- the **goal progress row** (above the composer, when a goal is active) `[doc]`
- the **subagent panel** (above the composer in the IDE surface) `[doc]`
- **Action buttons** (project actions "appear in the ChatGPT desktop app top bar") `[doc]`

`@` = files/context/skills/sources, `$` = skills, `/` = commands. `[doc]` A dedicated **"Add files and more /"** menu is a documented, separately-complaining-about surface. `[doc]` `[observed in GitHub issue #19747]`

### 1.4 Auxiliary panes and how they resize/collapse

| Pane | Trigger | Behavior |
|---|---|---|
| **Review / diff pane** | `Ctrl/Cmd+Shift+G` ("Open review tab"); `/review` | Right-hand column. Contains scope selector, file list, inline diff, inline comments, Git actions. |
| **File tree** | `Cmd/Ctrl+Shift+E` ("Toggle file tree") | "percentage-based file tree resizing" shipped 2026-03-18 — resizing is **proportional, not pixel-snapped**. `[doc]` |
| **Integrated terminal** | `Ctrl+` (backtick) ("Toggle terminal") | **Bottom panel**, not a right column. `Cmd/Ctrl+J` = "Toggle bottom panel". Clear with `Ctrl+L`/`Cmd+K` when focused. |
| **Bottom panel (generic)** | `Cmd/Ctrl+J` | Hosts the terminal; separate from the terminal toggle. |
| **Browser** | `Cmd/Ctrl+T`; `Cmd+.` toggles browse/comment mode | Workspace **tab** in a tabbed right column, with address bar, history search, and browse/comment mode. |
| **Artifact / file preview** | automatic on task completion | Sidebar previews PDF, spreadsheet, document, presentation; `.html` opens as interactive preview with source-view toggle. |
| **Image viewer** | 2026-07-30 | "Focused view" and "Canvas view" toggle, cross-image comments. |
| **Sources panel** | 2026-04-09 | "Git summary and Sources section in the thread side panel." |

**Two distinct layout modes for the tabbed column** — this is a genuinely good idea: `[doc]`
- **Full view** — the tab fills the column (`Cmd+Shift+F`).
- **Split view** — chat and tab side by side (`Cmd+Shift+B` cycles Full / Split / hidden tabs).
- `Cmd+Ctrl+B` "switches between chat and tabs (In full view; shows or hides tabs in split view)."

**Tab persistence is deliberate:** "Keep tab widths and scroll positions more stable as you close browser tabs in full and split views" (2026-09-11); "Preserved thread scroll position per conversation and unread state across windows" (2026-04-09). `[doc]`

**Sidebar resize:** `Cmd/Ctrl+B` toggles the sidebar. The file tree's own resize is percentage-based. `[doc]`

### 1.5 What is *not* a pane (a real IA gap)

There is **no persistent "tasks" or "runs" table anywhere** in the app. Thread status is distributed across: sidebar row glyphs, the Activity bell view, the goal progress row, and system notifications. There is no unified "queued / running / needs-input / done / failed" board. This is the app's most significant IA weakness (see §6.2).

---

## 2. Core Screens

### 2.1 New-task entry

Two distinct entry paths, which is a real IA decision: `[doc]`

1. **With a project** — "In Codex, select **New chat** beside **Recents** to start without a project." So the sidebar's New-chat control is the projectless path.
2. **Without a project** — `/task` ("Start a chat without a project").
3. **Quick chat** — a separate icon on the right of **New chat**; `Cmd+Ctrl+N`. Opens an *ordinary ChatGPT chat* that does **not** appear in the Codex sidebar. `[doc]` `[observed]`
4. **Standalone chat** — `Cmd+Ctrl+O`, Codex-only. `[doc]`
5. **Worktree entry** — "In the new chat view, select **Worktree** under the composer", then choose a starting branch below the composer. `[doc]`

The new-chat view's pre-send configuration surface (below/under the composer) is: **environment** (Local / Worktree / Cloud), **starting branch**, **local environment setup script**, **permissions**, **model + reasoning effort**. `[doc]`

Onboarding includes example prompts surfaced as copyable chips: "Tell me about this project", "Build a classic Snake game in this repo.", "Find and fix bugs in my codebase with minimal, high-confidence changes." `[doc]`

**Import affordance:** `26.608` added "Import to Codex flows for importing supported setup from Claude Code and Claude Cowork, including during onboarding." `[doc]`

### 2.2 Active run view

The transcript is the run view. Structure, per the docs and changelog: `[doc]`

- **Assistant messages** with markdown, code blocks, and inline **annotations** (select text → annotate → "ask Codex to revise selected content"). Since `26.707`.
- **Tool/command activity** rendered inline. `26.320` shipped "Floating Composer v2"; `26.318` added a "branded loading shimmer while the app starts."
- **Subagent threads** — "The app surfaces each subagent thread so you can inspect its work and the summary returned to the main chat." Expandable, stoppable, individually openable. `[doc]`
- **Diff stats surfaced in the composer** — `26.325`: "surfaced subagent diff stats in the composer." `[doc]`
- **Plan mode** — `/plan` makes the agent "investigate and propose an approach before editing." `[doc]`
- **Side chats** — `/side` "Start a temporary side chat without interrupting the main chat" (`Cmd+Ctrl+S`). This is the explicit answer to "ask a question mid-run without disturbing state." `[doc]`
- **Goal progress row** above the composer with pause / resume / edit / clear buttons. `[doc]`

Measured body-text density in the main column: **13px glyph band with a consistent 36px line pitch** (22–23px inter-line gap). `[observed]` At 1× capture scale in a 1639px-wide window, that is a comfortable-but-airy reading rhythm, notably looser than a terminal or an IDE diff view.

### 2.3 Review / approval of a proposed change

Two separate surfaces, which is correct — the docs are careful to distinguish them: `[doc]`

**(a) The review pane (local Git changes).** "Open the review pane to understand what changed, give line-specific feedback, and decide what to stage, revert, commit, or push."

- **Scope selector** in the review header: **Unstaged** (default), **Staged**, **Commit**, **Branch**, **Last turn**. `[doc]`
- The pane reflects **the whole repo state, not just what Codex edited** — it includes your own edits and any other uncommitted changes. This is stated explicitly and is an honesty win. `[doc]`
- **Repository selector** in the review header for multi-repo projects; "Last turn" shows "All repos." `[doc]`
- **Inline comments**: hover a line → a **`+`** button appears → write feedback. `@mentions` and skill mentions work inside review comments (`26.226`). Comments are collapsible (`26.406`).
- **Git actions at three granularities**: entire diff (header **Stage all** / **Revert all**), per file, per hunk. `[doc]`
- **Mark as viewed** per file, scoped to the displayed revision. `[doc]`
- **Diff-state persistence**: "better diff batching and preserved diff and search state" (`26.417`); "preserved diff and search state" after refresh. `[doc]`
- `Cmd+F` seeds from the current text selection, "which makes searching reviews and diffs faster." (`26.318`) `[doc]`
- **Diff size cap removed** in `26.206` for large reviews. `[doc]`
- Interaction grammar, precisely specified: `[doc]`
  - Click file name → open in your editor (editor configurable: VS Code, Zed, TextMate, VS…).
  - Click file name **background** → expand/collapse the diff.
  - `Cmd`-click a single line → open that line in the editor.
  - Expand/collapse **all** diffs (shipped on iOS `1.2026.160`). `[doc]`
- Review results render **as inline comments in the review pane** (not a separate findings list). `[doc]`
- `/review` offers **Review against a base branch** or **Review uncommitted changes**; findings come back "without changing your working tree." `[doc]`
- **Detached review mode** (`26.406`; Settings > General > Code review) runs `/review` in a separate chat so the main transcript stays clean. `[doc]`

**(b) The Code Review plugin (remote pull requests).** `[doc]`
- Open from the sidebar (pinnable), connect a source account, pick a PR from the sidebar or paste a link.
- Two sub-views: **Summary** (description, activity, comments, checks, **Threads**, **Stack**) and **Changes** (files + diff + comments).
- **Personal inbox** with filters **Assigned to me** / **Assigned to my team** / **Authored by me**.
- **Pinned** section for PRs you want to return to.
- **Checks** section: `Fix` attaches failing checks to the chat, composing a message *without sending it*. `[doc]`
- **Submit review** → Comment / Approve / Request changes, sent to the provider. `[doc]`
- Review comments are **draft by default** — "Keep drafts in chat until you're ready: posting in Summary or Changes sends the comment to the source provider immediately." The docs warn about this footgun explicitly. `[doc]`
- **Review instructions** (gear next to **Review with Codex**) define criteria + reporting format across all reviews; per-review overrides go in the PR chat. `[doc]`
- PR status badges in task rows: **draft, open, merged, closed** (`26.227`). `[doc]`
- PR activity timeline + PR-page commenting + push choices in the push modal (`26.409`). `[doc]`
- PR Chat (`26.707`): "Send inline review feedback, inspect proposed patches, and edit, accept, or reject them without leaving the app."

### 2.4 Diff inspection

Covered above. Distinctive details worth stealing: `[doc]` `[observed]`
- Scope is explicit and user-chosen (Unstaged/Staged/Commit/Branch/Last turn) rather than inferred.
- Per-hunk stage/revert.
- The pane admits it shows *your* changes too — honest framing.
- Click-to-open-in-editor at three levels (file / line / selection).
- Project rows show `+n -n` inline, so diff state is visible from the sidebar without entering the pane. `[observed]`

### 2.5 Task history

- Sidebar **Recents** (implicit) + **Threads**/Projects section + **Pinned** (for projects and for PRs). `[doc]` `[observed]`
- **Archive** (`Cmd+Shift+A`) rather than delete; "Archive chats" from a project's menu archives the whole project. `26.323` added one-click archive-all for a project. Archived chats live in **Settings > Archived chats** with dates + project context, and are restorable. `[doc]`
- **Search chats** — with "expanded matching" it searches **chat content and Git branch names** (e.g. `fix/login-redirect`). `26.323` added it to the sidebar. `[doc]`
- **Find in chat** (`Cmd+F`) is per-chat only; does not search across chats. `[doc]`
- Jump-to-recent: `Cmd+Opt+1–6` (recent) and `Cmd+1–9` (positional). `[doc]`
- `26.320` fixed archive freezes — a tell that archiving was a hot spot.

### 2.6 Settings

`Cmd/Ctrl+,`; also deep-linkable per-section. Sections: `[doc]`
General (multiline via `Cmd+Enter`, **Prevent sleep while running**, **Follow-up behavior**) · **Profile** (lifetime/peak tokens, streaks, longest task, token activity charts, shareable profile cards) · **Keyboard Shortcuts** · **Notifications** · **Appearance** · **Pets** · **Browser** · **Computer Use** · **Personalization** (Friendly / Pragmatic / None + custom instructions → `AGENTS.md`) · **Suggested prompts** · **Memories** · **Archived chats**.

A parallel **Developer settings** surface holds the engineering-grade controls: `[doc]`
- **Project and terminal behavior** — where files open, **how much command output appears in chats**, where terminal tabs open.
- **Git** — branch naming, force-push policy, commit-message and PR-description prompts.
- **Code review** — Review delivery: **Inline** vs **Detached**.
- **Browser developer mode** — full CDP access.
- **Agent configuration** — inherited from `config.toml`; app/IDE/CLI share the same layers.
- **Integrations and MCP**, **Worktrees** (retention limit, worktree root).

Settings **search** spans panels including Git and pets (`26.608`). Custom instructions write straight into `AGENTS.md`. `[doc]`

### 2.7 Model picker

- Shortcut `Ctrl+Shift+M`, or `/model` in the composer. `[doc]`
- The app's model control sits **beneath the composer** and combines **model + reasoning effort in one control**. `[doc]`
- **Ultra** is a distinct mode: "goes beyond a single-agent run. It uses subagents to accelerate complex work." `[doc]`
- **Security coupling is real and worth copying:** selecting an approved model can *silently* switch the permissions control to **Approve for me** — "selecting an approved Daybreak model automatically switches the permissions control to **Approve for me**." Model selection never overrides org policy. `[doc]`
- Selecting **Full Access** with an approved security model triggers a **model-specific warning** recommending Approve-for-me instead, plus a dialog when combining Full Access with Ultra (`26.707`). `[doc]`
- `/reasoning` sets reasoning effort independently; `/fast` toggles a Fast service tier; `/personality` sets Friendly/Pragmatic. `[doc]`
- The **web** surface uses a different metaphor: a **Power** setting with **Faster** / **Smarter** / **Advanced** presets. The desktop app exposes the raw model + effort. `[doc]`

---

## 3. Interaction Patterns

### 3.1 Streaming / turn rendering

- Markdown + code blocks + Mermaid (inline diagram rendering, iOS `1.2026.195`) + LaTeX (iOS `1.2026.160`). `[doc]`
- **Loading affordance:** a "branded loading shimmer" at app start (`26.318`). `[doc]`
- **Tool activity is stylized and explicitly worked on**: "Improved tool activity styling and progress indicators" (iOS `1.2026.188`); "Improved rendering of MCP tool calls" (`26.226`); "Improved thread and tool rendering" (later release). `[doc]`
- **Floating Composer v2** (`26.320`) — the composer reflows during a turn rather than sitting inert. `[doc]`
- **Interrupted-stream UX:** `↑` with an empty composer restores the previous prompt. `[doc]`
- **Diff stats land in the composer** while a turn runs, so progress is legible without scrolling. `[doc]`
- Reasoning summaries: in the **CLI TUI**, "live reasoning summaries in the status row and completion timestamps after successful turns" (`0.155.0`). `[doc]` — *the desktop app's equivalent is not documented.*
- **Compact composer gauge** for reasoning effort shipped on iOS (`1.2026.230`). `[doc]`

### 3.2 Approval prompts / tool-permission UX

This is the app's most carefully designed system, and the most transferable part. `[doc]`

**Two independent controls, explicitly separated in the docs:** "Sandboxing and approvals are different controls that work together. The sandbox defines technical boundaries. The approval policy decides when the agent must stop and ask before crossing them."

**The composer permissions menu** offers: **Ask for approval**, **Approve for me** (auto-review), **Full access**, and named/custom permission profiles. `[doc]`

**Sandbox modes** (`sandbox_mode`): `read-only`, `workspace-write` (default low-friction), `danger-full-access`. Native enforcement per OS: Seatbelt on macOS, `bubblewrap` on Linux/WSL2, native Windows sandbox under PowerShell. `[doc]`

**Approval policies** (`approval_policy`): `on-request` (default), `never`. `untrusted` is retired. `[doc]`

**Permission profiles** (Beta): named policies combining filesystem + network rules. Built-ins: `:read-only`, `:workspace`, `:danger-full-access`. Admins can restrict selectable profiles via `allowed_permission_profiles` — "omitted profiles are denied, including omitted built-ins and profiles added in future Codex versions." `[doc]`

**Approval interaction itself:**
- `Enter` = approve, `Esc` = decline, when an approval request is open. `[doc]`
- Approval offers carry **different scopes** (e.g. once vs. for the session), and the docs instruct: "choose the narrowest scope that lets the task continue." `[doc]`
- "Don't ask again" handling for MCP approval panels (`26.325`). `[doc]`
- MCP approval choices distinguish **allowing in the current chat vs. across chats** (iOS `1.2026.160`). `[doc]`
- **Don't grant blanket access:** "Keep the project boundary as the default; use separate projects or worktrees instead of broadening access across unrelated repositories." And rules should be narrow: "Prefer precise command prefixes such as `["cargo", "test"]` over broad patterns such as `["python"]` or `["curl"]`." `[doc]` — this is unusually good guidance and a good model for surfacing in-product.
- An **in-app trust review flow for hooks** ships in `26.506`. `[doc]`

**Auto-review (reviewer-agent swap):** a separate reviewer agent evaluates eligible boundary-crossing requests and returns a rationale. It is explicitly *not* a permission grant: "It does not expand `writable_roots`, enable network access, or weaken protected paths." `[doc]`
- **Rejection circuit breaker, per turn:** interrupts after **3 consecutive denials** or **10 denials in a rolling window of the last 50 reviews**. Emits a warning and aborts with an interrupt. Any non-denial resets the consecutive counter. `[doc]`
- **Denial semantics differ from errors:** the main agent is told "Do not pursue the same outcome via workaround, indirect execution, or policy circumvention. Continue only with a materially safer alternative. Otherwise, stop and ask the user." `[doc]`
- `/approve` opens an **Auto-review Denials** picker to grant **one retry** of a specific denied action; up to 10 recent denials per task are retained. The retry **still goes through auto-review**, and the reviewer can deny again. `[doc]`
- Hidden assistant reasoning is **not** shown to the reviewer — it sees retained chat items and tool evidence only. `[doc]`
- Computer Use **app-level** approvals always surface to the human, even under auto-review. `[doc]`

### 3.3 Interrupt and steer

- **Steer-during-run is a first-class setting.** Settings > General > **Follow-up behavior**: choose whether a message sent while the agent works should **steer the current run** or **wait for the next run**. `[doc]` This is a genuinely important knob that most agent UIs bury.
- **Side chat** (`/side`, `Cmd+Ctrl+S`) — ask a question without interrupting the main chat. `[doc]`
- **Goal progress row** above the composer: pause / resume / edit goal / clear goal, plus follow-up messages to steer. `[doc]`
- **Interrupt the turn**: a circuit-breaker abort is a first-class, warned outcome (see auto-review above). The CLI has opt-in `instant_interrupt` for steering during model responses (`0.159.0`). `[doc]`
- **Stop a running review** from the review chat. `[doc]`
- **Stop all active subagents** from the subagent panel. `[doc]`
- **Review queue**: "When an Automation finishes, the results land in a review queue so you can jump back in and continue working if needed." `[doc]`
- `26.707`: "Improved permission handling when resuming tasks or sending follow-ups."

### 3.4 Background / queued work

- **Worktrees are the parallelism primitive.** "Think of Local as the foreground and Worktree as the background." `[doc]`
- **Handoff** moves a chat *and its code* between Local and Worktree, handling the Git operations safely (Git forbids one branch checked out in two worktrees). Each chat keeps its **same associated worktree** over time. `[doc]`
- Managed worktrees are **detached HEAD** by default, created under `$CODEX_HOME/worktrees`, so parallel chats don't pollute branches. `[doc]`
- **Retention:** keeps the most recent **15** Codex-managed worktrees by default (configurable, or disable auto-delete). Never auto-deletes a worktree tied to a **pinned** chat, an **in-progress** chat, or a **permanent** worktree. Auto-deletes on chat archive or when over the limit. **Before deleting, it saves a snapshot** and offers restore when you reopen the chat. `[doc]` — this is a thoughtful design: destructive GC with a recovery path.
- `.worktreeinclude` copies gitignored setup files (`.env` etc.) into new worktrees; source symlinks are skipped and existing files are never overwritten. `AGENTS.override.md` is copied automatically. `[doc]`
- **Scheduled tasks** run locally in the project dir or on **dedicated background worktrees** for Git repos so they don't conflict. Since `26.312` you can choose local vs. worktree execution, set custom reasoning levels and models, and start from templates. `[doc]`
- **Thread automations** wake the same thread on a schedule, preserving conversation context. `[doc]`
- **Background monitoring:** **Prevent sleep while running** (Settings > General); Pets or system notifications to signal a chat needs input or is ready for review. `[doc]`
- `26.608`: tray usage-limit surfacing; `26.325` Windows system tray menu so Codex stays resident after the last window closes.

### 3.5 Notifications

- Desktop turn-completion alerts: **never / only while in background / always**, plus separate toggles for **permission** and **question** notifications. OS-level permission prompt. `[doc]`
- **Activity view** (bell in sidebar) lists chats that are **unread, running, or waiting for your response**; filterable to **Work / Chat / Pinned / Scheduled**; "Mark all as read." `[doc]`
- **Next chat needing attention**: `Cmd/Ctrl+Opt+A` — a single keystroke to jump to the thread that is actually blocked on you. `[doc]`
- Unread state: `Shift+Esc` clears **all** unread indicators; `Cmd+Shift+U` marks a chat unread; unread state is **preserved across windows** and reconnects. `[doc]`
- **Pets** as an ambient status channel: a floating companion shows **Running / Needs input / Ready / Blocked** — i.e. the four-state model is the pet's own state machine. `[doc]`
- CLI `notify` config runs an external program on turn completion. `[doc]`

### 3.6 Keyboard shortcuts and command palette

`Cmd/Ctrl+Shift+P` or `Cmd/Ctrl+K` opens the **command menu**. `[doc]`

**Fully remappable, with a keystroke-search mode.** Settings > Keyboard Shortcuts: "search by command name or switch the search field into keystroke mode and press the shortcut you want to find," plus **reset to defaults**. `[doc]` — the inverse-keystroke search is a nice touch.

Complete documented shortcut inventory (macOS / Windows), condensed: `[doc]`

| Group | Shortcut | Action |
|---|---|---|
| **Command menu** | `⌘⇧P` / `⌘K` | Open command menu |
| **Settings** | `⌘,` | Open settings |
| **Shortcut help** | `⌘/` | Open keyboard shortcuts |
| **Open folder** | `⌘O` | Add/select project |
| **Sidebar** | `⌘B` | Toggle sidebar |
| **Bottom panel** | `⌘J` | Toggle bottom panel (Codex) |
| **Terminal** | `` ⌃` `` | Toggle integrated terminal |
| **Clear terminal** | `⌃L` / `⌘K` | Clear (when focused) |
| **Clear unread** | `⇧Esc` | Clear all unread indicators |
| **Font size** | `⌘+` / `⌘-` / `⌘0` | Increase / decrease / reset |
| **New chat** | `⌘N` / `⌘⇧O` | New chat |
| **Standalone chat** | `⌘⌥O` | New standalone chat (Codex) |
| **Quick chat** | `⌘⌥N` | Quick chat (ChatGPT) |
| **Archive** | `⌘⇧A` | Archive chat |
| **Unread** | `⌘⇧U` | Mark chat unread |
| **Pin** | `⌘⌥P` | Pin/unpin chat |
| **Rename** | `⌘⌥R` | Rename chat |
| **Side chat** | `⌘⌥S` | Open side chat (Codex) |
| **Find in chat** | `⌘F` / `⌘G` / `⌘⇧G` | Find / next / previous |
| **Prev/next chat** | `⌃⇧Tab` / `⌃Tab` | Previous / next chat or tab |
| **Needs attention** | `⌘⌥A` | Next chat needing attention |
| **Recent chat 1–6** | `⌘⌥1–6` | Open recent chat |
| **Go to chat 1–9** | `⌘1–9` | Positional jump |
| **Model picker** | `⌃⇧M` | Open model picker |
| **Project picker** | `⌘⌥⇧O` | Open project picker |
| **Voice / dictation** | `⌃⇧V` / `⌃⇧D` | Start voice / dictation |
| **Restore prompt** | `↑` | Restore previous prompt (empty composer) |
| **Approve / decline** | `⏎` / `Esc` | Approve / decline open request |
| **Surface switch** | `⌃1` / `⌃2` / `⌃3` | Chat / Work / Codex |
| **Activity view** | `⌘⌥U` | Toggle Activity view |
| **File search** | `⌘P` | Search files (Codex) |
| **File tree** | `⌘⇧E` | Toggle file tree |
| **Review tab** | `⌃⇧G` | Open review tab |
| **Layout** | `⌘⌥B` | Switch chat ↔ tabs |
| **Layout cycle** | `⌘⇧B` | Full / split / hidden tabs |
| **Full view** | `⌘⇧F` | Enter/exit full view |
| **Browser tab** | `⌘T` | Open browser tab |
| **Browse/comment** | `⌘.` | Toggle browser mode |
| **Env action** | `⌘⇧D` | Run environment action 1 |
| **Appshot** | `⌘⌘` (both keys) | Capture frontmost app window |
| **Copy cwd** | `⌘⇧C` | Copy working directory |
| **Copy thread path** | `⌘⌥⇧C` | Copy conversation path |
| **Copy session ID** | `⌘⌥C` | Copy session ID |
| **Deep link** | `⌘⌥L` | Copy chat deep link |

Notable: **"Search chats" has no default shortcut** — users must assign one. Given search now matches content and branch names, that's an oversight. `[doc]`
A `codex://` URL scheme is retained for deep links, documented per-surface: `codex://threads/<id>`, `codex://new?prompt=&path=&originUrl=`, `codex://settings/connections/ssh/add?name=`, `codex://skills`, `codex://automations`, `codex://plugins/install/...`, `codex://pets/install?...`. `[doc]`

### 3.7 Slash commands (composer)

`/` in the composer. Skills are invoked with `$`, files/context with `@`. Enabled skills also appear in the slash list; custom prompts appear as `/prompts:<name>`. `[doc]`

`/approve` · `/cloud` · `/cloud-environment` · `/compact` · `/fast` · `/feedback` · `/fork` · `/goal` · `/ide-context` · `/init` · `/local` · `/mcp` · `/memories` · `/model` · `/pet` · `/personality` · `/plan` · `/project` · `/reasoning` · `/review` · `/side` · `/status` · `/task` · `/worktree` `[doc]`

`/status` is the diagnostics command: "Show the chat ID, context usage, and rate limits" (app); CLI `/status` shows "the active model, approval policy, writable roots, and token usage." `[doc]`

Note the design tension visible in the command list: `/local` and `/cloud` both exist alongside a Local/Cloud toggle, and `/cloud-environment` is a third, separate concern. The model is drifting. `[doc]`

---

## 4. Run / Agent State Model

### 4.1 The canonical four states

The **Pets** doc states the app's own status vocabulary for a chat: `[doc]`

> **Running** · **Needs input** · **Ready** · **Blocked**

The **Notifications** doc uses a parallel framing: chats that are "**unread, running, or waiting for your response**." `[doc]`

The iOS changelog confirms these are the real indicator set: "Improved status indicators for **running threads, queued prompts, side chats, and subagents**." `[doc]`

**Reconciled state model (as surfaced in the UI):** `[observed]` `[doc]`

| State | Sidebar glyph | Other surface |
|---|---|---|
| **Running** | solid blue filled circle | pet status "Running"; running filter in Activity view |
| **Needs input / Blocked** | **green pill "Awaiting approval"** | pet "Needs input" / "Blocked"; Activity "waiting for your response"; `⌘⌥A` jumps here |
| **Ready** | plain row, no badge | pet "Ready"; "results land in a review queue" for automations |
| **Queued** | (no documented desktop glyph) | documented on iOS as a status indicator for "queued prompts" |
| **Failed / errored** | **not surfaced as a first-class state** | only via activity text, `26.325`'s "retry failed tasks" (web, 2025) |
| **Paused (goal)** | goal progress row | pause/resume buttons above composer |
| **Subagent** | surfaced as its own thread | subagent panel above composer, stoppable |

### 4.2 Per-thread metadata rows carry

`[observed]`
- status glyph (leading)
- title, truncated with `…`
- status pill where applicable
- expand chevron
- **right-aligned dimmed relative timestamp** (`5m`, `7m`)
- for project rows: **inline `+n -n` diff stats** in green/red

### 4.3 Elapsed time

**Weak spot.** The documented status vocabulary contains no elapsed-time concept, and I observed no elapsed timer in the sidebar. Elapsed time appears only in the **CLI TUI**, which shows "live reasoning summaries in the status row and **completion timestamps** after successful turns." `[doc]` For long-running goals ("hours or even days") the desktop app has no documented elapsed indicator. **Design gap worth fixing in a rebuild.** `[doc]`

### 4.4 Token / cost

- **Profile section**: "activity insights, **lifetime tokens, peak tokens, streaks, your longest task, and token activity**" with token activity charts and shareable profile cards. `[doc]`
- **`/status`** in the composer: chat ID, **context usage**, rate limits. `[doc]`
- **Trays and badges surface usage limits** (Windows system tray `26.401`, "tray usage-limit surfacing" `26.417`). `[doc]`
- **Context window pressure is bounded in skills**: the skill list is capped at **2% of the model's context window, or 8,000 characters** when unknown. `[doc]`
- **Per-turn token cost is not surfaced in the transcript.** Docs describe `/compact` for context management and `/status` for totals, but there is no per-turn cost readout in the desktop run view. **Design gap.** `[doc]`

### 4.5 Per-step progress

Progress is **not** a timeline/stepper. There is no milestone or step model in the desktop app. What exists instead: `[doc]`

- **Plan mode** (`/plan`) — a pre-execution plan the agent proposes before editing.
- **Goal progress row** — pause/resume/edit/clear above the composer (the closest thing to a progress affordance).
- **Subagent threads** — each surfaced and inspectable, expandable to see status, collectively stoppable.
- **Diff stats in the composer** while a turn runs.
- **Thread automations** surface run results into a **review queue**.
- **Task sidebar** (during a run) "can surface the agent's **plan, sources, generated files, and chat summary**." `[doc]`
- On iOS, "context-aware suggestions … surface follow-ups and tasks you may want to resume." `[doc]`

**This is the single biggest structural difference from a Mission-Control-style UI.** Codex's progress model is *artifact-centric* (what changed, what's blocked) rather than *step-centric* (step 3 of 7). It is honest and low-noise, but it gives an operator no forward visibility into a long run's shape. `26.707` partially addressed this: "Made task and subagent activity easier to follow while Codex works." `[doc]`

---

## 5. Visual Design Language

### 5.1 Theming system (a genuinely strong feature)

Since `26.312` (2026-03-12), Appearance is fully user-definable. The documented control surface, read directly from the Settings page: `[doc]`

- **Base theme**: Light / Dark / Match system
- **Colors**: Accent, Background, Foreground
- **Fonts**: UI font, Code font — with a **separate code font size** ("base size used for code across chats and diffs")
- **UI font size** and **Code font size** sliders
- **Translucent sidebar** toggle
- **Contrast** slider
- **Use pointer cursors** toggle ("Change the cursor to a pointer when hovering over interactive elements")
- **Import / Copy theme** — themes are shareable as text (`1 const themePreview : ThemeConfig = { surface: "sidebar", accent: "#2563eb", contrast: 42 };`)

The theme config is a small, legible JSON-ish object: `surface` (`sidebar` | `sidebar-elevated`), `accent`, `contrast` (numeric).

**Built-in themes documented with exact tokens:** `[doc]`

| Theme | Accent | Background | Foreground | Code font | Contrast |
|---|---|---|---|---|---|
| **Codex (light)** | `#0285FF` | `#FFFFFF` | `#0D0D0D` | `ui-monospace, "SFMono…` | 45 |
| **Codex (dark)** | `#339CFF` | `#181818` | `#FFFFFF` | `ui-monospace, "SFMono…` | 60 |
| **Catppuccin (light)** | `#8839EF` | `#EFF1F5` | `#4C4F69` | — | 45 |
| **Dracula (dark)** | `#FF79C6` | `#282A36` | `#F8F8F2` | — | 60 |

Default UI font stack: `-apple-system, BlinkMac…` (system UI font); code font `ui-monospace, "SFMono…`. Dracula's UI font is **Inter**. `[doc]`
Default base sizes: **UI 14px, code 14px**. `[doc]`
Bundled **Raycast themes** shipped `26.324`. `[doc]`
A **command-palette theme switcher** shipped `26.417`. `[doc]`

### 5.2 Measured palette (official product screenshots)

Dark capture, `codex-windows-dark.webp`: `[observed]`

| Token | Measured | Matches docs? |
|---|---|---|
| Main pane background | `#181818` (42% of light-mode analogue; 1.54% of pixels here) | **exactly** the documented dark Background |
| Sidebar base | `#1A1F39` (bluish-tinted, translucent) | consistent with the "Translucent sidebar" toggle + `surface: "sidebar"` |
| Composer card | `#242424` | one elevation step above `#181818` |
| Selected row fill | `#1F1F2D` over `#1A1F39` | ~5% lightness lift — very low contrast |
| Pane divider | `#3B3D44` band, ~3px | neutral grey, no accent |
| Body text | `#FFFFFF` primary; `#8E8E8E` / `#6F6F6F` secondary; `#C2C2C2` tertiary | 3-step text ramp |
| Wallpaper bleed-through | `#4E45A0`→`#6A60D8` gradient, `#8976EB` | the window is **translucent**; the desktop wallpaper shows through |

Light capture, `codex-windows-light.webp`: `[observed]`

| Token | Measured |
|---|---|
| Main pane background | `#FFFFFF` (45.8% of pixels) |
| Sidebar base | `#F3F3F3` |
| Pane divider | `#8E96B6` |
| Accent | `#684FE9` (brand purple in the chrome/wallpaper) |
| Selection tint | `#DAE2F7` (4.25% — a cool blue-grey, the dominant saturated hue) |

**Diff colors (from the light capture's project rows):** additions **green** — `#C3E1CD` background, `#65AF69` text; deletions **red/salmon** — `#E5C8CA` background, `#DC7971` text. `[observed]`
In the dark capture the equivalents are deeply desaturated: `#1E331F` / `#3A7B3D` green, `#351A18` / `#892F28` red — i.e. **low-saturation, low-luminance diff tints**, not saturated GitHub-style pastels. `[observed]`

The **hero marketing shot** shows a saturated indigo/violet brand gradient (`#222944` over `#111111`) — a **different, more decorative** palette from the shipping app. Do not treat marketing shots as the product's visual language. `[observed]`

### 5.3 Spacing and density

Measured from the dark capture (1× scale): `[observed]`
- **Body line pitch: 36px** (13px glyph band + 23px gap), extremely consistent across ~25 measured lines.
- **Composer: 724 × 70px** collapsed, floating, **0.67 × main-column width**, **radius ≈ 6–8px**.
- **Sidebar: 556px**, i.e. **33.9%** of window width — a wide sidebar. This is a direct consequence of putting Recents *and* the Projects tree in one column without a sub-tab.
- **Pane divider: ~3px**, neutral `#3B3D44`.

### 5.4 Borders vs. shadows

**Borders/dividers, essentially no shadows.** `[observed]`
- Pane separation is by a thin neutral vertical rule, not shadow.
- The composer is a flat fill one step lighter than the page. **No drop shadow detected.**
- Selection is a flat, very-low-contrast fill, **not** a left accent bar and not an outline.
- This is a deliberate, coherent "native desktop app" register rather than a "floating web card" register.

### 5.5 Corner radius

Measured ~**6–8px** on the composer card. The sidebar's selected-row fill probed to a similar small radius (inset 7 → 0 over ~7px). Small, consistent radii throughout — closer to macOS system UI than to a consumer chat bubble. `[observed]`

### 5.6 Distinctive motifs

1. **The floating centered composer** — narrow (67% of column), elevated by a single fill step, carrying the entire run configuration beneath the input. The single most identifiable visual/IA signature. `[observed]`
2. **Inline diff stats in the sidebar** (`+1 -9`, green/red) — diff state visible from the navigation rail. `[observed]`
3. **Status pills and glyphs in the recents list** — green "Awaiting approval" pill, blue filled dot, sparkle for agent-derived. `[observed]`
4. **Translucency as a product feature** — the desktop wallpaper bleeds through the window; a documented toggle. `[doc]` `[observed]`
5. **Shareable themes as plain text** with a `const themePreview : ThemeConfig = {…}` snippet. A clever, developer-native theme-sharing format. `[doc]`
6. **The "surface" model** — Chat / Work / Codex as switchable modes of one app (`Ctrl+1/2/3`), each with its own sidebar contents. `[doc]`
7. **Annotation-driven refinement** — the same select-and-annotate gesture works on code, Markdown, rendered websites, PDFs, spreadsheets, and slides. A strong unifying interaction. `[doc]`
8. **Pets** — animated companion floating over the app with a four-state status readout. `[doc]`

---

## 6. Strengths and Weaknesses

### 6.1 Strengths (specific)

1. **The review pane is the best part of the product.** Explicit scope selector (Unstaged/Staged/Commit/Branch/Last turn), per-hunk stage/revert, three levels of click-to-open-in-editor, "Mark as viewed" scoped to a revision, diff/search state preserved across refresh, and — critically — an explicit statement that the pane shows *the whole repo state, not just what the agent edited*. Honesty about scope is rare and valuable. `[doc]`
2. **Inline review comments as the primary feedback channel.** Hover → `+` → comment on a line, then a follow-up message. Docs note "Because comments are line-specific, Codex can respond more precisely than with a general instruction," and instruct users to follow up explicitly so intent is unambiguous. This is a better feedback loop than a separate findings panel. `[doc]`
3. **Sandbox and approval are cleanly separated concepts**, with the boundary-vs-when-to-ask distinction made explicit in the docs and mirrored in the UI (permissions menu under the composer; sandbox decided by project). `[doc]`
4. **Auto-review is a reviewer swap, not a permission grant** — and the docs say so in those words, with a documented circuit breaker (3 consecutive / 10-of-50), distinct denial semantics ("do not pursue the same outcome via workaround"), timeouts treated separately from denials, one-retry override that still re-enters review, and private reasoning excluded from the reviewer's input. This is unusually rigorous. `[doc]`
5. **Steer-vs-queue as an explicit user setting** (Follow-up behavior). Most agent UIs pick one silently. `[doc]`
6. **Side chats** (`/side`) decouple curiosity from interruption. `[doc]`
7. **Worktree lifecycle is production-grade**: detached HEAD by default, 15-worktree default retention, no auto-delete for pinned/in-progress/permanent worktrees, **snapshot-before-delete with restore**. `[doc]`
8. **Handoff** moves chat *and* code between Local and Worktree, respecting Git's one-branch-per-worktree constraint, and each chat returns to its same worktree. `[doc]`
9. **Full keyboard remapping with inverse keystroke search** and a reset-to-defaults. `[doc]`
10. **Theming is unusually deep and shareable** — accent/background/foreground, separate UI and code fonts *and* separate code font size, contrast, translucency, and import/export-as-text. `[doc]`
11. **`/status` as a first-class diagnostic** (chat ID, context usage, rate limits). `[doc]`
12. **Command menu on `⌘K`/`⌘⇧P` with a documented `codex://` deep-link scheme**, so external tooling can drive navigation. `[doc]`
13. **Slash commands span the whole lifecycle** — `/plan` → `/goal` → `/fork` → `/worktree` → `/review` → `/compact`. That's a real workflow vocabulary, not a feature grab-bag. `[doc]`
14. **Layout state is preserved obsessively** — per-conversation scroll position, unread state across windows, tab widths and scroll positions when closing browser tabs, diff and search state across refresh, in-progress message edits across thread switches, unfinished comments on response text across chat switches. Each of these was a separate fix in the changelog, which tells you they're real pain points being systematically retired. `[doc]`
15. **Security UX is honest about limits**: Computer Use runs in the *foreground* on Windows ("expect ChatGPT to move the pointer, type, and take over"), and the browser profile is separate from your real browser session. `[doc]`

### 6.2 Weaknesses (specific)

1. **No unified run-state board.** There is no place that lists all runs with status, elapsed time, and cost. State is fragmented across sidebar glyphs, the Activity bell, the goal row, and OS notifications. For the app's own stated purpose — "a command center for agents," "supervising coordinated teams of agents" — this is the central IA failure. The Activity view is the closest thing and it is a filter, not a dashboard. `[doc]` `[observed]`
2. **No step/milestone timeline.** Progress is artifact-centric (what changed, what's blocked), never step-centric. For a goal that "can run for hours or even days," there is no forward visibility into shape, no step 3-of-7, no ETA. `26.707`'s "task and subagent activity easier to follow" concedes the gap. `[doc]`
3. **No elapsed time and no per-turn token cost in the run view.** Profile gives lifetime/peak totals; `/status` gives context usage; neither is per-turn, and the transcript shows neither. `[doc]`
4. **The sidebar is 34% of window width and overloaded.** Recents + fixed nav (New chat, Search, Plugins, Automations, Codex mobile) + Projects all compete. Users are formally asking for this to be fixed: GitHub issue **#27042** requests collapsing the Recents section, capping it at 1–2 rows before "Show more," making "Show more" a compact dropdown, an icon-only nav mode, and the ability to hide low-frequency nav items. `[doc]` `[observed]` `[secondary — issue text, not a shipped response]`
5. **No unified "failed" state.** Errors surface as text inside the transcript. There is no red row, no failed badge, no retry affordance in the desktop sidebar. The "retry failed tasks" button exists only on the **web** environment page (2025). `[doc]` `[observed]`
6. **Surface feature-flag instability is user-visible and damaging.** GitHub issue **#19747**: Automations and Codex Mobile disappeared from the sidebar after a model update, reappeared days later, and vanished again; the "Add files and more /" menu stopped exposing expected actions; language switching didn't apply. Same platform, same account class, different results across accounts in issue **#18774** ("Another account on Codex Desktop show…"). Capability is being gated by server-side flags the UI doesn't represent, so navigation items appear and vanish. `[doc]`
7. **Rendering regressions on the secondary platform.** Issue **#18774** (macOS Intel): clipped sidebar/workspace selector, Plugins page rendering with large blank areas and skeletal cards, and a malformed slash-command overlay with clipped side edges. Issue **#22801** (`26.513.20950`): sidebar text became noticeably bolder after an update — suspected font fallback — plus a Desktop Pet that grew with no size control. Two separate releases, both shipping unintended typography changes. `[doc]`
8. **Long-conversation and diff scrolling were repeatedly unstable.** Fixes for "thread jumpiness, **sidebar jitter**, and diff scrolling" (`26.227`), a dedicated pass on the same (`26.304`), "thread jumpiness" again, "right-panel reset" (`26.406`), diff whitespace handling (`26.410`), and scrolling in PR reviews (`26.707`). **Sidebar jitter in a developer tool is a serious defect** — the nav rail must be rock-steady. `[doc]`
9. **"Search chats" ships with no default keyboard shortcut** despite now matching full chat content and Git branch names. Users must bind one themselves in Settings. `[doc]`
10. **Command surface is fragmenting.** `/local`, `/cloud`, and `/cloud-environment` are three overlapping execution-target commands, alongside a Local/Cloud toggle in the UI and a Local↔Worktree **Handoff** action and a `/worktree` command. Four overlapping mechanisms for "where does this run." `/task` vs. `New chat` beside Recents vs. `/new` is a similar three-way overlap for "start a chat." `[doc]`
11. **Approval scope choices can encourage over-granting.** The docs warn to pick the narrowest scope and to prefer `["cargo","test"]` over `["python"]` — which implies the UI does offer broad, tempting choices ("Don't ask again", "for the session", Full access). Good guidance in docs does not fix a permissive default in the prompt. `[doc]`
12. **"Ask for approval" is the stated Windows path to get sandbox protections** — "To apply sandbox protections in either mode, select **Ask for approval** beneath the composer before sending messages." Coupling *safety* to an *approval verb* is a confusing mental model, since `never` + `workspace-write` is the safer configuration. `[doc]`
13. **Full Access with Ultra shows a dialog, but Full Access alone shows only a warning** with a link to configure reviewer policy. The safer option is recommended, not enforced, even though `Full access` is a single click away and `26.707` had to *add* the warning. `[doc]`
14. **Window chrome regressed repeatedly.** `26.304` shipped "startup reliability and **keyboard zoom** behavior" fixes; `26.417` added **per-window zoom**; `26.727` added "keep tab widths and scroll positions more stable as you close browser tabs." A desktop app that cannot reliably keep its own layout stable undermines the whole "command center" premise. `[doc]`
15. **Product identity churn is visible in the docs themselves.** The same app is "Codex app," "ChatGPT desktop app," and "ChatGPT Learn"; several doc pages still carry stale titles ("Settings | ChatGPT Learn" under a ChatGPT desktop app heading) and the app now has Pets, Dots, Sites, Space, and Codex Micro alongside Codex. The IA burden of "what is this app" is real, and surface sprawl is leaking into the command set. `[doc]`

---

## 7. Transferable Patterns for a Rebuild

Highest-value takeaways, in priority order:

1. **Put the whole run configuration beneath the composer** — permissions, model+effort, execution environment, starting branch. One surface, always in the same place, above the send button.
2. **Make the review scope explicit and user-chosen**, and state plainly that the diff includes the user's own edits.
3. **Per-hunk staging + three levels of click-to-open-in-editor** (file name / file background / `Cmd`-click line).
4. **Line-anchored feedback as the primary review channel**, followed by an explicit follow-up message.
5. **Ship steer-vs-queue as a setting, not a hidden default.**
6. **Surface diff stats in the navigation rail** (`+n -n`) so state is visible without entering the review pane.
7. **Separate "what the agent may do" (sandbox) from "when it must ask" (approval policy)" in both the model and the UI copy.**
8. **If you auto-approve with a reviewer agent, add a circuit breaker, treat denials as non-errors, and keep a one-retry override that still re-enters review.**
9. **Make destructive GC recoverable** — snapshot before deleting a worktree, offer restore.
10. **Shareable themes as text**, with separate UI/code fonts and a separate code font size.
11. **Add what Codex lacks:** a unified run board, a step timeline, elapsed time, and per-turn token cost. These are the four highest-value additions a Mission-Control-style UI can make over Codex's model.

## 8. Evidence Gaps

- **I could not obtain a clean, unobstructed screenshot of the review/diff pane, the approval prompt, the model picker, or the Activity view.** The official screenshots I obtained show the sidebar and a chat-plus-composer view; the docs' "Illustrations" are referenced as image assets I could not resolve to files in the page HTML. All statements about those surfaces in this report are `[doc]`-sourced (they come from official written documentation, which is unusually detailed) and **not** visually verified. Diff *colors* are `[observed]` but from the **sidebar's inline `+n -n` stats**, not from the diff pane itself.
- **Typographic specifics beyond the two font stacks and the 14px/14px base sizes are not documented** — no type scale, no line-height spec, no weight ramp. The 36px measured line pitch is from one capture at one window size and is not a spec.
- **Shadow usage is inferred from absence** in a flat-fill UI at 1× scale; sub-pixel shadows would not be detectable in my analysis. Claim is "no visible drop shadow at this scale," not "no shadows."
- **Third-party design write-ups were not usable.** Searches for critiques returned mostly low-substance SEO summaries of the launch blog; the substantive criticism in §6.2 items 4–7 comes from **GitHub issues in `openai/codex`**, which are primary user reports but not official documentation — flagged accordingly.
- **ChatGPT Work / Dots / Sites / Space** were out of scope; they share the shell described in §1 but have their own panes and are not covered.
