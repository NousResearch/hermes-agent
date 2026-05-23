# Hermes Agent Self-Knowledge

## Identity

Hermes Agent is a tool-using AI assistant and automation runtime created by Nous Research. It operates through CLI, TUI, API, scheduler, gateway platforms, skills, tools, and plugins.

This document is the repo-grounded self-inventory for Hermes. Hand-written sections explain identity and operating principles; AUTO sections are regenerated from code so Hermes does not rely on stale prose when describing its own capabilities.

## Operating Principles

- Treat code and configured registries as the source of truth.
- Keep generated inventory local, deterministic, and free of network calls.
- Never include secret values, tokens, API keys, or private user data in this document.
- Prefer compact summaries over exhaustive dumps.
- Preserve hand-written context when refreshing generated sections.

## Capabilities at a Glance

<!-- AUTO-START: capabilities -->
| Tool | Toolset | Description |
|---|---|---|
| browser_back | browser | Navigate back to the previous page in browser history. Requires browser_navigate to be called first. |
| browser_click | browser | Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first. |
| browser_console | browser | Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requires browser_navigate to be called first. When 'expression' is provided, evaluates JavaScript in the page context and returns the result — use this for DOM inspection, reading page state, or extracting data programmatically. |
| browser_get_images | browser | Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first. |
| browser_navigate | browser | Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer curl via the terminal tool or web_extract; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating. |
| browser_press | browser | Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first. |
| browser_scroll | browser | Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first. |
| browser_snapshot | browser | Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 8000 chars are truncated or LLM-summarized. Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot — use this to refresh after interactions that change the page, or with full=true for complete content. |
| browser_type | browser | Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first. |
| browser_vision | browser | Take a screenshot of the current page and analyze it with vision AI. Use this when you need to visually understand what's on the page - especially useful for CAPTCHAs, visual verification challenges, complex layouts, or when the text snapshot doesn't capture important visual information. Returns both the AI analysis and a screenshot_path that you can share with the user by including MEDIA:<screenshot_path> in your response. Requires browser_navigate to be called first. |
| browser_cdp | browser-cdp | Send a raw Chrome DevTools Protocol (CDP) command. Escape hatch for browser operations not covered by browser_navigate, browser_click, browser_console, etc. **Requires a reachable CDP endpoint.** Available when the user has run '/browser connect' to attach to a running Chrome, or when 'browser.cdp_url' is set in config.yaml. Not currently wired up for cloud backends (Browserbase, Browser Use, Firecrawl) — those expose CDP per session but live-session routing is a follow-up. Camofox is REST-only and will never support CDP. If the tool is in your toolset at all, a CDP endpoint is already reachable. **CDP method reference:** https://chromedevtools.github.io/devtools-protocol/ — use web_extract on a method's URL (e.g. '/tot/Page/#method-handleJavaScriptDialog') to look up parameters and return shape. **Common patterns:** - List tabs: method='Target.getTargets', params={} - Handle a native JS dialog: method='Page.handleJavaScriptDialog', params={'accept': true, 'promptText': ''}, target_id=<tabId> - Get all cookies: method='Network.getAllCookies', params={} - Eval in a specific tab: method='Runtime.evaluate', params={'expression': '...', 'returnByValue': true}, target_id=<tabId> - Set viewport for a tab: method='Emulation.setDeviceMetricsOverride', params={'width': 1280, 'height': 720, 'deviceScaleFactor': 1, 'mobile': false}, target_id=<tabId> **Usage rules:** - Browser-level methods (Target.*, Browser.*, Storage.*): omit target_id and frame_id. - Page-level methods (Page.*, Runtime.*, DOM.*, Emulation.*, Network.* scoped to a tab): pass target_id from Target.getTargets. - **Cross-origin iframe scope** (Runtime.evaluate inside an OOPIF, Page.* targeting a frame target, etc.): pass frame_id from the browser_snapshot frame_tree output. This routes through the CDP supervisor's live connection — the only reliable way on Browserbase where stateless CDP calls hit signed-URL expiry. - Each stateless call (without frame_id) is independent — sessions and event subscriptions do not persist between calls. For stateful workflows, prefer the dedicated browser tools or use frame_id routing. |
| browser_dialog | browser-cdp | Respond to a native JavaScript dialog (alert / confirm / prompt / beforeunload) that is currently blocking the page. **Workflow:** call ``browser_snapshot`` first — if a dialog is open, it appears in the ``pending_dialogs`` field with ``id``, ``type``, and ``message``. Then call this tool with ``action='accept'`` or ``action='dismiss'``. **Prompt dialogs:** pass ``prompt_text`` to supply the response string. Ignored for alert/confirm/beforeunload. **Multiple dialogs:** if more than one dialog is queued (rare — happens when a second dialog fires while the first is still open), pass ``dialog_id`` from the snapshot to disambiguate. **Availability:** only present when a CDP-capable backend is attached — Browserbase sessions, local Chrome via ``/browser connect``, or ``browser.cdp_url`` in config.yaml. Not available on Camofox (REST-only) or the default Playwright local browser (CDP port is hidden). |
| clarify | clarify | Ask the user a question when you need clarification, feedback, or a decision before proceeding. Supports two modes: 1. **Multiple choice** — provide up to 4 choices. The user picks one or types their own answer via a 5th 'Other' option. 2. **Open-ended** — omit choices entirely. The user types a free-form response. Use this tool when: - The task is ambiguous and you need the user to choose an approach - You want post-task feedback ('How did that work out?') - You want to offer to save a skill or update memory - A decision has meaningful trade-offs the user should weigh in on Do NOT use this tool for simple yes/no confirmation of dangerous commands (the terminal tool handles that). Prefer making a reasonable default choice yourself when the decision is low-stakes. |
| execute_code | code_execution | Run a Python script that can call Hermes tools programmatically. Use this when you need 3+ tool calls with processing logic between them, need to filter/reduce large tool outputs before they enter your context, need conditional branching (if X then Y else Z), or need to loop (fetch N pages, process N files, retry on failure). Use normal tool calls instead when: single tool call with no processing, you need to see the full result and apply complex reasoning, or the task requires interactive user input. Available via `from hermes_tools import ...`: web_search(query: str, limit: int = 5) -> dict Returns {"data": {"web": [{"url", "title", "description"}, ...]}} web_extract(urls: list[str]) -> dict Returns {"results": [{"url", "title", "content", "error"}, ...]} where content is markdown read_file(path: str, offset: int = 1, limit: int = 500) -> dict Lines are 1-indexed. Returns {"content": "...", "total_lines": N} write_file(path: str, content: str) -> dict Always overwrites the entire file. search_files(pattern: str, target="content", path=".", file_glob=None, limit=50) -> dict target: "content" (search inside files) or "files" (find files by name). Returns {"matches": [...]} patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict Replaces old_string with new_string in the file. terminal(command: str, timeout=None, workdir=None) -> dict Foreground only (no background/pty). Returns {"output": "...", "exit_code": N} Limits: 5-minute timeout, 50KB stdout cap, max 50 tool calls per script. terminal() is foreground-only (no background or pty). Scripts run in the session's working directory with the active venv's python, so project deps (pandas, etc.) and relative paths work like in terminal(). Print your final result to stdout. Use Python stdlib (json, re, math, csv, datetime, collections, etc.) for processing between tool calls. Also available (no import needed — built into hermes_tools): json_parse(text: str) — json.loads with strict=False; use for terminal() output with control chars shell_quote(s: str) — shlex.quote(); use when interpolating dynamic strings into shell commands retry(fn, max_attempts=3, delay=2) — retry with exponential backoff for transient failures |
| computer_use | computer_use | Universal macOS desktop control via cua-driver. Works with any tool-capable model (Anthropic, OpenAI, OpenRouter, local vLLM, etc.). Background computer-use: does NOT steal the user's cursor or keyboard focus. |
| cronjob | cronjob | Manage scheduled cron jobs with a single compressed tool. Use action='create' to schedule a new job from a prompt or one or more skills. Use action='list' to inspect jobs. Use action='update', 'pause', 'resume', 'remove', or 'run' to manage an existing job. To stop a job the user no longer wants: first action='list' to find the job_id, then action='remove' with that job_id. Never guess job IDs — always list first. Jobs run in a fresh session with no current-chat context, so prompts must be self-contained. If skills are provided on create, the future cron run loads those skills in order, then follows the prompt as the task instruction. On update, passing skills=[] clears attached skills. NOTE: The agent's final response is auto-delivered to the target. Put the primary user-facing content in the final response. Cron jobs run autonomously with no user present — they cannot ask questions or request clarification. Important safety rule: cron-run sessions should not recursively schedule more cron jobs. |
| delegate_task | delegation | Spawn one or more subagents in isolated contexts. Description is rebuilt at every get_definitions() call to reflect the user's current delegation limits. |
| discord | discord | Read and participate in a Discord server. Available actions: search_members(guild_id, query) — find members by name prefix fetch_messages(channel_id) — recent messages; optional before/after snowflakes create_thread(channel_id, name) — create a public thread; optional message_id anchor Use the channel_id from the current conversation context. Use search_members to look up user IDs by name prefix. |
| discord_admin | discord_admin | Manage a Discord server via the REST API. Available actions: list_guilds() — list servers the bot is in server_info(guild_id) — server details + member counts list_channels(guild_id) — all channels grouped by category channel_info(channel_id) — single channel details list_roles(guild_id) — roles sorted by position member_info(guild_id, user_id) — lookup a specific member list_pins(channel_id) — pinned messages in a channel pin_message(channel_id, message_id) — pin a message unpin_message(channel_id, message_id) — unpin a message delete_message(channel_id, message_id) — delete a message rename_thread(channel_id, name) — rename an existing thread by thread channel ID add_role(guild_id, user_id, role_id) — assign a role remove_role(guild_id, user_id, role_id) — remove a role Call list_guilds first to discover guild_ids, then list_channels for channel_ids. Runtime errors will tell you if the bot lacks a specific per-guild permission (e.g. MANAGE_ROLES for add_role). |
| feishu_doc_read | feishu_doc | Read Feishu document content |
| feishu_drive_add_comment | feishu_drive | Add a whole-document comment |
| feishu_drive_list_comment_replies | feishu_drive | List comment replies |
| feishu_drive_list_comments | feishu_drive | List document comments |
| feishu_drive_reply_comment | feishu_drive | Reply to a document comment |
| patch | file | Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. Returns a unified diff. Auto-runs syntax checks after editing. REPLACE MODE (mode='replace', default): find a unique string and replace it. REQUIRED PARAMETERS: mode, path, old_string, new_string. PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. REQUIRED PARAMETERS: mode, patch. |
| read_file | file | Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM\|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are rejected; use offset and limit to read specific sections of large files. NOTE: Cannot read images or binary files — use vision_analyze for images. |
| search_files | file | Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents. Content search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts. File search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time. |
| write_file | file | Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out). |
| yb_query_group_info | hermes-yuanbao | Query basic info about a group (called '派/Pai' in the app), including group name, owner, and member count. |
| yb_query_group_members | hermes-yuanbao | Query members of a group (called '派/Pai' in the app). Use this tool when you need to @mention someone, find a user by name, list bots (including Yuanbao AI), or list all members. IMPORTANT: You MUST call this tool before @mentioning any user, because you need the exact nickname to construct the @mention format. |
| yb_search_sticker | hermes-yuanbao | Search the built-in Yuanbao sticker (TIM face / 表情包) catalogue by keyword. Returns the top matching candidates with sticker_id, name, and description. Use this BEFORE yb_send_sticker to discover the right sticker_id. Sticker = 贴纸 = TIM face — NOT a message reaction. Prefer sending a sticker over bare Unicode emoji when reacting/expressing emotion. |
| yb_send_dm | hermes-yuanbao | Send a private/direct message (DM) to a user in a group, with optional media files. This tool automatically looks up the user by name in the group member list and sends the message. Use this when someone asks to privately message / 私信 / DM a user. Supports text, images, and file attachments. You can also provide user_id directly if already known. |
| yb_send_sticker | hermes-yuanbao | Send a built-in sticker (TIMFaceElem / 贴纸表情) to the current Yuanbao chat. Call yb_search_sticker first if you don't know the sticker_id/name. Sticker = 贴纸 = TIM face — NOT a message reaction. CRITICAL: Whenever the user asks you to send a sticker / 贴纸 / 表情包, you MUST use this tool. DO NOT draw a PNG via execute_code / Pillow / matplotlib and then call send_image_file — that produces a fake 'sticker' image instead of a real TIM face and is the WRONG path. If no suitable sticker_id is known, call yb_search_sticker first. When the recent thread shows users sending stickers, prefer matching that tone by replying with a sticker instead of (or in addition to) text. |
| ha_call_service | homeassistant | Call a Home Assistant service to control a device. Use ha_list_services to discover available services and their parameters for each domain. |
| ha_get_state | homeassistant | Get the detailed state of a single Home Assistant entity, including all attributes (brightness, color, temperature setpoint, sensor readings, etc.). |
| ha_list_entities | homeassistant | List Home Assistant entities. Optionally filter by domain (light, switch, climate, sensor, binary_sensor, cover, fan, etc.) or by area name (living room, kitchen, bedroom, etc.). |
| ha_list_services | homeassistant | List available Home Assistant services (actions) for device control. Shows what actions can be performed on each device type and what parameters they accept. Use this to discover how to control devices found via ha_list_entities. |
| image_generate | image_gen | Generate high-quality images from text prompts. The underlying backend (FAL, OpenAI, etc.) and model are user-configured and not selectable by the agent. Returns either a URL or an absolute file path in the `image` field; display it with markdown ![description](url-or-path) and the gateway will deliver it. |
| kanban_block | kanban | Transition the task to blocked because you need human input to proceed. ``reason`` will be shown to the human on the board and included in context when someone unblocks you. Use for genuine blockers only — don't block on things you can resolve yourself. |
| kanban_comment | kanban | Append a comment to a task's thread. Use for durable notes that should outlive this run (questions for the next worker, partial findings, rationale). Ephemeral reasoning doesn't belong here — use your normal response instead. |
| kanban_complete | kanban | Mark your current task done with a structured handoff for downstream workers and humans. Prefer ``summary`` for a human-readable 1-3 sentence description of what you did; put machine-readable facts in ``metadata`` (changed_files, tests_run, decisions, findings, etc). At least one of ``summary`` or ``result`` is required. If you created new tasks via ``kanban_create`` during this run, list their ids in ``created_cards`` — the kernel verifies them so phantom references are caught before they leak into downstream automation. |
| kanban_create | kanban | Create a new kanban task, optionally as a child of the current one (pass the current task id in ``parents``). Used by orchestrator workers to fan out — decompose work into child tasks with specific assignees, link them into a pipeline, then complete your own task. The dispatcher picks up the new tasks on its next tick and spawns the assigned profiles. |
| kanban_heartbeat | kanban | Signal that you're still alive during a long operation (training, encoding, large crawls). Call every few minutes so humans see liveness separately from PID checks. Pure side effect — no work changes. |
| kanban_link | kanban | Add a parent→child dependency edge after both tasks already exist. The child won't promote to 'ready' until all parents are 'done'. Cycles and self-links are rejected. |
| kanban_list | kanban | List Kanban task summaries so an orchestrator profile can discover work to route. Supports the same core filters as the CLI: assignee, status, tenant, include_archived, and limit. Returns compact rows with ids, title, status, assignee, priority, parent/child ids, and counts. Bounded to 50 rows by default, 200 max, with truncation metadata. Also recomputes ready tasks before listing, matching the CLI. Orchestrator-only — dispatcher-spawned task workers never see this tool. |
| kanban_show | kanban | Read a task's full state — title, body, assignee, parent task handoffs, your prior attempts on this task if any, comments, and recent events. Use this to (re)orient yourself before starting work, especially on retries. The response includes a pre-formatted ``worker_context`` string suitable for inclusion verbatim in your reasoning. |
| kanban_unblock | kanban | Move a blocked Kanban task back to ready. Orchestrator-only — only profiles with the kanban toolset can unblock routed work; dispatcher-spawned task workers never see this tool. |
| memory | memory | Save durable information to persistent memory that survives across sessions. Memory is injected into future turns, so keep it compact and focused on facts that will still matter later. WHEN TO SAVE (do this proactively, don't wait to be asked): - User corrects you or says 'remember this' / 'don't do that again' - User shares a preference, habit, or personal detail (name, role, timezone, coding style) - You discover something about the environment (OS, installed tools, project structure) - You learn a convention, API quirk, or workflow specific to this user's setup - You identify a stable fact that will be useful again in future sessions PRIORITY: User preferences and corrections > environment facts > procedural knowledge. The most valuable memory prevents the user from having to repeat themselves. Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO state to memory; use session_search to recall those from past transcripts. If you've discovered a new way to do something, solved a problem that could be necessary later, save it as a skill with the skill tool. TWO TARGETS: - 'user': who the user is -- name, role, preferences, communication style, pet peeves - 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned ACTIONS: add (new entry), replace (update existing -- old_text identifies it), remove (delete -- old_text identifies it). SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state. |
| send_message | messaging | Send a message to a connected messaging platform, or list available targets. IMPORTANT: When the user asks to send to a specific channel or person (not just a bare platform name), call send_message(action='list') FIRST to see available targets, then send to the correct one. If the user just says a platform name like 'send to telegram', send directly to the home channel without listing first. |
| mixture_of_agents | moa | Route a hard problem through multiple frontier LLMs collaboratively. Makes 5 API calls (4 reference models + 1 aggregator) with maximum reasoning effort — use sparingly for genuinely difficult problems. Best for: complex math, advanced algorithms, multi-step analytical reasoning, problems benefiting from diverse perspectives. |
| session_search | session_search | Search your long-term memory of past conversations, or browse recent sessions. This is your recall -- every past session is searchable, and this tool summarizes what happened. TWO MODES: 1. Recent sessions (no query): Call with no arguments to see what was worked on recently. Returns titles, previews, and timestamps. Zero LLM cost, instant. Start here when the user asks what were we working on or what did we do recently. 2. Keyword search (with query): Search for specific topics across all past sessions. Returns LLM-generated summaries of matching sessions. USE THIS PROACTIVELY when: - The user says 'we did this before', 'remember when', 'last time', 'as I mentioned' - The user asks about a topic you worked on before but don't have in current context - The user references a project, person, or concept that seems familiar but isn't in memory - You want to check if you've solved a similar problem before - The user asks 'what did we do about X?' or 'how did we fix Y?' Don't hesitate to search when it is actually cross-session -- it's fast and cheap. Better to search and confirm than to guess or ask the user to repeat themselves. Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), phrases for exact match ("docker networking"), boolean (python NOT java), prefix (deploy*). IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses sessions that only mention some terms. If a broad OR query returns nothing, try individual keyword searches in parallel. Returns summaries of the top matching sessions. |
| skill_manage | skills | Manage skills (create, update, delete). Skills are your procedural memory — reusable approaches for recurring task types. New skills go to ~/.hermes/skills/; existing skills can be modified wherever they live. Actions: create (full SKILL.md + optional category), patch (old_string/new_string — preferred for fixes), edit (full SKILL.md rewrite — major overhauls only), delete, write_file, remove_file. On delete, pass `absorbed_into=<umbrella>` when you're merging this skill's content into another one, or `absorbed_into=""` when you're pruning it with no forwarding target. This lets the curator tell consolidation from pruning without guessing, so downstream consumers (cron jobs that reference the old skill name, etc.) get updated correctly. The target you name in `absorbed_into` must already exist — create/patch the umbrella first, then delete. Create when: complex task succeeded (5+ calls), errors overcome, user-corrected approach worked, non-trivial workflow discovered, or user asks you to remember a procedure. Update when: instructions stale/wrong, OS-specific failures, missing steps or pitfalls found during use. If you used a skill and hit issues not covered by it, patch it immediately. After difficult/iterative tasks, offer to save as a skill. Skip for simple one-offs. Confirm with user before creating/deleting. Good skills: trigger conditions, numbered steps with exact commands, pitfalls section, verification steps. Use skill_view() to see format examples. Pinned skills are protected from deletion only — skill_manage(action='delete') will refuse with a message pointing the user to `hermes curator unpin <name>`. Patches and edits go through on pinned skills so you can still improve them as pitfalls come up; pin only guards against irrecoverable loss. |
| skill_view | skills | Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter. |
| skills_list | skills | List available skills (name + description). Use skill_view(name) to load full content. |
| process | terminal | Manage background processes started with terminal(background=true). Actions: 'list' (show all), 'poll' (check status + new output), 'log' (full output with pagination), 'wait' (block until done or timeout), 'kill' (terminate), 'write' (send raw stdin data without newline), 'submit' (send data + Enter, for answering prompts), 'close' (close stdin/send EOF). |
| terminal | terminal | Execute shell commands on a Linux environment. Filesystem usually persists between calls. Do NOT use cat/head/tail to read files — use read_file instead. Do NOT use grep/rg/find to search — use search_files instead. Do NOT use ls to list directories — use search_files(target='files') instead. Do NOT use sed/awk to edit files — use patch instead. Do NOT use echo/cat heredoc to create files — use write_file instead. Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell. Foreground (default): Commands return INSTANTLY when done, even if the timeout is high. Set timeout=300 for long builds/scripts — you'll still get the result in seconds if it's fast. Prefer foreground for short commands. Background: Set background=true to get a session_id. Two patterns: (1) Long-lived processes that never exit (servers, watchers). (2) Long-running tasks with notify_on_complete=true — you can keep working on other things and the system auto-notifies you when the task finishes. Great for test suites, builds, deployments, or anything that takes more than a minute. For servers/watchers, do NOT use shell-level background wrappers (nohup/disown/setsid/trailing '&') in foreground mode. Use background=true so Hermes can track lifecycle and output. After starting a server, verify readiness with a health check or log signal, then run tests in a separate terminal() call. Avoid blind sleep loops. Use process(action="poll") for progress checks, process(action="wait") to block until done. Working directory: Use 'workdir' for per-command cwd. PTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL). Do NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. Pipe git output to cat if it might page. |
| todo | todo | Manage your task list for the current session. Use for complex tasks with 3+ steps or when the user provides multiple tasks. Call with no parameters to read the current list. Writing: - Provide 'todos' array to create/update items - merge=false (default): replace the entire list with a fresh plan - merge=true: update existing items by id, add any new ones Each item: {id: string, content: string, status: pending\|in_progress\|completed\|cancelled} List order is priority. Only ONE item in_progress at a time. Mark items completed immediately when done. If something fails, cancel it and add a revised item. Always returns the full current list. |
| text_to_speech | tts | Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured (built-in providers like edge/openai or custom command providers under tts.providers.<name>), not model-selected. |
| video_analyze | video | Analyze a video from a URL or local file path using a multimodal AI model. Sends the video to a video-capable model (e.g. Gemini) for understanding. Use this for video files — for images, use vision_analyze instead. Supports mp4, webm, mov, avi, mkv, mpeg formats. Note: large videos (>20 MB) may be slow; max ~50 MB. |
| video_generate | video_gen | (rebuilt at get_definitions() time — see _build_dynamic_video_schema) |
| vision_analyze | vision | Load an image into the conversation so you can see it. Accepts a URL, local file path, or data URL. When your active model has native vision, the image is attached to your context directly and you read the pixels yourself on the next turn — call this any time the user references an image (filepath in their message, URL in tool output, screenshot from the browser, etc.). For non-vision models, falls back to an auxiliary vision model that returns a text description. |
| web_extract | web | Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs (arxiv papers, documents, etc.) — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized and capped at ~5000 chars per page. Pages over 2M chars are refused. If a URL fails or times out, use the browser tool to access it instead. |
| web_search | web | Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and "exact phrase" may work when the backend supports them. |
| x_search | x_search | Search X (Twitter) posts, profiles, and threads using xAI's built-in X Search tool. Use this for current discussion, reactions, or claims on X rather than general web pages. Available when xAI credentials are configured (SuperGrok OAuth or XAI_API_KEY). |
<!-- AUTO-END: capabilities -->

## Toolsets

<!-- AUTO-START: toolsets -->
Core default tools: 48

| Toolset | Description | Tools | Includes |
|---|---|---|---|
| browser | Browser automation for web interaction (navigate, click, type, scroll, iframes, hold-click) with web search for finding URLs | 13 | - |
| clarify | Ask the user clarifying questions (multiple-choice or open-ended) | 1 | - |
| code_execution | Run Python scripts that call tools programmatically (reduces LLM round trips) | 1 | - |
| computer_use | Background macOS desktop control via cua-driver — screenshots, mouse, keyboard, scroll, drag. Does NOT steal the user's cursor or keyboard focus. Works with any tool-capable model. | 1 | - |
| cronjob | Cronjob management tool - create, list, update, pause, resume, remove, and trigger scheduled tasks | 1 | - |
| debugging | Debugging and troubleshooting toolkit | 2 | web, file |
| delegation | Spawn subagents with isolated context for complex subtasks | 1 | - |
| discord | Discord read and participate tools (fetch messages, search members, create threads) | 1 | - |
| discord_admin | Discord server management (list channels/roles, pin messages, assign roles) | 1 | - |
| feishu_doc | Read Feishu/Lark document content | 1 | - |
| feishu_drive | Feishu/Lark document comment operations (list, reply, add) | 4 | - |
| file | File manipulation tools: read, write, patch (with fuzzy matching), and search (content + files) | 4 | - |
| hermes-acp | Editor integration (VS Code, Zed, JetBrains) — coding-focused tools without messaging, audio, or clarify UI | 29 | - |
| hermes-api-server | OpenAI-compatible API server — full agent tools accessible via HTTP (no interactive UI tools like clarify or send_message) | 35 | - |
| hermes-bluebubbles | BlueBubbles iMessage bot toolset - Apple iMessage via local BlueBubbles server | 48 | - |
| hermes-cli | Full interactive CLI toolset - all default tools plus cronjob management | 48 | - |
| hermes-cron | Default cron toolset - same core tools as hermes-cli; gated by `hermes tools` | 48 | - |
| hermes-dingtalk | DingTalk bot toolset - enterprise messaging platform (full access) | 48 | - |
| hermes-discord | Discord bot toolset - full access (terminal has safety checks via dangerous command approval) | 50 | - |
| hermes-email | Email bot toolset - interact with Hermes via email (IMAP/SMTP) | 48 | - |
| hermes-feishu | Feishu/Lark bot toolset - enterprise messaging via Feishu/Lark (full access) | 53 | - |
| hermes-gateway | Gateway toolset - union of all messaging platform tools | 0 | hermes-telegram, hermes-discord, hermes-whatsapp, hermes-slack, hermes-signal, hermes-bluebubbles, hermes-homeassistant, hermes-email, hermes-sms, hermes-mattermost, hermes-matrix, hermes-dingtalk, hermes-feishu, hermes-wecom, hermes-wecom-callback, hermes-weixin, hermes-qqbot, hermes-webhook, hermes-yuanbao |
| hermes-homeassistant | Home Assistant bot toolset - smart home event monitoring and control | 48 | - |
| hermes-matrix | Matrix bot toolset - decentralized encrypted messaging (full access) | 48 | - |
| hermes-mattermost | Mattermost bot toolset - self-hosted team messaging (full access) | 48 | - |
| hermes-qqbot | QQBot toolset - QQ messaging via Official Bot API v2 (full access) | 48 | - |
| hermes-signal | Signal bot toolset - encrypted messaging platform (full access) | 48 | - |
| hermes-slack | Slack bot toolset - full access for workspace use (terminal has safety checks) | 48 | - |
| hermes-sms | SMS bot toolset - interact with Hermes via SMS (Twilio) | 48 | - |
| hermes-telegram | Telegram bot toolset - full access for personal use (terminal has safety checks) | 48 | - |
| hermes-webhook | Webhook toolset - receive and process external webhook events | 48 | - |
| hermes-wecom | WeCom bot toolset - enterprise WeChat messaging (full access) | 48 | - |
| hermes-wecom-callback | WeCom callback toolset - enterprise self-built app messaging (full access) | 48 | - |
| hermes-weixin | Weixin bot toolset - personal WeChat messaging via iLink (full access) | 48 | - |
| hermes-whatsapp | WhatsApp bot toolset - similar to Telegram (personal messaging, more trusted) | 48 | - |
| hermes-yuanbao | Yuanbao Bot 元宝消息平台工具集 - 群信息、成员查询、私聊、贴纸表情 | 53 | - |
| homeassistant | Home Assistant smart home control and monitoring | 4 | - |
| image_gen | Creative generation tools (images) | 1 | - |
| kanban | Kanban multi-agent coordination — only active when the agent is spawned by the kanban dispatcher (HERMES_KANBAN_TASK env set). The dispatcher runs inside the gateway by default; see `kanban.dispatch_in_gateway` in config.yaml. Lets workers mark tasks done with structured handoffs, block for human input, heartbeat during long ops, comment on threads, and (for orchestrators) list, unblock, and fan out tasks. | 9 | - |
| memory | Persistent memory across sessions (personal notes + user profile) | 1 | - |
| messaging | Cross-platform messaging: send messages to Telegram, Discord, Slack, SMS, etc. | 1 | - |
| moa | Advanced reasoning and problem-solving tools | 1 | - |
| safe | Safe toolkit without terminal access | 0 | web, vision, image_gen |
| search | Web search only (no content extraction/scraping) | 1 | - |
| session_search | Search and recall past conversations with summarization | 1 | - |
| skills | Access, create, edit, and manage skill documents with specialized instructions and knowledge | 3 | - |
| spotify | Native Spotify playback, search, playlist, album, and library tools | 7 | - |
| terminal | Terminal/command execution and process management tools | 2 | - |
| todo | Task planning and tracking for multi-step work | 1 | - |
| tts | Text-to-speech: convert text to audio with Edge TTS (free), ElevenLabs, OpenAI, or xAI | 1 | - |
| video | Video analysis and understanding tools (opt-in, not in default toolset) | 1 | - |
| video_gen | Video generation tools. Single ``video_generate`` tool covers text-to-video (prompt only) and image-to-video (prompt + image_url) — the active backend auto-routes. Configure via ``hermes tools`` → Video Generation. | 1 | - |
| vision | Image analysis and vision tools | 1 | - |
| web | Web research and content extraction tools | 2 | - |
| x_search | Search X (Twitter) posts and threads via xAI's built-in x_search Responses tool. Available when xAI credentials are configured (SuperGrok OAuth or XAI_API_KEY). Off by default; enable in `hermes tools` → X (Twitter) Search. | 1 | - |
| yuanbao | Yuanbao platform tools - group info, member queries, DM, stickers | 5 | - |
<!-- AUTO-END: toolsets -->

## Slash Commands

<!-- AUTO-START: slash_commands -->
| Command | Category | Scope | Description |
|---|---|---|---|
| /busy | Configuration | cli | Control what Enter does while Hermes is working |
| /codex-runtime | Configuration | cli+gateway | Toggle codex app-server runtime for OpenAI/Codex models |
| /config | Configuration | cli | Show current configuration |
| /fast | Configuration | cli+gateway | Toggle fast mode — OpenAI Priority Processing / Anthropic Fast Mode (Normal/Fast) |
| /footer | Configuration | cli+gateway | Toggle gateway runtime-metadata footer on final replies |
| /indicator | Configuration | cli | Pick the TUI busy-indicator style |
| /model | Configuration | cli+gateway | Switch model for this session |
| /personality | Configuration | cli+gateway | Set a predefined personality |
| /reasoning | Configuration | cli+gateway | Manage reasoning effort and display |
| /skin | Configuration | cli | Show or change the display skin/theme |
| /statusbar | Configuration | cli | Toggle the context/model status bar |
| /verbose | Configuration | cli | Cycle tool progress display: off -> new -> all -> verbose |
| /voice | Configuration | cli+gateway | Toggle voice mode |
| /yolo | Configuration | cli+gateway | Toggle YOLO mode (skip all dangerous command approvals) |
| /quit | Exit | cli | Exit the CLI (use --delete to also remove session history) |
| /commands | Info | gateway | Browse all commands and skills (paginated) |
| /copy | Info | cli | Copy the last assistant response to clipboard |
| /debug | Info | cli+gateway | Upload debug report (system info + logs) and get shareable links |
| /gquota | Info | cli | Show Google Gemini Code Assist quota usage |
| /help | Info | cli+gateway | Show available commands |
| /image | Info | cli | Attach a local image file for your next prompt |
| /insights | Info | cli+gateway | Show usage insights and analytics |
| /paste | Info | cli | Attach clipboard image from your clipboard |
| /platform | Info | gateway | Pause, resume, or list a failing gateway platform |
| /platforms | Info | cli | Show gateway/messaging platform status |
| /profile | Info | cli+gateway | Show active profile name and home directory |
| /update | Info | gateway | Update Hermes Agent to the latest version |
| /usage | Info | cli+gateway | Show token usage and rate limits for the current session |
| /whoami | Info | cli+gateway | Show your slash command access (admin / user) |
| /agents | Session | cli+gateway | Show active agents and running tasks |
| /approve | Session | gateway | Approve a pending dangerous command |
| /background | Session | cli+gateway | Run a prompt in the background |
| /branch | Session | cli+gateway | Branch the current session (explore a different path) |
| /clear | Session | cli | Clear screen and start a new session |
| /compress | Session | cli+gateway | Manually compress conversation context |
| /deny | Session | gateway | Deny a pending dangerous command |
| /goal | Session | cli+gateway | Set a standing goal Hermes works on across turns until achieved |
| /handoff | Session | cli | Hand off this session to a messaging platform (Telegram, Discord, etc.) |
| /history | Session | cli | Show conversation history |
| /new | Session | cli+gateway | Start a new session (fresh session ID + history) |
| /queue | Session | cli+gateway | Queue a prompt for the next turn (doesn't interrupt) |
| /redraw | Session | cli | Force a full UI repaint (recovers from terminal drift) |
| /restart | Session | gateway | Gracefully restart the gateway after draining active runs |
| /resume | Session | cli+gateway | Resume a previously-named session |
| /retry | Session | cli+gateway | Retry the last message (resend to agent) |
| /rollback | Session | cli+gateway | List or restore filesystem checkpoints |
| /save | Session | cli | Save the current conversation |
| /sessions | Session | cli+gateway | Browse and resume previous sessions |
| /sethome | Session | gateway | Set this chat as the home channel |
| /snapshot | Session | cli | Create or restore state snapshots of Hermes config/state |
| /status | Session | cli+gateway | Show session info |
| /steer | Session | cli+gateway | Inject a message after the next tool call without interrupting |
| /stop | Session | cli+gateway | Kill all running background processes |
| /subgoal | Session | cli+gateway | Add or manage extra criteria on the active goal |
| /title | Session | cli+gateway | Set a title for the current session |
| /topic | Session | gateway | Enable or inspect Telegram DM topic sessions |
| /undo | Session | cli+gateway | Remove the last user/assistant exchange |
| /browser | Tools & Skills | cli | Connect browser tools to your live Chrome via CDP |
| /cron | Tools & Skills | cli | Manage scheduled tasks |
| /curator | Tools & Skills | cli+gateway | Background skill maintenance (status, run, pin, archive, list-archived) |
| /kanban | Tools & Skills | cli+gateway | Multi-profile collaboration board (tasks, links, comments) |
| /plugins | Tools & Skills | cli | List installed plugins and their status |
| /reload | Tools & Skills | cli | Reload .env variables into the running session |
| /reload-mcp | Tools & Skills | cli+gateway | Reload MCP servers from config |
| /reload-skills | Tools & Skills | cli+gateway | Re-scan ~/.hermes/skills/ for newly installed or removed skills |
| /skills | Tools & Skills | cli | Search, install, inspect, or manage skills |
| /tools | Tools & Skills | cli | Manage tools: /tools [list\|disable\|enable] [name...] |
| /toolsets | Tools & Skills | cli | List available toolsets |
<!-- AUTO-END: slash_commands -->

## Gateway Platforms

<!-- AUTO-START: gateway_platforms -->
| Platform | Adapter |
|---|---|
| api_server | gateway/platforms/api_server.py |
| base | gateway/platforms/base.py |
| bluebubbles | gateway/platforms/bluebubbles.py |
| dingtalk | gateway/platforms/dingtalk.py |
| discord | gateway/platforms/discord.py |
| email | gateway/platforms/email.py |
| feishu | gateway/platforms/feishu.py |
| feishu_comment | gateway/platforms/feishu_comment.py |
| feishu_comment_rules | gateway/platforms/feishu_comment_rules.py |
| helpers | gateway/platforms/helpers.py |
| homeassistant | gateway/platforms/homeassistant.py |
| matrix | gateway/platforms/matrix.py |
| mattermost | gateway/platforms/mattermost.py |
| msgraph_webhook | gateway/platforms/msgraph_webhook.py |
| signal | gateway/platforms/signal.py |
| signal_rate_limit | gateway/platforms/signal_rate_limit.py |
| slack | gateway/platforms/slack.py |
| sms | gateway/platforms/sms.py |
| telegram | gateway/platforms/telegram.py |
| telegram_network | gateway/platforms/telegram_network.py |
| webhook | gateway/platforms/webhook.py |
| wecom | gateway/platforms/wecom.py |
| wecom_callback | gateway/platforms/wecom_callback.py |
| wecom_crypto | gateway/platforms/wecom_crypto.py |
| weixin | gateway/platforms/weixin.py |
| whatsapp | gateway/platforms/whatsapp.py |
| yuanbao | gateway/platforms/yuanbao.py |
| yuanbao_media | gateway/platforms/yuanbao_media.py |
| yuanbao_proto | gateway/platforms/yuanbao_proto.py |
| yuanbao_sticker | gateway/platforms/yuanbao_sticker.py |
<!-- AUTO-END: gateway_platforms -->

## Voice / STT / TTS Loop

<!-- AUTO-START: voice_loop -->
Voice input routes platform audio through gateway callbacks, STT, agent response generation, optional TTS, and voice playback.

| Surface | Status |
|---|---|
| tools/transcription_tools.py | present |
| tools/tts_tool.py | present |
| tools/voice_mode.py | present |
| gateway/platforms/discord.py | present |
| gateway/run.py | present |
<!-- AUTO-END: voice_loop -->

## Skills and Agent Profiles

<!-- AUTO-START: skills_profiles -->
Total skills discovered: 267

Profiles: profile-chad, profile-ffm-ceo, profile-heidi, profile-hmg-ceo, profile-main, profile-ohp-ceo, profile-scout, profile-scribe, profile-virtuity-ceo, profile-zentry-architecture, profile-zentry-build, profile-zentry-pm, profile-zentry-qa, profile-zentry-ux

| Category | Skills |
|---|---|
| apple | 10 |
| autonomous-ai-agents | 8 |
| creative | 38 |
| data-science | 2 |
| devops | 6 |
| dogfood | 2 |
| email | 2 |
| gaming | 4 |
| github | 12 |
| hermes-agents | 14 |
| mcp | 2 |
| media | 10 |
| mlops | 18 |
| note-taking | 2 |
| openclaw-imports | 78 |
| productivity | 19 |
| red-teaming | 2 |
| research | 10 |
| smart-home | 2 |
| social-media | 2 |
| software-development | 22 |
| yuanbao | 2 |
<!-- AUTO-END: skills_profiles -->

## Plugins and Integrations

<!-- AUTO-START: plugins_integrations -->
Credential-like environment keys present: 5 names redacted by design.

| Plugin | Path |
|---|---|
| __pycache__ | plugins/__pycache__ |
| browser | plugins/browser |
| context_engine | plugins/context_engine |
| disk-cleanup | plugins/disk-cleanup |
| example-dashboard | plugins/example-dashboard |
| google_meet | plugins/google_meet |
| hermes-achievements | plugins/hermes-achievements |
| image_gen | plugins/image_gen |
| kanban | plugins/kanban |
| memory | plugins/memory |
| model-providers | plugins/model-providers |
| observability | plugins/observability |
| platforms | plugins/platforms |
| spotify | plugins/spotify |
| teams_pipeline | plugins/teams_pipeline |
| video_gen | plugins/video_gen |
| web | plugins/web |
<!-- AUTO-END: plugins_integrations -->

## Recent Activity

<!-- AUTO-START: recent_activity -->
- bdc2113b5 2026-05-17 fix(xai): wire schema sanitizer into post-refactor build_api_kwargs
- 2551f0813 2026-05-17 fix(schema_sanitizer): strip pattern/format from Responses-format tools for xAI compatibility
- 532b209f0 2026-05-17 fix(run_agent): scope kimi tool-reasoning trigger to host, not model name substring
- af7b38d78 2026-05-17 test(voice_cli): drop stale ≥1 requirement for force=True error _vprint calls
- 0b491c466 2026-05-17 fix(model_switch): preserve explicit custom-provider model list when no api_key
- bfcab25dc 2026-05-17 test(tools_config): align post_setup parametrize with current browser provider catalog
- f27416dc8 2026-05-17 fix(cli): include send in _BUILTIN_SUBCOMMANDS for plugin discovery gating
- dfc6ea72c 2026-05-17 test(gateway): include direct_messages_topic_id in telegram DM metadata assertions
- 06924e827 2026-05-17 test(gateway): accept trust_env in fake aiohttp ClientSession lambdas
- e66a3e86e 2026-05-17 chore(acp): bump registry manifest to 0.14.0 matching pyproject
- 822e92edb 2026-05-17 fix(aux): default OpenRouter auxiliary to gemini-3-flash-preview
- e3f7ff112 2026-05-16 test(xai-oauth): pin PKCE token-exchange wire format
- cb53c40e4 2026-05-16 fix(xai-oauth): echo code_challenge in token POST so PKCE exchange succeeds
- bc7c608d5 2026-05-16 fix(gateway): ignore inaccessible service path dirs
- 1a82b7a1f 2026-05-16 fix(tests): stabilize xai env and provider parity
- 73df32921 2026-05-15 fix(doctor): flag missing credentials for active openrouter provider
- a2cc30544 2026-05-17 chore(release): map vaddisrinivas for #26394 salvage
- 7847a58b3 2026-05-15 fix(docker): preload messaging gateway deps
- 4a7cd2e16 2026-05-15 fix(codex): allow kanban worker board writes
- ee7cd1028 2026-05-17 chore(release): map hehehe0803 email for #26212 salvage
- 280c63ce9 2026-05-16 fix(mcp): prevent parallel-safe prefix collisions
- 874dad5cc 2026-05-16 test(delegation): add regression test for runtime missing 'provider' key
- 84667cbc2 2026-05-16 fix(delegation): preserve configured_provider name when runtime returns 'custom'
- 08a66b2ae 2026-05-17 Merge pull request #27489 from NousResearch/bb/tui-composer-cursor-drift-v2
- 3f01e9493 2026-05-17 chore(release): AUTHOR_MAP entries for batch salvage group 6 contributors
- 74031e1e2 2026-05-17 fix(dashboard): respect HERMES_BASE_PATH in WebSocket URLs (#25547)
- 714b3b2bd 2026-05-17 fix(web_server): pass proxy_headers=False to uvicorn.run so the dashboard's loopback gate sees the real connection peer
- 4afd479f5 2026-05-13 fix(gateway): use service restart path in Docker/Podman containers
- 55d6a1636 2026-05-17 fix(agent): honor provider timeout config in streaming API calls
- 2f28b60a4 2026-05-16 fix(send_message): preserve Slack and Matrix thread targets resolved from channel directory
- d5a0815c3 2026-05-16 fix(transports): use monotonic deadlines in codex app-server turn loop
- 37286a5bc 2026-05-17 chore(release): map QuenVix, Mind-Dragon, soynchux emails for Tier 4 salvage
- d0f551b44 2026-05-17 fix(doctor): show xAI OAuth login state in hermes doctor Auth Providers section
- 016893f5e 2026-05-17 feat(status): show xAI OAuth login state in hermes status
- e10bb9dff 2026-05-17 fix(doctor): isolate per-provider OAuth imports to prevent fallback regression
- e89d78ff0 2026-05-17 fix(doctor): suppress stale XAI_API_KEY issue when xAI OAuth is healthy
- caac54796 2026-05-17 chore: revert unrelated package-lock + nix hash churn to keep PR diff minimal
- 711f46e4b 2026-05-17 review(tui): update stale comment refs to renamed visualLines helper
- 220736f41 2026-05-17 chore(nix): refresh ui-tui npmDeps hash after wrap-ansi direct-dep drop
- 8c78f533d 2026-05-17 review(tui): route cursorLayout through @hermes/ink wrapAnsi shim (Bun runtime parity)
<!-- AUTO-END: recent_activity -->

## Open Questions / Unknowns

- Which generated sections should be included in the always-on slim prompt summary versus available only on demand?
- Which plugin integrations should be summarized as first-class capabilities versus implementation details?
- When stable, should CI drift checks become strict or remain advisory?

## Pointers

- `AGENTS.md` — development guide and source map for Hermes internals.
- `tools/registry.py` — runtime tool registry and discovery path.
- `toolsets.py` — named toolset definitions and default core tools.
- `hermes_cli/commands.py` — slash command registry shared by CLI and gateway surfaces.
- `agent/prompt_builder.py` — system prompt assembly.
- `gateway/` — messaging gateway, platform adapters, and voice routing.
