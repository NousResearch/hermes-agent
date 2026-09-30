# initiate-setup: mapping from the current onboarding

Developer notes for the v0 draft. Not read by the setup bot. Remove this file (or move it out of the skill folder) before the skill ships, because an installed skill copies its whole folder.

Sources mapped: `apps/desktop/src/store/onboarding-script.ts` (the runbook), `apps/desktop/src/components/onboarding-chat/setup-profile.ts` (the task-chat runbook), `components/onboarding-chat/{directive.tsx,cards/*,assembly.ts,options.tsx,first-build.ts,signpost.ts}`, `store/onboarding-{answers,capabilities,plugins,plugin-outcomes}.ts`, `store/machine.ts`, `electron/machine-profile.ts`, `hermes_cli/setup_profile.py` (`SETUP_SOUL`). Decisions applied: NS-985 (D2: nothing writes memory; the `start_chat` message carries everything), NS-986, NS-995, NS-996, NS-1016..NS-1020, and Sid's rulings of 2026-09-30: the setup profile's tools are `setup_choose`, `start_chat`, `apply_layout`, `gui_tour` and `manage_connections` (no `manage_catalog`); nothing installs in the setup chat, and the setup chat connects only apps the user asks to connect now; plugin installs and the other connects happen in the task chat's first turn (one install card, one connect card), asked for by the handoff message; there is no theme beat (the `theme` kind stays for requests); the app plays the welcome, name and accent beats through the real `setup_choose` path (P18), and the model runs apps, plugins, layout, the tour offer and the task from the scan and what it knows about the user; one persona, `SETUP_SOUL`, keeps the voice; `setup_choose` answers mirror `clarify`; the skill stays in `optional-skills/`.

## 1. Beats and directives to new primitives

| Current beat / directive | Where it lives today | New primitive | Skill section |
|---|---|---|---|
| Greeting row (app-written `guidedGreeting.line` + `nameSuggestion(loginName)`) | `assembly.ts::pickOnboardingGreeting`, i18n | The app plays the welcome and `setup_choose kind:"question"` for the name (P18), one option = `account.suggested_name`; the bot opens it only when the history has no name answer | Opening |
| `::onboarding{step="name" value}` (saves `answers.name`) | `directive.tsx` DATA_STEPS | Nothing. The answer is the `setup_choose` result in history; it reaches the task chat in the `start_chat` message | Opening, Beat 7 part 2 |
| `::onboarding{step="look"}` (LookCard, accent swatches + custom picker) | `cards/setup.tsx::LookCard` | `setup_choose kind:"accent"`, `options:[]`, played by the app (P18) | Opening |
| `::onboarding{step="look" value="#hex"}` (custom colour in text) | `cards/setup.tsx::LookCard` | `setup_choose kind:"accent"` with one `{id:"#rrggbb"}` option (open question 4) | Opening |
| (none) | - | No theme beat (Sid, D7). `setup_choose kind:"theme"` only when the user asks for light or dark | Failure handling |
| `::onboarding{step="connectors"}` (one card: connectors + plugins; summary sent as `[setup] apps I use...`) | `cards/setup.tsx::ConnectorsCard` | Two cards: `setup_choose kind:"connectors"` and `kind:"plugins"`, both `multi_select`; a pick only records it and feeds the handoff message | Beats 1, 2 |
| `manage_connections` status + connect in the setup chat when asked | runbook "CONNECTING, IF THEY ASK" | Kept: the setup profile has the `connections` toolset; the setup chat connects only apps the user asks to connect now, the rest wait for the task chat | Beat 1 |
| `::onboarding{step="layout"}` (LayoutCard: preset + interface mode + window grow) | `cards/setup.tsx::LayoutCard`, `assembly.ts::assembleChatOnboarding` | `setup_choose kind:"layout"`; `apply_layout` only for a layout asked for in words | Beat 3 |
| Step 4 model-picker explanation | runbook | Dropped; the tour's model picker stop covers it, in both presets | - |
| `::ask` "Want a look around first?" + `gui_tour` targets/start | runbook step 4 | `setup_choose kind:"tour"` (rows filled by the backend) + one `gui_tour` `start` with `preset` `quick` or `full` (the app's own tour) | Beat 4 |
| `::ask` fork, `input="true"` | runbook step 5, `forkOptions()` | `setup_choose kind:"fork"`; the backend fills `fork.options` computed by `scripts/host_facts.py` | Beat 5 |
| `::ask` "What sounds better?" (Something else) | runbook, `forkFallbackOptions()` | The same `fork` call: "Something else" opens `fork.fallback_options` in the backend | Beat 5 |
| Machine branch: one question on main use | runbook step 6 | `setup_choose kind:"question"`, options Work / Gaming / School / Creative / A bit of everything | Beat 6 |
| `::onboarding{step="working" value}` (saves `answers.context`) | `directive.tsx` DATA_STEPS | Nothing. History + `start_chat` message part 2 | Beat 6, Beat 7 |
| `::onboarding{step="first" options}` (FirstBuildCard, 2-4 pills, each <= 60 chars, fallback pill) | `cards/build.tsx::FirstBuildCard` | `setup_choose kind:"question"` with 3-4 options suited to pills | Beat 6 |
| Install beat: one `manage_catalog` install batch in the setup chat | runbook `installBeat()` | Moved to the task chat (Sid, D4): handoff part 4 asks its first turn for ONE `manage_catalog` install call with only the plugins the task needs (all picked for machine setup or a task naming the app; `plugin_tasks` bring their own), then ONE `manage_connections` connect call | Beat 7 part 4 |
| `::onboarding{step="handoff" task brief plan}` + HandoffCard + `requestSetupHandoff` + hidden runbook seed in the new session | `cards/build.tsx::HandoffCard`, `setup-profile.ts` | `start_chat {message, title, profile: primary_profile}`; the plan runbooks become paragraphs of the visible message | Beat 7 |
| `[setup] handoff complete` note | `setup-profile.ts::buildHandoffCompleteNote` | `start_chat` result `started` | Beat 8 |
| Handoff-failed note + "Retry first build" button | runbook step 8, HandoffCard | `start_chat` result `rejected`; the bot says so and retries once, with the same `profile`, when the user says yes | Beat 8 |
| `[setup] <summary>` hidden user rows after each card | `cards/frame.tsx` | `setup_choose` tool results | Tool rules |
| `::onboarding{step="progress" title}` in the task chat | `setup-profile.ts`, ProgressCard | Dropped (directives are deleted). No replacement (open question 9) | - |
| `::ask` "Does this match what you wanted?" in the task chat | `setup-profile.ts` | Handoff message part 6 asks the task chat to ask (it has `clarify`) | Beat 7 part 6 |
| `::ask` "Want me to run this?" (machine-setup) in the task chat | `MACHINE_SETUP_RUNBOOK` | Machine-setup paragraph of the handoff message | Beat 7 |
| `[setup] checkpoint` note after 8 and 20 tool calls in the task chat | `first-build.ts` | No carrier (open question 9) | - |
| Post-handoff tour of the profile rail and sessions list | `signpost.ts::showHandoffTour` | Beat 8 line says where the setup chat lives. The rail tour itself is not in the skill: after `start_chat` the user may already be in the new chat (open question 6) | Beat 8 |
| "Skip this for now" fork option | runbook step 6 | Fork id `skip` | Beat 6 |
| Skip button (applies `basic` layout, marks skipped) | `assembly.ts::skipChatOnboarding` | Renderer / backend marker (NS-1016). The bot handles "I want to leave" in words | Failure handling |

## 2. Guidance to its new home

"Skill" = `SKILL.md`. "Persona" = `SETUP_SOUL` in `hermes_cli/setup_profile.py`, the one voice copy (the skill has no Voice section). "Handoff msg" = text the skill tells the bot to put in the `start_chat` message.

### `onboarding-script.ts`

| Guidance | Home | Notes |
|---|---|---|
| `VOICE_RULES` | Persona | The style rules the SOUL lacked (em dashes, exclamation marks, stock lines, AI diction, no announcing) moved into its last bullet; the pick acknowledgment rules stay in the skill's tool rules |
| `PLAIN_SPEECH` | Persona | |
| `FIRST_USE_GUIDANCE` 1 (first AI agent app, explain when useful) | Skill, Shape it to them; compact copy in Handoff msg part 6 | |
| `FIRST_USE_GUIDANCE` 2 (memory / skill save primer) | Not in the skill | The setup bot has no memory or skill tools. The task chat has no carrier for it now (open question 8) |
| `FIRST_USE_GUIDANCE` 3 (model, model picker, local, search route) | Skill, Shape it to them | Search-route verification adapted: the bot cannot verify, so it defers to the task chat |
| `FIRST_USE_GUIDANCE` 4 (machine age heuristic) | Skill, Shape it to them | |
| `PERSONA` (4 paragraphs, front desk, concrete say/don't-say pairs) | Persona | One copy (Sid, 2026-09-30). The say/don't-say pairs live on as the skill's acknowledgment examples |
| Language paragraph (write in the OS language; directive names stay English) | Skill, Ground rule 3 + tool rule on ids vs labels | Directive names are gone; option ids replace them |
| Never call yourself "Setup" | Persona; off desktop, Skill, Surface fallback | Off desktop `/initiate-setup` runs in the user's own profile, where `SETUP_SOUL` does not load |
| "This message is invisible" | Skill, Ground rule 1 | `/initiate-setup` is now a visible user row; the skill still forbids naming the mechanics |
| RULE 1, never think out loud | Skill, Ground rule 1 | |
| RULE 2, images | Dropped | The setup bot has no image tool |
| RULE 3, one question per turn | Skill, tool rules; plain text: Surface fallback | `setup_choose` blocks, so "per turn" became "one card at a time". The name/working exception is obsolete |
| RULE 4, card beats carry no tool calls | Inverted | Cards ARE tool calls now. "Never repeat a successful call" kept |
| "Your first message has already been sent" | Kept in spirit | The app plays the opening (P18); the bot's first message follows the accent answer |
| Suggested-name acceptance ("sure", "yes") | Skill, Opening | Name source changed: OS full name only (NS-986), never the login handle |
| Steps 1-8 | App opening + Skill, Beats 1-8 | See section 1 |
| Sign-in nudge when not signed in | Skill, Beat 1 | Keyed on builder fact `guest_free_tier` (today: `record.free_tier_route`) |
| Custom colour | Skill, Opening | |
| Tour branches, `targets` first, stable targets, one `start` | Skill, Beat 4 | One `start` with a built-in `preset`; `targets` + `steps` only for one specific part later |
| Local models flow (Settings, Providers, Local Models) | Skill, Shape it to them | |
| Fresh-machine and Spark fork text | Skill, Beat 5 | Signals computed by the script |
| General-idea branch, first-task card rules, connector examples | Skill, Beat 6 | Compressed |
| "Their tap or typed task is the decision" | Skill, Beat 6 | |
| "When a plugin fits" (Hermes interface as first build) | Dropped | Gated on `desktop_plugins_root`, which the builder never sends (open question 10) |
| Connector-dependent tasks welcome, no mock inbox | Skill, Beat 6 | |
| Plugin tasks (Blender, NVIDIA) | Script `plugin_tasks` + Skill, Beats 6-7 | |
| `installBeat()` | Handoff msg part 4 (task chat's first turn) | Task-needed selection, plugin tasks count as picked, unneeded picks named but not installed, one batch, exact ids, one approval card, no second offer after a skip |
| Handoff line, task <= 40, brief <= 200, plan attr | Skill, Beat 7 | `title` <= 40; the brief is now the full message |
| Step 8, `[setup]` notes | Skill, Beat 8 | Tool results replace notes |
| Fenced code block for drafts | Skill, Ground rule 4 | |
| General `::ask` rules (2-6 options, act on the pick, no prose options) | Skill, tool rules | `setup_choose` allows up to 12 |
| Exactness rule for scripted options | Skill, tool rules | Ids instead of label matching |
| Tool-turn shape example | Skill, Ground rule 2 | The forced "Two seconds" line is gone: `SETUP_SOUL` says not to announce an action; the call, then one result line |
| Never end on a promise | Skill, Ground rule 2 | |
| Memory paragraph (cards persist answers; handoff saves to profile) | Replaced | NS-985 D2: nothing writes memory; Beat 7 says the message carries everything |
| `[setup]` picks: acknowledge, never the same phrase twice | Skill, tool rules | |
| "Someone just walked in" | Persona ("pleased they came in") | |

### `onboarding-capabilities.ts` (CATALOG EVIDENCE)

Dropped from the skill: the `/initiate-setup` builder never sends `catalog_evidence` (open question 1). What survives without it: one option per detected app, a connection-free choice, and never replacing the machine fork (Skill, Beat 6); the no-route-around rule (Handoff msg build plan).

### `setup-profile.ts` (task-chat runbook, now the `start_chat` message)

| Guidance | Home | Notes |
|---|---|---|
| "You are Hermes. The welcome chat opened this session..." / invisible | Dropped | The message is visible and written as the user's ask |
| Name, context | Handoff msg part 2 | |
| Apps they use; check status; never require an unconnected one | Handoff msg parts 3-4 + build plan | Apps connected during setup are named; the rest of the task's apps connect in the first turn |
| Go signal (next message) | Changed | `start_chat` submits the message and the turn starts at once |
| "You'll ask for permissions as you go" | Handoff msg part 6 | |
| Capabilities block | Dropped | The task chat can read its own catalog |
| `FIRST_USE_GUIDANCE` | Handoff msg part 6 (compact) | Memory/skill primer has no carrier (open question 8) |
| `connectFirstRunbook` | Handoff msg part 4 (connect list) | Was limited to `CONNECTOR_LEAD_ORDER` slugs; now the apps the task needs that setup did not connect |
| `NO_AUTH_RULE` | Handoff msg build paragraph (empty connect list) | |
| `MACHINE_SETUP_RUNBOOK` + `machineDescription()` | Handoff msg machine-setup paragraph + `signals.description` | Every rule kept, in first person |
| `pluginRunbook` (desktop plugin) | Dropped | The interface plan needs `desktop_plugins_root`, which the builder never sends (open question 10) |
| `pluginsRunbook` (installed / offered / not offered; `tool_search`, `skill_view`; "do not install plugins yourself") | Handoff msg part 4 | The task chat now installs the task's plugins itself in its first turn; `tool_search` and `skill_view` guidance kept |
| `::onboarding{step="progress"}` | Dropped | Directives deleted |
| First-pass review ask | Handoff msg part 6 | |
| `PLAIN_SPEECH` in the task chat | No carrier | Open question 8 |
| `buildHandoffCompleteNote` | `start_chat` result + Beat 8 | |

### `SETUP_SOUL` (`hermes_cli/setup_profile.py`)

The one voice copy (Sid, 2026-09-30); it gained the voice rules it lacked. The check-in guidance there ("when you check in, look at what has changed...") has no carrier while the setup bot has no session/connector/cron tools (open question 11).

## 3. Renderer-only plumbing, left out of the skill

Directive parsing and `DATA_STEPS`/`STEP_CARDS`; `FUNNEL_STEPS` metrics (`recordOnboarding('guide_look'...)`, which need a new hook on `setup_choose` kinds); `$onboardingAnswers` localStorage and `committed` receipts; handoff receipts, `persisted-handoff.ts`, `requestSetupHandoff` guards; chat-solo layout, `hermes:window:size` window sizing (`electron/window-growth.ts`), `restorePreviousLayout`; interface-mode switch on layout pick (`LAYOUTS[].mode`); `accentsFor()` swatches and `NOUS_ACCENT`; `CONNECTOR_LEAD_ORDER` ordering and `CONNECTOR_PICKER_HIDDEN` (discord, discordbot, microsoft_teams); the 12-minus-plugins cap and search field; card footnote "Nothing connects or installs yet"; FirstBuildCard dedupe, 60-char filter, 4-pill cap, fallback pill; `firstTaskTitle` 28-char truncation; `reasoning: minimal` on the free tier route; `SETUP_CHAT_TITLE` and session adoption; catalog prefetch; guide loading UI and gate; `skipChatOnboarding`.

## 4. Behaviour changes versus today

- Name suggestion uses the OS full name only; the login handle is never offered (NS-986). On this Mac (`pw_gecos == login`) no name is suggested.
- The app plays the welcome, name and accent (P18); the bot starts at the apps card and shapes the rest from the scan.
- No theme beat.
- Connectors and plugins are two cards; a pick only records it.
- The setup chat connects only apps the user asks to connect now and installs nothing; the task chat's first turn installs and connects what the task needs.
- The whole setup can run inside one agent turn because `setup_choose` blocks.
- `start_chat` targets `primary_profile`, not always `default`, and a retry never drops it.
- A later `/initiate-setup` after a handoff (`setup_completed_at` set) skips to the fork for a new task.
- The first-task handoff carries its whole runbook as visible text.

## 5. Host facts: renderer field to script field

| Renderer (`electron/machine-profile.ts`, `store/machine.ts`) | Script (`scripts/host_facts.py`) | Source |
|---|---|---|
| `ageDays` (home birthtime) | `account.home_age_days` | `os.stat(home).st_birthtime`; home from the user database, not `HOME`. Linux: null (no birth time in `os.stat`) |
| `machineLooksNew()` (<= 21 days) | `signals.machine_state`, `signals.looks_new` | `looks_new` is `machine_state == "fresh"`, from the user scan's `install.install_state`; without a scan, home-folder age (<= 21 days fresh, < 120 settling) |
| `username` + `machineUserName()` filter | `account.suggested_name` | `pwd` GECOS / `GetUserNameExW(NameDisplay)`; dropped when equal to the login or handle-like (digits, underscores, all lowercase) |
| `locale` (`app.getLocale()`) + `machineLanguageName()` | `account.locale`, `account.locale_is_english` | CoreFoundation preferred language / `GetUserDefaultLocaleName` / `/etc/locale.conf`. The model names the language from the tag |
| `nvidia` (Chromium GPU list) | `machine.gpu_class`, `signals.has_nvidia_gpu` | `hermes_platform.host.facts.gpu_class()` |
| `model` (`/proc/device-tree/model`) + `machineIsSpark()` | `signals.is_spark` | `products.is_nvidia_arm_soc()`, win32+arm64+nvidia, or `dgx|spark|gb10` in `facts.cpu_model()` (which falls back to the device-tree model) |
| `machineSetupLeads()` | `signals.machine_setup_leads` | |
| `machineKind()` | `signals.machine_kind` | |
| `machineDescription()` | `signals.description` | model string is the CPU model, not the chassis model |
| `platform`, `release`, `arch` | `machine.os_family`, `machine.os_release`, `machine.native_arch` | `facts.os_family()`, `platform.release()`, `facts.native_arch()` |
| `forkOptions()`, `forkFallbackOptions()`, `pluginForkOptions()` | `fork`, `plugin_tasks` | same order and labels, plus stable ids |
| (none) | `machine.cpu_model`, `ram_gb`, `wsl`, `container` | new context for ideas |
| (none) | `scan` | the user scan in `scripts/userscan/` (T1 pass, cached in `HERMES_HOME/insights/profile.json` for 24 h while its L1 detectors match), interpreted in code |

## 6. Open questions

Resolved by Sid (2026-09-30): skill location (stays in `optional-skills/`); the `setup_choose` answer shape (mirrors `clarify`: an id, a list of ids with `multi_select`, or free text); the tour beat (restored with `gui_tour`); plugin installs and connects (in the task chat, except connect-now on request); no theme beat; one persona (`SETUP_SOUL`).

1. **Who supplies which fact.** `primary_profile`, `guest_free_tier` and `setup_completed_at` (from the setup profile's marker) are session facts the builder adds. `catalog_evidence` and `desktop_plugins_root` are client facts the builder never sends, so the skill no longer reads them; bring them back only with a builder that sends them. `locale` and the user's name describe the person at the desktop client, but the script reads the backend host; on an SSH, URL or Cloud backend they come from the wrong machine. Should the desktop client send locale and name to the builder?
2. **Install profile versus handoff profile.** Resolved: the task chat runs in `primary_profile` and installs there itself (NS-1015), so no setup-time install can land in the wrong profile.
4. **Accent.** Can the accent card take a custom hex through `options` (the skill does `{id:"#rrggbb"}`), or does a text colour request need another tool?
5. **Empty plugins card.** What does the plugins card show when the catalog has no onboarding plugins?
6. **Post-handoff rail tour.** `signpost.ts` shows the profile rail and sessions list after the handoff. `gui_tour` could do it, but after `start_chat` the user may be in the new chat, not watching the setup chat. Keep it in the app, or have the bot run it before `start_chat`?
7. **Layout.** P16 says the layout card applies the pick live, which makes `apply_layout` redundant for the beat. Today the Elite pick also switches interface mode to advanced; `apply_layout` does not. The card uses `sidebar-left` while the skip path uses `basic`. Which ids should the card and `apply_layout` expose?
8. **Guidance with no carrier in the task chat.** The memory/skill primer (`FIRST_USE_GUIDANCE` 2), `PLAIN_SPEECH` and the voice rules reached the task chat through the hidden runbook. With D2 only the visible `start_chat` message reaches it. The handoff message now carries the whole plan runbook as visible user text, which is long. Alternative: plan skills (machine-setup, desktop-plugin, first-build) in the primary profile, named in a short message.
9. **Task-chat progress and check-ins.** `::onboarding{step="progress"}` and the `[setup] checkpoint` notes after 8 and 20 tool calls have no replacement.
10. **Interface first task.** Dropped from the skill until the builder sends the app-level desktop plugin folder, which only the Electron client knows (`desktopPluginsRoot`). The old runbook names a `building-hermes-desktop-plugins` skill that does not exist in the tree.
11. **SOUL check-ins.** `SETUP_SOUL` tells the bot to look at sessions, connectors and scheduled jobs before checking in. The setup profile's tools cannot read those.
12. **One turn or many.** `setup_choose` blocks, so the whole setup can run in one agent turn. What happens on a card timeout, a closed window mid-card, or a relaunch with a pending card? The skill says: take the default and continue from the first unanswered beat.
13. **Other surfaces.** P13 runs `/initiate-setup` on CLI, TUI and messaging too, in the user's own profile with full tools. The skill falls back to plain-text questions, skips the tour without `gui_tour`, and starts the task in the same chat without `start_chat`. Should it use the extra tools those sessions have?
14. **Size.** `SKILL.md` is under 25 KB (~245 lines) after the 2026-09-30 trim; the handoff plan paragraphs are most of it.
15. **Host facts gaps.** Linux has no home birth time in `os.stat`, so a fresh Linux machine never leads with machine setup (Electron used `statx`). Name and locale are read in the skill script, not in `hermes_platform.host`, which covers hardware only. Should `hermes_platform.host` grow a birth-time helper and account facts?
