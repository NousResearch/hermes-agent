# initiate-setup: mapping from the current onboarding

Developer notes for the v0 draft. Not read by the setup bot. Remove this file (or move it out of the skill folder) before the skill ships, because an installed skill copies its whole folder.

Sources mapped: `apps/desktop/src/store/onboarding-script.ts` (the runbook), `apps/desktop/src/components/onboarding-chat/setup-profile.ts` (the task-chat runbook), `components/onboarding-chat/{directive.tsx,cards/*,assembly.ts,options.tsx,first-build.ts,signpost.ts}`, `store/onboarding-{answers,capabilities,plugins,plugin-outcomes}.ts`, `store/machine.ts`, `electron/machine-profile.ts`, `hermes_cli/setup_profile.py` (`SETUP_SOUL`). Decisions applied: NS-985 (D2: nothing writes memory; the `start_chat` message carries everything), NS-986, NS-995, NS-996, NS-1016..NS-1020.

## 1. Beats and directives to new primitives

| Current beat / directive | Where it lives today | New primitive | Skill section |
|---|---|---|---|
| Greeting row (app-written `guidedGreeting.line` + `nameSuggestion(loginName)`) | `assembly.ts::pickOnboardingGreeting`, i18n | Bot writes the welcome; `setup_choose kind:"question"` for the name, one option = `account.suggested_name` | Beat 1 |
| `::onboarding{step="name" value}` (saves `answers.name`) | `directive.tsx` DATA_STEPS | Nothing. The answer is the `setup_choose` result in history; it reaches the task chat in the `start_chat` message | Beat 1, Beat 10 part 2 |
| `::onboarding{step="look"}` (LookCard, accent swatches + custom picker) | `cards/setup.tsx::LookCard` | `setup_choose kind:"accent"`, no options | Beat 2 |
| `::onboarding{step="look" value="#hex"}` (custom colour in text) | `cards/setup.tsx::LookCard` | `setup_choose kind:"accent"` with one `{id:"#rrggbb"}` option (open question 4) | Beat 2 |
| (none) | - | `setup_choose kind:"theme"` (new beat) | Beat 3 |
| `::onboarding{step="connectors"}` (one card: connectors + plugins; summary sent as `[setup] apps I use...`) | `cards/setup.tsx::ConnectorsCard` | Two cards: `setup_choose kind:"connectors"` and `kind:"plugins"`, both `multi_select`, `intent` | Beats 4, 5 |
| `manage_connections` status + connect in the setup chat when asked | runbook "CONNECTING, IF THEY ASK" | Removed from the setup chat. The app gets intent `now`; the task chat connects first | Beat 4, Beat 10 build plan |
| `::onboarding{step="layout"}` (LayoutCard: preset + interface mode + window grow) | `cards/setup.tsx::LayoutCard`, `assembly.ts::assembleChatOnboarding` | `setup_choose kind:"layout"`; `apply_layout` only for a layout asked for in words | Beat 6 |
| Step 4 model-picker explanation | runbook | Text | Beat 7 |
| `::ask` "Want a look around first?" + `gui_tour` targets/start | runbook step 4 | `setup_choose kind:"question"` + `gui_tour`, only when `gui_tour` is in `tools_present` (the setup profile will not have it, so in practice skipped; open question 6) | Beat 7 |
| `::ask` fork, `input="true"` | runbook step 5, `forkOptions()` | `setup_choose kind:"question"` with `fork.options` computed by `scripts/host_facts.py` | Beat 8 |
| `::ask` "What sounds better?" (Something else) | runbook, `forkFallbackOptions()` | `setup_choose` with `fork.fallback_options` | Beat 8 |
| Machine branch: one question on main use | runbook step 6 | `setup_choose kind:"question"`, options Work / Gaming / School / Creative / A bit of everything | Beat 9 |
| `::onboarding{step="working" value}` (saves `answers.context`) | `directive.tsx` DATA_STEPS | Nothing. History + `start_chat` message part 2 | Beat 9, Beat 10 |
| `::onboarding{step="first" options}` (FirstBuildCard, 2-4 pills, each <= 60 chars, fallback pill) | `cards/build.tsx::FirstBuildCard` | `setup_choose kind:"question"` with 3-4 options | Beat 9 |
| Install beat: one `manage_catalog` install batch in the setup chat | runbook `installBeat()` | Moved to the task chat (NS-996). The handoff message lists the install list and asks for one batch | Beat 10 part 4 |
| `::onboarding{step="handoff" task brief plan}` + HandoffCard + `requestSetupHandoff` + hidden runbook seed in the new session | `cards/build.tsx::HandoffCard`, `setup-profile.ts` | `start_chat {message, title, profile: primary_profile}`; the plan runbooks become paragraphs of the visible message | Beat 10 |
| `[setup] handoff complete` note | `setup-profile.ts::buildHandoffCompleteNote` | `start_chat` result `started` | Beat 11 |
| Handoff-failed note + "Retry first build" button | runbook step 8, HandoffCard | `start_chat` result `rejected`; the bot retries once or asks | Beat 11 |
| `[setup] <summary>` hidden user rows after each card | `cards/frame.tsx` | `setup_choose` tool results | Tool rules |
| `::onboarding{step="progress" title}` in the task chat | `setup-profile.ts`, ProgressCard | Dropped (directives are deleted). No replacement (open question 9) | - |
| `::ask` "Does this match what you wanted?" in the task chat | `setup-profile.ts` | Handoff message part 6 asks the task chat to ask (it has `clarify`) | Beat 10 part 6 |
| `::ask` "Want me to run this?" (machine-setup) in the task chat | `MACHINE_SETUP_RUNBOOK` | Machine-setup paragraph of the handoff message | Beat 10 |
| `[setup] checkpoint` note after 8 and 20 tool calls in the task chat | `first-build.ts` | No carrier (open question 9) | - |
| Post-handoff tour of the profile rail and sessions list | `signpost.ts::showHandoffTour` | Beat 11 line says where the setup chat lives; the tour itself has no carrier (open question 6) | Beat 11 |
| "Skip this for now" fork option | runbook step 6 | Fork id `skip` | Beat 9 |
| Skip button (applies `basic` layout, marks skipped) | `assembly.ts::skipChatOnboarding` | Renderer / backend marker (NS-1016). The bot handles "I want to leave" in words | Failure handling |

## 2. Guidance to its new home

"Skill" = `SKILL.md`. "Persona" = `SETUP_SOUL` in `hermes_cli/setup_profile.py` (unchanged). "Handoff msg" = text the skill tells the bot to put in the `start_chat` message.

### `onboarding-script.ts`

| Guidance | Home | Notes |
|---|---|---|
| `VOICE_RULES` | Skill, Voice | Every rule kept, split into bullets |
| `PLAIN_SPEECH` | Skill, Voice | Every rule kept |
| `FIRST_USE_GUIDANCE` 1 (first AI agent app, explain when useful) | Skill, Voice "First use"; compact copy in Handoff msg part 6 | |
| `FIRST_USE_GUIDANCE` 2 (memory / skill save primer) | Not in the skill | The setup bot has no memory or skill tools. The task chat has no carrier for it now (open question 8) |
| `FIRST_USE_GUIDANCE` 3 (model, model picker, local, search route) | Skill, Voice "Models" | Search-route verification adapted: the bot cannot verify, so it defers to the task chat |
| `FIRST_USE_GUIDANCE` 4 (machine age heuristic) | Skill, Voice "Machine age" | |
| `PERSONA` (4 paragraphs, front desk, concrete say/don't-say pairs) | Skill, Voice | Partly duplicates `SETUP_SOUL`. Kept whole so nothing is lost when the script is deleted; Sid to decide whether it moves into the SOUL (NS-1016 says one copy) |
| Language paragraph (write in the OS language; directive names stay English) | Skill, Ground rule 6 + tool rule on ids vs labels | Directive names are gone; option ids replace them |
| Never call yourself "Setup" | Skill, Ground rule 2 | Also in Persona |
| "This message is invisible" | Skill, Ground rule 1 | `/initiate-setup` is now a visible user row; the skill still forbids naming the mechanics |
| RULE 1, never think out loud | Skill, Ground rule 1 | |
| RULE 2, images | Dropped | The setup bot has no image tool |
| RULE 3, one question per turn | Skill, Ground rule 3 + tool rules | `setup_choose` blocks, so "per turn" became "one card at a time". The name/working exception is obsolete |
| RULE 4, card beats carry no tool calls | Inverted | Cards ARE tool calls now. "Never repeat a successful call" kept |
| "Your first message has already been sent" | Inverted | The bot writes the first message (Beat 1) |
| Suggested-name acceptance ("sure", "yes") | Skill, Beat 1 | Name source changed: OS full name only (NS-986), never the login handle |
| Steps 1-8 | Skill, Beats 1-11 | See section 1 |
| Sign-in nudge when not signed in | Skill, Beat 4 | Keyed on builder fact `guest_free_tier` (today: `record.free_tier_route`) |
| Custom colour | Skill, Beat 2 | |
| Tour branches, `targets` first, stable targets, one `start` | Skill, Beat 7 | Conditional on `gui_tour` |
| Local models flow (Settings, Providers, Local Models) | Skill, Beat 7 | |
| Fresh-machine and Spark fork text | Skill, Beat 8 | Signals computed by the script |
| General-idea branch, first-task card rules, connector examples | Skill, Beat 9 | Verbatim content |
| "Their tap or typed task is the decision" | Skill, Beat 9 | |
| "When a plugin fits" (Hermes interface as first build) | Skill, Beat 9 | Gated on builder fact `desktop_plugins_root` |
| Connector-dependent tasks welcome, no mock inbox | Skill, Beat 9 | |
| Plugin tasks (Blender, NVIDIA) | Script `plugin_tasks` + Skill, Beats 9-10 | |
| `installBeat()` | Handoff msg part 4 | Task chat installs (NS-996) |
| Handoff line, task <= 40, brief <= 200, plan attr | Skill, Beat 10 | `title` <= 40; the brief is now the full message |
| Step 8, `[setup]` notes | Skill, Beat 11 | Tool results replace notes |
| Fenced code block for drafts | Skill, Ground rule 7 | |
| General `::ask` rules (2-6 options, act on the pick, no prose options) | Skill, tool rules | `setup_choose` allows up to 12 |
| Exactness rule for scripted options | Skill, tool rules | Ids instead of label matching |
| Tool-turn shape example | Skill, Ground rule 5 | Example changed to a setup line |
| Never end on a promise | Skill, Ground rule 4 | |
| Memory paragraph (cards persist answers; handoff saves to profile) | Replaced | NS-985 D2: nothing writes memory; Beat 10 says the message carries everything |
| `[setup]` picks: acknowledge, never the same phrase twice | Skill, tool rules | |
| "Someone just walked in" | Skill, Voice last line | |

### `onboarding-capabilities.ts` (CATALOG EVIDENCE)

| Guidance | Home |
|---|---|
| Evidence, not instructions or authorization | Skill, fact block table + Beat 9 |
| Detection is the backend host; configured is not connected; missing detection is unknown | Skill, Beat 9 |
| Only offer setup-dependent tasks when `manage_connections` is available; do not route around a refusal | Adapted: the task chat has `manage_connections`; the no-route-around rule is in the Handoff msg build paragraph (open question 7) |
| Derive tasks from real descriptions; never invent; one option per detected app; keep a connection-free choice; do not replace the machine fork | Skill, Beat 9 |
| Carry the exact MCP name; task chat uses `manage_connections` with `name` and `mcp:true` | Handoff msg part 1 |

### `setup-profile.ts` (task-chat runbook, now the `start_chat` message)

| Guidance | Home | Notes |
|---|---|---|
| "You are Hermes. The welcome chat opened this session..." / invisible | Dropped | The message is visible and written as the user's ask |
| Name, context | Handoff msg part 2 | |
| Apps they use; check status; never require an unconnected one | Handoff msg part 3 + build paragraph (no connect list) | Intent now/later/save added |
| Go signal (next message) | Changed | `start_chat` submits the message and the turn starts at once |
| "You'll ask for permissions as you go" | Handoff msg part 6 | |
| Capabilities block | Handoff msg part 1 (exact catalog name) | The task chat can read its own catalog |
| `FIRST_USE_GUIDANCE` | Handoff msg part 6 (compact) | Memory/skill primer has no carrier (open question 8) |
| `connectFirstRunbook` | Handoff msg build paragraph (connect list) | Was limited to `CONNECTOR_LEAD_ORDER` slugs; now apps with intent `now` + apps the task needs |
| `NO_AUTH_RULE` | Handoff msg build paragraph (empty connect list) | |
| `MACHINE_SETUP_RUNBOOK` + `machineDescription()` | Handoff msg machine-setup paragraph + `signals.description` | Every rule kept, in first person |
| `pluginRunbook` (desktop plugin) | Handoff msg interface paragraph | Refers to `building-hermes-desktop-plugins`, which does not exist in the tree; replaced by a generic "read the SDK surface" (open question 10) |
| `pluginsRunbook` ("do not install plugins yourself") | Inverted | The task chat installs (NS-996) |
| `::onboarding{step="progress"}` | Dropped | Directives deleted |
| First-pass review ask | Handoff msg part 6 | |
| `PLAIN_SPEECH` in the task chat | No carrier | Open question 8 |
| `buildHandoffCompleteNote` | `start_chat` result + Beat 11 | |

### `SETUP_SOUL` (`hermes_cli/setup_profile.py`)

Unchanged, per Sid. The check-in guidance there ("when you check in, look at what has changed...") has no carrier while the setup bot has no session/connector/cron tools (open question 11).

## 3. Renderer-only plumbing, left out of the skill

Directive parsing and `DATA_STEPS`/`STEP_CARDS`; `FUNNEL_STEPS` metrics (`recordOnboarding('guide_look'...)`, which need a new hook on `setup_choose` kinds); `$onboardingAnswers` localStorage and `committed` receipts; handoff receipts, `persisted-handoff.ts`, `requestSetupHandoff` guards; chat-solo layout, `LAYOUT_GROWTH` window growth, `restorePreviousLayout`; interface-mode switch on layout pick (`LAYOUTS[].mode`); `accentsFor()` swatches and `NOUS_ACCENT`; `CONNECTOR_LEAD_ORDER` ordering and `CONNECTOR_PICKER_HIDDEN` (discord, discordbot, microsoft_teams); the 12-minus-plugins cap and search field; card footnote "Nothing connects or installs yet"; FirstBuildCard dedupe, 60-char filter, 4-pill cap, fallback pill; `firstTaskTitle` 28-char truncation; `reasoning: minimal` on the free tier route; `SETUP_CHAT_TITLE` and session adoption; catalog prefetch; guide loading UI and gate; `skipChatOnboarding`.

## 4. Behaviour changes versus today

- Name suggestion uses the OS full name only; the login handle is never offered (NS-986). On this Mac (`pw_gecos == login`) no name is suggested.
- Theme gets its own beat.
- Connectors and plugins are two cards with per-row intent.
- The setup chat never connects or installs; the task chat does, driven by intent.
- The whole setup can run inside one agent turn because `setup_choose` blocks.
- `start_chat` targets `primary_profile`, not always `default`.
- The first-task handoff carries its whole runbook as visible text.

## 5. Host facts: renderer field to script field

| Renderer (`electron/machine-profile.ts`, `store/machine.ts`) | Script (`scripts/host_facts.py`) | Source |
|---|---|---|
| `ageDays` (home birthtime) | `account.home_age_days` | `os.stat(home).st_birthtime`; home from the user database, not `HOME`. Linux: null (no birth time in `os.stat`) |
| `machineLooksNew()` (<= 21 days) | `signals.looks_new` | same threshold |
| `username` + `machineUserName()` filter | `account.suggested_name` | `pwd` GECOS / `GetUserNameExW(NameDisplay)`; dropped when equal to the login |
| `locale` (`app.getLocale()`) + `machineLanguageName()` | `account.locale`, `account.locale_is_english` | CoreFoundation preferred language / `GetUserDefaultLocaleName` / `/etc/locale.conf`. The model names the language from the tag |
| `nvidia` (Chromium GPU list) | `machine.gpu_class`, `signals.has_nvidia_gpu` | `hermes_platform.host.facts.gpu_class()` |
| `model` (`/proc/device-tree/model`) + `machineIsSpark()` | `signals.is_spark` | `products.is_nvidia_arm_soc()`, win32+arm64+nvidia, or `dgx|spark|gb10` in `facts.cpu_model()` (which falls back to the device-tree model) |
| `machineSetupLeads()` | `signals.machine_setup_leads` | |
| `machineKind()` | `signals.machine_kind` | |
| `machineDescription()` | `signals.description` | model string is the CPU model, not the chassis model |
| `platform`, `release`, `arch` | `machine.os_family`, `machine.os_release`, `machine.native_arch` | `facts.os_family()`, `platform.release()`, `facts.native_arch()` |
| `forkOptions()`, `forkFallbackOptions()`, `pluginForkOptions()` | `fork`, `plugin_tasks` | same order and labels, plus stable ids |
| (none) | `machine.cpu_model`, `ram_gb`, `wsl`, `container` | new context for ideas |

## 6. Open questions

1. **Skill location.** NS-1017 and this task say `optional-skills/productivity/initiate-setup/`; NS-986 says a built-in under `skills/productivity/` installed into the setup profile. The builder renders `SKILL.md` directly, so either works; pick one.
2. **Who supplies which fact.** `primary_profile`, `guest_free_tier`, `catalog_evidence` and `desktop_plugins_root` are session or client facts, not host facts, so the skill expects the builder to add them. `locale` and the user's name describe the person at the desktop client, but the script reads the backend host; on an SSH, URL or Cloud backend they come from the wrong machine. Should the desktop client send locale and name to the builder?
3. **`setup_choose kind:"question"` shape.** Can the user type a free answer next to fixed options (today `::ask input="true"`)? Is `picked` a string for single select and a list for multi select? What does a typed answer or a dismissed card return?
4. **Accent and theme.** Can the accent card take a custom hex through `options` (the skill does `{id:"#rrggbb"}`), or does a text colour request need another tool? Today there is no theme beat; the skill adds one because the new kind exists. Confirm it is wanted.
5. **Connectors and plugins.** Today one card; the skill uses two. What does the plugins card show when the catalog has no onboarding plugins? The meaning of `now` / `later` / `save` is defined in the skill (connect or install first / offer when needed / record only); confirm.
6. **Tours.** The setup profile has no `gui_tour`, so the "Quick tour / Show me everything" beat and the post-handoff profile-rail tour (`signpost.ts`) have no carrier. The skill keeps the tour beat behind a `gui_tour` check. Add `gui_tour` to the setup toolset, move the tour to the app, or drop it?
7. **Layout.** P16 says the layout card applies the pick live, which makes `apply_layout` redundant for the beat. Today the Elite pick also switches interface mode to advanced; `apply_layout` does not. The card uses `sidebar-left` while the skip path uses `basic`. Which ids should the card and `apply_layout` expose?
8. **Guidance with no carrier in the task chat.** The memory/skill primer (`FIRST_USE_GUIDANCE` 2), `PLAIN_SPEECH` and the voice rules reached the task chat through the hidden runbook. With D2 only the visible `start_chat` message reaches it. The handoff message now carries the whole plan runbook as visible user text, which is long. Alternative: plan skills (machine-setup, desktop-plugin, first-build) in the primary profile, named in a short message.
9. **Task-chat progress and check-ins.** `::onboarding{step="progress"}` and the `[setup] checkpoint` notes after 8 and 20 tool calls have no replacement.
10. **Interface first task.** It needs the app-level desktop plugin folder, which only the Electron client knows (`desktopPluginsRoot`). The current runbook names a `building-hermes-desktop-plugins` skill that does not exist in the tree.
11. **SOUL check-ins.** `SETUP_SOUL` tells the bot to look at sessions, connectors and scheduled jobs before checking in. With only `setup_choose`, `start_chat` and `apply_layout`, it cannot.
12. **One turn or many.** `setup_choose` blocks, so the whole setup can run in one agent turn. What happens on a card timeout, a closed window mid-card, or a relaunch with a pending card? The skill says: take the default and continue from the first unanswered beat.
13. **Other surfaces.** P13 runs `/initiate-setup` on CLI, TUI and messaging too, in the user's own profile with full tools. The skill falls back to plain-text questions and starts the task in the same chat when `start_chat` is absent. Should it also skip beats that only make sense on desktop, or use the extra tools there?
14. **Size.** `SKILL.md` is about 33k characters and ~310 lines, above the ~200-line guide and above today's 27k runbook, because nothing was cut and the task-chat runbooks moved in. Sid plans to trim.
15. **Host facts gaps.** Linux has no home birth time in `os.stat`, so a fresh Linux machine never leads with machine setup (Electron used `statx`). Name and locale are read in the skill script, not in `hermes_platform.host`, which covers hardware only. Should `hermes_platform.host` grow a birth-time helper and account facts?
