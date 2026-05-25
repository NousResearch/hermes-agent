# Idea Workflow Interview Question Bank

Use this bank to ask better questions during idea capture. Ask one question at a time. Prefer the smallest question that unlocks the next useful artifact.

## Mode selection

- "Do you want this as a quick Lite capture, or should I take it through the Full build-ready workflow?"
- "Is this something you may actually want built soon, or just an idea to save?"
- "Should I optimize for speed or completeness?"

## Lite mode: fast capture questions

Use 3-5 of these, then draft.

1. "What should we call this idea for now?"
2. "What is the one-sentence version?"
3. "Who is this for?"
4. "What problem does it solve?"
5. "What is the main thing it should do?"
6. "What would make it feel useful enough to keep?"
7. "What should it definitely not become?"
8. "Do you want me to save this as a rough note now?"

## Full mode: product philosophy

- "What belief or frustration is behind this idea?"
- "What should this product make easier, calmer, faster, safer, or more fun?"
- "If this works perfectly, what changes for the user?"
- "What should the product refuse to do, even if competitors do it?"
- "Should this feel like a power tool, an assistant, an overview page, a notebook, a game, or something else?"
- "What is the taste level: minimal, playful, professional, dense, visual, conversational, etc.?"

## User and audience

- "Who is the first real user?"
- "Is this for you personally, a small group, a team, or a public audience?"
- "What does the user already do today instead?"
- "What is annoying, slow, fragile, or scattered about the current workflow?"
- "How technical is the first user?"
- "What does the user already know when they arrive?"

## Core job and user flows

- "What is the first job this product must do well?"
- "Walk me through the main user flow from opening the app to getting value."
- "What is the smallest successful session?"
- "What should happen the first time a new user opens it?"
- "What should happen on a normal repeat visit?"
- "What action should always be one click or one command away?"
- "What should the user never have to leave the product to do?"

## Scope and MVP boundaries

- "What is absolutely required for version one?"
- "What sounds exciting but should wait?"
- "What feature would make the project too big too early?"
- "What can be manual behind the scenes for v1?"
- "What can be mocked or stubbed for the first prototype?"
- "What is the smallest version that would still prove the idea?"

## UX and information architecture

- "What are the main screens or areas?"
- "Should the product organize around people, projects, tasks, time, files, objects, or something else?"
- "What should be visible at a glance?"
- "What details can be hidden until needed?"
- "Where should the user take action?"
- "What should the empty state teach or invite?"
- "What would make the interface feel trustworthy?"

## Data, integrations, and platform needs

Ask these after product direction is clear.

- "What data does the product need to store?"
- "Where should that data live: local-only, self-hosted, Cloudflare, AWS, another cloud, or undecided for now?"
- "If cloud-hosted, is the goal managed simplicity, low cost, portability, enterprise readiness, or something else?"
- "What data comes from outside systems?"
- "Does anything need to sync across devices or users?"
- "Are there APIs, files, email accounts, calendars, devices, CLIs, or services it must connect to?"
- "Does it need login/auth, or can it be local/single-user first?"
- "Does it need real-time updates, or is refresh/polling fine?"
- "What data would be sensitive or private?"

## Platform targets

Ask these before implementation planning so the build handoff does not assume the wrong app shape.

- "Should this be web-only, or should it also have a Windows app, Mac app, or cross-platform desktop app?"
- "Does this need a mobile app, or is responsive/mobile web enough?"
- "If mobile is needed, is iOS, Android, or both required?"
- "Does any platform need offline support, notifications, local filesystem access, menu bar/tray behavior, background processes, or native integrations?"
- "Which platform is the first MVP target, and which platforms are explicitly later?"

## App topology, auth, and secrets

- "Is this one app, or multiple pieces such as desktop app + hosted web app + API + worker?"
- "Which piece owns capture/input, storage, processing, viewing, sharing, and admin actions?"
- "Who can log in: local admin only, multiple users, public viewers with links, teams, or no login?"
- "What should public sharing allow, and what must remain admin-only?"
- "Which secrets/API keys are needed, where should they live, and which clients must never receive them?"
- "Are simple env-file credentials acceptable for an MVP, or should the default recommendation use a stronger auth/session model?"

## Technical constraints

- "Do you already have a preferred stack or hosting target?"
- "Does this need to run locally, hosted, mobile, desktop, browser-only, or some combination?"
- "Does it need offline support?"
- "What should be cheap/simple to operate?"
- "Are there existing repos, tools, scripts, or services this must fit into?"
- "What would make maintenance painful later?"

## Technical defaults: recommend, then confirm

Use this section after product direction, data location, and platform targets are clear enough. Do not ask the user to pick every technology from a blank slate. First propose sensible defaults, then ask what they want to accept or change.

Recommended prompt:

"Based on the idea so far, my default technical recommendation is: <short stack summary>. Do you want to accept this as the default, or change any part of it?"

Cover these areas as needed:

- "Database/storage: I recommend <SQLite/Postgres/MySQL/DynamoDB/Cloudflare D1/KV/R2/S3/local files/no database> because <reason>. Accept or change?"
- "Backend/runtime: I recommend <Node/Python/Go/serverless/local-only/etc.> because <reason>. Accept or change?"
- "Frontend/UI: I recommend <React/Next/Vite/Svelte/native desktop/etc.> because <reason>. Accept or change?"
- "Auth/users: I recommend <no auth/local auth/magic link/OAuth/provider> because <reason>. Accept or change?"
- "Hosting/deployment: I recommend <local/Cloudflare/AWS/Vercel/Fly/self-hosted/etc.> because <reason>. Accept or change?"
- "Files/objects: I recommend <local filesystem/S3/R2/database blobs/no file storage> because <reason>. Accept or change?"
- "Background jobs/queues: I recommend <none/cron/worker queue/Cloudflare Queues/SQS/etc.> because <reason>. Accept or change?"
- "Realtime/sync: I recommend <none/polling/WebSocket/SSE/platform realtime> because <reason>. Accept or change?"
- "Search: I recommend <database search/SQLite FTS/Postgres FTS/Meilisearch/Algolia/none> because <reason>. Accept or change?"
- "Observability/logging: I recommend <console logs/local logs/Sentry/OpenTelemetry/provider logs> because <reason>. Accept or change?"
- "Testing: I recommend <unit/integration/e2e/manual demo script> because <reason>. Accept or change?"

If the user says "use your suggestions" or seems unsure, record the recommendations as accepted defaults and continue. If the user overrides any item, preserve the override explicitly in the design doc and implementation handoff.

## Quality, testing, and done criteria

- "How will we know v1 works?"
- "What should be tested manually before calling it done?"
- "What should be covered by automated tests?"
- "What failure would make the product untrustworthy?"
- "What demo should the build agent be able to run at the end?"
- "What must be true before someone can honestly say 'done'?"

## Risks and open questions

- "What is the riskiest assumption?"
- "What part is most likely to be harder than it sounds?"
- "What decision are you intentionally postponing?"
- "What could make this not worth building?"
- "What should the spec say if we do not know yet?"

## Override phrase

If the user says **GREENLIGHT NEXT STAGE**, stop asking questions and move to the next artifact. Carry unresolved issues forward under **Open Questions** or **Assumptions**.
