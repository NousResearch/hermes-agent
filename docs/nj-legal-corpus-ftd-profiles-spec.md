# NJ Legal Corpus Autonomous FTD Profiles Spec

Status: Phase 0 implementation/specification.
Owner chain: Benjamin -> John Galt/default profile -> component FTD profiles.
Target repo: `/Users/johngalt/Projects/nj-legal-corpus` (`Enragedsaturday/nj-legal-corpus`, private).

## 1. Goal

Run two separate autonomous Hermes Full-Time Developer profiles:

1. `nj-case-law-ftd` — NJ legal corpus case-law engine.
2. `nj-statutes-ftd` — NJ legal corpus statutes pipeline.

Each profile is a long-lived autonomous FTD project manager. Each has isolated Hermes state, identity, memory bank, gateway token, sessions, logs, and work area, while coordinating inside the same private monorepo.

The default John Galt profile remains superior in the chain of command. Component FTDs notify Galt at major checkpoints. Galt decides merge timing and main-branch integration.

## 2. Verified Hermes capabilities

### 2.1 Profiles isolate runtime state

Hermes profiles use `HERMES_HOME` as the profile root. Named profiles live under:

```text
~/.hermes/profiles/<profile>/
```

Each profile owns its own:

- `config.yaml`
- `.env`
- `SOUL.md`
- `memories/`
- `sessions/`
- `skills/`
- `cron/`
- state DB / gateway PID / logs

Hermes docs confirm profile commands such as:

```bash
hermes profile create coder
coder chat
coder gateway start
coder gateway install
```

and describe separate gateway processes and separate profile `.env` files.

### 2.2 Gateway profile/token isolation

Verified in code:

- `hermes -p <profile> ...` resolves the profile with `hermes_cli.profiles.resolve_profile_env()` and sets `HERMES_HOME` before loading `.env` and gateway config.
- `gateway.config.load_gateway_config()` loads the active profile's `config.yaml`, then `_apply_env_overrides()` reads `DISCORD_BOT_TOKEN`, `DISCORD_HOME_CHANNEL`, and related Discord env vars.
- `gateway.platforms.discord.DiscordAdapter.start()` acquires a scoped lock using scope `discord-bot-token` and the configured token.
- `gateway.status.acquire_scoped_lock()` prevents two local gateway processes from using the same external identity simultaneously.

Implication: per-profile Discord isolation is supported, but each running Discord gateway profile must use a distinct Discord bot token. Reusing the default profile token is wrong; the second gateway will be blocked or Discord-side behavior will be undefined.

### 2.3 Kanban assigned-profile routing

Verified and strengthened with a targeted test:

- Kanban dispatcher `_default_spawn()` spawns assigned tasks with profile routing via `-p <assignee>` and sets `HERMES_PROFILE=<assignee>`.
- Added test assertion in `tests/hermes_cli/test_kanban_boards.py::TestWorkerSpawnEnv::test_default_spawn_sets_env_vars` proving the command contains `-p teknium` and env has `HERMES_PROFILE=teknium`.

This means a task assigned to `nj-case-law-ftd` or `nj-statutes-ftd` should execute under that profile, not under default Galt.

### 2.4 FTD PM direct-spawn profile routing

The existing FTD scripts needed a small patch. Directly spawned FTD PM runners now support profile selection.

Patched files under `~/.hermes/scripts/`:

- `ftd_lib.py`
  - added `read_config_text()` and `normalize_profile_name()` helpers
  - `spawn_pm_runner()` now reads/stores `pm_profile`
  - runner command now includes `-p <pm_profile> chat`
  - runner env now includes `HERMES_PROFILE=<pm_profile>` and profile-resolved env when available
  - PM prompt no longer hardcodes "You are John Galt"; it names the active Hermes PM profile as authoritative
- `ftd_start.py`
  - added `--pm-profile`
  - reads `.fulltime-dev/config.yaml` key `pm_profile` when CLI arg is absent
  - persists and prints PM profile
- `~/.hermes/scripts/tests/test_ftd_control_plane.py`
  - added test proving `spawn_pm_runner()` launches `-p ftd-benjamin-graham chat` and sets `HERMES_PROFILE`

Tests passed: `21 passed` for `~/.hermes/scripts/tests/test_ftd_control_plane.py`.

### 2.5 Hindsight memory-bank isolation

Verified in code without exercising local Hindsight/Ollama load:

- Hindsight provider loads config from `$HERMES_HOME/hindsight/config.json` first, then legacy `~/.hindsight/config.json`, then env.
- Hindsight supports `bank_id_template` with placeholders, and existing tests prove that mechanism.
- For these two production profiles, use **static `bank_id` per profile**, not a template. Static IDs avoid ambiguity around `agent_identity` during Kanban-spawned or direct-spawned runs.

Expected banks:

- `nj-legal-corpus-nj-case-law-ftd`
- `nj-legal-corpus-nj-statutes-ftd`
- default Galt remains separate (`galt-default` or current configured default)

Pre-flight only; do not run full Hindsight smoke tests casually:

```bash
curl -fsS http://127.0.0.1:9177/version
lsof -nP -iTCP:9177 -sTCP:LISTEN
```

Acceptable bind is localhost, e.g. `127.0.0.1:9177`, not public `*:9177`. Local Hindsight can be slow and uses a dedicated local Ollama path; keep concurrency bounded and avoid unnecessary retain/reflect tests.

## 3. Profile design

### 3.1 Profile names

Use lowercase Hermes-safe profile IDs:

```text
nj-case-law-ftd
nj-statutes-ftd
```

Avoid generic names like `case-law` or `statutes`; they do not carry enough ownership/context in logs, Kanban, memory banks, and Discord bot identities.

### 3.2 Worktrees and branches

Use separate git worktrees, not a shared checkout.

Reason: the repo is a shared monorepo, and these agents will run asynchronously. A single checkout creates avoidable working-tree conflicts. Separate worktrees isolate uncommitted files, branch state, test artifacts, and agent scratch work.

Actual layout used after setup:

```text
/Users/johngalt/Projects/nj-legal-corpus                 # canonical checkout/main integration
/Users/johngalt/Projects/nj-legal-corpus-case-engine     # case-law worktree
/Users/johngalt/Projects/nj-legal-corpus-statutes        # statutes worktree
```

Actual branches:

```text
ftd/case-law-engine
ftd/statutes
```

Galt owns final merge to `main`.

### 3.3 Shared infrastructure rule

Only one shared local infrastructure stack should run at a time.

Policy:

- Component FTDs may start required local services only when absent.
- If a service is already healthy and compatible with the next sprint, leave it running.
- Do not restart shared resources between sprints unless config changed, state is corrupt, or health checks fail.
- If both profiles need the same DB/search/cache service, it remains shared and long-lived.
- Profile-specific dev servers are allowed only when port-isolated and necessary.
- Galt resolves infrastructure contention.

Each profile must include this rule in `SOUL.md` / project instructions.

## 4. Manual steps Benjamin must do

These require human credentials/Discord ownership. I should not fabricate them or print secrets.

### 4.1 Create two Discord bot applications

In the Discord Developer Portal, create two bot applications:

1. `Galt NJ Case Law FTD`
2. `Galt NJ Statutes FTD`

For each bot:

- enable required gateway intents: Message Content, DM messages, guild/server messages, and voice states. Server Members is required only if allowlists use usernames or role lookup instead of numeric IDs.
- generate/copy its bot token
- invite it to `Galt's Gulch` with the minimum bot permissions required for reading/responding in the chosen channels/threads
- do not reuse the default Galt bot token
- do not paste tokens into Discord/chat; put them directly into profile `.env` files

External basis: Discord's bot documentation says a bot user is added from an application's Bot tab, and the bot is invited using an OAuth2 URL with bot scope and required permissions. Bot tokens are credentials; security guidance consistently says to store them in environment variables/secrets and never commit/share them.

### 4.2 Decide profile Discord surfaces

Actual Discord layout:

- category: `nj-legal-corpus-ftd`
- control channel: `nj-ftd-control`
- case-law home: `nj-case-law-ftd-home`
- statutes home: `nj-statutes-ftd-home`
- shared infra: `nj-ftd-shared-infra`
- threads: `case-law-checkpoints`, `case-law-questions`, `statutes-checkpoints`, `statutes-questions`, `merge-review-and-approvals`, `shared-stack-status`

Set `DISCORD_HOME_CHANNEL` per profile to its durable home channel. Do not pin the gateway home to a thread by default; use threads for checkpoint/question substreams.

## 5. Implementation procedure

### 5.1 Create profiles

Run from the default John Galt profile; verify first:

```bash
hermes profile list
```

Then create profiles:

```bash
hermes profile create nj-case-law-ftd --clone
hermes profile create nj-statutes-ftd --clone
```

`--clone` copies `config.yaml`, `.env`, `SOUL.md`, and selected memory files from the source profile. Immediately neutralize inherited gateway tokens before any gateway command can run:

```bash
python3 - <<'PY'
from pathlib import Path
for profile in ['nj-case-law-ftd', 'nj-statutes-ftd']:
    env = Path.home()/'.hermes'/'profiles'/profile/'.env'
    lines = env.read_text().splitlines() if env.exists() else []
    out = []
    seen = False
    for line in lines:
        if line.startswith('DISCORD_BOT_TOKEN='):
            out.append('DISCORD_BOT_TOKEN=')
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append('DISCORD_BOT_TOKEN=')
    env.write_text('\n'.join(out) + '\n')
    print(profile, 'DISCORD_BOT_TOKEN cleared')
PY
```

Then edit identity/config in each profile. Do not leave cloned `SOUL.md` as generic Galt. Inherited `USER.md` may remain if it only describes Benjamin; remove Galt-specific operational instructions from component profile memories if present.

### 5.2 Add profile `.env` secrets manually

Case-law profile:

```bash
$EDITOR ~/.hermes/profiles/nj-case-law-ftd/.env
```

Required entries, with real values inserted manually:

```dotenv
DISCORD_BOT_TOKEN=<case-law-discord-bot-token>
DISCORD_HOME_CHANNEL=<coordination-channel-id>
DISCORD_HOME_CHANNEL_THREAD_ID=<optional-case-law-thread-id>
```

Statutes profile:

```bash
$EDITOR ~/.hermes/profiles/nj-statutes-ftd/.env
```

```dotenv
DISCORD_BOT_TOKEN=<statutes-discord-bot-token>
DISCORD_HOME_CHANNEL=<coordination-channel-id>
DISCORD_HOME_CHANNEL_THREAD_ID=<optional-statutes-thread-id>
```

### 5.3 Configure profile working directories

Use explicit profile flags instead of relying on wrapper aliases:

```bash
hermes -p nj-case-law-ftd config set terminal.cwd /Users/johngalt/Projects/nj-legal-corpus-worktrees/case-law
hermes -p nj-statutes-ftd config set terminal.cwd /Users/johngalt/Projects/nj-legal-corpus-worktrees/statutes
```

Actual setup used the existing clean worktrees instead:

```bash
hermes -p nj-case-law-ftd config set terminal.cwd /Users/johngalt/Projects/nj-legal-corpus-case-engine
hermes -p nj-statutes-ftd config set terminal.cwd /Users/johngalt/Projects/nj-legal-corpus-statutes
```

### 5.4 Create worktrees

From canonical repo:

```bash
cd /Users/johngalt/Projects/nj-legal-corpus
mkdir -p /Users/johngalt/Projects/nj-legal-corpus-worktrees
git fetch origin
git worktree add -b ftd/nj-case-law-engine /Users/johngalt/Projects/nj-legal-corpus-worktrees/case-law main
git worktree add -b ftd/nj-statutes /Users/johngalt/Projects/nj-legal-corpus-worktrees/statutes main
```

If `main` is not the canonical integration branch, substitute the actual base branch after checking. Lowercase `-b` is deliberate: it fails if a branch already exists. Do not use uppercase `-B` unless you intentionally want to reset an existing branch to the base commit.

### 5.5 Configure Hindsight per profile

Create profile-scoped Hindsight configs:

```bash
mkdir -p ~/.hermes/profiles/nj-case-law-ftd/hindsight
mkdir -p ~/.hermes/profiles/nj-statutes-ftd/hindsight
```

Write `config.json` for each. Use the shared local Hindsight API, but distinct static banks.

For case law:

```json
{
  "mode": "local_external",
  "api_url": "http://127.0.0.1:9177",
  "bank_id": "nj-legal-corpus-nj-case-law-ftd",
  "budget": "mid",
  "recall_budget": "mid",
  "memory_mode": "hybrid",
  "retain_tags": ["source_system:hermes-agent", "project:nj-legal-corpus", "component:case-law"],
  "recall_tags": ["project:nj-legal-corpus", "component:case-law"],
  "recall_tags_match": "all",
  "retain_async": true,
  "timeout": 360
}
```

For statutes, same but `component:statutes` and bank `nj-legal-corpus-nj-statutes-ftd`.

Use static `bank_id` per profile to avoid any ambiguity about `agent_identity` naming. Before writing these files, verify the shared local API is already up: `curl -fsS http://127.0.0.1:9177/version`.

### 5.6 Configure FTD project PM profile

In each worktree, add/update `.fulltime-dev/config.yaml`:

Case-law worktree:

```yaml
pm_profile: nj-case-law-ftd
max_sprints: 1
```

Statutes worktree:

```yaml
pm_profile: nj-statutes-ftd
max_sprints: 1
```

`max_sprints: 1` is a first-launch safety setting, not the target steady state. After the first supervised checkpoint proves routing/gateway/memory behavior, raise it or restart explicitly for subsequent sprints. `pm_profile` is now consumed by `ftd_start.py` / `ftd_lib.py`.

### 5.7 Install/start gateways

After tokens are present:

```bash
hermes -p nj-case-law-ftd gateway install
hermes -p nj-statutes-ftd gateway install
hermes -p nj-case-law-ftd gateway start
hermes -p nj-statutes-ftd gateway start
hermes gateway list
```

If running manually instead of services:

```bash
hermes -p nj-case-law-ftd gateway run
hermes -p nj-statutes-ftd gateway run
```

Do not start either gateway until its `.env` has a distinct bot token.

### 5.8 Start FTD loops

Canonical start command for Phase 0 is the patched lower-level script. It takes the repo path as a positional argument:

```bash
python3 ~/.hermes/scripts/ftd_start.py /Users/johngalt/Projects/nj-legal-corpus-worktrees/case-law --pm-profile nj-case-law-ftd
python3 ~/.hermes/scripts/ftd_start.py /Users/johngalt/Projects/nj-legal-corpus-worktrees/statutes --pm-profile nj-statutes-ftd
```

It spawns the PM runner and returns after initialization. Use `~/.hermes/scripts/ftd_status.py` / watchdog state for liveness, and do not start a second runner when state is `ACTIVE`.

## 6. Permissions and MCP

### 6.1 Default tool posture

Each FTD profile should have:

- terminal/file/search/git tools for local repo work
- browser/dogfood tools only when needed for UI/API verification
- GitHub CLI access only if the profile needs issues/PR status
- no cloud processing of sensitive identifiers
- no push-to-main authority
- no package installation without Galt/user approval unless project policy explicitly permits it

### 6.2 MCP

Do not add new MCP servers blindly.

Allowed after explicit setup:

- GitHub MCP/gh for issue/PR metadata if needed
- local-only DB/search MCP for project infrastructure if it does not expose sensitive data outside localhost
- filesystem/local repo access scoped to the profile worktree

Rejected by default:

- broad cloud-memory MCP with case/statute content
- remote DB MCP carrying CJIS/sensitive identifiers
- MCP servers that duplicate already-running shared infrastructure

## 7. Coordination protocol

### 7.0 Kanban ownership

Each FTD worktree gets its own FTD board created by `ftd_start.py` from the repo path/project ID. The board DB is owned by the dispatching FTD control plane and is passed into spawned workers through `HERMES_KANBAN_DB`; this is intended. Do not manually create duplicate boards in the component profiles unless Galt explicitly decides to split the control plane.

For cross-profile coordination, Galt/default maintains the integration view: component FTDs report checkpoint summaries to Discord and their own board; Galt creates any cross-component integration tasks and assigns them explicitly. The component FTDs should not create tasks for each other except to request coordination.


### 7.1 Component boundaries

Case-law FTD owns:

- case-law ingestion/enrichment/search/ranking/evaluation
- CourtListener enrichment only where already in scope
- case-law engine tests and dogfooding

Statutes FTD owns:

- statutes ingestion/parsing/versioning/search/evaluation
- statute-specific tests and dogfooding

Shared files require coordination:

- repo-level config
- shared schema/migrations
- shared API contracts
- shared frontend/search UX
- shared infrastructure scripts

### 7.2 Merge protocol

A checkpoint means a coherent, tested unit of work with: summary, changed files, tests run, known failures, infrastructure state, and next recommended action. Notification mechanism is Discord via the profile gateway to the configured home thread/channel, plus the FTD board task status/summary.

1. FTD reaches checkpoint and notifies Galt.
2. Galt reviews diff, tests, and risk.
3. Galt may request fixes from the component profile.
4. Galt integrates to main only when the checkpoint is coherent and passing.
5. If case-law/statutes branches conflict, Galt does the integration pass, not either component FTD unilaterally.

## 8. Component profile `SOUL.md` minimum template

Each component profile's `SOUL.md` must be rewritten after cloning. Minimum content:

```markdown
# Identity
You are the NJ Legal Corpus <Case Law Engine|Statutes> Full-Time Developer profile.
You are subordinate to John Galt/default. Benjamin is the project owner.

# Scope
Own only <case-law engine|statutes pipeline> work in `/Users/johngalt/Projects/nj-legal-corpus`.
Coordinate before touching shared schema, repo-wide config, shared API contracts, or infrastructure scripts.

# Operating rules
- Use your assigned worktree and branch only.
- Do not merge to `main`; Galt performs integration.
- Do not push to protected branches.
- Do not install packages or restart shared infrastructure unless required and allowed by project config or Galt.
- Keep one shared infrastructure stack running when healthy; do not thrash services between sprints.
- At checkpoints, report summary, tests, failures, infra state, and requested Galt decision.
```

Branch protection on `main` is the technical enforcement for no-push-to-main. If branch protection is absent, this is only prompt/process control and should be treated as a material risk.

## 9. Verification checklist

Before declaring ready:

```bash
hermes profile list
python3 - <<'PY'
from pathlib import Path
for profile in ['nj-case-law-ftd', 'nj-statutes-ftd']:
    cfg = Path.home()/'.hermes'/'profiles'/profile/'config.yaml'
    print('==', profile, '==')
    in_terminal = False
    for line in cfg.read_text().splitlines():
        if line.startswith('terminal:'):
            in_terminal = True
            continue
        if in_terminal and line and not line.startswith(' '):
            break
        if in_terminal and 'cwd:' in line:
            print(line.strip())
PY
```

Check token presence without printing values:

```bash
python3 - <<'PY'
from pathlib import Path
for profile in ['nj-case-law-ftd', 'nj-statutes-ftd']:
    env = Path.home()/'.hermes'/'profiles'/profile/'.env'
    text = env.read_text() if env.exists() else ''
    token = ''
    for line in text.splitlines():
        if line.startswith('DISCORD_BOT_TOKEN='):
            token = line.split('=', 1)[1].strip()
    print(profile, 'env_exists=', env.exists(), 'discord_token_set=', bool(token) and not token.startswith('<'))
PY
```

Check gateway status:

```bash
hermes gateway list
hermes -p nj-case-law-ftd gateway status --deep
hermes -p nj-statutes-ftd gateway status --deep
```

Check worktree separation:

```bash
git -C /Users/johngalt/Projects/nj-legal-corpus worktree list
git -C /Users/johngalt/Projects/nj-legal-corpus-worktrees/case-law branch --show-current
git -C /Users/johngalt/Projects/nj-legal-corpus-worktrees/statutes branch --show-current
```

Targeted tests already run in this Phase 0 pass:

```bash
python3 -m pytest ~/.hermes/scripts/tests/test_ftd_control_plane.py -q
python3 -m pytest tests/hermes_cli/test_kanban_boards.py tests/hermes_cli/test_kanban_db.py -q -o 'addopts='
```

## 10. Remaining risks

1. Discord bot token creation is manual and cannot be bypassed safely.
2. The FTD scripts live under `~/.hermes/scripts`, outside the Hermes repo checkout. Treat them as local operational code unless moved into the repo.
3. Hindsight local API can bottleneck if both profiles aggressively retain/recall. Keep `retain_async=true`, bounded budgets, and avoid unnecessary smoke tests.
4. Worktree branch integration is a Galt responsibility; component FTDs must not merge each other or main.
5. A cloned profile may inherit too much default Galt personality/config. Rewrite `SOUL.md` and project instructions explicitly.

## 11. Recommended next action

After Benjamin creates the two Discord bots and inserts tokens, Galt should perform the profile/worktree setup and run the verification checklist before launching autonomous FTD loops.
