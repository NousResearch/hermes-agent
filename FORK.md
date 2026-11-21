# Team-Sneakymouse fork

`origin` = `Team-Sneakymouse/hermes-agent`, `upstream` = `NousResearch/hermes-agent`. We track
upstream's latest stable release and carry the smallest possible delta on top. Read this before a
sync, a deploy, or adding a new integration.

## What the fork carries

| Delta | Where | Conflicts with upstream? | Retire when |
|---|---|---|---|
| Our plugins | `custom_plugins/*`, `plugins/pocketbase/` | Never (new paths) | n/a |
| Conversation hook seam | `gateway/conversation_plugins.py` (contract), `plugins/platforms/discord/adapter_receipts.py` (Discord receipts) + call sites below | Yes — the only real conflict surface | Upstream ships equivalent admission / routing / delivery-receipt hooks |
| Discord clarify fix | `plugins/platforms/discord/adapter.py` (PR #5) | Occasionally | Upstream fixes lossless clarify options |
| Update restart hand-back fix | `hermes_cli/gateway.py::_prepare_profile_gateway_update_restart` (PR #13) | Occasionally | Upstream takes the same fix |
| Dashboard `--external-supervisor` | `hermes_cli/dashboard_procs.py::_restart_killed_backends`, `subcommands/dashboard.py`, `main_dashboard.py::_route_named_profile_dashboard` (PR #16) | Occasionally | Upstream ships an external-supervisor declaration for `dashboard`/`serve` |
| Fork CI | `.github/workflows/fork-ci.yml`, `.github/scripts/fork_sync_health.py`, `tests/scripts/test_fork_sync_health.py` | Never (new paths) | n/a |

Hook call sites in upstream files (the conflict checklist):
`gateway/platforms/base.py`, `gateway/run_turn_prepare.py`, `gateway/session.py`,
`gateway/session_recovery.py`, `gateway/slash_commands_session.py`, `hermes_cli/plugins.py`,
`hermes_cli/plugins_dispatch.py`, `hermes_cli/plugin_isolation.py`,
`plugins/platforms/discord/adapter.py`.
Contract tests: `tests/gateway/test_plugin_conversation_contract.py`,
`tests/gateway/test_discord_conversation_plugin.py`,
`tests/plugins/platforms/test_discord_delivery_receipts.py` (real discord.py against a loopback
Discord API; it lives outside `tests/gateway/` because that conftest mocks the SDK). Docs:
`website/docs/developer-guide/gateway-conversation-plugins.md`.

`git diff --stat "$(git describe --tags --abbrev=0 --match 'v*' --exclude '*+*')" -- . ':!custom_plugins' ':!plugins/pocketbase'`
(the fork against the release it last synced; needs that tag fetched) must show only the above.
Anything else that appears is an accidental fork drift; fix it.

## Weekly upstream sync

Sync to upstream's latest published release (`vX.Y.Z`), not `main`'s tip. Release tags are commits
on upstream `main` (no release branches), so a tag sync is an ordinary merge, but a release has
passed upstream's stable gate (install E2E, Docker, Nix, live Windows, packaged-transition and
acceptance lanes) that a random `main` commit hasn't; release candidates that fail it are tagged
`abandoned-rc.*`. Releases land every few days, so the lag is small. Never sync to an `rc.*`,
`abandoned-rc.*` or `+canary` tag. If we need a fix that is on `main` but not yet released, sync to
that `main` commit instead and say so in the PR; the next release sync merges cleanly over it.

Merge, never rebase: rebasing rewrites our merged PRs and throws away earlier resolutions.

```sh
TAG=$(gh -R NousResearch/hermes-agent release view --json tagName -q .tagName)   # latest stable release
git fetch origin && git fetch --no-tags upstream tag "$TAG"
git switch -c chore/sync-upstream-$TAG origin/main
git merge-tree --write-tree --name-only HEAD "$TAG" | grep CONFLICT   # preview
P='conversation_plugins|conversation_lane|gateway_message_admission|gateway_session_route|gateway_delivery_receipt|prepare_conversation_route|preceding_message_ids|report_delivery|delivery_turn|bind_delivery_session|conversation_reset_boundary|plugin_profile_scope|delivery_receipt_trace'
INV="$(mktemp -d)"
git grep -c -E "$P" -- gateway hermes_cli plugins/platforms | sort > "$INV/before.txt"
git merge --no-ff "$TAG" -m "chore(sync): merge NousResearch $TAG"
# resolve (rules below), then:
git grep -c -E "$P" -- gateway hermes_cli plugins/platforms | sort > "$INV/after.txt"
diff "$INV/before.txt" "$INV/after.txt"   # every difference must be explained
```

After the merge, disable any workflow upstream added (§ Fork CI):

```sh
gh -R Team-Sneakymouse/hermes-agent workflow list --all --limit 500 --json path,state \
  --jq '.[] | select(.state == "active") | .path | split("/") | last' |
  grep -vxE 'fork-ci\.yml|osv-scanner\.yml' |
  xargs -r -n1 gh -R Team-Sneakymouse/hermes-agent workflow disable
```

Verify, then open a PR into `origin/main` (title `chore(sync): merge NousResearch <tag> (<sha>)`).
Fork CI runs all of this on the PR; locally:

```sh
HERMES_PYTHON=.venv/bin/python scripts/run_tests.sh \
  tests/gateway/test_plugin_conversation_contract.py tests/gateway/test_discord_conversation_plugin.py \
  tests/gateway/test_discord_clarify_lossless.py tests/gateway/test_discord_clarify_buttons.py \
  tests/plugins/platforms/test_discord_delivery_receipts.py \
  tests/plugins/test_pocketbase.py tests/plugins/test_pocketbase_api_scope.py \
  tests/hermes_cli/test_gateway_external_supervisor.py tests/hermes_cli/test_dashboard_external_supervisor.py \
  tests/scripts/test_fork_sync_health.py
for d in custom_plugins/*/; do .venv/bin/python -m unittest discover -s "$d" -p 'test_*.py'; done
python scripts/check            # every guard + the health ratchet
python .github/scripts/fork_sync_health.py --base origin/main   # the health ratchet as Fork CI gates a sync
```

(Windows: `.venv/Scripts/python.exe`, with `PYTHONUTF8=1`.) A sync makes the health ratchet judge
every upstream file it brings in, which can outlast `scripts/check`'s 900s per-check cap on a slow
host, and fails on upstream's own code wherever upstream added or moved a file. Judge it as Fork CI
does, with the last command above: a sync is measured against `origin/main` and against the release,
and only findings new against both block (upstream's code is clean against the release, fork debt
against `main`). A sync to an unreleased upstream `main` commit gets the plain ratchet. GitHub can
take ~20 minutes to attach a push to a large sync PR; if Fork CI hasn't started, run it from the
Actions tab (`gh workflow run fork-ci.yml --ref <branch>`). Windows hosts also fail a few path/file-lock tests that pass on
Linux (`file:///C:/...` media paths, todo-assessment `test_gateway_handoff`); compare against
`main` in a scratch worktree before blaming the sync. Then read each `custom_plugins/*/README.md` "compatibility
fence" section and grep the merged tree for the upstream behaviour it names.

Build `.venv` once with `python -m pm.build_env --source . --out .venv --group dev --group test --extra discord`
(Python 3.14; rebuild when `uv.lock` changes). Never run the update or gateway test files on the
bot host: they scan the machine's real processes, and the live gateway is one of them. Merge the
sync PR with a merge commit, not a squash, then deploy it (§ Deploy): merging alone changes nothing
on the bot. Don't let syncs lapse: three weeks of drift was ~9.7k commits and 5 conflicts, and
production once sat two merged syncs behind.

### Fork CI

Upstream's workflows run on Blacksmith runners the fork can't use: their jobs queue for a day and
are cancelled. Rather than edit ~50 upstream files every sync, they are disabled in the repo's
Actions settings (the command above; settings aren't files, so they never conflict), except
`osv-scanner.yml`, which runs on GitHub's runners. `fork-ci.yml` runs the verification above on
every PR into `main` and every push to `main`, on a GitHub-hosted runner. It skips upstream's
own test suite on purpose: upstream already ran it, and our delta is what the contract tests
cover. The code-health ratchet runs outside `scripts/check` there, because its 900s per-check cap
is too short for a sync PR on a small runner.

### Conflict rules (learned the hard way)

- **Upstream refactors win.** Upstream constantly splits god files into `<stem>_<topic>.py`
  siblings. Take their side (`git checkout --theirs <file>`) and re-apply our one-liner where the
  code now lives (`git grep` for the surrounding function). Example: `_hmwa_resolve_session` moved
  from `run_turn.py` to `run_turn_prepare.py`, so `bind_delivery_session` moved with it.
- **Prefer one choke point over per-path wiring.** Discord receipts come from a single
  `http_trace` passed to `commands.Bot(...)`, not a call in every send method, so upstream can
  rewrite or add send paths without conflicts. If a conflict ever touches that `Bot(...)` call,
  keep the `http_trace=` kwarg.
- **Clean auto-merges can still drop a call site.** Never skip the before/after inventory diff.
- **Same name ≠ same feature.** Upstream's `gateway.session_ingress.admit_message` is queue
  durability, not our plugin admission veto. Read the code before retiring a hook.
- **Check whether each hook is still needed:** diff `VALID_HOOKS` in `hermes_cli/plugins.py` and
  skim upstream commits touching the checklist files. If upstream now covers a hook, port the
  plugin to it and delete our seam in the same sync PR. Update the table above.
- **A clean merge can still break a plugin.** Upstream tightens behaviour our plugins rely on
  without touching our files. Example: since `5018d437cc`, `ctx.inject_message(origin=...)` is
  refused for a chat with no admitted human message, which broke the todo-assessment one-click
  start. These are deliberate security boundaries: adapt the plugin, never patch core around
  them. If the plugin can't be adapted in time, disable it on the host
  (`hermes plugins disable <name>`) and mark its README, rather than holding the sync back.
  `discord-todo-assessment` has been disabled this way since the 2026-10-25 sync.
- Keep upstream's style changes (`Set[str]` → `set[str]`, string annotations dropped) in our lines.

## Deploy

Judge every upstream change against the host as it is, not as it might be:

- `ssh mothermouse@10.0.0.103`, checkout `~/.hermes/hermes-agent` on fork `main`, one `default`
  profile (no `profiles/`, so no multiplex), Discord allowlist in `.env` (`DISCORD_ALLOWED_USERS`,
  `DISCORD_ALLOWED_ROLES`), no `discord.allow_from` in `config.yaml`.
- The gateway belongs to root's `/usr/local/sbin/hermes-gateway-supervisor`, which runs
  `gateway run --external-supervisor` in a loop and restarts at once on exit 75. The dashboard
  (port 9119) belongs to root's `/usr/local/sbin/hermes-dashboard-supervisor`, which must run
  `dashboard --external-supervisor ...`; without the flag `hermes update` respawns it detached
  and the supervisor loops on exit 75 ("port in use").
- Enabled plugins include catalog plugins the fork doesn't track (`hermes-dreaming`,
  `homeassistant`, `infisical`); no fork test covers them. `discord-todo-assessment` is disabled.
- `updates.pre_update_backup` is off, and the shell-command approvals are decided by an auxiliary
  model, not a Discord button.

**Before merging the sync PR,** load the host's plugins against the merged tree in a scratch home
that holds only `config.yaml` and `plugins/` (no `.env`, no databases), from a scratch checkout of
the sync branch on the host:

```sh
S=$(mktemp -d -p ~/projects); mkdir "$S/home"; cp ~/.hermes/config.yaml "$S/home/"; cp -a ~/.hermes/plugins "$S/home/"
HERMES_HOME="$S/home" "$(ls -d ~/.hermes/tools/python-3.14*/bin/python3 | tail -1)" -m pm.build_env --source . --out "$S/venv" --group dev --extra discord
HERMES_HOME="$S/home" "$S/venv/bin/python" -c 'from hermes_cli.plugins import get_plugin_manager as g
m = g(); m.discover_and_load()
for n, p in sorted(m._plugins.items()): print(n, p.error or "ok")' | grep -v "not enabled"
rm -rf "$S"
```

Every enabled plugin must print `ok`.

**Deploy** (interrupts the bot for a minute or two; it drains running turns first, up to ~30 min):

```sh
grep -E 'inbound message|response ready' ~/.hermes/logs/gateway.log | tail -2   # idle?
git -C ~/.hermes/hermes-agent fetch origin main   # hermes update gives its fetch only 300s
hermes update --backup --yes                       # --backup: the host's automatic one is off
```

**Verify:**

- the update ends with `✓ default (pid <PID>) @ <merged sha> — up to date`;
- `ps -o ppid= -p <PID>` is the supervisor (`pgrep -f hermes-gateway-supervisor`), and
  `~/.hermes/logs/gateway-sysv.log` shows one `starting Hermes gateway` for the update, not a stream.
  A parent of `1` means `hermes update` relaunched the gateway itself (the bug PR #13 fixed): when
  the bot is idle, `kill -TERM <PID>` and the supervisor starts its own within seconds;
- `gateway.log` shows `✓ discord connected` and no `ERROR` since the restart;
- the update prints `PID <PID> handed back to its external supervisor` for the dashboard, and the
  process on port 9119 (`ss -ltnp | grep 9119`) has `hermes-dashboard-supervisor` as its parent.
  A parent of `1` means the supervisor lacks `--external-supervisor`: `kill -TERM` that PID.

Then smoke-test in Discord and read `gateway.log` for each: a clarify button and an approval button,
both in a guild channel and in a thread (use a PocketBase change for the approval; shell commands
are auto-decided); `/new` then `/resume` in a thread; optionally, a non-allowlisted user's mention
stays unanswered.

**Rollback** (not rehearsed yet): reset the checkout to the update receipt's `pre_update.sha`
(`~/.hermes/logs/update_receipts/update_*.json`), check the dependencies with `hermes pm doctor`,
restore data with `hermes import <the --backup zip>`, then `kill -TERM` the gateway so the
supervisor restarts it on the old code.

## Writing a new integration

The root `AGENTS.md` footprint ladder is upstream's and starts by extending existing code. In the fork,
where every line added to an upstream file is a future sync conflict, use this ladder instead and stop
at the first rung that works:

1. **Plugin on existing surface:** a directory in `custom_plugins/<name>/` (`plugin.yaml`,
   `__init__.py`, `README.md`, tests next to it) using existing hooks / `ctx` methods. Zero
   conflict cost. Strongly preferred. Use only documented `ctx`/hook behaviour, write an
   integration test through the real gateway path, including the runtime production builds
   (`initialize_gateway_runtime` / session authority; a bare `GatewayRunner` skips the policy
   that refused todo-assessment), since that test is the sync admission check, and list any
   behaviour you depend on that upstream might tighten under a "compatibility fence" heading in
   the README.
2. **Need a seam that doesn't exist?** Add a *generic* hook, never plugin-specific logic in core:
   - contract types and dispatch go in our own module (`gateway/conversation_plugins.py` or a new
     `gateway/<topic>_plugins.py`), never inline in upstream files;
   - each upstream file gets a 1–3 line, late-imported call into that module and nothing more;
   - register the hook in `VALID_HOOKS` (`hermes_cli/plugins.py`), in `_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS`
     (`hermes_cli/plugins_dispatch.py`) if it is a policy gate, and in `HOST_DEGRADED_HOOKS`
     (`hermes_cli/plugin_isolation.py`) if its payload can't cross the plugin host;
   - add a contract test in `tests/gateway/` that drives the real code path, and add the new call
     sites to the checklist and the inventory regex above.
3. **Upstream it if it's generic.** A seam with a working consumer (our plugin) meets upstream's
   bar ("no hooks without a concrete consumer"). Every accepted hook is one less file to resolve
   each week.

Don't patch upstream behaviour directly in a core file for a local preference. Make it a plugin
setting, or carry it as a clearly scoped fix PR that you'd also offer upstream (like PR #5).
