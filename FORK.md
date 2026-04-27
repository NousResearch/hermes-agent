# pebble-tech/hermes-agent

Fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) used as the deploy target for [pebble-tech/nudge-hermes-agents](https://github.com/pebble-tech/nudge-hermes-agents) customer VPSes. `main` is a runtime-ready integration branch — clone it and it works.

> **As of 2026-04-27:** one open upstream PR — [NousResearch/hermes-agent#16445](https://github.com/NousResearch/hermes-agent/pull/16445) — carried via `feat/whatsapp-forward-owner-messages-upstream`. Once merged, drop it from `FEATURE_BRANCHES` and from the fork. Otherwise `main = upstream/main + ops-overlay + open feature branches`. The fork stays alive as a stable deploy target so customer VPSes pull from a known SHA on our schedule rather than racing upstream.

## Branches

| Branch | Contents | Rebased daily | Use |
|---|---|---|---|
| `main` | Rebuilt each sync run as: upstream `main` + `ops-overlay` + every open `FEATURE_BRANCHES` entry. | yes (rebuilt) | **Deploy from here.** `git clone && git checkout main`. |
| `ops-overlay` | Upstream `main` + one commit containing `.github/workflows/sync-upstream.yml` and this `FORK.md`. | yes | Source of truth for fork-level ops. |

Other branches and tags are inherited from the initial fork and are **not** maintained — they will drift over time. Only the two branches above are rebased by the sync workflow.

## How the fork stays current

A scheduled GitHub Actions workflow ([`.github/workflows/sync-upstream.yml`](.github/workflows/sync-upstream.yml)) runs daily at 03:00 UTC and on manual dispatch. Each run:

1. Fetches `NousResearch/hermes-agent:main`.
2. Rebases `ops-overlay` onto `upstream/main`.
3. Rebases each branch in `FEATURE_BRANCHES` onto `upstream/main`. (Currently empty — see history below.)
4. Rebuilds `main` from scratch: checks out `upstream/main`, then cherry-picks `ops-overlay` followed by each feature branch's commits.
5. Installs `hermes-agent` from the rebuilt `main` via `uv`.
6. Runs focused tests:
   - `tests/gateway/test_pre_gateway_dispatch.py` (regression for the hook contract our gateway-policy plugin depends on)
   - `tests/gateway/test_session.py` (regression for WhatsApp canonical session keying)
   - `tests/hermes_cli/test_plugins.py`
7. Clones `pebble-tech/hermes-plugin-gateway-policy` and runs its test suite (smoke-test catches plugin-side drift).
8. On success → `push --force-with-lease` every source branch and `main`.
9. On **any** failure → opens (or comments on) a `sync-failure`-labelled issue with the run URL and upstream SHA. Never force-pushes a broken state.

### Why focused tests, not the full suite

The full upstream `pytest tests/` would catch noise unrelated to our deploy target (flaky tests, missing env, etc.) and slow the feedback loop. The focused tests cover the contracts our customer plugins depend on: the `pre_gateway_dispatch` hook firing, WhatsApp session keying staying canonical, plugin manifests loading.

## Adding a new fork patch

1. Branch off `upstream/main`, make the commit, open an upstream PR.
2. Push the branch to this fork as `feature/<name>`.
3. Add `feature/<name>` to the `FEATURE_BRANCHES` env var in the sync workflow. Commit via `ops-overlay`.
4. Next scheduled run (or manual dispatch) folds it into `main`.

When an upstream PR merges, drop its branch from `FEATURE_BRANCHES` via an `ops-overlay` commit and delete the branch from the fork. The cherry-pick becomes a no-op once upstream absorbs the change.

## Manually recovering from a sync failure

The workflow files an issue when it can't complete. Fix locally:

```bash
git fetch upstream
# Identify the failing source branch from the issue body, then:
git checkout <branch>              # ops-overlay or feature/*
git rebase upstream/main           # resolve conflicts
python -m pytest tests/gateway/test_pre_gateway_dispatch.py \
                 tests/gateway/test_session.py \
                 tests/hermes_cli/test_plugins.py -q
git push --force-with-lease pebble <branch>
```

Close the open `sync-failure` issue. The next scheduled run will rebuild `main` on top of the healthy source branches.

## Consuming this fork

### As a runtime / deploy target

```bash
git clone https://github.com/pebble-tech/hermes-agent.git
cd hermes-agent   # main (default branch) = integration build
```

Or install directly via pip:

```bash
pip install "git+https://github.com/pebble-tech/hermes-agent.git@main#egg=hermes-agent[all]"
```

### As a Hermes Agent contributor

Use `NousResearch/hermes-agent` directly. This fork is purely a stable deploy target for our VPSes; it carries no patches that aren't either already upstream or on their way upstream.

## Upstream PR history

| PR | Branch | Status |
|---|---|---|
| [NousResearch/hermes-agent#13445](https://github.com/NousResearch/hermes-agent/pull/13445) | `feature/pre-gateway-dispatch` (deleted) | Merged via [#15050](https://github.com/NousResearch/hermes-agent/pull/15050) on 2026-04-24 |
| [NousResearch/hermes-agent#14904](https://github.com/NousResearch/hermes-agent/pull/14904) | `feature/whatsapp-dm-canonical-session-key` (deleted) | Merged via [#15191](https://github.com/NousResearch/hermes-agent/pull/15191) on 2026-04-24, plus an upstream follow-up that extracted the JID/LID helpers into `gateway/whatsapp_identity.py`. |
| [NousResearch/hermes-agent#16445](https://github.com/NousResearch/hermes-agent/pull/16445) | `feat/whatsapp-forward-owner-messages-upstream` | **Open.** Opt-in `WHATSAPP_FORWARD_OWNER_MESSAGES` flag + `MessageEvent.metadata["whatsapp_from_owner"]` propagation. Default off, no behavior change. |
