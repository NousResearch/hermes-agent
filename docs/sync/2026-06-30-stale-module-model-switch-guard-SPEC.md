# Spec — Recurring "gateway is running stale code / restart to switch models" error

**Status:** investigation + options (decision = Ace's call)
**Author:** Apollo · 2026-06-30
**Trigger:** Ace flagged two Discord messages that "keep appearing":

- `1521621285002674317` (2026-06-30 20:59) — running `eba89180b7`, disk `19cccaf841`
- `1521674808218226759` (2026-07-01 00:32) — running `bb701866fa`, disk `04f94c35fb`

Both fired on `/model claude-app/claude-opus-4-8`.

---

## 1. What the message actually is

It is **not** an error in the classic sense — it is a deliberate **safety guard**
that refuses a *model switch* when the gateway process is running older code than
what is currently on disk.

Path: `gateway/slash_commands.py::_model_switch_skew_guard()` →
`gateway/code_skew.py::detect_code_skew()`.

Mechanism:

1. At gateway boot, `record_boot_fingerprint()` (called once in
   `gateway/run.py:~19332`) snapshots the checkout revision into the module-global
   `_boot_fingerprint` (a `git:<ref>:<sha>` string read cheaply from `.git`, no
   subprocess — `hermes_cli/main.py::_read_git_revision_fingerprint`).
2. On **every `/model` switch**, `detect_code_skew()` re-reads the *current* disk
   fingerprint and compares. If `disk != boot`, it returns `(boot_rev, disk_rev)`
   and the switch is refused with the message Ace sees.
3. Scope is intentionally narrow: **only `/model`** is guarded, because a model
   switch is the known highest-risk trigger for a first-time lazy import landing a
   freshly-pulled consumer module against a stale cached dependency → the cryptic
   `cannot import name 'env_float' from 'utils'` class of crash. (See
   `tests/test_stale_utils_module_import.py`.)

So the guard is **working as designed**. The question is not "why is it broken" —
it's "why does the precondition (disk moved out from under a long-lived gateway)
keep happening."

## 2. Why it keeps recurring (root cause)

All fleet gateways run from **one shared editable checkout**:
`~/.hermes/hermes-agent` (venv `-e` install; every `ai.hermes.gateway-*` plist's
`ProgramArguments` points python at `…/hermes-agent/venv/bin/python -m
hermes_cli.main`). A gateway loads its Python modules into memory at boot and holds
them for its whole lifetime.

Meanwhile that same working tree is **the active development tree**. Over the two
timestamps Ace flagged, the live tree's `HEAD` moved repeatedly (git reflog shows
~20 HEAD moves on 2026-06-30 alone): feature-branch checkouts, `fork/main`
fast-forwards, and the per-job-cron / mem0-e2e work all landing on `main`. Each
disk move re-arms the guard. The next `/model` Ace types after any disk move trips
it.

Concretely, the two instances:

| Time | boot sha (gateway) | disk sha (tree) | what moved the disk |
|------|--------------------|-----------------|---------------------|
| 20:59 | `eba89180b` | `19cccaf84` | cron per-job-fallback + sync work landed after that gateway booted |
| 00:32 | `bb70186 6f` | `04f94c35f` | mem0-e2e / cron-fallback feature commits landed after boot |

**This is a self-inflicted, expected consequence of doing active dev in the same
checkout the live gateways run from.** It is not a code bug and not a merge
regression. It will keep happening as long as (a) gateways are long-lived and
(b) the shared tree keeps moving under them without a gateway restart.

## 3. Contributing factor — the honest caveat

The guard is a *coarse* signal: it fires on **any** disk-sha change, even a change
that touches zero modules the model-switch path will lazily import (a docs edit, a
locale file, a cron test). So a fraction of these refusals are **false alarms** —
the switch would have been perfectly safe. The guard trades precision for safety
(a false "restart me" is cheap; a stale-import crash mid-conversation is not), and
that trade is defensible — but it is why Ace sees it "keep appearing" even on
innocuous disk moves.

## 4. Options (Ace decides — this is a taste/intent call, not labor)

### Option A — Operational discipline only (no code change)
Stop doing dev commits in the live `~/.hermes/hermes-agent` tree; do sync/feature
work in an isolated worktree (as this very parity merge does) and only advance the
live tree at a deliberate deploy point, immediately followed by a gateway restart
(`safe-gateway-restart` skill). The guard then only ever fires in the brief window
between "disk advanced" and "gateway restarted," which is expected and correct.
- **Pro:** zero code risk; the guard stays maximally safe.
- **Con:** relies on operator discipline; doesn't help the moments Ace *has* pulled
  and simply hasn't restarted yet — he still gets refused.

### Option B — Make the guard precise (fire only on real risk)
Narrow `detect_code_skew()` from "any sha differs" to "a **Python module under the
import roots** differs since boot." Compute the boot fingerprint over the mtimes/
hashes of the `*.py` tree (or `git diff --name-only <boot>..<disk> -- '*.py'`
restricted to the packages the gateway imports), so a docs/locale/test-only disk
move no longer trips it.
- **Pro:** kills the false-alarm class; Ace only sees it when a switch is genuinely
  unsafe. Contract-testable (assert a docs-only delta → no skew; a `utils.py` delta
  → skew).
- **Con:** more logic in a safety path; must stay conservative (any doubt → still
  refuse). A `git diff` per `/model` is cheap but not free.

### Option C — Auto-heal instead of refuse (highest effort, matches autonomy doctrine)
On detecting skew at `/model`, don't just refuse — **schedule a safe self-restart**
(the Option-A/Option-B restart machinery already exists: per-session drain + resume)
so the gateway reloads disk code and the model switch proceeds after the bounce,
with a single "reloaded to `<sha>`, switching now" message instead of a dead-end
"go restart me yourself."
- **Pro:** removes the human-in-the-loop chore entirely — the system self-resolves
  the stuck state, which is exactly the fleet autonomy doctrine (page a human only
  for a *decision*, not for labor a machine can do).
- **Con:** a `/model` that silently triggers a gateway restart is a bigger behavior
  change; must preserve the in-flight session (drain+resume) and must not loop if
  the restart itself doesn't advance the sha. Needs the most care + proof.

### Recommendation
Ace tends to descope / keep a human in the loop for irreversible-ish behavior, and
he values the guard's safety. Suggested split:
- Adopt **Option A** immediately as standing practice (it's already how this parity
  merge is being run — isolated worktree, deploy-point restart).
- If the false-alarm noise still annoys, do **Option B** (precise skew) as a small,
  well-tested upstream-shaped change — it's the surgical fix with the best
  safety/precision trade.
- Treat **Option C** as a separate, later autonomy upgrade only if Ace wants the
  gateway to self-heal the switch rather than be told to restart.

## 5. Scope / non-goals
- Not changing the fact that a model switch on genuinely-stale code should be
  blocked — that guard's *purpose* is correct and stays.
- Not touching the boot-fingerprint recording site.
- Any change lands via the fork PR flow (isolated worktree → PR → CI green → Ace's
  deploy gate), never hot-patched into the live tree.

## 6. Ground-truth references (for whoever implements)
- Guard entry: `gateway/slash_commands.py::_model_switch_skew_guard`
- Skew detector + boot snapshot: `gateway/code_skew.py`
- Fingerprint reader (worktree-aware): `hermes_cli/main.py::_read_git_revision_fingerprint`
- Boot-record call site: `gateway/run.py` (search `record_boot_fingerprint`)
- Existing regression for the crash it prevents: `tests/test_stale_utils_module_import.py`
- Deploy/restart machinery to reuse for Option C: `safe-gateway-restart` skill +
  the per-session drain/resume in `gateway/run.py`.
