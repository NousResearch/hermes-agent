# AI Factory worktree admission gate

HER-95 adds a machine-wide worktree admission gate to the existing `factory_lane.py` registry layout. It does not introduce a second registry: owners still live under `registry/locks/<KEY>/owner.json`, lane journals under `registry/lanes/<KEY>.jsonl`, and all worktree conflict decisions canonicalize `realpath(worktree)`.

## Commands

### Owner claim / pre-build hard gate

```bash
python scripts/factory_lane.py \
  --registry /Users/jeanyoder/Documents/Jean-AI-Memory/Agent-Shared/AI-Factory/registry \
  admit HER-95 \
  --mode owner \
  --hard \
  --agent default \
  --profile default \
  --session <session-id> \
  --gateway-session-key <platform:chat:thread> \
  --owner-pid <long-lived-agent-pid> \
  --worktree /Users/jeanyoder/Documents/GitHub/_worktrees/hermes-her-95-worktree-admission-gate
```

Hard owner admission:
- serializes the final decision under `.worktree-admission.lock` to close preflight→claim TOCTOU races;
- refuses a second live owner for the same canonical worktree, even if the second owner uses another issue key;
- refreshes heartbeat for the same owner/session; a same-session re-claim onto a
  *different* worktree that is already owned by another lane is refused (never
  rewrites the owner, never creates two owners for one worktree);
- stores `profile` and `gateway_session_key` when supplied, but rejects a
  secret-like `gateway_session_key` (anything matching token/password/api_key/…)
  before writing owner.json — no tokens or chat secrets are ever persisted;
- records the transported **parent** process identity (`--owner-pid`, optional
  `--owner-start-time`) instead of the ephemeral `factory_lane.py` subprocess
  pid, so liveness/reclaim stays correct when the gate is driven as a subprocess;
  a dead `--owner-pid` (or a start-time that does not match it) is refused;
- refuses dirty ownerless git worktrees before a build, without resetting or deleting anything.

`--owner-pid` defaults to the running `factory_lane.py` process only when
omitted (standalone CLI use). When the gate is invoked from a launcher/gateway
subprocess, always transport the long-lived agent pid so a claim never persists a
pid that dies the instant the subprocess exits.

### Reviewer read-only admission

```bash
python scripts/factory_lane.py --registry <registry> admit HER-95 \
  --mode reviewer --agent opus-reviewer --session <session-id> \
  --worktree <worktree> --json
```

Reviewer mode is read-only: it may report the current owner, but it never creates or mutates `owner.json`.

### Advisory SessionStart

```bash
python scripts/factory_lane.py --registry <registry> hook-session-start \
  --repo <worktree> --agent hermes-immo --session <session-id>
```

If the same worktree is owned by another live session, the hook prints a bounded `STOP: worktree already owned ...` advisory. If the registry is absent or corrupt, the hook fails open with exit 0 and no output.

### Stale recovery

```bash
python scripts/factory_lane.py --registry <registry> claim HER-96 \
  --agent default --session <session-id> --worktree <worktree> \
  --reclaim-worktree --ttl-hours 2
```

Recovery only succeeds when the previous owner process is stale (`not_found`, `zombie`, or PID `reused`), heartbeat TTL is expired, and the worktree has been inactive for 24h. Otherwise the claim fails closed.

### Business-profile domain guard

For a métier profile such as `hermes-immo`, pass a bounded domain prefix set before hard owner admission:

```bash
python scripts/factory_lane.py --registry <registry> admit SCA-740 \
  --mode owner --hard --agent hermes-immo --profile hermes-immo \
  --domain-prefixes JYI,HER --session <session-id> --worktree <worktree>
```

A key outside the allowed prefixes is refused before any owner file is written. This is the canary path for generic `continue` prompts arriving in a business gateway.

## Runtime wiring — real pre-mutation gate (`pre_tool_call` hook)

The `admit` / `claim` CLI takes *ownership*; the runtime gate that actually runs
**before every build-capable tool** is `scripts/factory_admission_hook.py`, wired
through the generic shell-hook bridge Hermes already loads at startup
(`agent.shell_hooks.register_from_config(load_config(), …)`, called from
`cli.py`, `hermes_cli/main.py`, and `gateway/run.py`). No core file changes: the
gate is a declarative, opt-in `hooks:` entry in `cli-config.yaml` (profile-aware).

```yaml
# ~/.hermes/cli-config.yaml (or a profile's cli-config.yaml)
hooks:
  pre_tool_call:
    - matcher: "terminal|patch|write_file|str_replace_editor|apply_patch"
      command: >-
        python3 /ABS/scripts/factory_admission_hook.py
        --registry /ABS/registry --agent default
```

At tool time the plugin manager calls the hook with the standard shell-hook stdin
payload (`hook_event_name`, `tool_name`, `tool_input`, `session_id`, `cwd`, …).
The hook is **read-only**: it resolves every effective mutation target — explicit
terminal `workdir`, file-tool `path`/`file_path`, Codex `apply_patch`
`changes[*].path`, and path arguments in terminal commands (absolute or relative
to the session cwd), then the session `cwd` fallback — to its git top-level
before calling `factory_lane.evaluate_admission_guard(...)`. This prevents a
session launched outside a worktree from bypassing admission by targeting it
through tool arguments. Only when the guard denies, the hook prints
`{"decision": "block", "reason": "..."}`. `agent/shell_hooks.py`
translates that into the canonical `{"action": "block", "message": …}` that
`hermes_cli.plugins.get_pre_tool_call_block_message()` (the exact call site in
`model_tools.handle_function_call`) uses to veto the tool before it executes.

Guard semantics:
- worktree owned by a **different live session** → block (one winner per worktree);
- same owning session → allowed (the hook never rewrites owner.json, never
  persists the hook subprocess pid);
- **business profile out of domain** → block automatically — the
  `--profile` / `--domain-prefixes` live in the hook command line (the profile's
  `cli-config.yaml`), so the denial no longer depends on a caller remembering to
  pass flags;
- ownerless worktree or absent/corrupt registry → advisory fail-open (the gate
  only constrains genuinely admitted lanes);
- an unexpected error in the advisory hook fails open (a gate bug must not freeze
  every tool), while a *detected* conflict fails closed.

For a business profile such as `hermes-immo`, the same block is produced with the
profile's own hook line:

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal|patch|write_file|str_replace_editor|apply_patch"
      command: >-
        python3 /ABS/scripts/factory_admission_hook.py
        --registry /ABS/registry --agent hermes-immo
        --profile hermes-immo --domain-prefixes JYI,HER
```

The hook is opt-in and consent-gated exactly like any other shell hook
(allowlist + `--accept-hooks` / `HERMES_ACCEPT_HOOKS` / `hooks_auto_accept`), and
it is skipped entirely under `--safe-mode`.

## AppSec hardening (exact-head review fixes)

The two exact-head reviews' blockers are closed and covered by
`tests/test_factory_lane_appsec.py` and `tests/test_factory_lane_integration.py`:

- **Ancestor symlink swap (`registry/locks`, `registry/lanes`).** Every
  claim/admit write descends the path with `openat(O_NOFOLLOW|O_DIRECTORY)` from a
  registry-root fd, re-validated per write, so a swapped ancestor fails the
  `openat` (`ELOOP`/`ENOTDIR`) and the write can never land outside the registry.
  On platforms without `renameat(dir_fd)` (macOS), the atomic replace re-checks
  `(st_dev, st_ino)` of the open fd against the textual path before renaming, and
  the temp file lives in the real directory so a late swap fails the rename.
- **Same-session rebind** onto an already-owned worktree is refused (guarded
  before any owner rewrite).
- **Secret-like `gateway_session_key`** is validated and rejected before write.
- **`process_start_time=None`** is classified `alive`, never `reused`, so a live
  owner without a recorded start baseline never becomes reclaimable.
- **Process identity** is transported from the long-lived parent (`--owner-pid`),
  never the ephemeral gate subprocess pid.

## Canary for Hermes Immo (no live restart in this task)

1. Use a temporary registry and two temporary git worktrees.
2. Claim one worktree as `default` on a product lane (transport a live
   `--owner-pid` so the owner is not immediately reclaimable).
3. Run `hook-session-start --agent hermes-immo --session continue` against the same worktree and verify the STOP advisory.
4. Run `admit --mode owner --hard --profile hermes-immo --domain-prefixes JYI,HER` for `SCA-740` and verify it refuses before owner creation.
5. Run `admit --mode reviewer` and verify it exits 0 without changing the owner JSON.
6. Wire `factory_admission_hook.py` into a *temporary* `cli-config.yaml` and
   confirm `hermes hooks test pre_tool_call` (or a scripted
   `get_pre_tool_call_block_message`) blocks a `terminal`/`patch` tool in the
   foreign-owned worktree and allows the owning session — this is what
   `tests/test_factory_lane_integration.py` automates.
7. Only after review/merge should a real gateway config enable this hook before
   launching build-capable agents. Do not restart `hermes-immo` from this canary.

## Rollback

This change is additive in the repo. The tracked files are
`scripts/factory_lane.py`, `scripts/factory_admission_hook.py`, this doc, and the
tests `tests/test_factory_lane_admission.py`,
`tests/test_factory_lane_appsec.py`, `tests/test_factory_lane_integration.py`.

The runtime wiring is **config-only and opt-in**: the gate is disabled until a
`hooks: pre_tool_call:` entry is added to `cli-config.yaml`. To roll the wiring
back, delete that `hooks` entry (and, optionally, revoke the hook from the
shell-hook allowlist with `hermes hooks revoke <command>`) — no core file is
touched, so nothing else changes.

To roll the code back before installation, remove the files above from the
branch. If a deployed copy has already been installed into
`~/.hermes/scripts/factory_lane.py`, restore the timestamped backup of that file
and remove the `hooks` entry. The registry remains reconstructible evidence; do
not delete `registry/` as part of rollback unless Jean explicitly asks for a
separate cleanup gate.
