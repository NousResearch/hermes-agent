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
  --worktree /Users/jeanyoder/Documents/GitHub/_worktrees/hermes-her-95-worktree-admission-gate
```

Hard owner admission:
- serializes the final decision under `.worktree-admission.lock` to close preflight→claim TOCTOU races;
- refuses a second live owner for the same canonical worktree, even if the second owner uses another issue key;
- refreshes heartbeat for the same owner/session;
- stores `profile` and `gateway_session_key` when supplied, but no tokens or chat secrets;
- refuses dirty ownerless git worktrees before a build, without resetting or deleting anything.

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

## Canary for Hermes Immo (no live restart in this task)

1. Use a temporary registry and two temporary git worktrees.
2. Claim one worktree as `default` on a product lane.
3. Run `hook-session-start --agent hermes-immo --session continue` against the same worktree and verify the STOP advisory.
4. Run `admit --mode owner --hard --profile hermes-immo --domain-prefixes JYI,HER` for `SCA-740` and verify it refuses before owner creation.
5. Run `admit --mode reviewer` and verify it exits 0 without changing the owner JSON.
6. Only after review/merge should a real gateway integration call this hard gate before launching build-capable agents. Do not restart `hermes-immo` from this canary.

## Rollback

This change is additive in the repo. To roll back before installation, remove `scripts/factory_lane.py`, this doc, and `tests/test_factory_lane_admission.py` from the branch. If a deployed copy has already been installed into `~/.hermes/scripts/factory_lane.py`, restore the timestamped backup of that file and remove only the new hook invocations. The registry remains reconstructible evidence; do not delete `registry/` as part of rollback unless Jean explicitly asks for a separate cleanup gate.
