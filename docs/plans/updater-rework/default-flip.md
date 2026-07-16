# Default Bundle Install (Phase 5 Task 5.4)

> **Status: GATED** — not yet active.
>
> **Gate:** phases 1-4 E2E all green in CI for 2+ consecutive weeks;
> no open P1s against adopt/apply; maintainer sign-off.

## What changes

When the gate is met:

1. `scripts/install.sh`: `BUNDLE_MODE` defaults to `true`; `--source`
   flag = the old path (clone + venv + deps). The `--bundle` flag
   becomes a no-op (it's the default).

2. `scripts/install.ps1`: same — bundle is the default.

3. The desktop bootstrap runner requests the bundle path via its stage
   protocol (the updater serves the stage manifest — phase 1 task 1.8).

4. Website docs (`installation.md` + zh-Hans mirror) describe two paths:
   - **Managed** (default): `install.sh` → downloads `hermes-updater` →
     `hermes-updater install` → managed slots. Updates via `hermes-updater apply`.
   - **Source** (`--source`): `install.sh --source` → git clone + venv +
     deps. Updates via `hermes update` (worktree flow). For developers.

## How to flip the default

```diff
- BUNDLE_MODE=false
+ BUNDLE_MODE=true
```

In `scripts/install.sh` (line ~83) and `scripts/install.ps1`.

Add `--source` flag to the arg parser:
```sh
--source)
    BUNDLE_MODE=false
    shift
    ;;
```

Update the help text to reflect that `--bundle` is the default and
`--source` is the developer path.

## Docs to update

- `website/docs/getting-started/installation.md` — describe managed vs ejected
- `website/docs/getting-started/installation.zh-Hans.md` — mirror (if exists)
- `website/docs/getting-started/updating.md` — describe `hermes-updater` updates
