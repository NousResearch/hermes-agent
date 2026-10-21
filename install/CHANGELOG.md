# Changelog — install

Release notes for the Hermes install and update path (`scripts/install.sh`,
`scripts/install.ps1`, `scripts/install.cmd`). Entries are newest-first.

Entries record operator-visible changes to install/update behaviour. A change
that only refactors internals does not belong here.

## Unreleased

### The installer names local work before it overwrites it

`install.sh` now runs a maintenance step at the top of the `repository` stage,
before the fetch and before anything moves the tree. If the checkout carries
local work, the installer prints what it found — one line per item, each with
its kind — and then acts on your call:

```
⚠ /home/you/.hermes/hermes-agent carries local work that this update would overwrite:
    modified agent/conversation_compression_deadline.py
    untracked notes.tmp
    branch: fix/ruaaaaa-5212-compaction-run-budget-cap (1 local commit(s))

  [s] stash and continue
  [a] abort; leave the working copy for you to handle
  [c] commit onto local/fix/ruaaaaa-5212-compaction-run-budget-cap and continue
  choice [s]:
```

**Why.** The step exists because the previous guards could not see the most
expensive case. `git status --porcelain` reports edited and untracked files,
but reports an **empty string** for a patch that is already committed on a
local branch — so a local fix applied as a commit produced no signal at all,
and the `reset --hard origin/<branch>` that follows a failed fast-forward
dropped it without a word. The branch line above is counted explicitly
(`rev-list --count HEAD --not --remotes --tags`) for exactly that case.

**Migration cost: none.** No operator action is required, and no configuration
changes. What is new is a pause that did not exist before:

- **Clean checkout** — nothing is printed and nothing changes.
- **Local edits or untracked files, no terminal** (`curl … | bash`, or
  `--non-interactive`) — the previous behaviour is unchanged: the changes are
  still autostashed as `hermes-install-autostash-<stamp>` and the install
  continues. The difference is that the list is now printed first and the
  disposition is written to the install log, instead of the stash appearing
  with no announcement.
- **Local edits or untracked files, with a terminal** — the installer now asks
  instead of deciding for you. The default (empty input, or `s`) is the
  previous behaviour: stash and continue.
- **`a` (abort)** — the run stops with failure class `local_patch_abort` and
  leaves the working copy exactly as it was. Nothing is stashed.
- **`c` (commit)** — uncommitted work is committed onto `local/<branch>` first,
  so the branch switch below leaves it alone. If the tree is already clean and
  the only local work is the branch's commits, they are left on their branch
  under that name.

The chosen disposition is appended to the install log (`$INSTALL_LOG`) as
`==> Local patch maintenance: <choice>`, so "was the patch rebased, parked, or
dropped" is answerable from the log alone the next morning.

Windows parity for this step is not in this change.