# Building a North Forge drive (admin)

This is the one-time, admin-side procedure for turning a blank USB/SSD into a
drive a non-technical recipient can plug in and run **with no installation, no
network, and no prompts**. It is written for the owner / a provisioning admin,
not for recipients.

Related: [`DECISION-2026-09-06-003`](../logs/ledger/decisions/DECISION-LOG.md)
(drive-native run-in-place install model),
[`DECISION-2026-09-09-001`](../logs/ledger/decisions/DECISION-LOG.md) (bundled
toolchain — this doc's step 3), `scripts/nf-setup.ps1` (tier / pinned-edition
provisioning), `scripts/bootstrap-north-forge.ps1` (first-launch venv build).

---

## The layout on a finished drive

Everything is a **sibling of the checkout folder**, so a drive-letter change moves
them together and nothing depends on a fixed path:

```
<drive>:\
├── Start North Forge.lnk          ← recipient double-clicks this
├── north-forge-agent\             ← the checkout (this repo)
├── north-forge-agent-toolchain\   ← STEP 3 - the bundled Python toolchain
│   ├── uv\uv.exe
│   ├── python\                     ← a python-build-standalone CPython 3.11
│   │   └── python.exe
│   └── THIRD-PARTY-NOTICES.txt
├── north-forge-agent-venv\        ← built on first launch by bootstrap
└── north-forge-agent-data\        ← HERMES_HOME, created on first launch
```

The `-venv` and `-data` folders are **not** prepared by the admin — the recipient's
first launch (`north-forge.cmd` → `nf-preflight.ps1` → `bootstrap-north-forge.ps1`)
builds them, using the bundled toolchain from step 3.

---

## Step 1 — put the checkout on the drive

Clone (or copy) this repo to `<drive>:\north-forge-agent\`. Keep it a real git
checkout so `git pull` still works for you later. Nothing about the checkout is
recipient-specific.

## Step 2 — provision tier / pinned edition (optional)

If this drive is a locked **Basic**-tier drive or ships a pinned edition, run the
Setup Run wizard now:

```powershell
cd <drive>:\north-forge-agent
powershell -ExecutionPolicy Bypass -File scripts\nf-setup.ps1
#   or non-interactive:
powershell -ExecutionPolicy Bypass -File scripts\nf-setup.ps1 -NonInteractive -Tier basic -Pin penny-pincher
```

This writes the signed `north-forge\provisioning.json` + on-drive key. A drive
with no provisioning record behaves like plain Hermes (un-provisioned = inert).
**The bundled toolchain and the venv build never touch this record** — see
"Interaction with tier/pin" below.

## Step 3 — stage the bundled toolchain (one-time master prep, then copy per drive)

`bootstrap-north-forge.ps1` builds the venv from a toolchain it resolves in this
order:

1. **bundled** — `<parent>\<leaf>-toolchain\uv\uv.exe` + `\python\python.exe`
   (this step). Used silently when present. Zero PATH use, zero network.
2. **host PATH** — `uv` / `python` on the admin's PATH. Used only as a fallback,
   and logged as `using host toolchain (admin/dev)` so it is never confused with
   the bundled path. **A recipient drive must not rely on this.**
3. neither → the existing hard error (`No 'uv' and no 'python' on PATH…`).

### 3a. Build the master toolchain once (on any machine with network)

```powershell
$master = "C:\nf-toolchain-master"      # keep this; reuse for every drive
New-Item -ItemType Directory -Force $master\uv, $master\python | Out-Null

# uv.exe - a single static binary. Pin the version you tested against.
$uvVer = "0.8.13"                        # example - use the version you validate
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$uvVer/uv-x86_64-pc-windows-msvc.zip" -OutFile $env:TEMP\uv.zip
Expand-Archive $env:TEMP\uv.zip -DestinationPath $master\uv -Force   # -> $master\uv\uv.exe

# A python-build-standalone CPython 3.11 (the distribution uv itself uses).
# Easiest: let uv fetch one into a known dir, then copy the tree out.
$env:UV_PYTHON_INSTALL_DIR = "$master\_uvpy"
& $master\uv\uv.exe python install 3.11
# copy the resulting cpython-3.11-* directory's CONTENTS so python.exe sits at
# $master\python\python.exe :
$src = Get-ChildItem "$master\_uvpy\cpython-3.11-*" -Directory | Select-Object -First 1
Copy-Item "$src\*" $master\python -Recurse -Force
Remove-Item "$master\_uvpy" -Recurse -Force
Remove-Item Env:\UV_PYTHON_INSTALL_DIR

# Sanity: fully offline venv build from the master, nothing on PATH.
$env:Path = "$env:SystemRoot\System32;$env:SystemRoot"     # scrub PATH
& $master\uv\uv.exe venv $env:TEMP\nf-smoke --python $master\python\python.exe
Test-Path $env:TEMP\nf-smoke\Scripts\python.exe            # -> True
Remove-Item $env:TEMP\nf-smoke -Recurse -Force
```

Copy `docs\toolchain\THIRD-PARTY-NOTICES.txt` from this repo into
`$master\THIRD-PARTY-NOTICES.txt` (licence notices for CPython / uv /
python-build-standalone — required, see that file).

### 3b. Copy the master onto each drive

```powershell
# <leaf> is the checkout folder's name, usually "north-forge-agent"
Copy-Item C:\nf-toolchain-master <drive>:\north-forge-agent-toolchain -Recurse
```

That's it. Verify on the drive itself, PATH scrubbed:

```powershell
$env:Path = "$env:SystemRoot\System32;$env:SystemRoot"
cd <drive>:\north-forge-agent
powershell -ExecutionPolicy Bypass -File scripts\bootstrap-north-forge.ps1
#   expect:  [bootstrap] toolchain uv=bundled python=bundled ...
#            toolchain: bundled drive toolchain  (<drive>:\north-forge-agent-toolchain)
#   then a venv is built and the post-install `import hermes_cli` check passes.
```

## Step 4 — hand off

The recipient plugs in the drive and double-clicks `Start North Forge.lnk`
(created/refreshed automatically). First launch builds `-venv` and `-data` from
the bundled toolchain; every later launch self-heals.

---

## Rules and constraints

- **The toolchain folder is never git-tracked and never fetched at recipient
  first-launch.** It is prepared by you and copied onto the drive. There is no
  auto-download path for it, by design (offline promise).
- **It is an input to venv creation, not a venv.** `bootstrap`'s ownership
  classifier (`scripts/lib/nf-venv-state.ps1`) and readiness probe
  (`scripts/lib/nf-readiness.ps1`) never look at it; the `-Force` /
  create / rebuild / refuse state table is entirely upstream of toolchain
  resolution. A `-Force` rebuild deletes `-venv`, never `-toolchain`.
- **Keep the bundled `python` a python-build-standalone build** (`python.exe` at
  its root, relocatable, carries its own stdlib). The python.org *embeddable*
  package is deliberately not a venv base without `._pth` surgery — do not use it
  here. See `D:\logs\CODEX-BUNDLED-PYTHON-FEASIBILITY.md`.
- **Refresh cadence:** rebuild the master when North Forge bumps its supported
  Python or when you want a newer uv, then re-copy to drives. A stale bundled
  interpreter still works; it is not auto-detected as stale today.
- **Interaction with tier/pin:** `bootstrap` and the toolchain resolution never
  read, write, or verify `north-forge\provisioning.json` or the on-drive key. The
  venv build is tier-agnostic; `nf-setup.ps1` (step 2) is the only writer of the
  provisioning record. Building or rebuilding the venv on a Basic-tier drive does
  not change its pin, its edition, or its signature.
