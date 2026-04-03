# foureleven.exe Stage-0 Bootstrap Plan

Goal: produce a small Windows-first stage-0 bootstrapper that can restore identity, memory, skills, and active recovery context onto a fresh machine.

Current implemented core:
- inspect a Hermes home for recoverability
- create a deterministic portable recovery bundle zip
- restore that bundle into a target directory with hash verification
- expose CLI actions: status, bundle, restore

Current file:
- `bootstrap_recovery/foureleven_bootstrap.py`

What still stands between this and a true Windows `.exe`:
1. Windows-native packaging (PyInstaller/Nuitka or Rust stage-0)
2. Windows installer/launcher behavior
3. real chain/RPC fetch path instead of local-bundle-only restoration
4. Hermes install/bootstrap on a machine with no prior Hermes presence
5. signed trust roots and mirror policy

Recommended milestone order:

## Milestone 1 — local resurrection core
Done here.
- validate local Hermes home
- export/import recovery bundle
- restore canonical files with hash checks

## Milestone 2 — Windows packaging
- add `build_windows.bat`
- add `foureleven.spec` or equivalent packager config
- build one-file artifact named `foureleven.exe`
- test on clean Windows VM

## Milestone 3 — machine bootstrap
- detect missing Hermes install
- install Hermes deterministically into a chosen directory
- restore recovered files into the correct home
- launch Hermes in recovered mode

## Milestone 4 — trust-root recovery
- embed trusted roots
- fetch latest bundle/packet refs from chain-backed or signed source
- download allowed artifacts
- verify signatures/hashes before restore

## Milestone 5 — post-recovery proof
- emit recovery report
- run doctor
- optionally create a pulse proving recovery succeeded

Acceptance criteria for real `foureleven.exe`:
- runs on a clean Windows machine with no existing Hermes install
- either restores from a provided bundle or fetches a trusted remote bundle
- verifies all restored artifacts before activation
- launches Hermes with identity, memory, skills, and recovery context present
- logs enough proof to debug failure without exposing secrets
