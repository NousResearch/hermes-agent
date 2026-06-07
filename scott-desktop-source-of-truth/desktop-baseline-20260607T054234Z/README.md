# Scott Hermes Desktop source of truth â€” 20260607T054234Z

This folder preserves the laptop-specific Hermes Desktop baseline before any upstream update attempt.

- Laptop repo: $repo
- Branch at capture: $branch
- HEAD: $head
- git describe: $describe
- Ahead/behind vs origin/main: $aheadBehind
- Packaged Desktop exe SHA256: $exeHash
- Full evidence backup: $outsideCase

Secret-bearing files are not committed here. The profile config copy is redacted.

Key behaviors to preserve:
1. scott-omega-profile is the only custom Scott Desktop profile shown/used.
2. Apollo and Omega custom providers resolve to 10.13.10.46:18820 and 10.13.10.23:18821.
3. Apollo requests include Teams continuity headers from the current Teams session.
4. Custom-provider model switching preserves the correct base_url and provider slug.
5. Session/sidebar lists are scoped to the selected Apollo/Omega model.
6. Packaged app is rebuilt into pps/desktop/release/win-unpacked/Hermes.exe after source changes.
