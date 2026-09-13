---
name: steam-intel-vulkan-fix
description: "Fix Steam game crashes on Intel GPUs by forcing D3D11."
version: 0.1.0
author: "Gerardo Chara (charanoway10), Hermes Agent"
license: MIT
platforms: [windows]
metadata:
  hermes:
    tags: [steam, gaming, intel, vulkan, crash, troubleshooting]
    related_skills: []
---

# Steam Intel Vulkan Crash Fix

Diagnoses Steam games that crash at startup with an Access Violation in Intel's Vulkan driver (`igvk64.dll`) and fixes them by forcing the DirectX 11 renderer through Steam launch options. Applies to KEX-engine titles (Quake rerelease, DOOM 64, etc.) and any Steam game whose crashlog points at `igvk64.dll`.

## When to Use

- A Steam game crashes at launch and its crashlog (or Windows Event Viewer) shows an Access Violation (`0xc0000005`) in `igvk64.dll` or `ig9icd64.dll`
- The game worked before a recent update and now dies during renderer/shader initialization on an Intel GPU
- User asks why a Steam game stopped launching on Intel graphics (Iris Xe / UHD / Arc)

Don't use for: games that crash in other modules (`.exe` code paths, third-party DLLs), or non-Steam games — same launch-options trick exists but the persistence mechanism differs.

## Prerequisites

- Steam installed on Windows and the game installed (Steam Library path: `C:\Program Files (x86)\Steam\steamapps\common\<game>\`)
- A crashlog or error text naming the faulting module (KEX engine writes `crashlog.txt` next to the game exe)
- Permission to close Steam briefly when editing `localconfig.vdf` (Steam overwrites it on exit if it is running)

## How to Run

1. Read the crashlog with `read_file(path=<game_dir>/crashlog.txt)` and confirm the faulting module is `igvk64.dll`.
2. Apply the fix (two routes, pick one):
   - **UI route (no file edits):** user adds `+r_rhirenderfamily d3d11` in Steam → game Properties → Launch Options.
   - **File route (agent-driven):** close Steam, edit `localconfig.vdf` (backup first), relaunch Steam.
3. Launch the game from Steam and verify the process survives startup and no new crashlog is written.

## Quick Reference

- Fault signature: `Access Violation Exception (0xc0000005) in module igvk64.dll`
- Fix: force D3D11 → `+r_rhirenderfamily d3d11`
- Per-game config: `C:\Program Files (x86)\Steam\userdata\<steamid>\config\localconfig.vdf`, section `"apps" → "<appid>"` → `"LaunchOptions"`
- `appid` for Quake: 2310 (find others via `steamdb.info` or the game's store URL)
- Verify: `tasklist | grep -i <game>` after 30 s; crashlog mtime must stay unchanged

## Procedure

1. **Confirm the faulting module.** Read the crashlog: `read_file(path="C:/Program Files (x86)/Steam/steamapps/common/<game>/crashlog.txt")`. Look for `in module igvk64.dll` (Intel Vulkan). Also confirm the GPU is Intel via `powershell.exe -NoProfile -Command 'Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | Format-List'` — a recent driver does NOT rule out the bug; the game update is usually the trigger.
2. **Check for a known bug.** Search the web for the game name + `igvk64.dll` + game version: `web_search(query="<game> crash igvk64.dll <version>")`. If a Steam Community thread exists, the fix is community-validated.
3. **Apply the launch option (UI route).** Ask the user to open Steam → game → Properties → Launch Options and paste `+r_rhirenderfamily d3d11`. This is the zero-risk path when the user is at the machine.
4. **Apply the launch option (file route, agent-driven).**
   a. Close Steam: `terminal(command="C:/Program Files (x86)/Steam/steam.exe -shutdown")`, then `taskkill //F //T //IM steam.exe` if it lingers.
   b. Locate the user config: `terminal(command="ls 'C:/Program Files (x86)/Steam/userdata'")` to get the `<steamid>` folder.
   c. Backup: `terminal(command="cp <vdf> <vdf>.bak")`.
   d. Find the app's block: `search_files(pattern='"<appid>"', path='<vdf>')`, then `patch` the block to add `"LaunchOptions" "+r_rhirenderfamily d3d11"` (tabs preserved; the VDF parser tolerates fuzzy whitespace matching).
   e. Relaunch Steam: `terminal(command="C:/Program Files (x86)/Steam/steam.exe -silent", background=true)`.
5. **Verify.** Launch the game via Steam (`powershell.exe -NoProfile -Command 'Start-Process "steam://rungameid/<appid>"'`), wait ~30 s, then confirm: process alive (`tasklist`) and crashlog mtime unchanged (`stat -c "%y" <crashlog>`). A NEW crashlog means the fix did not take — check the LaunchOptions landed and the game is not using a different renderer flag.
6. **Document.** If the user keeps an Obsidian vault, log the case (problem, evidence, fix, how to revert) and commit.

## Pitfalls

- **Do not edit `localconfig.vdf` while Steam runs** — Steam rewrites it on exit and silently discards your edit. Always close Steam first and back up.
- **Launching the game exe directly from a shell is not a valid test** — without Steam's launcher context the process exits cleanly (no crashlog written), which looks like success but proves nothing. Always test through `steam://rungameid/<appid>`.
- **`igvk64.dll` crash ≠ old driver.** Recent Intel drivers still hit this on broken game builds; check the game's changelog/forums before blaming the driver.
- **Rendering API flag differs per engine.** KEX uses `r_rhirenderfamily`; other engines use `-d3d11`, `-dx11`, or a config file. Search the game's forums for the correct flag.
- **bash cannot always exec game exes directly** (MSYS permission quirks on `Program Files` paths) — use PowerShell `Start-Process` or Steam for launching.
- Reverting later: delete the `LaunchOptions` line (or the whole string) once the vendor fixes the Vulkan path.

## Verification

- [ ] Crashlog names `igvk64.dll` as faulting module before the fix
- [ ] `LaunchOptions` present under `"apps" → "<appid>"` in `localconfig.vdf`
- [ ] Game process alive 30 s after launch via Steam (no crash dialog, no new crashlog)
- [ ] `stat` on crashlog shows unchanged mtime after a successful launch
