---
name: busy-terminal
description: "Joke screensaver: fake coding or Hollywood hacking."
version: 1.4.0
author: "Luke The Dev (@iamlukethedev), Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [creative, screensaver, terminal, ascii, fun, animation, matrix, hacker]
    category: creative
---

# Busy Terminal Skill

Fills a terminal with invented activity, cycling scenes at random until
stopped. Two profiles: `developer` fakes honest work (editor, build, tests,
git) and `hacker` fakes the movie kind (digital rain, a multi-window war room
with a password-matching dialog, an intrusion ending in ACCESS GRANTED, a key
crack). `mixed` interleaves both. It is a screensaver in the `cmatrix` /
`genact` / Hollywood-hacker tradition.

Nothing it prints is real. It reads no files, runs no commands, and opens no
sockets; every path, SHA, and byte count is generated, and the hacker
profile's "targets" are RFC 5737 documentation addresses on example.* hosts,
unroutable by construction. It reports no status about the user's actual work
and should never be presented as if it did.

## When to Use

Trigger on any of these, without asking a follow-up question:

- "pretend I'm working" / "make it look like I'm working" / "look busy"
  / "fake coding" → `--profile developer`
- "hacker mode" / "make me look like a movie hacker" / "Hollywood hacking"
  / "matrix mode" / "digital rain" / "several windows" / "the screensaver"
  / "busy terminal" → `--profile hacker`
- Ambiguous, or the user is testing the feature → `--profile hacker` (the
  war room opens first). Never launch a bare `--window --duration` and rely
  on an old default — always pass `--profile` explicitly.

Do not reach for this when the user wants real build, test, or git output —
run the real thing with `terminal` instead. Never describe its output as if it
reflected real work; it is invented and reports nothing about the repository.

## Prerequisites

Python 3.9+. No third-party packages, no API keys, no network.

Colour needs a terminal that understands ANSI escapes. Output that is piped or
redirected degrades to plain text automatically, as does setting `NO_COLOR`.

## How to Run

Always pass `--window`. Run it through the `terminal` tool:

```bash
python3 ~/.hermes/skills/creative/busy-terminal/scripts/busy_terminal.py \
  --window --profile hacker --duration 600
```

`--window` opens a fresh terminal window on the user's screen, then returns
immediately. Without it the animation writes into the agent's captured pipe,
where there is no TTY to animate and — at the default unbounded duration — the
turn never ends.

Pick a `--duration` so the window cannot outlive the user's attention. Ten
minutes is a good default; they can Ctrl-C sooner.

Variants worth offering:

```bash
... --window --profile developer --duration 600   # fake coding, no rain
... --window --scene warroom --duration 180       # just the multi-window poster
... --window --scene matrix                       # just the rain
```

The user can also run it themselves in any terminal, without `--window`.

## Quick Reference

| Flag | Default | Effect |
|------|---------|--------|
| `--profile` | `hacker` | Scene set: `hacker`, `developer`, or `mixed` |
| `--duration` | `0` | Seconds to run; `0` means until Ctrl-C |
| `--speed` | `1.0` | Time multiplier — `2` is twice as fast |
| `--scene` | cycle | Pin one scene on repeat (overrides the profile) |
| `--seed` | random | Reproducible run |
| `--no-color` | off | Plain text, no ANSI escapes |
| `--window` | off | Open a new terminal window and return — use this from an agent |

| Scene | Profile | What it shows |
|-------|---------|---------------|
| `code` | developer | Editor pane, line numbers, source typed and highlighted |
| `build` | developer | Vite / Cargo / Docker output, progress bar, artifact sizes |
| `tests` | developer | Pytest-style dots, pass–fail summary, occasional flake retry |
| `git` | developer | Commit, push with delta compression, CI checks going green |
| `matrix` | hacker | Digital rain with bright heads and dimming trails |
| `warroom` | hacker | Rain behind live panes (red/green accents), four floating dialogs |
| `intrusion` | hacker | Port scan, brute-force, ACCESS GRANTED banner, exfil bar |
| `crack` | hacker | Hex spray, then an AES key locking in byte by byte |

## Procedure

1. Pick the profile from the user's phrasing (see When to Use); default to
   `hacker` when it is ambiguous. Always pass `--profile` on the command
   line — do not omit it.
2. Run the script with `--window`, the profile, and a `--duration` via
   `terminal`. The war room opens first on the hacker rotation.
3. Tell the user a new window opened and that Ctrl-C in it stops the show.
4. Suggest full screen and a larger font if they want it to fill the display —
   the matrix rain especially earns it.

Scenes never repeat back to back; `next_scene` excludes the one that just
played, so the cycle reads as varied rather than random-looking.

## Pitfalls

- **Forgetting `--window` hangs the turn.** The default duration is unbounded,
  so a captured run never returns and the user sees nothing.
- It owns the pane it runs in. `--window` gives it its own, which is why that
  is the agent's path.
- Backgrounding it (`terminal(background=True)`) is not a substitute — the
  output is the entire feature and a background process writes it nowhere
  visible.
- On Linux `--window` needs a terminal emulator on PATH; it raises
  `NoTerminalError` naming the ones it tried. Over plain SSH with no emulator,
  fall back to telling the user to run it without `--window`.
- Under 40 columns the editor pane and artifact table wrap badly. `Console`
  floors the width at 40, but a genuinely tiny terminal still looks cramped.
- `--speed` scales pauses, not content. Very high values (>10) reduce it to a
  wall of text with no rhythm.
- The matrix rain and the war room need ANSI cursor addressing; piped or
  `--no-color` output degrades them to scrolling lines on purpose.
- The war room shines at generous sizes — panes that would come out under
  14×4 cells are dropped, and the three corner dialogs (perimeter alert,
  proxy chain, exfil meter) only appear at 70×20 or larger. The password
  centerpiece always shows.
- The hacker profile is theatre, not technique — keep targets and loot inside
  the shipped fictional pools if you ever extend the pools.

## Verification

- Output appears within a second and keeps scrolling
- Over a few minutes every scene in the chosen profile appears, none twice in
  a row
- `--profile hacker` reaches the ACCESS GRANTED banner and a fully locked key
- Ctrl-C exits cleanly and the cursor comes back (no invisible prompt)
- `--no-color` output contains no `\033[` sequences
- Two runs with the same `--seed` produce the same transcript
