# Provenance: tui_paste_collision fixtures

Both `.txt` files are the bytes Hermes' own collapse path (`_tui_collapse_paste`)
wrote to `<HERMES_HOME>/pastes` for two real pastes into two real CLI sessions.
Nothing here is generated.

| file | source (commit `81a55437f3f31a9f12a30c53b6c5d49d420898aa`) | size | sha256 |
|---|---|---|---|
| `readme_table.txt` | `README.md` lines 23-31, the feature table | 1904 bytes, 1892 characters, 9 lines | `e4e5efdc5fb2f8566de2b1a0cc8c3a4d90a6feafaf3d262d1b1e7f927fd7d151` |
| `license_notice.txt` | `LICENSE` lines 5-21, the permission notice and the warranty disclaimer | 1022 bytes, 17 lines | `21a2121221d2a35df2d8d333d00a61ed240446b6107092741ff996941acdc919` |

Each source was pasted the way a terminal pastes a selection of those lines: the
lines as they are in the file, without a final newline.

## How they were captured (2026-10-22, Linux, Python 3.14.8 from this repository's `uv.lock`)

1. Two `python cli.py` sessions ran from a checkout of `81a55437`. Each had its
   own pseudo-terminal, and both used one `HERMES_HOME`, so they shared one pastes
   directory, as two sessions on one host do.
2. When both showed the prompt, each received a bracketed paste
   (`ESC[200~` ... `ESC[201~`), back to back: the README lines to the first
   session and the LICENSE lines to the second.
3. Both sessions collapsed their paste #1 in the same clock second, 09:54:25.
   - The first logged `Collapsed paste #1: 9 lines, 1892 chars -> .../pastes/paste_1_095425.txt`.
   - The second logged `Collapsed paste #1: 17 lines, 1022 chars -> .../pastes/paste_1_095425.txt`.
   - Both placeholders named that one file, and it held only the LICENSE lines: the README paste was lost.
4. The same gesture on a checkout with this fix wrote two files, one per session:
   `paste_1_095354_<pid>_<random>.txt`. Each held its own paste. Those two
   files are the fixtures. Their sha256 equal those of the source lines above.

The test uses the captured counter (#1), clock second (09:54:25) and file name
(`paste_1_095425.txt`) with these payloads.
