# External (vendored) skills

Skills deployed into the live skills directory (`~/.hermes/skills/`) that are
**not** part of this repo's source tree because they are third-party packs with
their own upstream and licence. They are intentionally not committed here; this
file is the restore recipe so the runtime is fully reproducible.

| Skill | Upstream | Licence | Pinned commit | Restore |
|-------|----------|---------|---------------|---------|
| `avoid-ai-writing` | https://github.com/conorbronsdon/avoid-ai-writing | MIT | `cbf885e087e8ec1168bc58dc603606a6e4bfacbd` (main) | `git clone https://github.com/conorbronsdon/avoid-ai-writing ~/.hermes/skills/avoid-ai-writing` then `git -C ~/.hermes/skills/avoid-ai-writing checkout cbf885e0` |

## Why these are not in-tree

They carry their own `.git` and `LICENSE`. Absorbing them would mix another
project's licensed code into KenseiAgent and lose the upstream link for updates.
Keeping them as documented external deps means KenseiAgent stays the single
source of truth for *its own* code while remaining a complete restore recipe for
the live runtime.
