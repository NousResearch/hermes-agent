# Gordon GBrain Path Layout

Session learning: Gordon wants GBrain organized like upstream `garrytan/gbrain` to avoid merge confusion and repeated ambiguity between Hermes native memory, gbrain source code, runtime DB/config, and content sources.

## Upstream gbrain convention

Original gbrain separates three axes:

- **Source repo / app code**: cloned implementation repo, e.g. `~/gbrain`.
- **Runtime config + local DB**: `~/.gbrain/config.json` and `~/.gbrain/brain.pglite` by default. If `GBRAIN_HOME=/opt/data`, gbrain appends `.gbrain`, so runtime path is `/opt/data/.gbrain/brain.pglite`.
- **Content sources**: independent repos/folders registered inside the brain DB with `gbrain sources add <id> --path <path>`. A source is selected by `--source`, `GBRAIN_SOURCE`, `.gbrain-source`, or registered `local_path` matching.

## Gordon Railway layout after this session

- **Hermes native memory hot-cache**: `/opt/data/memories`
  - Keep this for Hermes built-in memory only: `MEMORY.md`, `USER.md`, lock files.
  - Do not call this gbrain content.
- **Durable gbrain-style markdown content source**: `/opt/data/gbrain-content`
  - Contains `SCHEMA.md`, `index.md`, `entities/`, `concepts/`, `projects/`, `queries/`, `raw/`, etc.
- **GBrain source repo copy**: `/opt/data/repos/gbrain`
  - Use this for upstream pulls/merge work, not `/opt/data/gbrain`.
- **GBrain runtime/config/DB convention**: `/opt/data/.gbrain`
  - Intended upstream-style runtime location; note it was root-owned in the observed Railway volume.
- **Legacy leftover**: `/opt/data/gbrain`
  - Root-owned gbrain source checkout still existed and could not be moved/deleted by the Hermes user. Avoid using it for future source work unless ownership is fixed.

## Migration/verification pattern

Before making changes:

```bash
printf 'HERMES_HOME=%s\n' "$HERMES_HOME"
for p in /opt/data/memories /opt/data/gbrain-content /opt/data/repos/gbrain /opt/data/gbrain /opt/data/.gbrain; do
  if [ -e "$p" ]; then stat -c '%A %U:%G %n' "$p"; else echo "$p missing"; fi
done
```

Safe split used in this session:

```bash
mkdir -p /opt/data/gbrain-content /opt/data/repos
# Leave /opt/data/memories/MEMORY.md and USER.md in place.
for item in SCHEMA.md index.md comparisons concepts entities hobbies log projects queries raw; do
  [ -e "/opt/data/memories/$item" ] && mv "/opt/data/memories/$item" /opt/data/gbrain-content/
done
# If /opt/data/gbrain is root-owned and cannot be moved, copy instead:
cp -a --no-preserve=ownership /opt/data/gbrain /opt/data/repos/gbrain
```

Verification:

```bash
git -C /opt/data/repos/gbrain status --short
find /opt/data/memories -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
find /opt/data/gbrain-content -maxdepth 1 -mindepth 1 -printf '%f\n' | sort

# Runtime readiness is separate from source/content layout:
for c in gbrain bun node npm; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "$c=$(command -v "$c")"
    "$c" --version 2>/dev/null | head -1 || true
  else
    echo "$c=missing"
  fi
done
```

Expected:

- `/opt/data/memories` has only native memory files plus locks/git metadata.
- `/opt/data/gbrain-content` has durable markdown content directories.
- `/opt/data/repos/gbrain` is a clean source checkout.
- `gbrain` on PATH means Hermes can invoke the CLI directly; a clean `/opt/data/repos/gbrain` checkout alone does **not** prove runtime CLI readiness. If `gbrain=missing` and `bun=missing`, the pull-in/layout is good but install/linking still remains before direct `gbrain` calls will work.

## Pitfalls

- Do not use `/opt/data/memories` as a durable gbrain content path; it collides with Hermes native memory and confuses future sessions.
- Do not do upstream gbrain merge/pull work in `/opt/data/gbrain`; it may be a root-owned legacy checkout. Prefer `/opt/data/repos/gbrain`.
- Do not assume `/opt/data/.gbrain` is writable; check ownership first.
- Original gbrain's “brain” is a DB/runtime concept, not merely a markdown folder named `gbrain`.
