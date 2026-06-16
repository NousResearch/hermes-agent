# openclaw-vendor

Hermes plugin that wires the in-tree **OpenClaw vendor mirror** (`vendor/openclaw-mirror/`) into your profile:

- **Extensions** (`hypura-harness`, `hypura-provider`, `vrchat-relay`) — skill sync + readiness metadata
- **Packages** (`AI-Scientist`, `ShinkaEvolve`, `ATLAS`, `a2ui`, `elan`, `nc-kart-proof`) — path/marker checks for existing core tools
- **Sibling plugins** (`questframe-fh6vr`, `book-to-skill`) — listed for enablement; not auto-enabled

## Enable

```yaml
# ~/.hermes/config.yaml
plugins:
  enabled:
    - openclaw-vendor
    # Optional companions:
    # - questframe-fh6vr
    # - book-to-skill
```

## Commands

```bash
hermes openclaw-vendor list      # manifest
hermes openclaw-vendor status    # mirror + skill links + tool markers
hermes openclaw-vendor install   # link all extension skills → ~/.hermes/skills/
hermes openclaw-vendor sync --force
hermes openclaw-vendor install --extension hypura-harness
```

Then `/skills reload` (or new session) and enable `openclaw` / `vrchat` toolsets via `hermes tools`.

## Claim boundary

This plugin **does not** port every OpenClaw TypeScript tool into the Hermes core schema. It exposes **skills** and reports **readiness** for packages already integrated via `tools/` and `hermes harness` / `hermes hypura`.

## Vendor sync (repo maintainers)

Refresh the mirror with:

```bash
py -3 scripts/sync_openclaw_vendor.py
```
