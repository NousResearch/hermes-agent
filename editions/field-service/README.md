# field-service edition

An **optional persona overlay**, not a replacement for the base persona.

- **Base persona** (always): repo-root [`../../SOUL.md`](../../SOUL.md) — the
  generic North Forge chassis voice: adaptive tone, honest about uncertainty,
  direct, industry-neutral.
- **This overlay**: [`SOUL.md`](SOUL.md) — the same principles in a plainer,
  first-person register written for field technicians and sales reps ("I'm North
  Forge. I help field technicians and sales reps get through their day…"). Shorter
  sentences, less engineer-to-engineer framing, more "meet you where you are".

## When to use it

Only when a deployment is specifically for field techs / reps and the plainer
register fits the audience better. For anything general-purpose, leave the base
`SOUL.md` in place — it already adapts its tone.

## How to apply it

It is a plain file swap into the **active** persona location, done at
deploy/config time — never by editing the repo:

```
copy editions\field-service\SOUL.md  %HERMES_HOME%\SOUL.md      # Windows
cp    editions/field-service/SOUL.md  "$HERMES_HOME/SOUL.md"     # POSIX
```

The repo-root `SOUL.md` stays the generic one. Installers seed a fresh
`$HERMES_HOME/SOUL.md` from `hermes_cli/default_soul.py` (upstream's code
constant); this overlay is applied after that, only if chosen.

This overlay carries **no** industry skill content and no access-tier logic — it
is voice only.
