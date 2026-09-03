---
name: setup-wizard-generator
description: "Generate a bash wizard guiding a human through manual setup."
version: 1.0.0
author: "Matt Pocock (mattpocock/skills, wizard) + Hermes Agent"
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [wizard, setup, onboarding, credentials, secrets, migration, bash, human-in-the-loop]
    related_skills: []
---

# Setup Wizard Generator

Generates an interactive bash **wizard**: a script that walks a human, step
by step, through a manual procedure that is tedious to do by hand and tedious
to re-explain every time. It opens each URL, says exactly what to click and
copy, captures the values, writes them where they belong (`.env`, GitHub
secrets), confirms at every stage, and shows how many stages are left.

Ported from mattpocock/skills' MIT-licensed `wizard` skill.

## When to Use

- Provisioning infrastructure or third-party services (Stripe, Supabase,
  DNS, OAuth apps) where only a human can click through the dashboard
- Setting up credentials, CI secrets, or repo variables
- One-off migrations or cutovers with irreversible human-gated steps
- Any procedure the user will hand to a teammate to run

Do NOT use for steps the agent can perform itself — do those directly.

## Prerequisites

- `bash`; `gh` CLI only if stages write GitHub secrets/variables
- The library template: `templates/template.sh` in this skill's directory

## Procedure

### 1. Scope the procedure

Work out every manual step the human must take and every value captured
along the way. Read the repo first, don't ask cold:

- Setup: `.env`, `.env.example`, `README`, `docker-compose*`, framework
  config, and `.github/workflows/*` (every `secrets.*` / `vars.*` reference
  is a value the wizard must produce).
- Migration/cutover: the current state, the target state, and the
  irreversible actions between them.

Show the user the ordered stage list and the values each produces; they may
add, drop, or reorder. Done when every stage is named in order and, for each
captured value, you know (a) where the human gets it, (b) where it's written
(`.env`, a GitHub secret, both, or nowhere), and (c) whether it's secret
(hidden entry) or public.

### 2. Map each stage's journey

For each stage, write the precise path a human follows: which URL to open,
what to do there, where the value is shown — e.g. "Dashboard → Developers →
API keys → Reveal test key → copy". Where you don't know the current UI or
exact command, say so and check the docs or ask — never invent steps that
may not exist.

### 3. Author the wizard

Copy `templates/template.sh` (from this skill's directory) to the target
path. Replace the example stage with one `stage` per step, in dependency
order. Set `TOTAL_STAGES` to the number of stages you wrote.

Library helpers: `stage`, `say`/`step`/`note`/`warn`, `open_url`,
`ask`/`ask_secret`, `write_env`, `set_secret`/`set_var`, `pause`/`confirm`,
`banner`, `finish`. The library above the `STAGES` marker is identical in
every wizard — never hand-edit it; that consistency is the point.

Hold the bar the template sets: open the URL before asking for its value,
`ask_secret` for anything secret, `write_env` every persisted value,
`set_secret` only what CI actually needs, and `confirm` before anything
irreversible. Each `stage` clears the screen — keep one focused task per
stage so nothing the human needs scrolls away.

A wizard is ephemeral by default: save it to a scratch or `scripts/` path,
delete it when the job's done. Commit it only when the user wants a
repeatable setup path living in the repo.

### 4. Verify and hand off

- `bash -n <script>`; run `shellcheck` if available; `chmod +x <script>`.
- Do NOT run it end-to-end yourself: it opens browsers and blocks on human
  input. Trace it statically: every value from step 1 is captured and lands
  where step 1 said, and every `set_secret` name exactly matches a
  `secrets.*` reference in CI.
- Tell the user how to run it. If it's repeatable, commit it and link it
  from the README.

## Pitfalls

1. **Editing the library section.** Everything above the `STAGES` marker is
   the wizard library; author only below it.
2. **Inventing dashboard paths.** Third-party UIs drift. If unsure of the
   click path, verify against current docs or flag it as approximate.
3. **`set_secret` for values CI doesn't use.** Only push to GitHub secrets
   what a workflow actually references.
4. **Running the wizard yourself.** It blocks on human input; static tracing
   plus `bash -n` is the verification.
5. **One mega-stage.** Screen clearing per stage means a long stage scrolls
   critical instructions away; split it.

## Verification

- [ ] Stage list confirmed with the user before authoring
- [ ] `bash -n` passes; script is executable
- [ ] Every captured value traced to its declared destination
- [ ] Every `set_secret` name matches a CI `secrets.*` reference
- [ ] Library section untouched from the template
