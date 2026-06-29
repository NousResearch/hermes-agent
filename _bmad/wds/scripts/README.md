# WDS Scaffold Scripts

Node.js scripts that enforce deterministic output from AI agents. Agents provide content via CLI flags; scripts produce structure.

All scripts use only Node.js stdlib (no external dependencies). Run from the project root.

---

## Scripts

### `wds-init-scenario.js` — Initialize a scenario

Creates the scenario folder and a README index file.

```bash
node src/scripts/wds-init-scenario.js \
  --scenario "01 New User Onboarding" \
  --description "New user first visit to account creation"
```

Output: `C-UX-Scenarios/01-new-user-onboarding/README.md`

---

### `wds-init-page.js` — Initialize a page spec

Creates a new page spec file with all required sections pre-filled with placeholders.

```bash
node src/scripts/wds-init-page.js \
  --page "01 Start" \
  --scenario "01 New User Onboarding" \
  --platform "Mobile web" \
  --visibility "Public"
```

Output:
- `C-UX-Scenarios/01-new-user-onboarding/01-start/01-start.md`
- `C-UX-Scenarios/01-new-user-onboarding/01-start/sketches/`

After creating all pages in a scenario, run `wds-nav.js` to wire up navigation links.

---

### `wds-nav.js` — Update navigation links

Scans pages in a scenario (sorted by name) and writes correct prev/next navigation rows into each page spec.

```bash
# One scenario
node src/scripts/wds-nav.js --scenario "01 New User Onboarding"

# All scenarios
node src/scripts/wds-nav.js --all
```

Run this after adding or removing pages, or after reordering page numbers.

---

### `wds-add-object.js` — Append an object spec

Appends a structured object spec block to a page spec under a named section.

```bash
node src/scripts/wds-add-object.js \
  --page "C-UX-Scenarios/01-new-user-onboarding/01-start/01-start.md" \
  --section "Hero" \
  --object "Primary Headline" \
  --component "H1 heading" \
  --se "Välkommen" \
  --en "Welcome" \
  --behavior "Static display"
```

Object ID is auto-derived: `start-hero-primary-headline`

The section heading (`### Section: Hero`) is created if it doesn't already exist.

---

### `wds-add-spacing.js` — Append a spacing object

Appends a spacing notation entry to the `## Spacing` section of a page spec.

```bash
node src/scripts/wds-add-spacing.js \
  --page "C-UX-Scenarios/01-new-user-onboarding/01-start/01-start.md" \
  --direction v \
  --type space \
  --size xl \
  --reason "major section boundary between hero and features"
```

Valid directions: `v` (vertical), `h` (horizontal)
Valid types: `space`, `separator`, `line`
Valid sizes: `zero`, `sm`, `md`, `lg`, `xl`, `2xl`, `3xl`, `flex`

Spacing ID is auto-derived: `start-v-space-xl`

---

### `wds-validate.js` — Validate page specs

Checks page spec files for structural correctness.

```bash
# Single page
node src/scripts/wds-validate.js \
  --page "C-UX-Scenarios/01-new-user-onboarding/01-start/01-start.md"

# All pages in a scenario
node src/scripts/wds-validate.js --scenario "01 New User Onboarding"

# All scenarios
node src/scripts/wds-validate.js --all
```

Validates:
- Required sections present
- Object IDs are kebab-case with correct page prefix
- No duplicate Object IDs
- Navigation rows (3 expected)
- Metadata table has all required properties
- Sketches folder exists
- SE + EN content present for each object

---

## How agents use these scripts

1. Agent calls `wds-init-scenario.js` with scenario name and description
2. Agent calls `wds-init-page.js` for each page in the scenario
3. Agent calls `wds-nav.js` to wire navigation after all pages exist
4. Agent calls `wds-add-object.js` for each UI object, providing Swedish and English content
5. Agent calls `wds-add-spacing.js` for each spacing decision
6. Agent calls `wds-validate.js` to confirm the spec is structurally correct before handoff

The agent never writes raw markdown — it only supplies content as flag values. The scripts own all structural decisions.

---

## File location convention

```
C-UX-Scenarios/
  {scenario-slug}/
    README.md
    {page-slug}/
      {page-slug}.md
      sketches/
        {page-slug}-concept.jpg
```

Example: `C-UX-Scenarios/01-new-user-onboarding/02-signup/02-signup.md`
