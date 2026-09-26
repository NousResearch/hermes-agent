# Protocol

### 1. Discover inventory

Enumerate global, built-in, tap-installed, and profile-local SKILL.md files, preserving resolved paths and duplicate names.

### 2. Validate bundles

Parse frontmatter and verify referenced scripts, references, templates, assets, examples, and tests exist.

### 3. Measure overlap

Require concrete shared triggers, tools, workflow steps, outputs, or references before flagging overlap.

### 4. Check staleness

For skills with a declared source, compare normalized local content to the current source and summarize meaningful changes.

### 5. Cross-reference usage

Inspect available usage metadata, active cron jobs, profile manifests, and explicit skill references. Treat absent tracking as unknown.

### 6. Classify findings

Separate broken, stale, overlapping, unneeded, ambiguous, and healthy skills.

### 7. Prepare decisions

Propose actions with evidence, risk, affected references, and an empty approve or deny decision field.

## Evidence discipline

Record the source, timestamp when relevant, scope, access limitations, and any contradiction for each load-bearing finding.
