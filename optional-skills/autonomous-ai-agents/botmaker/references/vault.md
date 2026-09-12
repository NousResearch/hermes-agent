# Vault (`Bots/`)

When to load: documenting a certified bot in the fleet vault (optional — skip if the user keeps no vault).

Botmaker owns `Bots/` in the fleet vault. This layer assumes an Obsidian vault because that is what we run; any notes system with frontmatter and links works — adapt the conventions, keep the rule: **nothing gets documented before certification.**

Vault path:

`<your-vault-path>`

Quote it in shell if it contains spaces (cloud-synced vault paths often do). File tools take the absolute path.

Load your notes skill (if you have one) for filesystem conventions. This file is the *bot* layer on top.

## Section owners

- `Bots/` — `@botmaker`: specialist notes, `making-bots.md`, the roster.
- Other sections belong to whoever writes them — a planning bot's `Plans/`, a nightly tidy cron's `Sessions/`. Read any section; write only your own. Do not own `Servers/` or `Services/` except links.

## When to write

**After certification.** Seeding a profile is not a ship. Uncertified bots do not get `Bots/<name>.md` or a roster row. Personas and the daily driver never get one at all — specialists only.

Exceptions are dated and expire; none should stay active. (Ours: botmaker's own note briefly existed as *certification pending* during bootstrap. Do not copy that onto children.)

The drift checker verifies rostered profiles exist in the selected installation. It does not require every profile to be rostered: certification determines membership, not directory discovery. Add the new specialist's row as part of the post-certification checklist.

## Notes

| Note | Frontmatter | Role |
|---|---|---|
| `Bots/<name>.md` | `type: bot-reference`, `created`, `updated`, `profile`, `host`, `status`, `tags` | Specialist reference after certification. Not a SOUL dump. Training loop + runbook pointer + independence pin + pitfalls. |
| `Bots/making-bots.md` | `type: living-guide` | Human guide + fleet roster + changelog. Roster table at top — add a row when a bot **ships**. Changelog at bottom, append-only. |
| `Home.md` | `type: vault-root` | One index line under **Bots**. Never append running logs. |
| `Sessions/YYYY-MM-DD-<slug>.md` | `type: session` | What we did this session. Wikilink the bot note + `making-bots`. One Index line, newest first. |

`status:` values: `certification-pending` \| `shipped — YYYY-MM-DD`. `host:` names the machine the profile lives on.

Match your vault's existing tone: YAML frontmatter, wiki-style links, short, no credentials. Once your first specialist is certified, its note becomes the worked example — steal structure, not facts.

## Roster

`making-bots.md` table:

```markdown
| `@name` | `name` | <model> / <provider> | one-line job |
```

`Home.md` **Bots** section: `[[name]] — \`@name\`, one-line job`.

The roster table on `making-bots.md` is the fleet-state source of truth — which profiles exist, current pins (including your default profile's), peer and memory coverage. Enumerate fleet state there and nowhere else; a fleet enumeration baked into a skill rots the day the fleet changes.

## One home per rule

Every rule has exactly one home — method in the skill tree (`$hermes_root/skills/autonomous-ai-agents/botmaker/`, write the canonical path; profile files are symlinks), fleet state in the roster table on `Bots/making-bots.md`, history in its changelog. When a lesson lands, patch its one owner and add one changelog line naming that owner.

If `making-bots.md` is fresher than this tree, backport before adding anything new — lockstep failed one-directionally for us until we adopted this rule.

Ops source of truth for a *child* bot is that child's skill, not the vault note. The vault note points at the skill.
