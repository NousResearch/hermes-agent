# pstack

Harness-neutral pointer to [pstack](https://github.com/cursor/plugins/tree/main/pstack) (Lauren Tan / poteto, MIT). The Cursor plugin is the full pack (`/poteto-mode`, `/how`, `/unslop`, `/setup-pstack`, …). This kit does **not** copy that tree.

Child repos that already installed pstack keep it: extra `skills/how`, `.cursor/skills/*`, plugin settings are `local` in `.verified-oss-loop/inventory.yml` and are never overwritten.

## If pstack is already installed

Use it for rigor. Typical entry: `/poteto-mode` with a checkable outcome. Still follow this protocol:

- One claim. Isolated branch.
- Fail, then pass (`skills/tdd/SKILL.md`).
- Smallest complete change (`skills/anti-slop/SKILL.md`).
- **Workers never merge.** pstack shipping/autopilot playbooks that land PRs are maintainer-only. They are not worker authority.

Do not run `/setup-pstack` or `/add-plugin pstack` from an unattended claim unless a human asked.

## If it is not installed

Do not vendor 100+ plugin files into the repo. Fall back to this kit: orient → TDD → anti-slop → verify. Maintainers who want the plugin install it in Cursor (`/add-plugin pstack`).

## Do not

- Copy pstack playbooks that merge `main`.
- Duplicate `AGENTS.md` into a second instruction file because a playbook asked.
- Treat plugin presence as evidence on the receipt.
