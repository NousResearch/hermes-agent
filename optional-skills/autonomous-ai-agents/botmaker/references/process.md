# Process

When to load: scaffolding, pinning, linking, or verifying a specialist bot profile — the full command mechanics.

Interview → SOUL draft → human sign-off → scaffold → child's Bot Chat (first look, failure path, client ping) → child rewrites runbook/MEMORY → vault.

You draft. The child earns. If you fill `MEMORY.md`, you faked the training loop.

## Installation root

Before any write, discover the root through the same Hermes CLI that owns this session. `hermes -p default config path` uses Hermes's root resolver even from a named profile; it respects custom homes. Do not overwrite `HERMES_HOME` before discovery.

```bash
unset hermes_root
root_config="$(hermes -p default config path)" &&
hermes_root="$(python3 -c 'from pathlib import Path; import sys; p = Path(sys.argv[1]); assert p.is_absolute() and p.name == "config.yaml", "unexpected config path"; print(p.resolve().parent)' "$root_config")" &&
test -d "$hermes_root"
```

If any step fails, stop. Verify the resolved root is the intended installation before creating profiles, writing config/SOUL/memory, or linking skills. Never fall back to another installation or create a replacement root to make a missing path work. `hermes_root` is a local shell variable, not a new Hermes setting; resolve it again in a fresh shell. File tools and persisted instructions need the concrete absolute path, not the literal variable. Keep the default SOUL at that root off-limits. Do not change approval settings to perform this discovery or provisioning.

## Independence

If the bot operates a service — CLI, HTTP API, GUI control socket, ComfyUI, whatever — **do not run the bot on that service's inference path.**

`@gpuops` set the pattern: when our default Hermes pointed its model at the local inference server, default died whenever that server died — so the operator that *manages* the server is pinned to a hosted provider. Pins change (ours have); current pins live in your vault roster (`Bots/making-bots.md`) — the invariant doesn't. A ComfyUI specialist is the same class: talks *to* the API; must still think when the GPU box is busy or down.

Same machine is fine; same inference path is not.

Pin `model.provider` / `model.default` with `hermes -p NAME config set`. Never hand-edit `config.yaml` — `config set` only (your human may hand-edit their own; you do not). Do not copy `auth.json` between profiles — shared pools already rotate. Never print `auth.api_key` / `secret_key`.

**Create leftover:** `hermes profile create` copies your default profile's model block, whatever it currently is (for us, historically a local `base_url` — e.g. `http://127.0.0.1:8000/v1`). After pinning you **must** unset `model.base_url` and `model.api_key`. Verify:

```text
hermes -p NAME config get model
```

Must print only `default` and `provider`. If a `base_url` survives, the "independent" bot rides default's provider.

**Peers after create (site-specific; adapt or delete).** Peer *validation* reads the shared root config; *delivery* reads the *sending profile's* config — a new bot will dispatch `message_agent` to a peer as valid and then die silently in delivery until the peer is registered on that profile. Register every peer the bot must reach, immediately after create (our fleet invariant: every rostered profile has every peer it must reach — roster: vault `Bots/making-bots.md`):

```text
hermes -p NAME peer add <peer> --url <peer-url> --key "$(head -1 <keyfile>)"
```

Verify: `hermes -p NAME peer list` shows the peer. Do not print the key. If your fleet has no external peers, delete this step.

**Honcho peer after create (only if you use the Honcho memory provider).** Never `--clone`, so no fresh AI peer. `hermes honcho sync` backfills (idempotent), then set the provider — create leaves it empty:

```text
hermes honcho sync
hermes -p NAME config set memory.provider honcho
```

Verify `hermes honcho peers` shows the profile with a real user peer (not `(not set)`). If `hermes honcho --target-profile NAME status` says not connected, inherit `apiKey`/`baseUrl`/`requestTimeout` from host `hermes` into `hermes_<name>` in `$hermes_root/honcho.json`. Do not print them. Docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers#new-profile-fresh-honcho-peer

## Scaffold sequence

Alias preflight (next section) first, then:

```text
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes profile create NAME --no-skills --description "one or two sentences"
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes -p NAME config set model.provider <provider>
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes -p NAME config set model.default <model>
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes -p NAME config unset model.base_url
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes -p NAME config unset model.api_key
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes profile describe NAME --text "same one or two sentences"
```

Site-specific follow-ups — ours, shown as worked examples; adapt or delete (details above):

```text
# external peer bridge:
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes -p NAME peer add <peer> --url <peer-url> --key "$(head -1 <keyfile>)"
# Honcho memory provider:
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes honcho sync
HERMES_HOME="${hermes_root:?resolve installation root first}" hermes -p NAME config set memory.provider honcho
```

Inside a named profile `$HERMES_HOME` is `$hermes_root/profiles/<name>`. Hermes already resolves sibling profiles against their installation root; the explicit prefixes above pin commands to the verified root rather than replacing it with a hardcoded default. Canonical skills stay under `$hermes_root/skills/`.

`hermes profile describe` writes the roster one-liner (`profile.yaml`) other bots read — not a group-chat log. It does not set the display **title**, which lives in `profile.yaml`: write `ui_meta.hermes-bots.title` there after describe (`config get ui_meta` reads empty until then). And don't "fix" `config get toolsets` off the wrapper key — the spec list is `platform_toolsets.cli`.

## --no-skills

**Alias preflight before `profile create NAME` — mandatory.** Hermes writes `~/.local/bin/<name>` (`exec hermes -p <name>`), and `Path.write_text()` follows symlinks, so a wrapper can overwrite a real CLI in place — war story: a profile that shared its name with an installed CLI overwrote the real 107MB binary living at the far end of a `~/.local/bin` symlink; the alias wrapper was written *through* the link, straight over the binary. Built-in `which` collision check is useless under launchd PATH (`/usr/bin:/bin:/usr/sbin:/sbin` — no `~/.local/bin`); do not trust this process's `which` either.

```text
zsh -lc 'which -a NAME'
# and, even if which is empty:
test -e "$HOME/.local/bin/NAME" || test -L "$HOME/.local/bin/NAME"
```

If **either** finds anything: `--no-alias`, or `hermes profile alias NAME --name CUSTOM` only after the same two checks pass for `CUSTOM`. `~/.local/bin/NAME` as a **symlink** is the binary-destroying case. Common landmine: some popular AI-coding CLIs install their command as such a symlink; so does any CLI whose entry point lives in, or resolves through, `~/.local/bin`. Hermes only reserves `hermes`, `default`, `test`, `tmp`. If a CLI later acts like Hermes (`exec hermes -p` in a tiny file): delete the impostor, reinstall the real tool, rename or drop the Hermes alias.

Never `--clone` / `--clone-from`. Desktop clone is the same bug (copies SOUL, skills, memory, leftover model block). Desktop create-empty is fine.

`--no-skills` writes `.no-bundled-skills`; `hermes update` will not dump the bundle on this profile. Essential `hermes-agent` still seeds as a **real copy** — leave it; sync owns it. File-linking it to canonical instead is also acceptable (two of our bots do; the link tracks canonical) — never a directory symlink.

A review-lane bot still needs its review skill **discoverable** on the profile even when it is not pinned. Any automation that injects `--skills <review-skill>` dies at init on a `--no-skills` tree that cannot resolve the name (`Unknown skill(s): <review-skill>`). Link the canonical skill file-level. Do not always-load it.

Extra skills are context tax and off-mission bait. Three is already a lot (`botmaker` itself: its own runbook + `hermes-agent`, plus a notes skill if you keep a vault). A coordinator seat may earn more; each one is still a tax — trim when context strain shows.

## Shared skills (file-level links)

Canonical tree: `$hermes_root/skills/<category>/<name>/` (default Hermes loads this).

Profile path: `$hermes_root/profiles/<bot>/skills/<category>/<name>/` — **real directories**, every *file* a relative symlink.

Never symlink the skill directory itself. Hermes venv is Python 3.11; `Path.rglob("SKILL.md")` does not descend directory symlinks → empty index → the bot cannot see the skill. File-level `SKILL.md` inside a real dir **is** found.

`skill_manage` delete refuses if the skill **dir** is a symlink (will not `rmtree` through a redirect). Sandbox skill mounts skip symlinks — acceptable for a runbook the bot reads; do not put secrets in the skill either way.

Linker (this skill):

```text
python3 "${hermes_root:?resolve installation root first}/skills/autonomous-ai-agents/botmaker/scripts/link_skill_tree.py" \
  "$hermes_root/skills/<category>/<name>" \
  "$hermes_root/profiles/<bot>/skills/<category>/<name>"
```

Writes through the profile path hit the same inode. If a rewrite unlinks first, `ls -l` shows a regular file instead of `l` — that is a second copy starting again. Relink; do not keep both. `skill_manage` patch on a linked `SKILL.md` does that unlink — patch canonical with the file `patch` tool instead. `skill_manage` `file_path` and `skill_view(..., file_path=…)` on `references/` both fail: `Path.resolve()` follows the file symlink into `$hermes_root/skills/…`, then `relative_to` the profile skill dir rejects it. Workaround: `read_file` the canonical path. Do not tell bots to `skill_view` linked references until Hermes allows it.

Child `SOUL.md` writes: the gate is the signature — mechanics in `soul-craft.md` §Human gate.

## Kanban intake (if your fleet provisions from cards)

Cards assigned to `botmaker` arrive as headless workers (`HERMES_KANBAN_*` set; no approval surface — that is why the write prompt is off for this profile; see `soul-craft.md` §Human gate, and `profile/config.yaml` in this repo). A human-signed spec on the card is the signature: no signature, no scaffold, no SOUL write. End every worker run with exactly one terminal kanban call (`kanban_request_review` **is** that call in review lanes). Strip `HERMES_KANBAN_*` before any sibling spawn or sibling kanban CLI — inherited env silently retargets boards (`env -u HERMES_KANBAN_DB -u HERMES_KANBAN_BOARD … hermes kanban --board <slug>`).

## Generate-the-experience

Do not hand the child a finished runbook and a stuffed MEMORY.

Kickoff in the child's Bot Chat:

1. First look at the real box or job (status script, live files, not vibes).
2. One failure path, practiced (the thing that will hurt at 2am).
3. Client ping if there is a client (`@hermes` was `@gpuops`'s check that bot-to-bot actually works).

The child patches its own skill + MEMORY. You may send one `message_agent` ("first look") and then get out.

Certified when it has done the job against the live system and rewritten the stale bits. A clean first look is enough if your human ships it ("document it in the vault"). Uncertified profiles do not get a roster row unless your human overrides. Do not invent a 2am failure to practice.

## Bot-to-bot

Useful as a **health primitive** (can this bot reach another bot?) and as a client-path check. Not as a substitute for the child doing the work.

Do not impersonate children. Do not DM them a finished MEMORY.

## Blast radius

Specialists + living guide + this skill. You are the bot coordinator — of bots, nothing else; `@hermes` is not. Touch a shipped child's SOUL only when your human explicitly sends you there. Never default Hermes SOUL. There is no doctor class — each specialist diagnoses its own job; you diagnose the fleet.

Do not spawn `botmaker-2`.
