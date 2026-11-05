# Bo2bot credentials — where the file lives

Hermes does **not** store your live credentials inside the skill folder. They
live on the host at a fixed path under your Hermes home directory.

## Host path (the real credentials file)

```
~/.hermes/secrets/bo2bot.env
```

The skill declares this in `SKILL.md` as `required_credential_files`:
`secrets/bo2bot.env` — that path is **relative to `~/.hermes/`**, not relative
to the installed skill directory.

| What you might look for | Exists? |
|-------------------------|---------|
| `~/.hermes/secrets/bo2bot.env` | **Yes — put credentials here** |
| `~/.hermes/skills/.../secrets/bo2bot.env` | **No — wrong location** |
| Hermes global config (LLM provider keys only) | Different file — not where Bo2bot credentials go |

## Template (safe to read — no live secrets)

Copy from either bundled sample:

- **In the installed skill:** `references/bo2bot.env.sample`
- **In the GitHub kit (before install):** `hermes/bo2bot.env.sample`

```bash
mkdir -p ~/.hermes/secrets
cp "${HERMES_SKILL_DIR}/references/bo2bot.env.sample" ~/.hermes/secrets/bo2bot.env
# edit four BO2BOT_* values, then:
chmod 600 ~/.hermes/secrets/bo2bot.env
```

If you downloaded `bo2bot.env` from the portal, copy that file directly to
`~/.hermes/secrets/bo2bot.env` instead.

## Verify (non-interactive)

```bash
python3 "${HERMES_SKILL_DIR}/scripts/bo2bot_cred_manager.py" --check
```

Exit 0 → proceed with login:

```bash
eval "$(bash "${HERMES_SKILL_DIR}/scripts/bo2bot-login.sh" --export)"
```

Do **not** ask your human to paste `BO2BOT_AUTH_KEY` into chat when
`--check` succeeds.
