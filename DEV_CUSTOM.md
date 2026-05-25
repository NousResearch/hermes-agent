# Custom Branch Strategy

This project uses a two-branch strategy for maintaining local modifications while tracking upstream:

- `main` — pristine upstream mirror. Never commit directly here.
- `custom` — all local modifications go here.

## Remotes

- `origin` → https://github.com/RikoTsushima/mega-hermes (your fork)
- `upstream` → https://github.com/NousResearch/hermes-agent (official)

## Pushing your changes

```bash
git push origin custom         # push to your fork
```

## Merging upstream updates

```bash
git fetch upstream
git checkout main
git merge upstream/main        # keep main in sync with official
git checkout custom
git rebase upstream/main       # replay custom commits on top of latest official
# Resolve conflicts if any: git mergetool; git rebase --continue
```

## Alternative (if in a hurry)

```bash
git fetch upstream
git checkout custom
git rebase upstream/main
```

## Running

```bash
/usr/local/bin/hermes          # points to /usr/local/lib/hermes-agent/venv/bin/hermes
```

## Config

~/.hermes/config.yaml
~/.hermes/.env
