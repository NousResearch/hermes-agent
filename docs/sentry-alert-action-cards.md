# Sentry alert triage action cards

Hermes webhook routes can attach action buttons to cross-platform deliveries by
adding an `actions` list under `deliver_extra`. Telegram renders these as inline
buttons. The first supported action is a Sentry `Create PR` button that stores
the webhook payload locally and starts a detached Codex task when an authorised
user taps it.

Example route:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        sentry-alerts:
          secret: "${SENTRY_WEBHOOK_SECRET}"
          prompt: |
            🚨 Sentry · {project}

            **Signal:** {title}
            **Issue:** {url}
            **Environment:** {environment}

            Suggested next step: review the issue and create a PR if this is a code-shaped production regression.
          deliver: telegram
          deliver_extra:
            chat_id: "8046206740"
            actions:
              - label: "Create PR"
                kind: sentry
                action: create_pr
```

Repo mapping lives at `~/.hermes/sentry_repo_map.json`:

```json
{
  "incremnt": "/home/colm/onemore",
  "scenr": "/home/colm/scenr"
}
```

When tapped, Hermes launches `scripts/sentry_create_pr_from_packet.py`, which
writes logs under `~/.hermes/sentry-actions/<id>/` and starts a fresh detached
tmux/Codex task. The generated prompt explicitly says to commit, push, AND create
PR, and to not merge.

Limitations:

- Telegram action buttons are implemented first.
- Discord delivery currently receives the text but not arbitrary action-card
  components; add Discord component rendering before relying on this in Discord.
- Sentry payload shape varies by integration, so route prompts should include the
  fields your alert actually sends.
