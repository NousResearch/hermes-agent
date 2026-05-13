# Requester GitHub Identity Mapping

Slack-triggered Hermes sessions can attribute PR-producing work to the human who requested it while the bot still opens the PR.

Mappings live in `config/requester_identities.json` and are keyed by Slack user ID:

```json
{
  "U0123456789": {
    "name": "Jane Engineer",
    "email": "jane.engineer@example.com",
    "github_login": "jane-engineer"
  }
}
```

When a Slack requester is mapped, Hermes injects the following into terminal subprocesses for that session:

- `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`
- `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL`
- `HERMES_REQUESTER_GITHUB_LOGIN`

Agents are also instructed to append a `Co-Authored-By: citizen-wall-e <bot-email>` trailer to every commit message and to start PR bodies with `Requested by @<github-login> via Slack`, assigning the PR to that login.

If a Slack requester is not mapped, Hermes fails fast with a clear reply instead of silently falling back to bot-only attribution. Unattended cron/local jobs have no Slack requester and keep the existing bot-authored behavior.

Use the email attached to the human's GitHub account; otherwise GitHub will not render their commit avatar.
