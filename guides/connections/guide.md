# Connections

A connection folder is your operating knowledge for a service, not proof of
access. Reach services through native CLI logins, configured secrets, MCP or
the browser. There is no connection store or connection-management tool.

The user's explicit instruction always overrides the defaults below.

## Check What Exists First

List `{profile_home}/connections/` and read the service's `manual.md` before operating
it. Follow its account and method instructions, then verify actual access.
The opening manual listing is frozen per conversation; a missing entry does
not mean no manual exists. A missing folder does not mean no login exists.

## Choosing the Surface

**Connect through the lightest official surface that covers the work.**
Research the provider's official documentation and judge coverage against the
ongoing work, not just the operation in front of you. Never connect by guesswork.

Prefer the official CLI when it covers the work; otherwise use the API, an
appropriate MCP server, or the browser. Do not choose a lighter surface that
cannot do the job. Using two surfaces is legitimate when neither covers it all.
An unofficial integration needs the user's go-ahead when no official surface
covers the work.

Use native Hermes MCP discovery and execution. Check the server's current tools
before relying on remembered names or arguments; do not assume mcporter exists.

Research settles the mechanics before the person hears a question: where to
sign in or create the key, and the minimal permissions the work needs. Ask the
person for intent and approval, not to research the provider for you.

## Whose Account

When the user says whose account to connect, that decides it. Otherwise
decide from what the task must reach: work you perform as yourself — your own
inbox, your own posts, your own workspace — is best served by an account of
your own, while work over other people's data — their calendars, their
meeting recordings, their inboxes — needs those people's accounts or keys,
because your account cannot see it. When the choice isn't obvious from the
task, recommend one and confirm before connecting.

## Authentication

Use the service's native login, the configured secret source, or MCP's supported
authorization flow. Keep CLI tokens in the CLI's own storage and private keys
in their intended local storage. API keys belong in the configured secret
manager or profile secret configuration, never in manuals, scripts or chat.
Check the environment variable the actual CLI or SDK expects.

For user-supplied secrets, direct the person to the local secret configuration
or configured secret manager. Do not invent a secure request link or ask them
to paste reusable secrets into chat. Device-login links and pairing codes can
be relayed; one-time verification codes are not saved as credentials.

Never print a secret to test it or assume output will redact it. Verify the
account and access through a harmless read-only operation. Environment tokens
can override a CLI's saved login; distinguish them when checking or changing
accounts.

Repair access through the method that established it. Do not switch surfaces
because one broke once. When asked to disconnect, revoke the provider grant
where supported and clear the relevant local authentication; deleting a manual
does neither.

## The Manual

After first verifying access, create `{profile_home}/connections/<service>/manual.md`
if it does not exist. Use one folder per service, even when it has several
accounts or access methods. There is no generated `credentials.md`.

At minimum, record the account, whose it is, what work it serves, and the
verified access method. Add only what the next session would otherwise have
to rediscover: working commands, endpoints, instance URLs, account-selection
steps and pitfalls. Name credential locations or environment variables, never
secret values. Do not record a login as working merely because it was started.

Keep supporting instructions in `references/` and reusable helpers in
`scripts/`, linked from the manual. How a service works belongs here; the work
an area owns belongs in its responsibility. Before finishing, correct wrong
instructions and record useful discoveries in the existing manual rather than
creating a competing one.
