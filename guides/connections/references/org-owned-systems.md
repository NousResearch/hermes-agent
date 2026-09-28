# Connecting the Organization's Own Systems

This applies when the organization owns the backend — an internal tool, a
CRM it built for itself, its own product's production database. A
third-party provider decides which surfaces you get; an owner can mint one
no provider offers: direct access to the database. For data work —
reading, analyzing, moving records — that is usually the lightest surface
available: one query per operation, database clients installed on the
computer, nothing for anyone to build. Recommend it over having the
organization stand up an API or MCP server just for you, unless they want
that surface for reasons of their own.

## The Scope Lives in a Credential, Never in a Promise

Recommend the scope a person has in the tool's own dashboard: read and
row-level writes, no schema changes — a dedicated role or key whose
permissions make a migration impossible rather than forbidden. The user
narrows this to read-only when the work is analysis, or widens it; their
word decides. Never accept an owner or superuser credential, whatever the
agreed scope — a powerful credential handled "carefully" is scope by
promise.

## Connecting Is Ordinary Research

How the database is reached is the backend's decision, and you research it
like any surface — the backend's own documentation, never guesswork. A
database that issues connection strings (Postgres and its managed
platforms) means a dedicated role and a URL: tell the user exactly what to
create and grant for the agreed scope — a precise ask gets filled in one
pass. A managed backend that exposes no database credential — a Convex
deployment, a Lovable Cloud project — is the provider telling you access
routes through its official surfaces: weigh its API, MCP server, and CLI
as `{guides_root}/connections/guide.md` describes, and prefer a read-only mode where
one is offered and the agreed scope is read-only.

The credential goes in native secret configuration like any other — one secret field named
so the injected variable reads naturally (`psql "$ACME_CRM_DATABASE_URL"`)
— with the organization as owner. Verify like every surface: one harmless
read proves the connection. A read cannot prove the scope — a superuser
URL passes it just as well — so ask the backend what the credential may
do, and one that turns out more powerful than agreed goes back for
reissue, never into careful hands.

## Afterwards

The org's particulars — the schema that mattered, the queries that worked,
the platform's quirks — are exactly what the service's manual is for.
Record them in `{profile_home}/connections/<service>/manual.md`; the credential stays in
native secret configuration.
