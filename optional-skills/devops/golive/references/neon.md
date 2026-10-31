# Neon

Automated database provider with mock coverage and an approved Netlify + Neon throwaway live run
on 2026-09-23. New Free-project creation, connection transfer to both hosting environments and
read-only database/role verification passed after the `--data=-` parser fix. Separately approved
schema, two-session API and browser CRUD/refresh checks also passed. Exact-resource cleanup later
passed under its own approval using a supervised fixture helper (Neon projects remain a manual
handoff in `golive teardown`).
This validates the tested Free organization and app, not every account or workload. It supplies
server-only pooled `db.url` and direct `db.directUrl`. It does not supply Supabase keys or migrate a
Supabase SDK application, and it does not provision Neon Auth, Data API or application schemas.

1. Have the human install `npm install -g neon` and run `neon auth` in a **separate terminal**.
   golive reuses the official CLI's login without reading its credential cache. The CLI must support
   `neon api`. Alternatively use `NEON_API_KEY`: on macOS run
   `credentials --prompt NEON_API_KEY --json` for private native entry. Their own editor is the
   fallback if unavailable, unsupported, or preferred; follow
   [How the human connects accounts](../SKILL.md#how-the-human-connects-accounts) for fallback,
   replacement and cancellation. Never print or
   inspect credential contents, pass keys on argv, or ask for a chat paste. An explicit token wins
   over the CLI login; fix a rejected token instead of silently switching identities.
2. Choose the exact organization (`neon.organizationId`, or non-secret `NEON_ORG_ID`). New project
   creation requires its API-confirmed `free` plan; several Free organizations need a choice. No
   upgrades, purchases or alternate-organization fallback are automated. Free quotas remain Neon
   enforced. New projects use `aws-us-east-2` unless `neon.region` is set, with fixed
   main/neondb/neondb_owner and 0.25 CU compute defaults. Do not set branch/database/role overrides
   for a new project.
3. For an existing project set `projects.db` to its exact ID, plus `neon.branchId`, `neon.database`
   and `neon.role`. Ask which existing branch the human wants; never infer a production/default
   branch. Same-name adoption without selectors is refused. These non-secret selectors appear in
   env approval. Both deployment targets use this selected branch; per-target branches are not
   automatically created.
4. Show the plan's account, org/project/region and connection selectors, obtain explicit approval,
   then apply. golive keeps returned credentials inside Secrets and transfers them directly to the
   chosen hosting env store. Never run a raw connection-string command in Hermes-visible output.
   A failed, timed-out or malformed creation response may still mean the create succeeded:
   inspect the exact approved organization and re-plan before another create. Pending project
   and operation IDs are saved for recovery; no password resets or automatic remote rollback.
5. `db-connection` uses one fixed read-only SQL transaction against the API-verified endpoint,
   and checks the exact database and role. It runs only on verified Free organizations. A pass
   proves connectivity from golive, **not** deployed app queries, schema, migrations, RLS or user
   isolation. Approve schema work separately and test real application behavior. Unsupported
   Supabase-specific checks remain skips.

Doctor success proves authentication reads, not create permission or quota. CLI failures emit only
fixed diagnostic categories; `unknown` does not mean the user needs to log in again. CLI 5.0.1
handler-only stdin testing initially missed the argv bug; exact parser coverage now verifies the
equals-form marker. The body stays in Secret stdin. Argument errors require checking syntax,
not automatically asking the user to upgrade.
If diagnostics remain unknown on a separately approved retry, return only an in-memory allowlisted
HTTP status/category. Never expose or save raw provider/debug output, and never retry a create just
to inspect its error.

Neon's official [CLI](https://neon.com/docs/reference/neon-cli),
[API](https://neon.com/docs/reference/api) and [remote MCP](https://mcp.neon.tech/mcp) are available.
MCP is optional and is not this adapter's transport. `neon init` or MCP setup can mint credentials
and change agent config; do not run it as a read-only login check.
