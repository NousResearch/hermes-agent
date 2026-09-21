# Directus 11 permissions — the chain that actually grants access

Directus 10 attached permissions to a role. Directus 11 does not. A role is only a
label until a **policy** reaches it:

```
directus_users.role ──► directus_roles
                              │
                     directus_access  (junction: role | user  ×  policy)
                              │
                       directus_policies
                              │
                    directus_permissions  (collection × action × fields × filter)
```

Read it the other way when debugging a 403: the request was refused because no
permission row matched, because no policy is linked, or because the link points at
the wrong role.

| Table | What it holds | Endpoint |
|-------|---------------|----------|
| `directus_roles` | name, description. **No permissions.** | `/roles` |
| `directus_policies` | `admin_access`, `app_access`, `enforce_tfa`, `ip_access` | `/policies` |
| `directus_access` | `role` **or** `user`, plus `policy` and `sort` | `/access` |
| `directus_permissions` | `policy`, `collection`, `action`, `fields`, `permissions`, `validation` | `/permissions` |

`admin_access: true` on a policy bypasses `directus_permissions` entirely — never
give it to an agent account. `app_access: true` only means "may sign in to the
Studio"; an API-only service account does not need it.

## The `fields` trap

`fields` on a permission row is not optional decoration:

| Value | Meaning |
|-------|---------|
| `["*"]` | every field, including ones added later |
| `["id", "status"]` | those two fields only; others are omitted from the response |
| `[]` | **the primary key only** — reads return `{"id": …}` and nothing else |
| `null` | same as `[]` in practice; treat it as unset, not as "all" |

An empty list is the usual cause of "the row exists but every field is missing".
`permissions create` defaults to `["*"]` and warns when the caller passes an
explicitly empty `--fields`.

## Row-level rules

`permissions` (the rule) filters which rows the action may touch; `validation`
filters what a write may contain. Both take the standard Directus filter object:

```json
{"agent_id": {"_eq": "$CURRENT_USER"}}
{"status": {"_in": ["queued", "running"]}}
{"_and": [{"created_at": {"_gte": "$NOW(-1 day)"}}, {"archived": {"_eq": false}}]}
```

Dynamic variables: `$CURRENT_USER`, `$CURRENT_ROLE`, `$CURRENT_POLICIES`,
`$CURRENT_ROLES`, `$NOW`. A rule of `{}` means "all rows".

## Public access

The public role does not exist as a row. Unauthenticated access is a policy linked
in `directus_access` with both `role` and `user` set to `null`. Creating such a
link is how a collection becomes world-readable — check for one before assuming a
collection is private.

## Static tokens

A static token is a column on the user record (`directus_users.token`). It never
expires, so it is what a long-running agent uses.

- Set it with `PATCH /users/{id}` and a `token` value. A SQL `UPDATE` on the column
  does **not** work — Directus hashes and caches the value on write.
- It is returned once, by the call that set it; afterwards the API shows `"**********"`.
- Clear it by patching `token` to `null`.
- A token inherits every policy linked to its user *and* to the user's role.

## Collections that exist in the database but not in Directus

A table created with raw SQL DDL is invisible to `/items` until Directus has a
`directus_collections` row and `directus_fields` rows for it. Either create the
collection through `/collections` (which does both), or register the existing table
afterwards. `POST /schema/apply` with a diff from `GET /schema/snapshot` is the
migration path between instances; it is not needed for a single new collection.

## Quick 403 triage

1. `check` — is the token valid at all, and is it admin?
2. `users list --role <id>` — is the account in the role you think?
3. `access list --role <id>` — is any policy linked to that role?
4. `permissions list --policy <id>` — is there a row for this collection *and* this
   action? `read` does not imply `update`.
5. Look at `fields` on that row — an empty list answers "why is the row empty?".
