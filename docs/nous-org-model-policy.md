# Honouring the Nous org model policy in the pickers

> **Audience:** Contributors touching Nous model selection
> **Source files:** `hermes_cli/auth.py` (`_login_nous`, `fetch_nous_models`,
> `_prompt_model_selection`), `hermes_cli/models.py` (`fetch_models_with_pricing`,
> `get_pricing_for_provider`, `union_with_portal_*`, `partition_nous_models_by_tier`),
> `hermes_cli/model_setup_flows.py` (`_model_flow_nous`),
> `hermes_cli/model_switch.py` (`list_authenticated_providers`),
> `hermes_cli/web_server.py` (`/api/model/recommended-default`),
> `hermes_cli/nous_account.py` (`_info_from_valid_jwt`)
> **Related:** Inference gateway PR #164 (filters `GET /v1/models` by org policy),
> NAS #941 (team admins restrict providers), NAS `openrouter-provider-map`
> (publishes the model→providers map the gateway filter needs)

## What changed upstream

A Nous team admin can restrict which models and which serving providers their
org may use. The inference gateway applies that policy to `GET /v1/models`, so
an authenticated catalog read returns only what the caller may actually reach.
Blocked models are **omitted** — the row is skipped, no marker field is added
(`api/src/handlers/models.ts:99-138`). An anonymous read is still allowed and
still returns the full catalog (`api/src/app.ts:309-313` — no auth middleware
on the route).

Two things bound how urgent this is.

**The gateway is authoritative and this is cosmetic.** Asking for a hidden
model is refused at request time with `403 model_blocked_by_org_policy`
(`api/src/middleware/model_entitlement_gate.ts:337-356`). The listing fails
open; the request gate fails closed. Nothing here is a security boundary — the
cost of a wrong list is a predictable 403, and the gateway PR states that
tradeoff deliberately. This document is only about the client showing the
right list.

**It is inert today.** PR #164 is merged, but is switched off until NAS
publishes the policy fields and the provider map, and the admin surface sits
behind the `org-model-policy` Vercel flag. The `openrouter-provider-map` branch
is the publisher half (a daily cron writing `openrouter_model_providers` to the
entitlement Redis). Until that lands, every caller — anonymous and
authenticated — gets the same unfiltered list. **No change here is verifiable
end to end yet; every test mocks the filtered response.**

## Where we stand

Four surfaces list Nous models. **None of them is filtered.**

| surface | builds its list from | filtered |
| --- | --- | --- |
| Login (`_login_nous`, `auth.py:9383`) | `get_curated_nous_model_ids()` ∪ Portal recommendations | no |
| `hermes model` (`_model_flow_nous`, `model_setup_flows.py:399`) | same | no |
| `/model` picker (`list_authenticated_providers`, `model_switch.py:3062`) | same | no |
| Dashboard onboarding (`web_server.py:7486`) | same | no |

All four seed from the docs-hosted manifest and union the Portal's
`recommended-models` endpoint. Neither source is authenticated, so org policy
has no effect on any list a user picks from.

`cached_provider_model_ids("nous")` — which *does* reach the authenticated
`fetch_nous_models` — is not consulted by any of them. The `/model` picker
handles nous in its own branch that deliberately bypasses it, and nous cannot
reach the generic pathway at `model_switch.py:2898` because line 2861 skips
every non-`api_key` provider. Its only caller for nous is the background
prefetch (`model_switch.py:2390`), which writes an entry nothing reads.

Two things that are already fine, and should stay that way:

- `nous` is **not** in `_MODELS_DEV_PREFERRED`, so no models.dev entries are
  merged on top of the live list.
- The nous fallback ladder in `provider_model_ids` is a *chain* (live →
  manifest → in-repo snapshot), not a merge, so a successful live fetch is
  used exclusively.

---

## Fix 0 — put auth state in the pricing cache key

**This is a prerequisite for fix 1, and worth landing on its own merits.**

**Problem.** `fetch_models_with_pricing` caches on the base URL alone, and the
cache check happens *above* the point where the `Authorization` header is built
(`models.py:2404`):

```python
cache_key = (base_url or "").rstrip("/")
if not force_refresh:
    cached = _cached_catalog(cache_key)
    if cached is not None:
        return cached
...
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
```

`_pricing_cache` is process-lifetime with no expiry for a non-empty result
(`models.py:2231-2253`). So whichever read of a given base URL lands first —
authenticated or anonymous — answers every later read in that process,
whatever key it passes. An anonymous read landing first (the auxiliary-model
path in fix 2 is one) makes a later authenticated read return an unfiltered
list without touching the network. A fix built on this cache looks like it
works and does not.

**Do.** Fold auth state into the cache key. Distinguishing authenticated from
anonymous is enough — the token value need not be in the key, and keeping it
out avoids hashing a secret.

**Do** update `agent/credits_tracker.py:257`, which reaches into the private
`_pricing_cache` dict assuming one entry per base URL.

**Test.** An anonymous read followed by an authenticated read of the same base
URL issues two requests and returns two different lists. Independently
testable today, unlike everything below.

## Fix 1 — narrow each list to the org's policy

**Problem.** All four surfaces build their list from
`get_curated_nous_model_ids()` unioned with the Portal's `recommended-models`
endpoint. Neither is authenticated, so org policy has no effect on the model a
user picks — which is the model they then use. The Portal endpoint compounds
it: it takes no auth and no parameters, returns one globally CDN-cached payload
for the whole platform, and is invalidated only by admin pricing edits — never
by a policy change. It can put a hidden model straight back into a list. There
is no policy-aware variant of it and no parameter that would make one.

Each surface, however, already fetches `/v1/models`.
`get_pricing_for_provider("nous")` calls `fetch_models_with_pricing`, which
reads that endpoint and returns `{model_id: {...}}`, and already resolves
credentials (`_resolve_nous_pricing_credentials`), so it is already the
authenticated read. Its keys are the reachable set.

**Do.** Use that set to *narrow* each list, keeping the curated order.
`nous_policy_allowed_ids()` obtains the set; `restrict_to_nous_policy()`
applies it. Both live in `models.py`, and the fetch reuses the pricing cache
entry the surface already populates, so no surface makes an extra request.

**Do not** replace a list with the response's keys. Every surface shows the
curated agentic list in curated order deliberately — the live catalog is a
large alphabetical dump of vendor-prefixed models, and swapping it in is the
regression `model_switch.py:3070` records.

**Do not** treat the policy set as subtract-only either. An allowlist can name
only models the curated manifest has never heard of, and intersecting alone
then empties the picker — strictly worse than an unfiltered list, because the
models the org may use are the ones dropped. Fall back to the reachable set
itself in exactly that case.

Only when the intersection is empty, only when the set is small enough to be
an allowlist rather than a whole catalog, and **only for the list a user picks
from**. Callers opt in per list. Any list whose emptiness carries meaning must
not get the rescue: a paid-tier user's unavailable list is legitimately empty,
and rescuing it reads that as "nothing survived" and fills the picker's
unavailable block with the entire reachable set.

A size cap alone does not make this safe — a jurisdiction filter can leave few
enough models to pass it — and neither does gating on an empty intersection,
because an intentionally empty input is indistinguishable from a fully filtered
one. The opt-in is what separates them.

**Do not** narrow a list on evidence that cannot support it.
`nous_policy_allowed_ids()` returns `None` — meaning "leave the list alone" —
in three cases, and each matters:

- **The org has no policy, or the token is too old to say.** Gated on the
  `policy_present` claim (fix 4). For an unrestricted org — the common case —
  filtering buys nothing and risks dropping a Portal recommendation the
  gateway catalog has not caught up on yet. This keeps the change a no-op for
  everyone the policy does not apply to.
- **Credential resolution failed**, so the read was anonymous and therefore
  unfiltered. A full catalog must not be mistaken for a filtered one. A stated
  degradation, not a silent one.
- **The read came back empty**, which is a fetch failure, not an org that may
  reach nothing.

A `:free` sibling is kept when its base model is reachable, mirroring the
gateway, which admits a row when any of its requestable ids passes and treats
anything unknown as a keep — "over-listing costs a 403 from the authoritative
gate, while hiding a row the gate would serve is unrecoverable from the client"
(`api/src/libs/catalog_policy.ts:74-78`). Prefer over-listing here too.

**Test.** With a policy hiding model X: X is absent from each of the four
lists, and no surface makes more Nous requests than it does today. With no
policy, with credentials broken, or with an empty read, every list is byte-for-
byte what it is today. A model the Portal flags as free but the org hides stays
out; curated ordering survives filtering.

## Fix 2 — audit the other readers of the pricing map

**Problem.** `fetch_models_with_pricing` is shared, so any caller that treats
its keys as "the models that exist" inherits whatever authentication the first
caller happened to have. Fix 0 stops the *authentication* from leaking between
callers; this fix is about which callers may treat the map as a source of ids
at all.

**Do.** Make the map a lookup *for* ids already in the list, never a source of
ids. Two consumers are already correct and should stay that way:
`partition_nous_models_by_tier` only looks up ids it was given, and the
`union_with_portal_*` pair only ever writes into the map — their id-widening
comes from the Portal endpoint (fix 2), not from the map.

The one that is wrong is `agent/auxiliary_client.py:869-908`
(`_fast_model_from_catalog`), which iterates the map's keys directly as its
candidate list off an anonymous read. Reachable for nous on the titling path,
where it can select a policy-hidden model that then 403s at request time.

**Test.** With credentials broken so the read falls back to anonymous, no
list grows.

## Fix 3 — stop prefetching the nous catalog

**Problem.** The background prefetch calls
`cached_provider_model_ids("nous", force_refresh=True)`
(`model_switch.py:2390`); nous is collected into it because
`_collect_authed_provider_slugs` treats any `auth.json` providers entry as
credentials regardless of `auth_type` (`model_switch.py:2519-2526`). Because
`force_refresh=True` skips the cache read and no nous surface reads the entry,
this is a live authenticated `/v1/models` round trip per picker open written to
a location nothing consults.

**Do.** Exclude nous from the prefetch and delete the write-only entry.

This replaces what an earlier draft proposed here — folding `org_id` into
`_credential_fingerprint` and shortening `_PROVIDER_MODELS_STALE_SERVE_MAX`
for nous (a single global constant, `models.py:4204`, with no per-provider
branching today). Both would have hardened a cache that, after fix 1, has no
nous readers to protect. If a future surface routes nous through
`cached_provider_model_ids` again, revisit the fingerprint then: it hashes
env-var values and `auth.json` mtime and carries no org signal
(`models.py:4277`), so two orgs on one machine can serve each other's list.

**Test.** Opening the `/model` picker makes no Nous `/v1/models` request beyond
the one the displayed list is built from.

## Fix 4 — the `policy_present` claim

**Problem.** Under omission a blocked model simply vanishes, which reads as
"Hermes does not support this" rather than "your org disallows it".

**Do.** Read the `policy_present` claim off the Nous OAuth access token and,
when it is `true`, show a single line stating that the org restricts which
models are available. No enumeration, no per-model marking.

The claim rides the same JWT as `org_id`
(`access-token-issuer.ts:552,595`, `token_use: "access"`) — the token the
client already decodes — and `_info_from_valid_jwt` already retains every
claim in `raw_claims` (`nous_account.py:600-647`), so surfacing it is one
typed field on `NousPortalAccountInfo` and no new request.

It is already widened to cover provider-only restrictions, not just model
allowlists (`nous-account-service/src/server/entitlement-snapshot.ts:478-480`).
Two NAS docs still describe it as allowlist-only and list the widening as
pending — they are stale; trust that expression.

**Do not** enumerate the blocked set. Model policy is allowlist-only —
`denyModels` is a dead column (`nous-account-service/src/server/model-policy.ts:230`)
— so an org that allows five models blocks the entire rest of the catalog.
Graying hundreds of rows is a worse UI than omitting them. An earlier draft
proposed deriving the blocked set by diffing the anonymous and authenticated
reads and feeding it to `_prompt_model_selection`'s `unavailable_models`; that
is the wrong shape twice over, because that picker carries one
`unavailable_message` for the whole list and cannot say "free-tier-gated" and
"policy-hidden" at once.

**Do not** report the absence of the claim as the absence of a policy. It is
tri-state: `true`, `false`, and absent, where absent means unknown — an older
mint, not an unrestricted org. The gateway rejects a corrupt (non-boolean)
claim outright rather than reading it as "no policy"
(`api/src/middleware/nas_jwt_auth.ts:179`). Show the line only on `true`.

**Known bound:** the claim is stamped at mint time, so it goes stale until the
next token refresh — the line can lag a policy change by up to the access
token's lifetime. Acceptable, and worth stating rather than rediscovering.

**Test.** With `policy_present` true the line shows; with it false or absent it
does not.

---

## Order

Fix 0 first: fix 1 is silently wrong without it, and it is the only piece
testable before NAS switches the feature on. Fix 4's claim gates fix 1, so the
two land together. Fix 1 is the correctness work — without it the policy is
bypassed on every surface a user picks from. Fix 2 keeps the pricing map from
becoming another way to widen a list. Fix 3 is a deletion that fix 1 makes
safe.

Run tests with `scripts/run_tests.sh` — not bare `pytest`.
