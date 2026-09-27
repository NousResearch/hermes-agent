# feat(feishu): user_access_token grant — cross-chat message search and history as the user

Closes #11540.

Adds the user-scoped Feishu/Lark authorization the issue asks for: `hermes feishu login|status|logout`,
a rotation-safe token store, and the two tools that give the grant a reason to exist
(`feishu_message_search`, `feishu_message_list`). Opt-in end to end — an install that never runs
`hermes feishu login` gains nothing in the tool schema and behaves exactly as before.

## One correction to the issue's premise

The issue asks for **RFC 8628 Device Authorization Grant**. Feishu does not offer one for user tokens.
Its published user-OAuth is the **authorization-code flow** (`accounts.*/open-apis/authen/v1/authorize`
→ `open.*/open-apis/authen/v2/oauth/token`), with optional PKCE; the token doc lists
`authorization_code` and `refresh_token` and no `device_code`.

What OpenClaw's Lark `device-flow.js` drives is Feishu's *app-registration* device flow at
`accounts.feishu.cn/oauth/v1/app/registration` — it mints an **application**, not a user token. Hermes
already implements that one, and has since scan-to-create landed:

```python
# plugins/platforms/feishu/adapter.py
_REGISTRATION_PATH = "/oauth/v1/app/registration"

def _begin_registration(domain: str = "feishu") -> dict:
    """Start the device-code flow. Returns device_code, qr_url, user_code, interval, expire_in."""
```

So "add RFC 8628" would have been re-adding a flow we have, against an endpoint that cannot return a
UAT. The gap the issue is actually pointing at — `grep -rn user_access_token` returns **zero** hits
tree-wide — is real, and this PR closes it with the grant type Feishu documents. The
`hermes feishu login` output says so explicitly, so nobody goes looking for a QR code.

## What lands

| File | Role |
|---|---|
| `tools/feishu_user_auth.py` (new) | The whole flow: authorize URL, loopback callback, code exchange, refresh, store, status, runtime resolver |
| `tools/feishu_user_tool.py` (new) | `feishu_message_search`, `feishu_message_list` |
| `plugins/platforms/feishu/feishu_cli.py` (new) | the `hermes feishu login` / `status` / `logout` verbs |
| `plugins/platforms/feishu/adapter.py` | registers the CLI command; host table now shared |
| `toolsets.py` | `feishu_user` toolset, folded into `hermes-feishu` |

### Footprint: rung 2 + rung 3, no new core tool

Per the ladder, the flow is a **CLI command** (`hermes feishu ...`, registered by the plugin via
`ctx.register_cli_command` — the `hermes photon` shape, zero core wiring) and the two reads are a
**service-gated named toolset**, never `_HERMES_CORE_TOOLS`. `check_fn` is
`feishu_user_auth.has_user_token`, so both tools are absent from the schema until a grant exists. The
gate is reachability for the profile — it reads that profile's `auth.json`, which is exactly what
`registry.check_fn_cache_scope()` keys its cache on — not session surface, so it belongs in `check_fn`
rather than a session-resolved toolset.

No new `HERMES_*` env var: the OAuth client *is* the bot app, because Feishu's token endpoint is
confidential-client only (`client_secret` is required even with PKCE), so `FEISHU_APP_ID` /
`FEISHU_APP_SECRET` are the only credentials involved and `hermes setup` already collects them.

### Where the grant lives

`~/.hermes/auth.json` → `providers.feishu-user`, through the existing
`_auth_store_lock` / `_load_auth_store` / `_store_provider_state` helpers — the same store Spotify
(the other non-inference OAuth service) and Photon use, so it inherits 0600 perms, the corruption
handling, and profile awareness for free. `set_active=False` always: a Feishu user grant must never
become the install's inference provider.

The provider id is `feishu-user`, not `feishu`: the bot credentials in `.env` are a different
principal with different authority, and collapsing them into one row would make "logged out" ambiguous.

## The invariant that shaped the code: **Feishu rotates the refresh token**

> 在获取新的 `user_access_token` 时会返回新的 `refresh_token`，原 `refresh_token` 立即失效

Every refresh mints a new refresh token and kills the old one *immediately*. A naive
resolve-then-refresh-then-save loses a grant the first time two sessions overlap: both read
`u-refresh-1`, both spend it, the loser is permanently logged out with no error the user can act on.

So the refresh happens **inside** the auth-store lock and the rotated pair is persisted before the
lock is released (`resolve_user_access_token`). A second resolver either blocks or reads the new
tokens; it can never observe the dead one. Two related rules fall out:

- **An omitted `refresh_token` in a refresh response keeps the stored one.** Blanking it on a
  response that simply didn't rotate would silently downgrade the grant to single-use.
- **A terminal refusal quarantines the dead material** (`_quarantine_flat_oauth_state`), so the next
  call fails fast with "run `hermes feishu login`" instead of spending a round-trip on a grant Feishu
  has already revoked. Both are pinned by tests.

## Two real consumers, not a store waiting for one

Both in `tools/feishu_user_tool.py`, both reachable the moment login succeeds:

1. **`feishu_message_search`** — `POST /open-apis/im/v1/messages/search`, scope `search:message`.
   Feishu accepts a **user token only** here, so this is the capability the bot's
   `tenant_access_token` structurally cannot reach, and the one the issue leads with. Filters by chat,
   sender and time range; returns `message_id` + `chat_id` so the agent can follow up with (2).
2. **`feishu_message_list`** — `GET /open-apis/im/v1/messages` as the user, which needs
   `im:message.p2p_msg:get_as_user` / `im:message.group_msg:get_as_user`. This is the issue's "learn
   from other agents" case concretely: another app's messages in a shared chat are visible where the
   bot's own view is not, and a test asserts exactly that (`sender_type == "app"` on a message Hermes
   never received).

Result rows are flattened and bounded — `display_info` arrives wrapped in `<h>…</h>` highlight markup
(stripped), and a post/card `body.content` is previewed at 500 chars per message rather than
truncating the page, so one large card cannot swallow the other 29 hits.

Feishu's own caps are enforced client-side (`page_size` clamps to 30 for search, 50 for list) and a
scope error (`99991672` / `99991679`) is rewritten to name the fix instead of echoing a numeric code.

### Scopes requested

`offline_access` (without it there is no refresh token and the grant dies in two hours),
`search:message`, `im:message:readonly`, `im:message.p2p_msg:get_as_user`,
`im:message.group_msg:get_as_user` — exactly what the two tools call, nothing speculative.
`--scope` overrides and the requested set is persisted with the grant, so nothing needs a config key.

### One host table, two flows

`_ONBOARD_ACCOUNTS_URLS` / `_ONBOARD_OPEN_URLS` moved out of `adapter.py` into
`tools/feishu_user_auth.py` (the leaf; the adapter facade imports them under their existing names, so
no call site or patch target changed). A Lark tenant whose app was registered on
`accounts.larksuite.com` must not be sent to consent on `accounts.feishu.cn`; a test asserts the two
modules hold the *same object* so the tables cannot drift.

## The one thing the user must do by hand

Feishu refuses an unregistered `redirect_uri`, so `http://127.0.0.1:43829/feishu/callback` has to be
added under **Security Settings → Redirect URLs** and a new app version published. That is why the
loopback port is a fixed default rather than OS-assigned — an ephemeral port would break every login
after the first. `--redirect-uri` overrides it, and `login` prints the exact URI to paste before it
prints the authorize URL. The remote-session SSH-forward hint is the shared
`_print_loopback_ssh_hint`, so an SSH install gets the tunnel command instead of a silent timeout.

## Validation

`scripts/run_tests.sh tests/gateway/ tests/plugins/` — **1129 files, 10673 passed, 1 failed, 46
skipped**. The one failure is
`tests/gateway/test_update_streaming.py::TestCmdUpdateGatewayMode::test_gateway_flag_enables_gateway_prompt_for_stash`,
which trips the real-`~/.hermes` I/O guard on this machine
(`/home/calelin/.hermes/installs/…/test-environment/…/hermes_cli/main.py`). Reproduced identically on
a clean `origin/main` worktree — unrelated to this change.

`ruff check` clean on every changed file. `scripts/check_compat_pointers.py` and
`scripts/check_profile_scope_patterns.py` clean (0 findings).

### New tests — 28 contracts, real HTTP end to end

`tests/tools/feishu_user_helpers.py` stands up a local HTTP server and the tests redirect **only the
host table** (`OPEN_BASE_URLS`), so real `httpx` drives real status codes, real form encoding, real
query strings and real JSON bodies. Mocking the transport is what hides the bugs this flow can
actually have (`filter` in the body vs. paging in the query string is one of them).

`tests/tools/test_feishu_user_auth.py` (11):
- consent lands on the *domain's* accounts host, with `S256` and the nonce (Lark → `accounts.larksuite.com`);
- the adapter's QR flow and the user grant hold the same host table object;
- `offline_access` is in the default scope set; scope strings de-duplicate and keep order;
- a redirect URI must be loopback with an explicit port;
- missing bot credentials point at `hermes setup`;
- **full login E2E** — the real loopback listener is driven by a real HTTP GET replaying the browser's
  redirect, and the exchange is asserted to carry `client_secret` *and* a `code_verifier` whose SHA-256
  matches the `code_challenge` that went out;
- a state mismatch aborts **before** the token endpoint is called (`exchange_code` is a tripwire);
- rotation: the new refresh token is on disk by the time the resolver returns;
- an omitted `refresh_token` keeps the stored one;
- a revoked grant is quarantined and the second call makes **no HTTP request at all** (asserted on the
  fake's request count);
- logout reports whether there was anything to forget.

`tests/tools/test_feishu_user_tool.py` (10): the search/paging split, no empty `filter` object, page-size
clamping at both ends, row flattening with highlight markup stripped, the scope-error rewrite, list
reading another app's message, an invalid `sort_type` dropped rather than 400'd, non-text bodies
previewed rather than dropped, the `check_fn` gate flipping with the stored grant, and the registration
contract (`feishu_user` toolset, in `hermes-feishu`, **not** in `_HERMES_CORE_TOOLS`, and *not* absorbed
by `feishu_drive`'s positional slice of `_FEISHU_TOOLS`).

`tests/plugins/platforms/test_feishu_user_cli.py` (7): every verb resolves through the **real** argparse
tree built by `hermes_cli.main._build_cli_parser()` — the only thing that catches the deferred-platform
`invalid choice` failure that bit `hermes photon` in #54678 — bare `hermes feishu` reports status, a
login confirmation prints scopes and expiry but **never** token material, a failure is a non-zero exit
with the reason on stderr, and status/logout reflect the stored grant.

## Docs

`website/docs/user-guide/messaging/feishu.md` gains a **User Access** section: the three commands, the
console redirect-URL + scope table, what the tools unlock, where the token lives, the rotation note,
and a warning that a UAT carries the user's own authority. Plus four troubleshooting rows. The two
reference pages and the hermes-agent skill's toolset table list `feishu_user`.

## Deliberately not in this PR

- **Per-Feishu-user token binding.** OpenClaw's own issue asks for a map from `open_id` → grant so a
  multi-user deployment can act as whoever is speaking. That is a different security model (whose
  authority does an inbound message carry?) and needs its own consent and audit design; one grant for
  the operator is the useful, arguable-in-isolation first step.
- **Routing the adapter's existing bot calls through the UAT.** Tempting — a user token also reaches
  docs and calendars the bot cannot — but silently escalating the authority behind calls the adapter
  already makes is a behaviour change to argue on its own, not a rider on the grant that enables it.
- **`hermes tools` toggle for `feishu_user`.** `feishu_doc` / `feishu_drive` are not in
  `CONFIGURABLE_TOOLSETS` either; `hermes feishu logout` is the on/off switch, and it revokes the
  capability rather than just hiding it.
