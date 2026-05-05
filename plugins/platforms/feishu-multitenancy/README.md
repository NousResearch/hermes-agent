# Feishu Multitenancy Plugin

Route one Feishu bot to multiple isolated Hermes profiles.

This bundled plugin is opt-in. It uses the `pre_gateway_dispatch` hook to
intercept Feishu messages before the default gateway dispatch, resolve the real
sender `open_id`, and run the message against a per-user Hermes profile.

## Enable

```bash
hermes plugins enable platforms/feishu-multitenancy
hermes gateway restart
```

## Configure Routes

Create one Hermes profile per tenant persona:

```bash
mkdir -p ~/.hermes/profiles/alice ~/.hermes/profiles/bob
$EDITOR ~/.hermes/profiles/alice/SOUL.md
$EDITOR ~/.hermes/profiles/bob/SOUL.md
```

Apply a route file:

```bash
python plugins/platforms/feishu-multitenancy/sync.py apply users.json
```

Example `users.json`:

```json
[
  {
    "user_id": "alice",
    "profile_name": "alice",
    "open_id": "ou_xxx",
    "union_id": "on_xxx"
  },
  {
    "user_id": "bob",
    "profile_name": "bob",
    "open_id": "ou_yyy",
    "union_id": "on_yyy"
  }
]
```

New routes should use Feishu `open_id` (`ou_*`). `union_id` remains available
for legacy migration rows.

## Shared Feishu App

The plugin expects the gateway/default Hermes home to own the Feishu app
credentials and per-user Feishu user access token files. Do not duplicate app
credentials into every profile.

```text
~/.hermes/config.yaml                 # shared Feishu app config
~/.hermes/feishu_uat/<open_id>.json   # per-user token files
~/.hermes/profiles/<profile>/         # isolated SOUL, .env, config, memory
```

## Auto-Provisioning

Auto-provisioning is enabled by default:

```bash
HERMES_MULTITENANCY_AUTO_PROVISION=1
```

An unseen sender `ou_new_user` gets a deterministic profile path such as:

```text
~/.hermes/profiles/feishu_ou_new_user/
```

The profile is seeded from the shared Hermes config where possible. If the
shared config does not define a model, configure the generated profile before
expecting AIAgent replies.

Disable auto-provisioning with:

```bash
HERMES_MULTITENANCY_AUTO_PROVISION=0
```

## Toolsets

When a profile sets `platform_toolsets.feishu`, the plugin defaults to merging
those explicit entries with Hermes' default Feishu toolsets. This preserves
general capabilities such as `web_search` / `web_extract` while still allowing
per-profile Feishu tool additions.

For a strictly narrowed schema, set:

```yaml
multitenancy:
  toolsets_mode: explicit
```

or export:

```bash
HERMES_MULTITENANCY_TOOLSETS_MODE=explicit
```

## Verify

```bash
hermes plugins list
sqlite3 ~/.hermes/multitenancy.db \
  'select open_id, profile_name, active from multitenancy_routing;'
```

Then send the same prompt from two Feishu users to the same bot. The gateway
log should show distinct sender `ou_*` values and distinct profile homes.

## Notes

- No Hermes core files are patched.
- The plugin uses Feishu `open_id` first and only falls back to alternate IDs
  for legacy rows.
- AIAgent runs in an isolated subprocess so tool calls can execute without
  blocking the gateway event loop.
