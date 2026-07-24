<!-- GENERATED FILE: edit shared/shopify-admin-onboarding/core.md or manifest.json, then run node scripts/sync-onboarding.mjs --write. -->
<!-- onboarding-contract: 1.0.0; source-sha256: 04dcbd74020bb1835353bfaa94e8cbd25da03e5745a1fc49655ebdef41255b9f -->
# Connect Your Store

Connect only when the requested translation or locale/market review needs Shopify Admin data.

## Choose the smallest access path

CSV preparation can stay local; do not request a store connection until the merchant wants Admin reads or writes.

### Quick connection (recommended)

Use this for a first run or occasional work. The merchant supplies a store address, never an access token or API secret.

1. Ensure Shopify CLI 3.93.0+ is available.
2. Create private `skill-hub.env` in the merchant's working directory with the skill's `init-env` command. Keep it ignored by Git; never place it in the skill folder.
3. Explain that Shopify will open a browser permission page, then run:

```text
shopify store auth --store <shop>.myshopify.com --scopes read_locales,write_locales,read_markets,write_markets,read_translations,write_translations --json
```

4. Wait for the merchant to approve the browser permission page, then run the skill's read-only `connection-check`.

If the CLI grant expires, is revoked, or lacks access, repeat this browser step with the exact active-task scopes. CLI mode never uses the Automation Token.

### Long-running connection

Use this only for a trusted local or server agent operating on the merchant's own store.

1. In Shopify admin, open **Developer Dashboard** → **Apps** → **Create app** → **Start from Dev Dashboard**.
2. Create the first app version with this exact, copyable comma-separated scope list:

```text
read_locales,write_locales,read_markets,write_markets,read_translations,write_translations
```

3. Release the version, copy the Client ID and Client Secret from **Settings** into private local/server configuration, then install the app from **Home** to the merchant's own store.
4. Configure the private runtime values:

```text
SKILL_HUB_SHOPIFY_ACCESS_METHOD=dev_dashboard_client_credentials
SKILL_HUB_SHOPIFY_STORE_DOMAIN=<shop>.myshopify.com
SKILL_HUB_SHOPIFY_CLIENT_ID=<private-client-id>
SKILL_HUB_SHOPIFY_CLIENT_SECRET=<private-client-secret>
```

The helper exchanges the client credentials for short-lived API access in memory. Never save, display, or paste the access token in chat.

If the merchant wants this trusted agent to release future approved permission updates without supplying another credential, configure `SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN` privately now. It is optional for current scopes, cannot read store data, expires after 1, 3, or 6 months, and never grants merchant consent.

Advanced merchants may enable all permissions only after a plain-language warning about the wider access. Empty permissions are valid but block store work until updated.

## Permission upgrades

CLI mode: rerun `shopify store auth` with the exact missing scopes and wait for browser approval.

Dev Dashboard mode is a two-consent flow:

1. Show only the missing scopes, the merchant-language reason, and the full copyable scope list.
2. Obtain approval for those exact scopes. If declined, continue only with a path supported by the current access.
3. Obtain separate approval to publish the app change. Only then synchronize the merchant's existing app under private `.skill-hub/` with `shopify app config link --path <private-.skill-hub-app-dir> --client-id <client-id>`; preserve unknown settings and never synthesize a replacement configuration.
4. Run `shopify app config validate --path <private-.skill-hub-app-dir> --json`. Inject `SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN` as `SHOPIFY_APP_AUTOMATION_TOKEN` only into the child process that runs `shopify app deploy --path <private-.skill-hub-app-dir> --allow-updates`. Never place it in arguments, files, logs, or GraphQL requests.
5. Tell the merchant to open the installed app in Shopify admin and approve the pending **Update/Approve permissions** action. Publishing a version does not grant consent.
6. Wait for propagation, refresh the short-lived token, and rerun a read-only connection check. If the scope remains absent, report `SCOPE_UPDATE_PENDING` and stop; never redeploy repeatedly.

“Silent automation” means private credential injection after approval; it never means silent scope expansion or consent.

## Safety boundary

Connection never replaces the skill's preview and explicit approval boundary before enabling languages, changing Markets, or registering translations.
