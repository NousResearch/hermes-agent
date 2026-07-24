const TOKEN_SKEW_MS = 60_000;
const tokenCache = new Map();

const CONFIG_KEYS = [
  "SKILL_HUB_SHOPIFY_ACCESS_METHOD",
  "SKILL_HUB_SHOPIFY_STORE_DOMAIN",
  "SKILL_HUB_SHOPIFY_CLIENT_ID",
  "SKILL_HUB_SHOPIFY_CLIENT_SECRET",
  "SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN",
  "SKILL_HUB_SHOPIFY_CLI_JS",
  "SKILL_HUB_SHOPIFY_API_VERSION",
];

export function mergeRuntimeEnv(fileEnv = {}) {
  const env = { ...fileEnv };
  for (const key of CONFIG_KEYS) {
    if (typeof process.env[key] === "string" && process.env[key].trim()) env[key] = process.env[key].trim();
  }
  return env;
}

export function isDevDashboardMode(env = {}) {
  return (env.SKILL_HUB_SHOPIFY_ACCESS_METHOD || "shopify_cli_oauth") === "dev_dashboard_client_credentials";
}

export function normalizeMyShopifyDomain(value) {
  const raw = String(value || "").trim();
  const adminMatch = raw.match(/admin\.shopify\.com\/store\/([^/\s?#]+)/i);
  const host = adminMatch ? `${adminMatch[1]}.myshopify.com` : raw.replace(/^https?:\/\//i, "").replace(/\/.*$/, "");
  if (!/^[a-z0-9][-a-z0-9]*\.myshopify\.com$/i.test(host)) {
    throw new Error("INVALID_STORE_DOMAIN: Provide a Shopify admin URL or your .myshopify.com address.");
  }
  return host.toLowerCase();
}

function requireClientCredentials(env) {
  const clientId = String(env.SKILL_HUB_SHOPIFY_CLIENT_ID || "").trim();
  const clientSecret = String(env.SKILL_HUB_SHOPIFY_CLIENT_SECRET || "").trim();
  if (!clientId || !clientSecret) {
    throw new Error("DEV_DASHBOARD_CREDENTIALS_REQUIRED: Add the Client ID and Client Secret to your private configuration, or choose quick browser connection instead.");
  }
  return { clientId, clientSecret };
}

function classifyTokenFailure(status, text) {
  if (/shop_not_permitted/i.test(text)) return "DEV_DASHBOARD_STORE_NOT_PERMITTED";
  if (status === 401 || status === 403) return "DEV_DASHBOARD_CREDENTIALS_INVALID";
  return "DEV_DASHBOARD_TOKEN_REQUEST_FAILED";
}

export async function getDevDashboardAccess(env, shopInput) {
  const shop = normalizeMyShopifyDomain(shopInput || env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const { clientId, clientSecret } = requireClientCredentials(env);
  const cacheKey = `${shop}:${clientId}`;
  const cached = tokenCache.get(cacheKey);
  if (cached && Date.now() < cached.expiresAt - TOKEN_SKEW_MS) return cached;

  const response = await fetch(`https://${shop}/admin/oauth/access_token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ grant_type: "client_credentials", client_id: clientId, client_secret: clientSecret }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.access_token || !payload.expires_in) {
    const detail = String(payload.error_description || payload.error || "");
    throw new Error(`${classifyTokenFailure(response.status, detail)}: Shopify could not create a store connection. ${detail}`.trim());
  }
  const access = {
    token: payload.access_token,
    scopes: String(payload.scope || "").split(",").map((scope) => scope.trim()).filter(Boolean),
    expiresAt: Date.now() + Number(payload.expires_in) * 1000,
  };
  tokenCache.set(cacheKey, access);
  return access;
}

export function assertRequiredScopes(grantedScopes, requiredScopes) {
  const granted = new Set(grantedScopes || []);
  const missing = String(requiredScopes || "").split(",").map((scope) => scope.trim()).filter(Boolean).filter((scope) => !granted.has(scope));
  if (missing.length) {
    throw new Error(`SCOPE_UPDATE_REQUIRED: This task needs: ${missing.join(",")}. Show the merchant this copyable list, ask approval, then update and approve the app permissions before retrying.`);
  }
}

export async function devDashboardGraphql(env, shopInput, query, variables = {}) {
  const shop = normalizeMyShopifyDomain(shopInput || env.SKILL_HUB_SHOPIFY_STORE_DOMAIN);
  const access = await getDevDashboardAccess(env, shop);
  const version = String(env.SKILL_HUB_SHOPIFY_API_VERSION || "2026-04");
  const response = await fetch(`https://${shop}/admin/api/${version}/graphql.json`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Shopify-Access-Token": access.token },
    body: JSON.stringify({ query, variables }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`DEV_DASHBOARD_GRAPHQL_FAILED: Shopify returned ${response.status}.`);
  if (payload.errors?.length) throw new Error(`DEV_DASHBOARD_GRAPHQL_ERRORS: ${JSON.stringify(payload.errors)}`);
  return { data: payload.data, scopes: access.scopes, version };
}
