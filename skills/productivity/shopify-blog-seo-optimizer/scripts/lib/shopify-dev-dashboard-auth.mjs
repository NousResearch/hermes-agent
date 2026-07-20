import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const tokenCache = new Map();
const TOKEN_SKEW_MS = 60_000;
const CONFIG_KEYS = [
  "SKILL_HUB_SHOPIFY_ACCESS_METHOD",
  "SKILL_HUB_SHOPIFY_STORE_DOMAIN",
  "SKILL_HUB_SHOPIFY_CLIENT_ID",
  "SKILL_HUB_SHOPIFY_CLIENT_SECRET",
  "SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN",
  "SKILL_HUB_SHOPIFY_CLI_JS",
];

export function parseEnv(text) {
  const values = {};
  for (const line of String(text || "").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const separator = trimmed.indexOf("=");
    if (separator < 0) continue;
    values[trimmed.slice(0, separator).trim()] = trimmed.slice(separator + 1).trim().replace(/^['"]|['"]$/g, "");
  }
  return values;
}

export async function loadShopifyConfig(envPath) {
  const fileEnv = await readFile(envPath, "utf8").then(parseEnv).catch(() => ({}));
  const env = { ...fileEnv };
  for (const key of CONFIG_KEYS) if (process.env[key]?.trim()) env[key] = process.env[key].trim();
  const rawDomain = String(env.SKILL_HUB_SHOPIFY_STORE_DOMAIN || "").trim();
  const adminMatch = rawDomain.match(/admin\.shopify\.com\/store\/([^/\s?&]+)/i);
  const domain = (adminMatch ? `${adminMatch[1]}.myshopify.com` : rawDomain)
    .replace(/^https?:\/\//i, "")
    .replace(/\/.*$/, "")
    .toLowerCase();
  if (!/^[a-z0-9][-a-z0-9]*\.myshopify\.com$/i.test(domain)) {
    throw new Error("INVALID_STORE_DOMAIN: provide a .myshopify.com domain or Shopify admin URL.");
  }
  return { ...env, SHOPIFY_STORE_DOMAIN: domain, SHOPIFY_API_VERSION: "2026-04" };
}

export function isDirectMode(env) {
  return env.SKILL_HUB_SHOPIFY_ACCESS_METHOD === "dev_dashboard_client_credentials";
}

function requireClientCredentials(env) {
  const clientId = String(env.SKILL_HUB_SHOPIFY_CLIENT_ID || "").trim();
  const clientSecret = String(env.SKILL_HUB_SHOPIFY_CLIENT_SECRET || "").trim();
  if (!clientId || !clientSecret) {
    throw new Error("DEV_DASHBOARD_CREDENTIALS_REQUIRED: set SKILL_HUB_SHOPIFY_CLIENT_ID and SKILL_HUB_SHOPIFY_CLIENT_SECRET in private skill-hub.env.");
  }
  return { clientId, clientSecret };
}

async function getClientCredentialsToken(env) {
  const key = `${env.SHOPIFY_STORE_DOMAIN}:${env.SKILL_HUB_SHOPIFY_CLIENT_ID}`;
  const cached = tokenCache.get(key);
  if (cached && Date.now() < cached.expiresAt - TOKEN_SKEW_MS) return cached;
  const { clientId, clientSecret } = requireClientCredentials(env);
  const response = await fetch(`https://${env.SHOPIFY_STORE_DOMAIN}/admin/oauth/access_token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ grant_type: "client_credentials", client_id: clientId, client_secret: clientSecret }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.access_token || !payload.expires_in) {
    const detail = payload.error_description || payload.error || `HTTP ${response.status}`;
    throw new Error(`DEV_DASHBOARD_TOKEN_REQUEST_FAILED: ${detail}`);
  }
  const token = {
    token: payload.access_token,
    scopes: String(payload.scope || "").split(",").map((value) => value.trim()).filter(Boolean),
    expiresAt: Date.now() + Number(payload.expires_in) * 1000,
  };
  tokenCache.set(key, token);
  return token;
}

export function assertRequiredScopes(granted, required) {
  const available = new Set(granted || []);
  const missing = String(required || "").split(",").map((value) => value.trim()).filter(Boolean).filter((value) => !available.has(value));
  if (missing.length) throw new Error(`SCOPE_UPDATE_REQUIRED: missing ${missing.join(",")}. Show the exact list and reason, obtain scope approval and separate app-release approval, publish only through the private Automation Token flow, then wait for Shopify admin Update/Approve permissions before retrying connection-check.`);
}

export function assertAnyScope(granted, alternatives) {
  const available = new Set(granted || []);
  if (alternatives.some((scope) => available.has(scope))) return;
  throw new Error(`SCOPE_UPDATE_REQUIRED: grant one of ${alternatives.join(" or ")}. Show the reason, obtain scope and separate app-release approval, wait for Shopify admin Update/Approve permissions, then retry.`);
}

export async function connectionStatus(env) {
  if (isDirectMode(env)) {
    const accessToken = await getClientCredentialsToken(env);
    const readScopes = ["read_content", "read_online_store_pages"];
    const writeScopes = ["write_content", "write_online_store_pages"];
    return {
      mode: "dev_dashboard_client_credentials",
      storeDomain: env.SHOPIFY_STORE_DOMAIN,
      apiVersion: env.SHOPIFY_API_VERSION,
      grantedScopeCount: accessToken.scopes.length,
      scopeStatus: {
        read: { granted: readScopes.filter((scope) => accessToken.scopes.includes(scope)), acceptable: readScopes },
        write: { granted: writeScopes.filter((scope) => accessToken.scopes.includes(scope)), acceptable: writeScopes },
      },
      tokenExpiresAt: new Date(accessToken.expiresAt).toISOString(),
    };
  }
  return {
    mode: "shopify_cli_oauth",
    storeDomain: env.SHOPIFY_STORE_DOMAIN,
    apiVersion: env.SHOPIFY_API_VERSION,
    grantedScopeCount: null,
    scopeStatus: { read: { granted: null, acceptable: ["read_content", "read_online_store_pages"] }, write: { granted: null, acceptable: ["write_content", "write_online_store_pages"] } },
    nextStep: `Run shopify store auth --store ${env.SHOPIFY_STORE_DOMAIN} --scopes read_content,write_content`,
  };
}

async function cliInvocation(env) {
  const candidates = [env.SKILL_HUB_SHOPIFY_CLI_JS, process.env.SKILL_HUB_SHOPIFY_CLI_JS];
  const npmRoot = await execFileAsync("npm", ["root", "-g"], { windowsHide: true }).then(({ stdout }) => stdout.trim()).catch(() => "");
  if (npmRoot) candidates.push(path.join(npmRoot, "@shopify", "cli", "bin", "run.js"));
  for (const candidate of candidates) if (candidate && await access(candidate).then(() => true).catch(() => false)) return { command: process.execPath, prefix: [candidate] };
  await execFileAsync("shopify", ["--version"], { windowsHide: true }).catch(() => {
    throw new Error("CLI_NOT_FOUND: install Shopify CLI 3.93.0+ or configure SKILL_HUB_SHOPIFY_CLI_JS.");
  });
  return { command: "shopify", prefix: [] };
}

async function cliGraphql(env, query, variables, mutation) {
  const temp = await mkdtemp(path.join(os.tmpdir(), "shopify-blog-seo-"));
  const queryFile = path.join(temp, "query.graphql");
  const variableFile = path.join(temp, "variables.json");
  const outputFile = path.join(temp, "output.json");
  try {
    const cli = await cliInvocation(env);
    await writeFile(queryFile, query, "utf8");
    await writeFile(variableFile, JSON.stringify(variables || {}), "utf8");
    const args = [...cli.prefix, "store", "execute", "--store", env.SHOPIFY_STORE_DOMAIN, "--query-file", queryFile, "--variable-file", variableFile, "--output-file", outputFile, "--json"];
    if (mutation) args.push("--allow-mutations");
    try {
      await execFileAsync(cli.command, args, { windowsHide: true, timeout: 180000, maxBuffer: 20 * 1024 * 1024 });
    } catch (error) {
      const detail = [error.stderr, error.stdout].filter(Boolean).join("\n");
      throw new Error(`CLI_GRAPHQL_FAILED: ${detail || error.message}. If authentication is required, run shopify store auth for ${env.SHOPIFY_STORE_DOMAIN}.`);
    }
    const raw = JSON.parse(await readFile(outputFile, "utf8"));
    if (raw.errors?.length) throw new Error(`CLI_GRAPHQL_ERRORS: ${JSON.stringify(raw.errors)}`);
    return raw.data || raw;
  } finally {
    await rm(temp, { recursive: true, force: true }).catch(() => {});
  }
}

export async function shopifyGraphql(env, query, variables = {}, options = {}) {
  if (!isDirectMode(env)) return cliGraphql(env, query, variables, Boolean(options.mutation));
  const accessToken = await getClientCredentialsToken(env);
  if (options.requiredScopes) assertRequiredScopes(accessToken.scopes, options.requiredScopes);
  if (options.requiredAnyScopes) assertAnyScope(accessToken.scopes, options.requiredAnyScopes);
  const response = await fetch(`https://${env.SHOPIFY_STORE_DOMAIN}/admin/api/${env.SHOPIFY_API_VERSION}/graphql.json`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Shopify-Access-Token": accessToken.token },
    body: JSON.stringify({ query, variables }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`GRAPHQL_HTTP_FAILED: Shopify returned HTTP ${response.status}.`);
  if (payload.errors?.length) throw new Error(`GRAPHQL_ERRORS: ${JSON.stringify(payload.errors)}`);
  return payload.data;
}
