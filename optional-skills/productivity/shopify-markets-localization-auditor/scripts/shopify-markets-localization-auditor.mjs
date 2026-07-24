#!/usr/bin/env node

import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createReadStream, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createInterface } from "node:readline";
import { assertRequiredScopes, devDashboardGraphql, isDevDashboardMode, mergeRuntimeEnv } from "./lib/shopify-dev-dashboard-auth.mjs";
import { fetchPublic } from "./lib/public-fetch.mjs";

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_ENV = "skill-hub.env";
const REQUIRED_SCOPES = "read_locales,write_locales,read_markets,write_markets,read_translations,write_translations,read_shipping,read_legal_policies,read_products";
const DEFAULT_RESOURCE_TYPES = [
  "PRODUCT",
  "COLLECTION",
  "PAGE",
  "BLOG",
  "ARTICLE",
  "SHOP",
  "SHOP_POLICY",
];
const LOCALE_BATCH_SIZE = 3;
const BULK_LOCALE_THRESHOLD = 4;
const RETRY_LIMIT = 3;
const THROTTLE_HEADROOM = 100;
const BULK_WATCH_TIMEOUT_MS = 60 * 60 * 1000;
const BULK_CONCURRENCY = 3;
let nextRequestAt = 0;
const SKIP_TRANSLATABLE_TYPES = new Set([
  "URI",
  "URL",
  "LINK",
  "LIST_URL",
  "LIST_LINK",
  "FILE_REFERENCE",
  "LIST_FILE_REFERENCE",
  "JSON",
  "JSON_STRING",
]);

function parseArgs(argv) {
  if (argv.length === 0 || argv.includes("--help") || argv.includes("-h")) {
    return {
      command: "help",
      env: DEFAULT_ENV,
      input: null,
      output: null,
      method: null,
      locales: null,
      transport: "auto",
      resume: false,
      execute: false,
      lang: "auto",
    };
  }

  const args = {
    command: argv[0],
    env: DEFAULT_ENV,
    input: null,
    output: null,
    method: null,
    locales: null,
    transport: "auto",
    resume: false,
    execute: false,
    lang: "auto",
  };

  for (let i = 1; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--env") {
      args.env = value;
      i += 1;
    } else if (key === "--input") {
      args.input = value;
      i += 1;
    } else if (key === "--output") {
      args.output = value;
      i += 1;
    } else if (key === "--method") {
      args.method = value;
      i += 1;
    } else if (key === "--locales") {
      args.locales = value;
      i += 1;
    } else if (key === "--transport") {
      args.transport = value;
      i += 1;
    } else if (key === "--resume") {
      args.resume = true;
    } else if (key === "--execute") {
      args.execute = true;
    } else if (key === "--lang") {
      args.lang = value;
      i += 1;
    }
  }

  const valid = ["help", "init-env", "connection-check", "audit", "report", "fix-plan", "apply"];
  if (!valid.includes(args.command)) {
    throw new Error(
      "Usage: node shopify-markets-localization-auditor.mjs <init-env|connection-check|audit|report|fix-plan|apply> [--env skill-hub.env] [--input file.json] [--output file] [--locales de,fr,ja] [--transport auto|standard|bulk] [--resume] [--lang zh-CN|en|auto] [--execute]",
    );
  }
  if (!["auto", "standard", "bulk"].includes(args.transport)) {
    throw new Error("--transport must be auto, standard, or bulk.");
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node shopify-markets-localization-auditor.mjs init-env --env skill-hub.env
  node shopify-markets-localization-auditor.mjs connection-check --env skill-hub.env
  node shopify-markets-localization-auditor.mjs audit --env skill-hub.env --output shopify-markets-localization-audit.json --transport auto --lang zh-CN
  node shopify-markets-localization-auditor.mjs audit --env skill-hub.env --output shopify-markets-localization-audit.json --resume
  node shopify-markets-localization-auditor.mjs report --input shopify-markets-localization-audit.json --output shopify-markets-localization-report.html --lang zh-CN
  node shopify-markets-localization-auditor.mjs fix-plan --input shopify-markets-localization-audit.json --output shopify-markets-localization-fix-plan.json
  node shopify-markets-localization-auditor.mjs apply --env skill-hub.env --input shopify-markets-localization-fix-plan.json
  node shopify-markets-localization-auditor.mjs apply --env skill-hub.env --input shopify-markets-localization-fix-plan.json --execute
`);
}

function parseEnv(text) {
  const env = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq === -1) continue;
    env[line.slice(0, eq).trim()] = line.slice(eq + 1).trim();
  }
  return env;
}

async function ensureGitignoreLine(line) {
  const file = ".gitignore";
  const existing = await readFile(file, "utf8").catch(() => null);
  if (existing === null) return { updated: false, reason: "missing .gitignore" };
  if (existing.split(/\r?\n/).includes(line)) return { updated: false, reason: "already ignored" };
  const next = existing.endsWith("\n") ? `${existing}${line}\n` : `${existing}\n${line}\n`;
  await writeFile(file, next, "utf8");
  return { updated: true };
}

async function initEnv(args) {
  const envFile = args.env || DEFAULT_ENV;
  let gitignoreCreated = false;
  const body = `# Skill Hub shared Shopify configuration
# Quick browser connection is the default. For a long-running agent, select
# dev_dashboard_client_credentials and add the Client ID and Client Secret.

SKILL_HUB_SHOPIFY_ACCESS_METHOD=shopify_cli_oauth
SKILL_HUB_SHOPIFY_STORE_DOMAIN=
# SKILL_HUB_SHOPIFY_CLIENT_ID=
# SKILL_HUB_SHOPIFY_CLIENT_SECRET=
# SKILL_HUB_SHOPIFY_APP_AUTOMATION_TOKEN=
`;

  const alreadyExists = existsSync(envFile);
  if (!alreadyExists) await writeFile(envFile, body, "utf8");
  if (!existsSync(".gitignore")) {
    await writeFile(".gitignore", `${envFile}\n`, "utf8");
    gitignoreCreated = true;
  }
  const gitignore = await ensureGitignoreLine(envFile);
  const currentEnv = parseEnv(await readFile(envFile, "utf8"));
  const placeholderDetected = Object.values(currentEnv).some((value) =>
    typeof value === "string" && (
      value.includes("your-store") ||
      value.includes("your-store")
    ));
  console.log(JSON.stringify({
    ok: true,
    envFile,
    created: !alreadyExists,
    gitignore: gitignoreCreated && gitignore.reason === "already ignored"
      ? { updated: false, reason: "created with env file entry" }
      : gitignore,
    placeholderDetected,
    warning: placeholderDetected
      ? "The env file still contains a placeholder store address. Replace it before connection-check."
      : null,
    requiredScopes: REQUIRED_SCOPES,
  }, null, 2));
}

function normalizeDomain(value) {
  const raw = value.trim();
  if (/admin\.shopify\.com\/store\//i.test(raw)) {
    const match = raw.match(/admin\.shopify\.com\/store\/([^\/\s?&]+)/i);
    if (match) return `${match[1].toLowerCase()}.myshopify.com`;
  }
  const url = raw.includes("://") ? new URL(raw) : new URL(`https://${raw}`);
  return url.host.toLowerCase();
}

async function resolveStoreDomainFromHtml(hostname) {
  const candidates = [
    `https://${hostname}`,
    `https://${hostname}/products`,
    `https://${hostname}/collections`,
  ];
  for (const url of candidates) {
    try {
      const response = await fetchPublic(url, {}, { timeoutMs: 8000 });
      if (!response.ok) continue;
      const html = await response.text();
      const match = html.match(/Shopify\.shop\s*=\s*"([^"]+\.myshopify\.com)"/i);
      if (match) return match[1].toLowerCase();
    } catch {}
  }
  return null;
}

function validateSafeUrl(value) {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Invalid protocol: "${parsed.protocol}". Only HTTP and HTTPS are allowed.`);
    }
    const hostname = parsed.hostname.toLowerCase();
    if (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "0.0.0.0" ||
      hostname.startsWith("10.") ||
      hostname.startsWith("192.168.") ||
      (hostname.startsWith("172.") &&
        Number(hostname.split(".")[1]) >= 16 &&
        Number(hostname.split(".")[1]) <= 31)
    ) {
      throw new Error(`Access to private address "${hostname}" is blocked.`);
    }
    return parsed.href;
  } catch (err) {
    throw new Error(`Invalid or unsafe URL "${value}": ${err.message}`);
  }
}

async function pathExists(filePath) {
  return access(filePath).then(() => true).catch(() => false);
}

async function resolveShopifyCliJs(env = {}) {
  const candidates = [];
  if (env.SKILL_HUB_SHOPIFY_CLI_JS || process.env.SKILL_HUB_SHOPIFY_CLI_JS) {
    candidates.push(env.SKILL_HUB_SHOPIFY_CLI_JS || process.env.SKILL_HUB_SHOPIFY_CLI_JS);
  }
  const npmRoot = await execFileAsync("npm", ["root", "-g"], { windowsHide: true })
    .then(({ stdout }) => stdout.trim())
    .catch(() => "");
  if (npmRoot) candidates.push(path.join(npmRoot, "@shopify", "cli", "bin", "run.js"));
  if (process.env.APPDATA) candidates.push(path.join(process.env.APPDATA, "npm", "node_modules", "@shopify", "cli", "bin", "run.js"));
  for (const candidate of candidates) {
    if (candidate && await pathExists(candidate)) return candidate;
  }
  const error = new Error("CLI_NOT_FOUND: Could not locate Shopify CLI JS entrypoint. Set SKILL_HUB_SHOPIFY_CLI_JS to @shopify/cli/bin/run.js.");
  error.code = "CLI_NOT_FOUND";
  throw error;
}

function normalizeCliJson(raw) {
  if (raw?.errors) return { errors: raw.errors };
  return raw?.data ? raw : { data: raw };
}

function classifyCliError(error, detail = "") {
  const text = `${detail}\n${error?.message || ""}`;
  if (error?.code === "CLI_NOT_FOUND") return { code: "CLI_NOT_FOUND", message: error.message };
  if (error?.code === "ENOENT" || error?.code === "EINVAL" || error?.code === "EFTYPE") return { code: "CLI_SPAWN_FAILED", message: error.message };
  if (/access denied|denied access/i.test(text)) return { code: "CLI_ACCESS_DENIED", message: text.trim() };
  if (/not authenticated|authentication required|run shopify store auth|authorization required|invalid.*token|token.*expired/i.test(text)) return { code: "CLI_AUTH_REQUIRED", message: text.trim() };
  if (/throttled|too many requests|\b429\b/i.test(text)) return { code: "CLI_THROTTLED", message: text.trim() };
  if (/fetch failed|request[\s\S]*(?:failed|aborted)|user\s+aborted\s+a\s+request|network|socket|econnreset|econnrefused|etimedout|enotfound|undici|\b5\d\d\b|internal.server.error/i.test(text)) return { code: "CLI_TRANSIENT_NETWORK", message: text.trim() };
  return { code: "CLI_SPAWN_FAILED", message: text.trim() || "Shopify CLI request failed." };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForRateBudget() {
  const waitMs = nextRequestAt - Date.now();
  if (waitMs > 0) await sleep(waitMs);
}

function observeThrottle(extensions) {
  const status = extensions?.cost?.throttleStatus;
  if (!status || !Number.isFinite(status.currentlyAvailable) || !Number.isFinite(status.restoreRate) || status.restoreRate <= 0) return;
  if (status.currentlyAvailable >= THROTTLE_HEADROOM) return;
  const waitMs = Math.ceil(((THROTTLE_HEADROOM - status.currentlyAvailable) / status.restoreRate) * 1000);
  nextRequestAt = Math.max(nextRequestAt, Date.now() + waitMs);
}

function isRetryableError(error) {
  return /CLI_THROTTLED|CLI_TRANSIENT_NETWORK|THROTTLED|INTERNAL_SERVER_ERROR|fetch failed|socket|econnreset|etimedout|\b429\b/i.test(error?.message || "");
}

function retryDelay(attempt, error) {
  const base = /THROTTLED|\b429\b/i.test(error?.message || "") ? 2000 : 800;
  return Math.min(15000, base * (2 ** attempt)) + Math.floor(Math.random() * 250);
}

async function withRetries(operation) {
  let lastError;
  for (let attempt = 0; attempt <= RETRY_LIMIT; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt === RETRY_LIMIT || !isRetryableError(error)) throw error;
      const delay = retryDelay(attempt, error);
      console.error(`[shopify] Temporary request failure; retrying in ${Math.ceil(delay / 1000)}s (${attempt + 1}/${RETRY_LIMIT}).`);
      await sleep(delay);
    }
  }
  throw lastError;
}

async function shopifyCliGraphql(env, query, variables = {}) {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "skill-hub-shopify-cli-"));
  const queryFile = path.join(tempDir, "query.graphql");
  const variableFile = path.join(tempDir, "variables.json");
  const outputFile = path.join(tempDir, "output.json");
  try {
    const cliJs = await resolveShopifyCliJs(env);
    await writeFile(queryFile, query, "utf8");
    await writeFile(variableFile, JSON.stringify(variables || {}), "utf8");
    const cliArgs = [
      cliJs,
      "store",
      "execute",
      "--store",
      env.SHOPIFY_API_DOMAIN,
      "--query-file",
      queryFile,
      "--variable-file",
      variableFile,
      "--output-file",
      outputFile,
      "--json",
    ];
    const isMutation = /(^|\n)\s*mutation\b/i.test(query);
    if (isMutation) cliArgs.push("--allow-mutations");

    const runOnce = async () => {
      await waitForRateBudget();
      await rm(outputFile, { force: true }).catch(() => {});
      await execFileAsync(process.execPath, cliArgs, {
        timeout: 180000,
        maxBuffer: 1024 * 1024 * 20,
        windowsHide: true,
      }).catch((error) => {
        const classified = classifyCliError(error, [error.stderr, error.stdout].filter(Boolean).map(String).join("\n"));
        throw new Error(`${classified.code}: ${classified.message}`);
      });
      if (!await pathExists(outputFile)) throw new Error("CLI_OUTPUT_MISSING: Shopify CLI did not create output JSON.");
      let json;
      try {
        json = normalizeCliJson(JSON.parse(await readFile(outputFile, "utf8")));
      } catch (error) {
        throw new Error(`CLI_JSON_PARSE_FAILED: ${error.message}`);
      }
      if (json.errors) throw new Error(`CLI_GRAPHQL_ERRORS: ${JSON.stringify(json.errors)}`);
      observeThrottle(json.extensions);
      return json.data;
    };
    // A mutation can succeed even when its CLI process times out. Do not repeat it implicitly.
    return isMutation ? await runOnce() : await withRetries(runOnce);
  } catch (error) {
    const authHint = /CLI_AUTH_REQUIRED/.test(error.message)
      ? `\nAuthenticate first: shopify store auth --store ${env.SHOPIFY_API_DOMAIN} --scopes ${REQUIRED_SCOPES} --json`
      : "";
    throw new Error(`${error.message}${authHint}`);
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}

async function shopifyCliBulkQuery(env, query, onNode) {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "skill-hub-shopify-bulk-"));
  const queryFile = path.join(tempDir, "query.graphql");
  const outputFile = path.join(tempDir, "output.jsonl");
  try {
    const cliJs = await resolveShopifyCliJs(env);
    await writeFile(queryFile, query, "utf8");
    const cliArgs = [
      cliJs, "store", "bulk", "execute", "--store", env.SHOPIFY_API_DOMAIN,
      "--query-file", queryFile, "--output-file", outputFile, "--watch",
    ];
    await waitForRateBudget();
    await rm(outputFile, { force: true }).catch(() => {});
    await execFileAsync(process.execPath, cliArgs, {
      timeout: BULK_WATCH_TIMEOUT_MS,
      maxBuffer: 1024 * 1024 * 20,
      windowsHide: true,
    }).catch((error) => {
      const classified = classifyCliError(error, [error.stderr, error.stdout].filter(Boolean).map(String).join("\n"));
      throw new Error(`${classified.code}: ${classified.message}\nBulk queries are not automatically resubmitted because the prior operation may still be running. Check it with: shopify store bulk status --store ${env.SHOPIFY_API_DOMAIN}`);
    });
    if (!await pathExists(outputFile)) throw new Error("CLI_BULK_OUTPUT_MISSING: Shopify CLI did not create a JSONL result file.");
    let nodeCount = 0;
    const lines = createInterface({ input: createReadStream(outputFile, { encoding: "utf8" }), crlfDelay: Infinity });
    for await (const line of lines) {
      if (!line) continue;
      let value;
      try { value = JSON.parse(line); } catch { throw new Error("CLI_BULK_JSONL_PARSE_FAILED: Shopify bulk result contained invalid JSONL."); }
      if (value?.resourceId) {
        onNode(value);
        nodeCount += 1;
      }
    }
    return nodeCount;
  } catch (error) {
    const authHint = /CLI_AUTH_REQUIRED/.test(error.message)
      ? `\nAuthenticate first: shopify store auth --store ${env.SHOPIFY_API_DOMAIN} --scopes ${REQUIRED_SCOPES} --json`
      : "";
    throw new Error(`${error.message}${authHint}`);
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}

async function loadEnv(envPath) {
  const text = await readFile(envPath, "utf8").catch(() => null);
  const env = mergeRuntimeEnv(text ? parseEnv(text) : {});
  if (!text && !env.SKILL_HUB_SHOPIFY_STORE_DOMAIN) {
    throw new Error(`Missing env file: ${envPath}. Current working directory: ${process.cwd()}. The env must be in the user's working directory, not the installed skill directory.`);
  }
  if (!["shopify_cli_oauth", "dev_dashboard_client_credentials"].includes(env.SKILL_HUB_SHOPIFY_ACCESS_METHOD || "shopify_cli_oauth")) throw new Error("Unsupported SKILL_HUB_SHOPIFY_ACCESS_METHOD. Use shopify_cli_oauth or dev_dashboard_client_credentials.");
  env.SHOPIFY_STORE_DOMAIN = env.SKILL_HUB_SHOPIFY_STORE_DOMAIN;
  if (!env.SHOPIFY_STORE_DOMAIN) throw new Error(`Missing SHOPIFY_STORE_DOMAIN in ${envPath}.`);

  env.SHOPIFY_STORE_DOMAIN = normalizeDomain(env.SHOPIFY_STORE_DOMAIN);
  if (!env.SHOPIFY_STORE_DOMAIN.endsWith(".myshopify.com")) {
    const resolved = await resolveStoreDomainFromHtml(env.SHOPIFY_STORE_DOMAIN);
    if (resolved) {
      console.error(`Note: resolved store domain from page HTML: ${env.SHOPIFY_STORE_DOMAIN} → ${resolved}`);
      env.SHOPIFY_STORE_DOMAIN = resolved;
    } else {
      throw new Error(`Could not resolve "${env.SHOPIFY_STORE_DOMAIN}" to a .myshopify.com domain. Provide your Shopify admin URL (https://admin.shopify.com/store/your-store) or .myshopify.com domain.`);
    }
  }
  env.SHOPIFY_API_DOMAIN = env.SHOPIFY_STORE_DOMAIN;
  env.SHOPIFY_API_VERSION = env.SKILL_HUB_SHOPIFY_API_VERSION || (isDevDashboardMode(env) ? "2026-04" : "shopify-cli");
  env.SHOPIFY_TRANSPORT = isDevDashboardMode(env) ? "dev_dashboard_client_credentials" : "shopify_cli";
  return env;
}

async function graphql(env, query, variables = {}) {
  if (isDevDashboardMode(env)) {
    const result = await devDashboardGraphql(env, env.SHOPIFY_API_DOMAIN, query, variables);
    assertRequiredScopes(result.scopes, REQUIRED_SCOPES);
    return result.data;
  }
  return shopifyCliGraphql(env, query, variables);
}

function writeJsonOutput(value, outputPath) {
  const text = JSON.stringify(value, null, 2);
  if (!outputPath) {
    console.log(text);
    return;
  }
  return mkdir(path.dirname(path.resolve(outputPath)), { recursive: true })
    .then(() => writeFile(outputPath, text, "utf8"))
    .then(() => console.log(JSON.stringify({ ok: true, output: outputPath }, null, 2)));
}

function parseLocales(raw) {
  if (!raw) return null;
  return raw.split(",").map((item) => item.trim()).filter(Boolean);
}

function resolveReportLang(requested, existing = null) {
  if (requested !== "auto") return requested;
  if (existing === "zh-CN" || existing === "en") return existing;
  return /^zh(?:-|$)/i.test(Intl.DateTimeFormat().resolvedOptions().locale || "") ? "zh-CN" : "en";
}

function textFromHtml(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function fetchHtml(url, headers = {}) {
  try {
    validateSafeUrl(url);
  } catch (e) {
    return { ok: false, status: 0, url, html: "", error: `Request blocked: ${e.message}` };
  }
  try {
    const response = await fetchPublic(url, {
      headers: {
        "User-Agent": "Selofy Skill Hub Markets Localization Auditor/1.0",
        "Accept-Language": "en-US,en;q=0.9",
        ...headers,
      },
    }, { timeoutMs: 15000 });
    const html = await response.text();
    return { ok: response.ok, status: response.status, url: response.url, html };
  } catch (error) {
    return { ok: false, status: 0, url, html: "", error: error.message };
  }
}

function extractPolicyLinks(html, origin) {
  const links = [];
  const regex = /href=["']([^"']+)["']/gi;
  let match;
  while ((match = regex.exec(html))) {
    const href = match[1];
    if (!/policy|refund|shipping|privacy|terms|return/i.test(href)) continue;
    try {
      links.push(new URL(href, origin).href);
    } catch {}
  }
  return [...new Set(links)];
}

async function crawlStorefront(shop) {
  const storefrontUrl = shop.primaryDomain?.url || `https://${shop.primaryDomain?.host || shop.myshopifyDomain}`;
  const home = await fetchHtml(storefrontUrl);
  const canonical = home.html.match(/<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i)?.[1] || null;
  const hreflangMatches = [...home.html.matchAll(/<link[^>]+hreflang=["']([^"']+)["'][^>]+href=["']([^"']+)["']/gi)];
  const hreflangs = hreflangMatches.map((match) => ({ locale: match[1], href: match[2] }));
  const policyLinks = extractPolicyLinks(home.html, storefrontUrl);

  return {
    storefrontUrl,
    homeStatus: home.status,
    canonical,
    hreflangCount: hreflangs.length,
    hreflangs,
    policyLinks,
  };
}

const OVERVIEW_QUERY = `#graphql
query MarketsAuditOverview {
  shop {
    name
    description
    myshopifyDomain
    currencyCode
    primaryDomain {
      host
      url
    }
    shopPolicies {
      id
      type
      title
      url
      body
    }
  }
  collections(first: 12, sortKey: UPDATED_AT) {
    nodes {
      title
      handle
    }
  }
  products(first: 12, sortKey: UPDATED_AT) {
    nodes {
      title
      handle
      productType
      tags
    }
  }
  shopLocales {
    name
    locale
    primary
    published
  }
  markets(first: 50) {
    nodes {
      id
      name
      handle
      enabled
      primary
      status
      currencySettings {
        localCurrencies
        baseCurrency {
          currencyCode
          currencyName
          enabled
        }
      }
      webPresence {
        id
        subfolderSuffix
        defaultLocale {
          locale
        }
        alternateLocales {
          locale
        }
        rootUrls {
          locale
          url
        }
      }
      regions(first: 250) {
        nodes {
          ... on MarketRegionCountry {
            id
            name
            code
          }
        }
      }
    }
  }
  deliveryProfiles(first: 50) {
    nodes {
      id
      name
      profileLocationGroups {
        countriesInAnyZone {
          country {
            code {
              countryCode
              restOfWorld
            }
            name
          }
          zone
        }
      }
    }
  }
}`;

function assertLocaleCode(locale) {
  if (!/^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$/.test(locale)) {
    throw new Error(`Unsupported locale code: ${locale}`);
  }
}

function buildTranslatableQuery(locales, { resourceType, bulk = false } = {}) {
  locales.forEach(assertLocaleCode);
  const translations = locales.map((locale, index) => `translation_${index}: translations(locale: ${JSON.stringify(locale)}) { key outdated }`).join("\n      ");
  const typeArgument = bulk ? `resourceType: ${resourceType}` : "resourceType: $resourceType, after: $after";
  const nodeSelection = `resourceId
      translatableContent { key type }
      ${translations}`;
  const connectionSelection = bulk
    ? `edges { node { ${nodeSelection} } }`
    : `nodes { ${nodeSelection} }
    pageInfo { hasNextPage endCursor }`;
  return `#graphql
query LocaleCoverage${bulk ? "Bulk" : "Page"}${bulk ? "" : "($resourceType: TranslatableResourceType!, $after: String)"} {
  translatableResources(first: ${bulk ? 250 : 100}, ${typeArgument}) {
    ${connectionSelection}
  }
}`;
}

const CONNECTION_QUERY = `#graphql
query SkillHubConnectionCheck {
  shop {
    name
    myshopifyDomain
  }
}`;

const APPLY_VERIFICATION_QUERY = `#graphql
query VerifyMarketsLocalizationActions {
  shopLocales { locale published }
  markets(first: 50) {
    nodes {
      id
      currencySettings { localCurrencies }
    }
  }
}`;

async function cmdConnectionCheck(args) {
  const env = await loadEnv(args.env);
  const data = await graphql(env, CONNECTION_QUERY);
  console.log(JSON.stringify({
    status: "OK",
    shop: data.shop.name,
    myshopifyDomain: data.shop.myshopifyDomain,
    transport: env.SHOPIFY_TRANSPORT || "admin_api",
  }, null, 2));
}

function createCoverageTotals(locales, resourceTypes, initialCoverage = []) {
  const initialByLocale = new Map(initialCoverage.map((item) => [item.locale, item]));
  return new Map(locales.map((locale) => {
    const initial = initialByLocale.get(locale);
    return [locale, initial ? {
      locale,
      resourceTypes,
      eligible: initial.eligible || 0,
      current: initial.current || 0,
      outdated: initial.outdated || 0,
      missing: initial.missing || 0,
      resources: initial.resources || 0,
      perType: Array.isArray(initial.perType) ? initial.perType : [],
    } : { locale, resourceTypes, eligible: 0, current: 0, outdated: 0, missing: 0, resources: 0, perType: [] }];
  }));
}

function createCoverageCounter(totalsByLocale, locales, resourceType) {
  const perTypeByLocale = new Map(locales.map((locale) => [locale, {
    resourceType, eligible: 0, current: 0, outdated: 0, missing: 0, resources: 0,
  }]));

  function consume(node) {
    for (const locale of locales) perTypeByLocale.get(locale).resources += 1;
    for (const content of node.translatableContent || []) {
      if (content.key === "handle" || SKIP_TRANSLATABLE_TYPES.has(content.type)) continue;
      locales.forEach((locale, index) => {
        const perType = perTypeByLocale.get(locale);
        perType.eligible += 1;
        const translations = node[`translation_${index}`] || [];
        const translation = translations.find((entry) => entry.key === content.key);
        if (!translation) perType.missing += 1;
        else if (translation.outdated) perType.outdated += 1;
        else perType.current += 1;
      });
    }
  }

  function commit() {
  for (const locale of locales) {
    const totals = totalsByLocale.get(locale);
    const perType = perTypeByLocale.get(locale);
    totals.eligible += perType.eligible;
    totals.current += perType.current;
    totals.outdated += perType.outdated;
    totals.missing += perType.missing;
    totals.resources += perType.resources;
    const existing = totals.perType.find((item) => item.resourceType === resourceType);
    if (existing) Object.keys(perType).forEach((key) => { if (key !== "resourceType") existing[key] += perType[key]; });
    else totals.perType.push(perType);
  }
  }

  return { consume, commit };
}

function countCoverageNodes(totalsByLocale, locales, resourceType, nodes) {
  const counter = createCoverageCounter(totalsByLocale, locales, resourceType);
  for (const node of nodes || []) counter.consume(node);
  counter.commit();
}

function finalizeCoverage(totalsByLocale) {
  return [...totalsByLocale.values()].map((totals) => ({
    ...totals,
    readinessPct: totals.eligible ? Math.round((totals.current / totals.eligible) * 100) : 0,
    gapPct: totals.eligible ? 100 - Math.round((totals.current / totals.eligible) * 100) : 0,
  }));
}

async function computeLocaleCoverageStandard(env, locales, resourceTypes, options = {}) {
  const totalsByLocale = createCoverageTotals(locales, options.allResourceTypes || resourceTypes, options.initialCoverage);
  for (const resourceType of resourceTypes) {
    for (let offset = 0; offset < locales.length; offset += LOCALE_BATCH_SIZE) {
      const localeBatch = locales.slice(offset, offset + LOCALE_BATCH_SIZE);
      const query = buildTranslatableQuery(localeBatch);
      let after = null;
      let hasNextPage = true;
      while (hasNextPage) {
        const page = await graphql(env, query, { resourceType, after });
        const connection = page.translatableResources;
        countCoverageNodes(totalsByLocale, localeBatch, resourceType, connection.nodes);
        hasNextPage = Boolean(connection.pageInfo?.hasNextPage);
        after = connection.pageInfo?.endCursor || null;
      }
    }
    await options.onResourceComplete?.(resourceType, finalizeCoverage(totalsByLocale));
  }
  return finalizeCoverage(totalsByLocale);
}

async function runWithConcurrency(items, limit, work) {
  let cursor = 0;
  let firstError = null;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      if (firstError) return;
      const index = cursor;
      cursor += 1;
      if (index >= items.length) return;
      try { await work(items[index]); } catch (error) { firstError ||= error; return; }
    }
  }));
  if (firstError) throw firstError;
}

async function computeLocaleCoverageBulk(env, locales, resourceTypes, options = {}) {
  const totalsByLocale = createCoverageTotals(locales, options.allResourceTypes || resourceTypes, options.initialCoverage);
  await runWithConcurrency(resourceTypes, BULK_CONCURRENCY, async (resourceType) => {
    console.error(`[audit] Bulk scanning ${resourceType} for ${locales.length} locale(s)...`);
    const counter = createCoverageCounter(totalsByLocale, locales, resourceType);
    const query = buildTranslatableQuery(locales, { resourceType, bulk: true });
    await shopifyCliBulkQuery(env, query, counter.consume);
    counter.commit();
    await options.onResourceComplete?.(resourceType, finalizeCoverage(totalsByLocale));
  });
  return finalizeCoverage(totalsByLocale);
}

function inferStoreProfile(data) {
  const collections = (data.collections?.nodes || []).map((item) => item.title).filter(Boolean);
  const products = (data.products?.nodes || []).filter(Boolean);
  const productTitles = products.map((item) => item.title).filter(Boolean);
  const productTypes = products.map((item) => item.productType).filter(Boolean);
  const productTags = products.flatMap((item) => Array.isArray(item.tags) ? item.tags : []).filter(Boolean);
  const evidenceLines = [
    data.shop?.name || "",
    data.shop?.description || "",
    ...collections,
    ...productTitles,
    ...productTypes,
    ...productTags,
  ].filter(Boolean);
  const categoryRules = [
    { key: "winter-sports", pattern: /(ski|skis|skiing|snowboard|winter sports?)/ },
    { key: "pet", pattern: /(pet|dog|cat|carrier|harness)/ },
    { key: "beauty", pattern: /(beauty|skincare|cosmetic)/ },
    { key: "jewelry", pattern: /(jewelry|necklace|ring|bracelet)/ },
    { key: "home", pattern: /(home|decor|kitchen|bedding|furniture)/ },
  ];
  const categoryScores = new Map(categoryRules.map((rule) => [rule.key, 0]));

  for (const line of evidenceLines) {
    const normalized = line.toLowerCase();
    for (const rule of categoryRules) {
      if (rule.pattern.test(normalized)) {
        categoryScores.set(rule.key, (categoryScores.get(rule.key) || 0) + 1);
      }
    }
  }

  const categoryLabels = [...categoryScores.entries()]
    .filter(([, score]) => score >= 2)
    .sort((a, b) => b[1] - a[1])
    .map(([key]) => key);

  let summary = "general ecommerce brand";
  if (categoryLabels.includes("winter-sports")) summary = "winter sports and ski gear brand";
  else if (categoryLabels.includes("pet")) summary = "pet-focused brand";
  else if (categoryLabels.includes("beauty")) summary = "beauty-focused brand";
  else if (categoryLabels.includes("jewelry")) summary = "jewelry-focused brand";
  else if (categoryLabels.includes("home")) summary = "home-focused brand";

  return {
    status: "store_profile_only",
    summary,
    categoryLabels,
    evidence: {
      shopName: data.shop?.name || "",
      shopDescription: data.shop?.description || "",
      collections,
      productTitles,
      productTypes,
      categoryScores: Object.fromEntries(categoryScores),
    },
    nextStep: "Complete external market research with at least 3 fresh category-country references before final country recommendations.",
  };
}

function normalizeOverview(data) {
  const localeMap = new Map((data.shopLocales || []).map((locale) => [locale.locale, locale]));
  const markets = (data.markets?.nodes || []).map((market) => {
    const countries = (market.regions?.nodes || []).map((country) => ({
      code: country.code,
      name: country.name,
    }));
    const defaultLocale = market.webPresence?.defaultLocale?.locale || null;
    const alternateLocales = (market.webPresence?.alternateLocales || []).map((entry) => entry.locale);
    return {
      id: market.id,
      name: market.name,
      handle: market.handle,
      enabled: market.enabled,
      primary: market.primary,
      status: market.status,
      defaultLocale,
      alternateLocales,
      allLocales: [defaultLocale, ...alternateLocales].filter(Boolean),
      countryCodes: countries.map((country) => country.code),
      countries,
      localCurrencies: Boolean(market.currencySettings?.localCurrencies),
      baseCurrencyCode: market.currencySettings?.baseCurrency?.currencyCode || null,
      baseCurrencyName: market.currencySettings?.baseCurrency?.currencyName || null,
      baseCurrencyEnabled: Boolean(market.currencySettings?.baseCurrency?.enabled),
      webPresenceId: market.webPresence?.id || null,
      subfolderSuffix: market.webPresence?.subfolderSuffix || null,
      rootUrls: market.webPresence?.rootUrls || [],
    };
  });

  const shippingCountries = new Set();
  const deliveryProfiles = (data.deliveryProfiles?.nodes || []).map((profile) => {
    const countries = [];
    for (const group of profile.profileLocationGroups || []) {
      for (const entry of group.countriesInAnyZone || []) {
        const countryCode = entry?.country?.code?.countryCode;
        if (!countryCode || entry.country.code.restOfWorld) continue;
        shippingCountries.add(countryCode);
        countries.push({
          code: countryCode,
          name: entry.country.name,
          zone: entry.zone,
        });
      }
    }
    return { id: profile.id, name: profile.name, countries };
  });

  return {
    shop: data.shop,
    storeProfile: inferStoreProfile(data),
    shopLocales: data.shopLocales || [],
    localeMap,
    markets,
    deliveryProfiles,
    shippingCountryCodes: [...shippingCountries],
  };
}

function chooseLocalesForAudit(overview, requestedLocales) {
  const primaryLocale = overview.shopLocales.find((locale) => locale.primary)?.locale || null;
  if (requestedLocales?.length) return requestedLocales.filter((locale) => locale !== primaryLocale);
  const locales = new Set();
  for (const locale of overview.shopLocales) {
    if (!locale.primary && locale.published) locales.add(locale.locale);
  }
  for (const market of overview.markets) {
    for (const locale of market.allLocales) {
      if (locale !== primaryLocale) locales.add(locale);
    }
  }
  return [...locales];
}

function buildFindings(overview, coverageByLocale, storefront) {
  const findings = [];
  const workingWell = [];
  const localeCoverageMap = new Map(coverageByLocale.map((entry) => [entry.locale, entry]));

  for (const locale of overview.shopLocales) {
    if (locale.primary && locale.published) {
      workingWell.push({
        title: `${locale.locale} is live`,
        summary: "The store's main language is enabled and visible to shoppers.",
        nextStep: "",
      });
    }
  }

  for (const market of overview.markets) {
    for (const localeCode of market.allLocales) {
      const locale = overview.localeMap.get(localeCode);
      if (!locale) {
        findings.push({
          level: "high",
          bucket: "fix-now",
          title: `${market.name} points to a language that is not enabled`,
          summary: `${localeCode} is attached to this market, but the store does not have it enabled.`,
          nextStep: "Enable the language first, then re-check the market setup.",
        });
      } else if (!locale.published) {
        findings.push({
          level: "high",
          bucket: "fix-now",
          title: `${market.name} uses a language that is still hidden`,
          summary: `${localeCode} exists, but shoppers cannot fully use it yet because it is not published.`,
          nextStep: "Publish the language, then refresh this market check.",
        });
      }
    }

    if (!market.webPresenceId && !market.primary) {
      findings.push({
        level: "medium",
        bucket: "tidy-next",
        title: `${market.name} has no dedicated web setup yet`,
        summary: "Shoppers can still reach the store, but the market does not have a clean URL setup for its local experience.",
        nextStep: "Review whether this market needs its own subfolder or domain strategy.",
      });
    }

    if (!market.localCurrencies && market.countryCodes.length > 1) {
      findings.push({
        level: "medium",
        bucket: "tidy-next",
        title: `${market.name} is sharing one money setup across several countries`,
        summary: "This is not always wrong, but it often leads to a less local buying experience.",
        nextStep: "Review whether local currencies should be turned on for this market.",
      });
    }

    const missingShipping = market.countryCodes.filter((code) => !overview.shippingCountryCodes.includes(code));
    if (missingShipping.length) {
      findings.push({
        level: "high",
        bucket: "fix-now",
        title: `${market.name} has countries with no shipping coverage`,
        summary: `${missingShipping.length} country${missingShipping.length === 1 ? "" : "ies"} in this market do not appear in any shipping zone.`,
        nextStep: "Add shipping coverage or remove those countries from the market until shipping is ready.",
      });
    } else if (market.countryCodes.length) {
      workingWell.push({
        title: `${market.name} has shipping coverage`,
        summary: "Every country in this market appears in at least one shipping zone.",
        nextStep: "",
      });
    }
  }

  for (const coverage of coverageByLocale) {
    if (coverage.readinessPct < 70) {
      findings.push({
        level: "high",
        bucket: "fix-now",
        title: `${coverage.locale} is not ready enough yet`,
        summary: `Only ${coverage.readinessPct}% of the checked text is up to date.`,
        nextStep: "Bring missing and old translations up to date before treating this language as finished.",
      });
    } else if (coverage.readinessPct < 90 || coverage.outdated > 0) {
      findings.push({
        level: "medium",
        bucket: "tidy-next",
        title: `${coverage.locale} still has cleanup work`,
        summary: `${coverage.missing + coverage.outdated} pieces of text are still missing or old.`,
        nextStep: "Tidy the remaining gaps so the whole language feels complete.",
      });
    } else {
      workingWell.push({
        title: `${coverage.locale} is in good shape`,
        summary: `Most checked text is current, with ${coverage.readinessPct}% ready.`,
        nextStep: "",
      });
    }
  }

  if (!storefront.canonical) {
    findings.push({
      level: "high",
      bucket: "fix-now",
      title: "The storefront is missing a canonical link",
      summary: "Search engines are not getting a clear main URL signal from the home page.",
      nextStep: "Add a canonical tag in the theme head.",
    });
  }

  if (storefront.hreflangCount === 0 && coverageByLocale.length > 0) {
    findings.push({
      level: "high",
      bucket: "fix-now",
      title: "Localized pages are missing language links for search engines",
      summary: "The home page does not expose `hreflang` alternate links.",
      nextStep: "Add `hreflang` support so search engines can understand the language versions.",
    });
  }

  const policyUrls = (overview.shop.shopPolicies || [])
    .map((policy) => policy.url)
    .filter(Boolean);
  if (!policyUrls.length) {
    findings.push({
      level: "medium",
      bucket: "tidy-next",
      title: "The store is light on policy pages",
      summary: "Key policy pages are missing from the Admin data.",
      nextStep: "Add policy pages so shoppers can understand shipping, returns, and privacy before buying.",
    });
  }

  if (policyUrls.length && storefront.policyLinks.length === 0) {
    findings.push({
      level: "medium",
      bucket: "tidy-next",
      title: "Policy pages exist but are hard to reach from the storefront",
      summary: "The storefront crawl did not find obvious policy links on the home page.",
      nextStep: "Add policy links to the footer or another consistent store area.",
    });
  }

  return { findings, workingWell };
}

function buildFixPlan(overview, coverageByLocale) {
  const actions = [];
  const actionIds = new Set();
  const localeMap = overview.localeMap;
  const weakLocales = coverageByLocale.filter((item) => item.readinessPct < 70);

  function pushAction(action) {
    if (actionIds.has(action.id)) return;
    actionIds.add(action.id);
    actions.push(action);
  }

  for (const market of overview.markets) {
    for (const localeCode of market.allLocales) {
      const locale = localeMap.get(localeCode);
      if (!locale) {
        pushAction({
          id: `enable-publish-${localeCode}`,
          type: "enable_publish_locale",
          title: `Turn on ${localeCode}`,
          summary: `This language is attached to a market but is not enabled in the store.`,
          reason: "The market cannot work cleanly until the language exists and is published.",
          risk: "low",
          executeSupported: true,
          variables: { locale: localeCode },
        });
      } else if (!locale.published) {
        pushAction({
          id: `publish-${localeCode}`,
          type: "publish_locale",
          title: `Publish ${localeCode}`,
          summary: `This language already exists, but shoppers still cannot fully use it.`,
          reason: "Publishing it is the smallest safe fix.",
          risk: "low",
          executeSupported: true,
          variables: { locale: localeCode },
        });
      }
    }

    if (!market.localCurrencies && market.countryCodes.length > 1) {
      pushAction({
        id: `local-currencies-${market.id}`,
        type: "enable_local_currencies",
        title: `Turn on local currencies for ${market.name}`,
        summary: "This market covers several countries but local currencies are off.",
        reason: "This often improves the local shopping experience. Shopify Payments support may still be required.",
        risk: "medium",
        executeSupported: true,
        variables: { marketId: market.id },
      });
    }
  }

  return {
    actions,
    explanation: actions.length === 0 && weakLocales.length
      ? `The main gaps are translation-readiness issues in ${weakLocales.map((item) => item.locale).join(", ")}. V1 does not auto-write translations, so these need manual translation work or a translation app instead of a direct Admin API fix.`
      : "",
  };
}

function groupFindings(findings, bucket) {
  return findings.filter((item) => item.bucket === bucket);
}

function uniqueStrings(values) {
  return [...new Set(values.filter(Boolean))];
}

function takeLabels(values, limit = 6) {
  if (values.length <= limit) return values;
  return [...values.slice(0, limit), `+${values.length - limit}`];
}

function buildLocaleStatusLabel(locale, zh) {
  if (locale.primary) return zh ? "\u4e3b\u8bed\u8a00" : "Primary";
  if (locale.published) return zh ? "\u5df2\u53d1\u5e03" : "Published";
  return zh ? "\u5df2\u6dfb\u52a0\u672a\u53d1\u5e03" : "Enabled, not published";
}

function buildMarketCurrencyLabel(market, zh) {
  const base = market.baseCurrencyCode || (zh ? "\u5e97\u94fa\u57fa\u7840\u5e01\u79cd" : "shop base currency");
  if (market.localCurrencies) {
    return zh ? "\u81ea\u52a8\u6309\u56fd\u5bb6\u663e\u793a\u672c\u5730\u8d27\u5e01" : "Auto local currencies by country";
  }
  if (market.countryCodes.length <= 1) {
    return zh ? `\u56fa\u5b9a\u4e3a ${base}` : `Fixed to ${base}`;
  }
  return zh ? `\u591a\u4e2a\u56fd\u5bb6\u5171\u7528 ${base}` : `Shared ${base} across countries`;
}

function buildMarketUrlLabel(market, zh) {
  if (market.webPresenceId && market.subfolderSuffix) {
    return zh ? `\u5b50\u76ee\u5f55 /${market.subfolderSuffix}/` : `Subfolder /${market.subfolderSuffix}/`;
  }
  if (market.webPresenceId && market.rootUrls.length) {
    const root = market.rootUrls[0]?.url || "";
    return zh ? `\u72ec\u7acb\u5165\u53e3 ${root}` : `Dedicated URL ${root}`;
  }
  return zh ? "\u8ddf\u968f\u4e3b\u7ad9\u5165\u53e3" : "Shared main storefront entry";
}

function buildExpansionIdeasFromAudit(audit, zh) {
  const publishedCount = audit.shopLocales.filter((item) => item.published && !item.primary).length;
  const remainingSlots = Math.max(0, 20 - publishedCount);
  const localeSet = new Set(audit.shopLocales.map((item) => item.locale));
  const weakLocales = audit.localeCoverage.filter((item) => item.readinessPct < 65);
  const sharedUrlMarkets = audit.markets.filter((market) => !market.webPresenceId);
  const sharedCurrencyMarkets = audit.markets.filter((market) => !market.localCurrencies && market.countryCodes.length > 1);
  const multilingualMarkets = audit.markets.filter((market) => market.allLocales.length > 1);
  const profile = audit.storeProfile || null;
  const ideas = [];

  if (profile) {
    ideas.push({
      title: zh ? "\u5e97\u94fa\u753b\u50cf" : "Store profile",
      summary: zh
        ? `\u57fa\u4e8e\u5e97\u540d\u3001\u63cf\u8ff0\u3001\u7cfb\u5217\u548c\u5546\u54c1\u6807\u9898\uff0c\u8fd9\u5bb6\u5e97\u5f53\u524d\u66f4\u50cf\u662f ${profile.summary}\u3002`
        : `Based on the shop name, description, collections, and product titles, this store currently looks like a ${profile.summary}.`,
      nextStep: zh
        ? "\u8fd9\u4e00\u6b65\u53ea\u662f\u5e97\u94fa\u753b\u50cf\uff0c\u4e0d\u7b49\u4e8e\u6700\u7ec8\u56fd\u5bb6\u5efa\u8bae\u3002\u771f\u6b63\u7684\u5e02\u573a\u5efa\u8bae\u8fd8\u9700\u8981\u8865\u5916\u90e8\u884c\u4e1a\u7814\u7a76\u3002"
        : "This is only the store profile, not the final country recommendation. External market research is still required.",
    });
  }

  ideas.push({
    title: zh ? "\u5148\u8865\u5916\u90e8\u7814\u7a76\uff0c\u518d\u7ed9\u56fd\u5bb6\u5efa\u8bae" : "Finish external research before country advice",
    summary: zh
      ? "\u5f53\u524d\u811a\u672c\u53ea\u5b8c\u6210\u4e86\u5e97\u94fa\u753b\u50cf\uff0c\u8fd8\u6ca1\u6709\u5b8c\u6210\u81f3\u5c11 3 \u4efd\u5916\u90e8\u884c\u4e1a\u8d44\u6599\u4ea4\u53c9\u9a8c\u8bc1\uff0c\u6240\u4ee5\u8fd9\u91cc\u4e0d\u76f4\u63a5\u8f93\u51fa\u786c\u6027\u7684\u56fd\u5bb6\u7ed3\u8bba\u3002"
      : "This script completes the store profile, but it does not complete the required 3-source external research step, so it should not output hard country conclusions yet.",
    nextStep: zh
      ? "\u4e0b\u4e00\u6b65\u8bf7\u56f4\u7ed5\u8fd9\u4e2a\u54c1\u7c7b\u8865\u81f3\u5c11 3 \u4efd\u6700\u65b0\u53ef\u4fe1\u8d44\u6599\uff0c\u518d\u51b3\u5b9a\u4f18\u5148\u56fd\u5bb6\u3002"
      : "Next, gather at least 3 fresh category-country references before finalizing priority countries.",
  });

  ideas.push({
    title: zh ? `\u8fd8\u53ef\u4ee5\u518d\u589e\u52a0 ${remainingSlots} \u4e2a\u5df2\u53d1\u5e03\u8bed\u8a00` : `You still have room for ${remainingSlots} more published language${remainingSlots === 1 ? "" : "s"}`,
    summary: zh ? "Shopify \u6700\u591a\u8fd8\u80fd\u7ee7\u7eed\u589e\u52a0\u8bed\u8a00\uff0c\u4f46\u5269\u4f59\u540d\u989d\u66f4\u9002\u5408\u7559\u7ed9\u771f\u6b63\u6709\u8f6c\u5316\u6f5c\u529b\u7684\u56fd\u5bb6\u3002" : "There is still room to add languages, but the remaining slots should go to the markets that can really convert.",
    nextStep: zh ? "\u4e0d\u8981\u5e73\u5747\u94fa\u5f00\uff0c\u4f18\u5148\u7ed9\u6700\u503c\u5f97\u505a\u6df1\u7684\u56fd\u5bb6\u548c\u8bed\u8a00\u3002" : "Do not spread evenly. Prioritize the countries and languages worth going deeper on.",
  });

  if (weakLocales.length) {
    ideas.push({
      title: zh ? "\u5148\u8865\u5f3a\u5df2\u6709\u8bed\u8a00\uff0c\u901a\u5e38\u6bd4\u7ee7\u7eed\u6269\u65b0\u5e02\u573a\u66f4\u5212\u7b97" : "Strengthening current languages may beat adding new markets",
      summary: zh ? `${weakLocales.map((item) => item.locale).join("\u3001")} \u8fd9\u7c7b\u8bed\u8a00\u8fd8\u6ca1\u6709\u771f\u6b63\u7ad9\u7a33\u3002` : `${weakLocales.map((item) => item.locale).join(", ")} still look too weak to treat as complete.`,
      nextStep: zh ? "\u5148\u628a\u8fd9\u4e9b\u8bed\u8a00\u505a\u5b8c\u6574\uff0c\u518d\u7ee7\u7eed\u6269\uff0c\u4f1a\u66f4\u7a33\u4e5f\u66f4\u7701\u5fc3\u3002" : "Finish these languages first before pushing broader expansion.",
    });
  }

  if (sharedUrlMarkets.length >= 4) {
    ideas.push({
      title: zh ? "\u91cd\u70b9\u56fd\u5bb6\u66f4\u9002\u5408\u6709\u81ea\u5df1\u7684\u5165\u53e3" : "Top markets deserve their own entry points",
      summary: zh ? "\u73b0\u5728\u4ecd\u6709\u4e0d\u5c11\u5e02\u573a\u5728\u8ddf\u968f\u4e3b\u7ad9\u5165\u53e3\uff0c\u8fd9\u4f1a\u5f71\u54cd\u672c\u5730\u611f\u548c\u56fd\u9645 SEO \u7684\u653e\u5927\u6548\u679c\u3002" : "A large share of markets still rely on the main storefront entry, which can limit local feel and SEO lift.",
      nextStep: zh ? "\u5148\u6311\u8f6c\u5316\u9ad8\u6216\u641c\u7d22\u91cf\u9ad8\u7684\u56fd\u5bb6\u505a\u5b50\u76ee\u5f55\u6216\u72ec\u7acb\u57df\u540d\u3002" : "Start by giving the highest-opportunity markets their own subfolder or dedicated domain path.",
    });
  }

  if (sharedCurrencyMarkets.length) {
    ideas.push({
      title: zh ? "\u591a\u56fd\u5e02\u573a\u8981\u540c\u65f6\u770b\u8bed\u8a00\u3001\u8d27\u5e01\u548c\u5b9a\u4ef7" : "Mixed-country markets need language, currency, and pricing together",
      summary: zh ? "\u591a\u56fd\u5171\u7528\u4e00\u4e2a\u5e02\u573a\u65f6\uff0c\u8fd0\u8425\u4e0a\u7701\u4e8b\uff0c\u4f46\u987e\u5ba2\u4f53\u9a8c\u4e0d\u4e00\u5b9a\u8db3\u591f\u672c\u5730\u3002" : "Operationally, one market can be easier to manage, but it can feel less local to buyers.",
      nextStep: zh ? "\u628a\u9ad8\u4ef7\u503c\u56fd\u5bb6\u4f18\u5148\u62c6\u51fa\u6765\uff0c\u7ed9\u66f4\u6e05\u6670\u7684\u8d27\u5e01\u548c\u7f51\u5740\u7b56\u7565\u3002" : "Split out the highest-value countries first for clearer pricing and local experience.",
    });
  }

  if (multilingualMarkets.length) {
    ideas.push({
      title: zh ? "\u591a\u8bed\u8a00\u56fd\u5bb6\u66f4\u8981\u91cd\u89c6\u9ed8\u8ba4\u8bed\u8a00" : "Multilingual countries need sharper default-language choices",
      summary: zh ? "\u50cf\u52a0\u62ff\u5927\u3001\u6bd4\u5229\u65f6\u8fd9\u7c7b\u5e02\u573a\uff0c\u9ed8\u8ba4\u8bed\u8a00\u987a\u5e8f\u4f1a\u76f4\u63a5\u5f71\u54cd\u987e\u5ba2\u7b2c\u4e00\u773c\u4f53\u9a8c\u548c\u641c\u7d22\u8868\u73b0\u3002" : "In multilingual countries, the default language choice influences both buyer clarity and SEO.",
      nextStep: zh ? "\u9ed8\u8ba4\u8bed\u8a00\u5148\u7ed9\u4e3b\u6d41\u5ba2\u7fa4\uff0c\u5176\u4ed6\u8bed\u8a00\u4f5c\u4e3a\u8865\u5145\u5165\u53e3\u3002" : "Use the dominant buyer language as default and keep the others as supporting paths.",
    });
  }

  if (localeSet.has("de") || localeSet.has("fr") || localeSet.has("it") || localeSet.has("es") || localeSet.has("nl")) {
    ideas.push({
      title: zh ? "\u6b27\u6d32\u5e02\u573a\u66f4\u9002\u5408\u7ee7\u7eed\u505a\u6df1" : "Europe usually rewards deeper localization",
      summary: zh ? "\u65e2\u7136\u5df2\u7ecf\u8986\u76d6\u591a\u79cd\u6b27\u6d32\u8bed\u8a00\uff0c\u63a5\u4e0b\u6765\u66f4\u503c\u5f97\u8865\u7684\u662f\u653f\u7b56\u3001\u9000\u8d27\u627f\u8bfa\u3001\u914d\u9001\u65f6\u6548\u548c\u672c\u5730\u5165\u53e3\uff0c\u800c\u4e0d\u53ea\u662f\u7ee7\u7eed\u52a0\u66f4\u591a\u56fd\u5bb6\u3002" : "Once several European languages are live, the next gains often come from policies, returns, delivery promises, and stronger local entry points rather than just adding more countries.",
      nextStep: zh ? "\u5148\u628a\u51e0\u4e2a\u6838\u5fc3\u6b27\u6d32\u5e02\u573a\u505a\u624e\u5b9e\uff0c\u518d\u8003\u8651\u7ee7\u7eed\u5916\u6269\u3002" : "Deepen the strongest European markets before adding more edge cases.",
    });
  }

  if (localeSet.has("ja") || localeSet.has("ko")) {
    ideas.push({
      title: zh ? "\u4e1c\u4e9a\u5e02\u573a\u66f4\u5403\u5b8c\u6574\u5ea6\u548c\u4fe1\u4efb\u611f" : "East Asia usually rewards polish and trust signals",
      summary: zh ? "\u65e5\u8bed\u3001\u97e9\u8bed\u5e02\u573a\u5f80\u5f80\u4e0d\u53ea\u662f\u5546\u54c1\u9875\u7ffb\u8bd1\uff0c\u914d\u9001\u3001\u9000\u8d27\u3001\u5ba2\u670d\u627f\u8bfa\u548c\u5e01\u79cd\u4f53\u9a8c\u4e5f\u5f88\u5173\u952e\u3002" : "For Japanese and Korean markets, policy clarity and checkout confidence often matter as much as product translation.",
      nextStep: zh ? "\u4f18\u5148\u8865\u5f53\u5730\u8bed\u8a00\u653f\u7b56\u9875\u3001\u914d\u9001\u627f\u8bfa\u548c\u79fb\u52a8\u7aef\u8def\u5f84\u3002" : "Prioritize local-language policies, delivery promises, and mobile navigation.",
    });
  }

  return ideas.slice(0, 8);
}

function buildReportModel(audit) {
  const fixPlan = Array.isArray(audit.fixPlan)
    ? { actions: audit.fixPlan, explanation: "" }
    : (audit.fixPlan || { actions: [], explanation: "" });
  const fixNow = groupFindings(audit.findings, "fix-now");
  const tidyNext = groupFindings(audit.findings, "tidy-next");
  const manualChecks = groupFindings(audit.findings, "manual");
  const lang = audit.reportLang || "zh-CN";
  const zh = lang.toLowerCase().startsWith("zh");
  const localeCoverageMap = new Map(audit.localeCoverage.map((entry) => [entry.locale, entry]));
  const localeAssignments = new Map();
  for (const market of audit.markets) {
    for (const locale of market.allLocales) {
      if (!localeAssignments.has(locale)) localeAssignments.set(locale, []);
      localeAssignments.get(locale).push(market.name);
    }
  }

  const marketsWithCountries = audit.markets.filter((market) => market.countryCodes.length > 0);
  const shippingGapMarkets = marketsWithCountries.filter((market) => market.countryCodes.some((code) => !audit.shippingCountryCodes.includes(code)));
  const coveredMarkets = marketsWithCountries.length - shippingGapMarkets.length;
  const weakLocales = audit.localeCoverage.filter((entry) => entry.readinessPct < 70);
  const cleanupLocales = audit.localeCoverage.filter((entry) => entry.readinessPct >= 70 && (entry.readinessPct < 90 || entry.outdated > 0));
  const strongLocales = audit.localeCoverage.filter((entry) => entry.readinessPct >= 90 && entry.outdated === 0);
  const marketsMissingWebPresence = audit.markets.filter((market) => !market.webPresenceId);
  const marketsWithDedicatedWeb = audit.markets.filter((market) => market.webPresenceId);
  const sharedCurrencyMarkets = audit.markets.filter((market) => !market.localCurrencies && market.countryCodes.length > 1);
  const dynamicCurrencyMarkets = audit.markets.filter((market) => market.localCurrencies);
  const hiddenLocaleIssues = audit.markets.flatMap((market) => market.allLocales
    .map((localeCode) => ({ market: market.name, localeCode, locale: audit.shopLocales.find((item) => item.locale === localeCode) || null }))
    .filter((entry) => !entry.locale || !entry.locale.published));

  let score = 100;
  score -= Math.min(weakLocales.length * 14, 28);
  score -= Math.min(shippingGapMarkets.length * 18, 36);
  score -= Math.min(hiddenLocaleIssues.length * 12, 24);
  score -= Math.min(sharedCurrencyMarkets.length * 4, 12);
  if (!audit.storefront.canonical) score -= 8;
  if (audit.localeCoverage.length > 0 && audit.storefront.hreflangCount === 0) score -= 8;
  if (marketsMissingWebPresence.length > Math.max(3, Math.floor(audit.markets.length * 0.5))) score -= 6;
  score = Math.max(42, score);

  const strengthCards = [
    {
      tone: "good",
      title: zh ? "\u914d\u9001\u8986\u76d6" : "Shipping coverage",
      value: `${coveredMarkets}/${marketsWithCountries.length || 0}`,
      detail: zh ? "\u5e02\u573a\u91cc\u7684\u56fd\u5bb6\u662f\u5426\u90fd\u5728\u914d\u9001\u8986\u76d6\u5185" : "Markets whose countries are covered by shipping",
      note: shippingGapMarkets.length
        ? (zh ? `${shippingGapMarkets.length} \u4e2a\u5e02\u573a\u8fd8\u6709\u914d\u9001\u7f3a\u53e3` : `${shippingGapMarkets.length} markets still have shipping gaps`)
        : (zh ? "\u76ee\u524d\u6ca1\u6709\u660e\u663e\u914d\u9001\u7f3a\u53e3" : "No obvious shipping gaps found"),
      chips: shippingGapMarkets.length ? takeLabels(shippingGapMarkets.map((market) => market.name)) : [],
    },
    {
      tone: "good",
      title: zh ? "\u8bed\u8a00\u4e0a\u7ebf\u60c5\u51b5" : "Published languages",
      value: `${audit.shopLocales.filter((item) => item.published).length}/${audit.shopLocales.length || 0}`,
      detail: zh ? "\u5df2\u53d1\u5e03\u5e76\u5bf9\u987e\u5ba2\u53ef\u89c1\u7684\u8bed\u8a00\u6570" : "Languages visible to buyers",
      note: weakLocales.length
        ? (zh ? `${weakLocales.length} \u4e2a\u8bed\u8a00\u8fd8\u6ca1\u51c6\u5907\u597d` : `${weakLocales.length} languages still look weak`)
        : (zh ? "\u5927\u90e8\u5206\u8bed\u8a00\u5df2\u7ecf\u5bf9\u5916\u53ef\u7528" : "Most languages are already live"),
      chips: takeLabels(audit.shopLocales.filter((item) => item.published).map((item) => item.locale)),
    },
    {
      tone: "good",
      title: zh ? "\u7f51\u5740\u672c\u5730\u5316" : "Localized URLs",
      value: `${marketsWithDedicatedWeb.length}/${audit.markets.length || 0}`,
      detail: zh ? "\u6709\u5355\u72ec\u5165\u53e3\u7684\u5e02\u573a\u6570" : "Markets with dedicated URL entry points",
      note: marketsMissingWebPresence.length
        ? (zh ? `${marketsMissingWebPresence.length} \u4e2a\u5e02\u573a\u4ecd\u8ddf\u968f\u4e3b\u7ad9\u5165\u53e3` : `${marketsMissingWebPresence.length} markets still share the main storefront entry`)
        : (zh ? "\u91cd\u70b9\u5e02\u573a\u5df2\u7ecf\u6709\u81ea\u5df1\u7684\u5165\u53e3" : "Key markets already have their own entry points"),
      chips: takeLabels(marketsWithDedicatedWeb.map((market) => market.name)),
    },
    {
      tone: "good",
      title: zh ? "\u641c\u7d22\u5f15\u64ce\u8bc6\u522b" : "Search signals",
      value: `${audit.storefront.canonical ? 1 : 0}/${audit.localeCoverage.length > 0 ? 2 : 1}`,
      detail: zh ? "canonical \u548c hreflang \u57fa\u7840\u4fe1\u53f7" : "Canonical and hreflang basics",
      note: audit.storefront.canonical
        ? (audit.storefront.hreflangCount > 0 || audit.localeCoverage.length === 0
          ? (zh ? "\u4e3b\u9875\u5173\u952e SEO \u4fe1\u53f7\u57fa\u672c\u9f50\u4e86" : "The homepage exposes the main localization signals")
          : (zh ? "canonical \u6709\u4e86\uff0c\u4f46 hreflang \u8fd8\u6ca1\u8865\u9f50" : "Canonical is present, but hreflang still needs work"))
        : (zh ? "\u4e3b\u9875\u8fd8\u7f3a\u5c11\u5173\u952e SEO \u4fe1\u53f7" : "The homepage is still missing a key SEO signal"),
      chips: audit.storefront.hreflangCount ? [zh ? `${audit.storefront.hreflangCount} \u6761 hreflang` : `${audit.storefront.hreflangCount} hreflang links`] : [],
    },
  ];

  const priorityCards = uniqueStrings([
    weakLocales.length ? "weak-locales" : "",
    shippingGapMarkets.length ? "shipping-gaps" : "",
    hiddenLocaleIssues.length ? "hidden-locales" : "",
    (!audit.storefront.canonical || (audit.localeCoverage.length > 0 && audit.storefront.hreflangCount === 0)) ? "seo-signals" : "",
  ]).map((key) => {
    if (key === "weak-locales") {
      return {
        tone: "bad",
        title: zh ? "\u7ffb\u8bd1\u51c6\u5907\u5ea6\u504f\u4f4e" : "Translation readiness is too low",
        value: `${weakLocales.length}`,
        detail: zh ? "\u8fd9\u4e9b\u8bed\u8a00\u73b0\u5728\u8fd8\u4e0d\u9002\u5408\u5f53\u6210\u5df2\u5b8c\u6210\u5e02\u573a" : "These languages are not ready to be treated as complete",
        note: zh ? "\u5148\u8865\u8fd9\u4e9b\u8bed\u8a00\uff0c\u518d\u8c08\u7ee7\u7eed\u6269\u65b0\u5e02\u573a\u4f1a\u66f4\u7a33\u3002" : "Strengthen these languages before expanding further.",
        chips: takeLabels(weakLocales.map((entry) => entry.locale)),
      };
    }
    if (key === "shipping-gaps") {
      return {
        tone: "bad",
        title: zh ? "\u6709\u5e02\u573a\u7f3a\u914d\u9001\u8986\u76d6" : "Some markets lack shipping coverage",
        value: `${shippingGapMarkets.length}`,
        detail: zh ? "\u5e02\u573a\u5df2\u5f00\uff0c\u4f46\u56fd\u5bb6\u8fd8\u6ca1\u5b8c\u5168\u843d\u5230\u914d\u9001\u533a" : "The market exists, but some countries still lack shipping-zone coverage",
        note: zh ? "\u8fd9\u7c7b\u95ee\u9898\u4f1a\u76f4\u63a5\u5f71\u54cd\u987e\u5ba2\u4e0b\u5355\u4f53\u9a8c\u3002" : "This directly affects whether people can buy smoothly.",
        chips: takeLabels(shippingGapMarkets.map((market) => market.name)),
      };
    }
    if (key === "hidden-locales") {
      return {
        tone: "bad",
        title: zh ? "\u5e02\u573a\u6302\u4e86\u8fd8\u6ca1\u516c\u5f00\u7684\u8bed\u8a00" : "A market points to a hidden language",
        value: `${hiddenLocaleIssues.length}`,
        detail: zh ? "\u8bed\u8a00\u5b58\u5728\uff0c\u4f46\u987e\u5ba2\u8fd8\u770b\u4e0d\u5230\uff0c\u6216\u8005\u8bed\u8a00\u8fd8\u6ca1\u771f\u7684\u542f\u7528" : "The language exists on paper but is not fully live for buyers",
        note: zh ? "\u5148\u628a\u8bed\u8a00\u542f\u7528\u5e76\u53d1\u5e03\uff0c\u518d\u7ee7\u7eed\u8c03\u6574\u5e02\u573a\u3002" : "Enable and publish the language before refining the market.",
        chips: takeLabels(hiddenLocaleIssues.map((entry) => `${entry.market} \u00b7 ${entry.localeCode}`)),
      };
    }
    return {
      tone: "bad",
      title: zh ? "\u641c\u7d22\u4fe1\u53f7\u8fd8\u6ca1\u8865\u9f50" : "Storefront search signals need work",
      value: `${Number(!audit.storefront.canonical) + Number(audit.localeCoverage.length > 0 && audit.storefront.hreflangCount === 0)}`,
      detail: zh ? "\u8fd9\u4f1a\u5f71\u54cd\u641c\u7d22\u5f15\u64ce\u8bc6\u522b\u4e0d\u540c\u8bed\u8a00\u9875\u9762" : "Search engines may not understand the localized storefront cleanly",
      note: zh ? "\u5148\u8865 canonical \u548c hreflang\uff0c\u518d\u8c08\u56fd\u9645 SEO \u653e\u5927\u3002" : "Fix canonical and hreflang before pushing harder on international SEO.",
      chips: uniqueStrings([
        !audit.storefront.canonical ? "canonical" : "",
        audit.localeCoverage.length > 0 && audit.storefront.hreflangCount === 0 ? "hreflang" : "",
      ]),
    };
  });

  const improveCards = [
    {
      tone: "warn",
      title: zh ? "\u8fd8\u6ca1\u505a\u72ec\u7acb\u5165\u53e3\u7684\u5e02\u573a" : "Markets still sharing the main storefront",
      value: `${marketsMissingWebPresence.length}`,
      detail: zh ? "\u4e0d\u662f\u9519\u8bef\uff0c\u4f46\u672c\u5730\u611f\u548c SEO \u5f80\u5f80\u4f1a\u5f31\u4e00\u4e9b" : "Not always wrong, but usually less local and weaker for SEO",
      note: zh ? "\u4f18\u5148\u628a\u91cd\u70b9\u56fd\u5bb6\u505a\u6210\u5b50\u76ee\u5f55\u6216\u72ec\u7acb\u57df\u540d\u5165\u53e3\u3002" : "Give the highest-value markets their own URL path first.",
      chips: takeLabels(marketsMissingWebPresence.map((market) => market.name)),
    },
    {
      tone: "warn",
      title: zh ? "\u591a\u56fd\u5171\u7528\u5355\u4e00\u5e01\u79cd" : "Several countries share one fixed currency",
      value: `${sharedCurrencyMarkets.length}`,
      detail: zh ? "\u8fd9\u4f1a\u8ba9\u4ef7\u683c\u4f53\u9a8c\u6ca1\u90a3\u4e48\u672c\u5730\u5316" : "This can make pricing feel less local",
      note: zh ? "\u91cd\u70b9\u770b\u591a\u56fd\u5e02\u573a\uff0c\u4e0d\u8981\u53ea\u76ef\u5355\u56fd\u5e02\u573a\u3002" : "Prioritize the multi-country markets here.",
      chips: takeLabels(sharedCurrencyMarkets.map((market) => market.name)),
    },
    {
      tone: "warn",
      title: zh ? "\u8bed\u8a00\u8fd8\u8981\u8865\u7ec6\u8282" : "Some languages still need cleanup",
      value: `${cleanupLocales.length}`,
      detail: zh ? "\u4e0d\u662f\u4e0d\u80fd\u4e0a\u7ebf\uff0c\u800c\u662f\u8fd8\u4e0d\u591f\u5b8c\u6574" : "These are usable, but not polished yet",
      note: zh ? "\u628a\u5269\u4f59\u7f3a\u53e3\u8865\u9f50\uff0c\u987e\u5ba2\u4f53\u9a8c\u4f1a\u66f4\u7edf\u4e00\u3002" : "Closing the remaining gaps will make the store feel more complete.",
      chips: takeLabels(cleanupLocales.map((entry) => entry.locale)),
    },
    {
      tone: "warn",
      title: zh ? "\u653f\u7b56\u9875\u5165\u53e3\u8fd8\u53ef\u4ee5\u66f4\u660e\u663e" : "Policy visibility can still improve",
      value: `${audit.storefront.policyLinks.length > 0 ? 0 : 1}`,
      detail: zh ? "\u524d\u7aef\u662f\u5426\u80fd\u8f7b\u677e\u627e\u5230\u914d\u9001\u3001\u9000\u8d27\u3001\u9690\u79c1\u8bf4\u660e" : "Whether shoppers can clearly reach shipping, returns, and privacy information",
      note: audit.storefront.policyLinks.length > 0
        ? (zh ? "\u9996\u9875\u5df2\u7ecf\u80fd\u770b\u5230\u4e00\u4e9b\u653f\u7b56\u5165\u53e3\u3002" : "The homepage already exposes some policy links.")
        : (zh ? "\u5efa\u8bae\u628a\u653f\u7b56\u5165\u53e3\u56fa\u5b9a\u653e\u5728 footer\u3002" : "A fixed footer policy area would be safer."),
      chips: audit.storefront.policyLinks.length ? takeLabels(audit.storefront.policyLinks, 2) : [],
    },
  ].filter((card) => card.value !== "0" || card.chips.length);

  const marketRows = audit.markets.map((market) => ({
    name: market.name,
    countriesLabel: zh ? `${market.countryCodes.length} \u4e2a\u56fd\u5bb6` : `${market.countryCodes.length} countries`,
    languageLabel: market.allLocales.length
      ? market.allLocales.join(", ")
      : (zh ? "\u8ddf\u968f\u4e3b\u7ad9\u9ed8\u8ba4\u8bed\u8a00" : "Uses the main storefront language"),
    currencyLabel: buildMarketCurrencyLabel(market, zh),
    webLabel: buildMarketUrlLabel(market, zh),
  }));

  const storefrontReminders = zh ? [
    {
      title: "\u987a\u624b\u68c0\u67e5\u524d\u7aef\u662f\u5426\u6709\u56fd\u5bb6/\u8bed\u8a00\u5207\u6362\u5668",
      summary: "\u8fd9\u9879\u5728\u4e0d\u540c\u4e3b\u9898\u91cc\u5dee\u5f02\u5f88\u5927\uff0c\u8fd9\u4efd\u62a5\u544a\u4e0d\u518d\u7ed9\u5b83\u505a\u81ea\u52a8\u5224\u5b9a\u3002",
      nextStep: "\u6253\u5f00\u9996\u9875\u548c\u79fb\u52a8\u7aef\u83dc\u5355\uff0c\u786e\u8ba4\u987e\u5ba2\u80fd\u65b9\u4fbf\u5207\u6362\u56fd\u5bb6\u548c\u8bed\u8a00\u3002"
    },
    {
      title: "\u987a\u624b\u68c0\u67e5\u540e\u53f0\u662f\u5426\u5f00\u542f\u81ea\u52a8\u5b9a\u5411",
      summary: "\u8fd9\u9879\u5728 Shopify \u540e\u53f0\u91cc\u7ba1\u7406\uff0c\u8fd9\u4efd\u62a5\u544a\u4e0d\u518d\u628a\u5b83\u5f53\u6210\u53ef\u7cbe\u51c6\u8bfb\u53d6\u7684\u914d\u7f6e\u9879\u3002",
      nextStep: "\u5230 Online Store > Preferences \u91cc\u786e\u8ba4\u81ea\u52a8\u5b9a\u5411\u8bbe\u7f6e\u3002"
    }
  ] : [
    {
      title: "Check whether the storefront has a clear country or language switcher",
      summary: "This report no longer treats selector visibility as a reliable automated pass/fail result.",
      nextStep: "Open the storefront and confirm shoppers can switch country and language easily."
    },
    {
      title: "Check whether automatic redirection is turned on in Shopify admin",
      summary: "This report no longer treats automatic redirection as a precise machine-readable setting.",
      nextStep: "Open Online Store > Preferences and confirm the automatic redirection settings."
    }
  ];

  return {
    title: zh ? `${audit.shop.name} \u56fd\u9645\u5316\u5ba1\u8ba1\u62a5\u544a` : `Markets and language audit for ${audit.shop.name}`,
    summaryText: zh
      ? "\u8fd9\u4efd\u62a5\u544a\u91cd\u70b9\u770b\u8bed\u8a00\u3001\u5e02\u573a\u3001\u914d\u9001\u3001\u7f51\u5740\u7b56\u7565\u548c\u56fd\u9645 SEO \u57fa\u7840\u6709\u6ca1\u6709\u771f\u6b63\u843d\u5730\u3002"
      : "This report checks whether shoppers are getting the right language, the right market setup, and a clear local buying experience.",
    shop: {
      name: audit.shop.name,
      domain: audit.shop.primaryDomain?.host || audit.shop.myshopifyDomain,
    },
    generatedAtLabel: new Date(audit.generatedAt).toLocaleString(zh ? "zh-CN" : "en-US", { dateStyle: "medium", timeStyle: "short" }),
    score,
    scoreText: zh
      ? (score >= 85 ? "\u6574\u4f53\u57fa\u7840\u4e0d\u9519\uff0c\u5269\u4e0b\u4e3b\u8981\u662f\u8865\u7ec6\u8282\u3002" : score >= 65 ? "\u6574\u4f53\u53ef\u7528\uff0c\u4f46\u8fd8\u6709\u51e0\u5757\u9700\u8981\u8865\u9f50\u3002" : "\u56fd\u9645\u5316\u57fa\u7840\u8fd8\u4e0d\u591f\u7a33\uff0c\u5efa\u8bae\u5148\u8865\u6838\u5fc3\u77ed\u677f\u3002")
      : (score >= 85 ? "The setup is in good shape, with a few improvements left." : score >= 65 ? "The store is workable, but several areas need attention." : "The store needs cleanup before the international setup feels reliable."),
    languageCount: audit.localeCoverage.length,
    languageCountText: zh ? "\u672c\u6b21\u68c0\u67e5\u7684\u8bed\u8a00\u6570" : "Checked in this run",
    marketCount: audit.markets.length,
    marketCountText: zh ? "\u672c\u6b21\u68c0\u67e5\u7684\u5e02\u573a\u6570" : "Active market structures reviewed",
    fixCount: (fixPlan.actions || []).length,
    fixCountText: zh ? "\u53ef\u9884\u89c8\u7684\u5b89\u5168\u4fee\u590d" : "Safe approval-based fixes ready",
    bigPicture: zh
      ? (fixNow.length ? "\u8fd9\u5bb6\u5e97\u5df2\u7ecf\u6709\u6bd4\u8f83\u5b8c\u6574\u7684\u56fd\u9645\u5316\u5e95\u5b50\uff0c\u4f46\u73b0\u5728\u7684\u77ed\u677f\u4e3b\u8981\u96c6\u4e2d\u5728\u7ffb\u8bd1\u624e\u5b9e\u5ea6\u3001\u90e8\u5206\u5e02\u573a\u5165\u53e3\u7b56\u7565\uff0c\u4ee5\u53ca\u5c11\u6570\u5e02\u573a\u7684\u914d\u7f6e\u7ec6\u8282\u3002" : "\u8fd9\u5bb6\u5e97\u7684\u56fd\u9645\u5316\u57fa\u7840\u5df2\u7ecf\u642d\u8d77\u6765\u4e86\uff0c\u5269\u4e0b\u4e3b\u8981\u662f\u628a\u91cd\u70b9\u5e02\u573a\u505a\u5f97\u66f4\u672c\u5730\u3001\u66f4\u6e05\u695a\u3002")
      : (fixNow.length ? "The store already has the building blocks for selling internationally, but some important gaps are still blocking a clean local experience." : "The foundation is solid. Most of the work left is polish and structure cleanup."),
    goodCount: strengthCards.filter((card) => Number.parseInt(card.value, 10) !== 0 || card.chips.length).length,
    fixNowCount: priorityCards.length,
    tidyCount: improveCards.length,
    strengthCards,
    priorityCards,
    improveCards,
    manualCards: storefrontReminders,
    languagesIntro: zh ? "\u8fd9\u91cc\u4e0d\u518d\u5c55\u5f00\u51e0\u767e\u6761\u6587\u5b57\u660e\u7ec6\uff0c\u800c\u662f\u7ed9\u6bcf\u4e2a\u8bed\u8a00\u4e00\u4e2a\u6574\u4f53\u51c6\u5907\u5ea6\u3002\u8fd9\u4e2a\u503c\u57fa\u4e8e\u672c\u6b21\u7eb3\u5165\u68c0\u67e5\u7684\u8d44\u6e90\u8ba1\u7b97\u3002" : "Each language gets one readiness score based on the resources included in this audit run.",
    localeRows: audit.localeCoverage.map((entry) => ({
      locale: entry.locale,
      statusLabel: buildLocaleStatusLabel(audit.shopLocales.find((item) => item.locale === entry.locale) || { primary: false, published: false }, zh),
      readinessLabel: zh ? `${entry.readinessPct}% \u51c6\u5907\u5ea6` : `${entry.readinessPct}% readiness`,
      assignedMarketsLabel: localeAssignments.has(entry.locale)
        ? localeAssignments.get(entry.locale).join(" / ")
        : (zh ? "\u8fd8\u6ca1\u5206\u914d\u5230\u5e02\u573a" : "Not attached to a market yet"),
    })),
    marketsIntro: zh ? "\u8fd9\u91cc\u4f1a\u770b\u6bcf\u4e2a\u5e02\u573a\u7684\u56fd\u5bb6\u5206\u7ec4\u3001\u8bed\u8a00\u3001\u8d27\u5e01\u548c\u7f51\u5740\u7b56\u7565\u3002" : "Markets are reviewed for language setup, country grouping, money setup, and URL strategy.",
    marketRows,
    shippingIntro: zh ? "\u914d\u9001\u8fd9\u91cc\u4e0d\u5217\u788e\u9879\uff0c\u76f4\u63a5\u770b\u5e02\u573a\u8986\u76d6\u7ed3\u679c\u3002" : "Shipping is summarized at the market level instead of listing every small finding.",
    shippingCards: [
      {
        tone: shippingGapMarkets.length ? "bad" : "good",
        title: zh ? "\u5e02\u573a\u914d\u9001\u8986\u76d6" : "Market shipping coverage",
        value: `${coveredMarkets}/${marketsWithCountries.length || 0}`,
        detail: zh ? "\u5df2\u7ecf\u5b8c\u5168\u8986\u76d6\u914d\u9001\u7684\u5e02\u573a\u6570" : "Markets fully covered by shipping",
        note: shippingGapMarkets.length
          ? (zh ? "\u8fd9\u4e9b\u5e02\u573a\u9700\u8981\u8865\u914d\u9001\u8986\u76d6\u6216\u8c03\u6574\u56fd\u5bb6\u8303\u56f4\u3002" : "These markets need shipping coverage or a tighter country list.")
          : (zh ? "\u8fd9\u5757\u76ee\u524d\u6ca1\u6709\u660e\u663e\u95ee\u9898\u3002" : "No obvious issue here."),
        chips: shippingGapMarkets.length ? takeLabels(shippingGapMarkets.map((market) => market.name)) : [],
      },
    ],
    storefrontIntro: zh ? "\u8fd9\u91cc\u770b\u7684\u662f\u987e\u5ba2\u771f\u5b9e\u4f1a\u770b\u5230\u7684\u5165\u53e3\u548c\u641c\u7d22\u4fe1\u53f7\u3002" : "These checks focus on what buyers and search engines can actually see.",
    storefrontCards: [
      {
        tone: audit.storefront.canonical ? "good" : "bad",
        title: zh ? "Canonical" : "Canonical",
        value: audit.storefront.canonical ? (zh ? "\u5df2\u53d1\u73b0" : "Found") : (zh ? "\u7f3a\u5931" : "Missing"),
        detail: zh ? "\u4e3b\u9875\u4e3b\u94fe\u63a5\u4fe1\u53f7" : "Homepage canonical signal",
        note: audit.storefront.canonical
          ? (zh ? "\u4e3b\u9875\u6709 canonical\u3002" : "The homepage exposes a canonical link.")
          : (zh ? "\u4e3b\u9875\u6ca1\u770b\u5230 canonical\u3002" : "No canonical link was found on the homepage."),
        chips: audit.storefront.canonical ? [audit.storefront.canonical] : [],
      },
      {
        tone: audit.storefront.hreflangCount > 0 || audit.localeCoverage.length === 0 ? "good" : "bad",
        title: zh ? "Hreflang" : "Hreflang",
        value: `${audit.storefront.hreflangCount || 0}`,
        detail: zh ? "\u4e3b\u9875\u66b4\u9732\u7ed9\u641c\u7d22\u5f15\u64ce\u7684\u8bed\u8a00\u94fe\u63a5\u6570" : "Homepage alternate-language links",
        note: audit.storefront.hreflangCount > 0
          ? (zh ? "\u641c\u7d22\u5f15\u64ce\u5df2\u7ecf\u80fd\u770b\u5230\u591a\u8bed\u8a00\u5165\u53e3\u3002" : "Search engines can already see alternate language paths.")
          : (zh ? "\u4e3b\u9875\u8fd8\u6ca1\u66b4\u9732\u591a\u8bed\u8a00\u5165\u53e3\u3002" : "The homepage is not exposing alternate language paths yet."),
        chips: takeLabels((audit.storefront.hreflangs || []).map((entry) => entry.locale), 8),
      },
      {
        tone: audit.storefront.policyLinks.length > 0 ? "good" : "warn",
        title: zh ? "\u653f\u7b56\u9875\u5165\u53e3" : "Policy links",
        value: `${audit.storefront.policyLinks.length || 0}`,
        detail: zh ? "\u9996\u9875\u53ef\u89c1\u7684\u653f\u7b56\u76f8\u5173\u5165\u53e3" : "Policy-related homepage links",
        note: audit.storefront.policyLinks.length > 0
          ? (zh ? "\u987e\u5ba2\u5df2\u7ecf\u80fd\u4ece\u524d\u7aef\u627e\u5230\u4e00\u4e9b\u653f\u7b56\u5185\u5bb9\u3002" : "Some policy links are visible on the storefront.")
          : (zh ? "\u5efa\u8bae\u8ba9\u914d\u9001\u3001\u9000\u8d27\u3001\u9690\u79c1\u5165\u53e3\u66f4\u660e\u663e\u3002" : "Policy visibility could be clearer from the storefront."),
        chips: takeLabels(audit.storefront.policyLinks, 3),
      },
    ],
    expansionIntro: zh ? "\u8fd9\u90e8\u5206\u4e0d\u662f\u5728\u627e bug\uff0c\u800c\u662f\u5728\u7ed9\u56fd\u9645\u5316\u7ecf\u8425\u4e0a\u7684\u5efa\u8bae\u3002" : "These are growth ideas, not bugs.",
    expansionIdeas: buildExpansionIdeasFromAudit(audit, zh),
    actionsIntro: zh ? "\u8fd9\u4e9b\u52a8\u4f5c\u53ef\u4ee5\u5148\u9884\u89c8\uff0c\u786e\u8ba4\u540e\u518d\u6267\u884c\u3002" : "These fixes are safe to preview now and execute later only after approval.",
    actions: fixPlan.actions || [],
    actionsEmptyNote: fixPlan.explanation || "",
    storefrontReminders,
    footerNote: zh
      ? "\u524d\u7aef\u5207\u6362\u5668\u548c\u81ea\u52a8\u5b9a\u5411\u8fd9\u4e24\u9879\u4e0d\u518d\u8ba1\u5206\uff0c\u53ea\u4f5c\u4e3a\u4f4e\u4f18\u5148\u7ea7\u4eba\u5de5\u63d0\u9192\u3002"
      : "Selector visibility and automatic redirection are no longer scored and are shown only as low-priority manual reminders.",
    lang,
  };
}

function checkpointPathFor(outputPath) {
  return `${path.resolve(outputPath)}.checkpoint.json`;
}

async function loadAuditCheckpoint(checkpointPath, locales, transport) {
  const checkpoint = JSON.parse(await readFile(checkpointPath, "utf8"));
  const expectedLocales = [...locales].sort().join(",");
  if (checkpoint.version !== 1 || checkpoint.transport !== transport || [...(checkpoint.locales || [])].sort().join(",") !== expectedLocales) {
    throw new Error("The audit checkpoint does not match the current locales or transport. Start a new audit without --resume.");
  }
  return checkpoint;
}

async function writeAuditCheckpoint(checkpointPath, checkpoint) {
  await mkdir(path.dirname(checkpointPath), { recursive: true });
  await writeFile(checkpointPath, JSON.stringify(checkpoint, null, 2), "utf8");
}

async function cmdAudit(args) {
  const env = await loadEnv(args.env);
  const raw = await graphql(env, OVERVIEW_QUERY);
  const overview = normalizeOverview(raw);
  const requestedLocales = parseLocales(args.locales);
  const localesToAudit = chooseLocalesForAudit(overview, requestedLocales);
  const requestedBulk = args.transport === "bulk" || (args.transport === "auto" && localesToAudit.length >= BULK_LOCALE_THRESHOLD);
  const useBulk = requestedBulk && !isDevDashboardMode(env);
  if (requestedBulk && isDevDashboardMode(env)) console.error("[audit] Long-running connection uses standard queries because Shopify CLI bulk mode needs browser authorization.");
  const transport = useBulk ? "bulk" : "standard";
  if (args.resume && !args.output) throw new Error("--resume requires --output so the audit checkpoint can be found.");
  const checkpointPath = args.output ? checkpointPathFor(args.output) : null;
  const checkpoint = args.resume ? await loadAuditCheckpoint(checkpointPath, localesToAudit, transport) : null;
  const completedResourceTypes = new Set(checkpoint?.completedResourceTypes || []);
  const pendingResourceTypes = DEFAULT_RESOURCE_TYPES.filter((item) => !completedResourceTypes.has(item));
  let checkpointWrite = Promise.resolve();
  const onResourceComplete = async (resourceType, localeCoverage) => {
    if (!checkpointPath) return;
    completedResourceTypes.add(resourceType);
    const snapshot = {
      version: 1,
      transport,
      locales: localesToAudit,
      completedResourceTypes: DEFAULT_RESOURCE_TYPES.filter((item) => completedResourceTypes.has(item)),
      localeCoverage,
      updatedAt: new Date().toISOString(),
    };
    checkpointWrite = checkpointWrite.then(() => writeAuditCheckpoint(checkpointPath, snapshot));
    await checkpointWrite;
  };
  console.error(`[audit] Locale coverage transport: ${transport}; ${pendingResourceTypes.length}/${DEFAULT_RESOURCE_TYPES.length} resource types remaining.`);
  const localeCoverage = useBulk
    ? await computeLocaleCoverageBulk(env, localesToAudit, pendingResourceTypes, { initialCoverage: checkpoint?.localeCoverage, allResourceTypes: DEFAULT_RESOURCE_TYPES, onResourceComplete })
    : await computeLocaleCoverageStandard(env, localesToAudit, pendingResourceTypes, { initialCoverage: checkpoint?.localeCoverage, allResourceTypes: DEFAULT_RESOURCE_TYPES, onResourceComplete });

  const storefront = await crawlStorefront(overview.shop);
  const { findings, workingWell } = buildFindings(overview, localeCoverage, storefront);
  overview.reportLang = resolveReportLang(args.lang);
  const fixPlan = buildFixPlan(overview, localeCoverage);

  const audit = {
    generatedAt: new Date().toISOString(),
    shop: overview.shop,
    shopLocales: overview.shopLocales,
    markets: overview.markets,
    deliveryProfiles: overview.deliveryProfiles,
    shippingCountryCodes: overview.shippingCountryCodes,
    storefront,
    storeProfile: overview.storeProfile,
    localeCoverage,
    findings,
    workingWell,
    fixPlan,
    localeCoverageTransport: transport,
    reportLang: resolveReportLang(args.lang),
  };

  await writeJsonOutput(audit, args.output);
  if (checkpointPath) await rm(checkpointPath, { force: true }).catch(() => {});
}

async function renderReport(audit, outputPath) {
  const templatePath = path.join(__dirname, "..", "templates", "report-template.html");
  const template = await readFile(templatePath, "utf8");
  const reportModel = buildReportModel(audit);
  const safeReportJson = JSON.stringify(reportModel)
    .replaceAll("<", "\\u003c")
    .replaceAll(">", "\\u003e")
    .replaceAll("&", "\\u0026");
  const html = template
    .replaceAll("{{REPORT_TITLE}}", reportModel.title.replace(/</g, "&lt;"))
    .replace("{{REPORT_DATA_JSON}}", safeReportJson);
  await mkdir(path.dirname(path.resolve(outputPath)), { recursive: true });
  await writeFile(outputPath, html, "utf8");
}

async function cmdReport(args) {
  if (!args.input) throw new Error("--input is required");
  if (!args.output) throw new Error("--output is required");
  const audit = JSON.parse(await readFile(args.input, "utf8"));
  audit.reportLang = resolveReportLang(args.lang, audit.reportLang);
  await renderReport(audit, args.output);
  console.log(JSON.stringify({ ok: true, output: args.output }, null, 2));
}

async function cmdFixPlan(args) {
  if (!args.input) throw new Error("--input is required");
  const audit = JSON.parse(await readFile(args.input, "utf8"));
  await writeJsonOutput(audit.fixPlan?.actions || [], args.output);
}

function assertNoUserErrors(payload, operation) {
  const userErrors = payload?.userErrors || [];
  if (userErrors.length) throw new Error(`${operation} failed: ${userErrors.map((item) => item.message).join("; ")}`);
}

async function applyAction(env, action) {
  if (action.type === "enable_publish_locale") {
    const enabled = await graphql(env, `#graphql
      mutation EnableLocale($locale: String!) {
        shopLocaleEnable(locale: $locale) {
          userErrors { field message }
        }
      }`, { locale: action.variables.locale });
    assertNoUserErrors(enabled.shopLocaleEnable, "Enable locale");

    const published = await graphql(env, `#graphql
      mutation PublishLocale($locale: String!) {
        shopLocaleUpdate(locale: $locale, shopLocale: { published: true }) {
          userErrors { field message }
        }
      }`, { locale: action.variables.locale });
    assertNoUserErrors(published.shopLocaleUpdate, "Publish locale");
    return published;
  }

  if (action.type === "publish_locale") {
    const published = await graphql(env, `#graphql
      mutation PublishLocale($locale: String!) {
        shopLocaleUpdate(locale: $locale, shopLocale: { published: true }) {
          userErrors { field message }
        }
      }`, { locale: action.variables.locale });
    assertNoUserErrors(published.shopLocaleUpdate, "Publish locale");
    return published;
  }

  if (action.type === "enable_local_currencies") {
    const updated = await graphql(env, `#graphql
      mutation EnableLocalCurrencies($id: ID!) {
        marketUpdate(id: $id, input: { currencySettings: { localCurrencies: true } }) {
          userErrors { field message }
        }
      }`, { id: action.variables.marketId });
    assertNoUserErrors(updated.marketUpdate, "Enable local currencies");
    return updated;
  }

  throw new Error(`Unsupported action type: ${action.type}`);
}

function verifyAppliedAction(action, data) {
  if (action.type === "enable_publish_locale" || action.type === "publish_locale") {
    const locale = (data.shopLocales || []).find((item) => item.locale === action.variables.locale);
    return locale?.published
      ? { title: action.title, status: "VERIFIED" }
      : { title: action.title, status: "NOT_VERIFIED", reason: "The locale is not published after the mutation." };
  }
  if (action.type === "enable_local_currencies") {
    const market = (data.markets?.nodes || []).find((item) => item.id === action.variables.marketId);
    return market?.currencySettings?.localCurrencies
      ? { title: action.title, status: "VERIFIED" }
      : { title: action.title, status: "NOT_VERIFIED", reason: "Local currencies are not enabled after the mutation." };
  }
  return { title: action.title, status: "SKIPPED", reason: "No verification rule for this action." };
}

async function cmdApply(args) {
  if (!args.input) throw new Error("--input is required");
  const parsed = JSON.parse(await readFile(args.input, "utf8"));
  const actions = Array.isArray(parsed) ? parsed : (parsed.actions || []);
  if (!args.execute) {
    console.log(JSON.stringify({
      mode: "preview",
      count: actions.length,
      actions: actions.map((action) => ({
        title: action.title,
        summary: action.summary,
        executeSupported: action.executeSupported,
      })),
    }, null, 2));
    return;
  }

  const env = await loadEnv(args.env);
  const results = [];
  for (const action of actions) {
    if (!action.executeSupported) {
      results.push({ title: action.title, status: "SKIPPED", reason: "execute not supported" });
      continue;
    }
    try {
      await applyAction(env, action);
      results.push({ title: action.title, status: "OK" });
    } catch (error) {
      results.push({ title: action.title, status: "ERROR", error: error.message });
    }
  }
  const successfulActions = actions.filter((action) => results.some((result) => result.title === action.title && result.status === "OK"));
  let verification = { status: "SKIPPED", results: [] };
  if (successfulActions.length) {
    try {
      const data = await graphql(env, APPLY_VERIFICATION_QUERY);
      verification = {
        status: "COMPLETE",
        results: successfulActions.map((action) => verifyAppliedAction(action, data)),
      };
    } catch (error) {
      verification = { status: "ERROR", error: error.message, results: [] };
    }
  }
  console.log(JSON.stringify({ mode: "execute", results, verification }, null, 2));
}

let args;
try {
  args = parseArgs(process.argv.slice(2));
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
const commands = {
  "help": () => printHelp(),
  "init-env": () => initEnv(args),
  "connection-check": () => cmdConnectionCheck(args),
  "audit": () => cmdAudit(args),
  "report": () => cmdReport(args),
  "fix-plan": () => cmdFixPlan(args),
  "apply": () => cmdApply(args),
};

Promise.resolve(commands[args.command]()).catch((error) => {
  console.error(error.message);
  process.exit(1);
});
