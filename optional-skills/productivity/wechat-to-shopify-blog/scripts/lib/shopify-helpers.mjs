import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import os from "node:os";
import path from "node:path";
import { assertRequiredScopes, devDashboardGraphql, isDevDashboardMode, mergeRuntimeEnv } from "./shopify-dev-dashboard-auth.mjs";

const execFileAsync = promisify(execFile);

export function parseEnv(text) {
  const env = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const separator = line.indexOf("=");
    if (separator >= 0) env[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }
  return env;
}

export async function loadShopifyEnv(envPath) {
  const text = await readFile(envPath, "utf8").catch(() => null);
  const env = mergeRuntimeEnv(text ? parseEnv(text) : {});
  if (!text && !env.SKILL_HUB_SHOPIFY_STORE_DOMAIN) throw new Error(`Missing env file: ${envPath}. Run from the folder containing skill-hub.env, pass --env <path-to-skill-hub.env>, or provide private runtime environment variables.`);
  if (!["shopify_cli_oauth", "dev_dashboard_client_credentials"].includes(env.SKILL_HUB_SHOPIFY_ACCESS_METHOD || "shopify_cli_oauth")) throw new Error("Unsupported SKILL_HUB_SHOPIFY_ACCESS_METHOD. Use shopify_cli_oauth or dev_dashboard_client_credentials.");
  const rawDomain = env.SKILL_HUB_SHOPIFY_STORE_DOMAIN;
  if (!rawDomain) throw new Error(`Missing SHOPIFY_STORE_DOMAIN in ${envPath}.`);
  const adminMatch = rawDomain.match(/admin\.shopify\.com\/store\/([^/\s?&]+)/i);
  const host = adminMatch ? `${adminMatch[1].toLowerCase()}.myshopify.com` : normalizeShopDomain(rawDomain);
  if (!/^[a-z0-9][-a-z0-9]*\.myshopify\.com$/i.test(host)) {
    throw new Error("Invalid store address. Provide a Shopify admin URL or .myshopify.com domain.");
  }
  return { ...env, SHOPIFY_STORE_DOMAIN: host, SHOPIFY_API_DOMAIN: host, SHOPIFY_TRANSPORT: isDevDashboardMode(env) ? "dev_dashboard_client_credentials" : "shopify_cli" };
}

function normalizeShopDomain(value) {
  const raw = value.trim();
  const url = raw.includes("://") ? new URL(raw) : new URL(`https://${raw}`);
  return url.host.toLowerCase();
}

async function pathExists(filePath) {
  return access(filePath).then(() => true).catch(() => false);
}

async function resolveCliInvocation(env = {}) {
  const candidates = [env.SKILL_HUB_SHOPIFY_CLI_JS, process.env.SKILL_HUB_SHOPIFY_CLI_JS];
  const npmRoot = await execFileAsync("npm", ["root", "-g"], { windowsHide: true }).then(({ stdout }) => stdout.trim()).catch(() => "");
  if (npmRoot) candidates.push(path.join(npmRoot, "@shopify", "cli", "bin", "run.js"));
  if (process.env.APPDATA) candidates.push(path.join(process.env.APPDATA, "npm", "node_modules", "@shopify", "cli", "bin", "run.js"));
  for (const candidate of candidates) if (candidate && await pathExists(candidate)) return { command: process.execPath, prefix: [candidate] };
  try {
    await execFileAsync("shopify", ["--version"], { windowsHide: true });
    return { command: "shopify", prefix: [] };
  } catch {
    throw new Error("CLI_NOT_FOUND: Install Shopify CLI 3.93.0+ or put `shopify` on PATH.");
  }
}

function classifyCliError(error, detail = "") {
  const text = `${detail}\n${error?.message || ""}`;
  if (/access denied|denied access/i.test(text)) return "CLI_ACCESS_DENIED";
  if (/not authenticated|authentication required|run shopify store auth|authorization required/i.test(text)) return "CLI_AUTH_REQUIRED";
  return "CLI_REQUEST_FAILED";
}

export async function shopifyCliGraphql(env, query, variables = {}) {
  if (isDevDashboardMode(env)) {
    const result = await devDashboardGraphql(env, env.SHOPIFY_API_DOMAIN, query, variables);
    assertRequiredScopes(result.scopes, "read_products,read_content,write_content,read_files,write_files");
    return result.data;
  }
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "skill-hub-shopify-cli-"));
  const queryFile = path.join(tempDir, "query.graphql");
  const variableFile = path.join(tempDir, "variables.json");
  const outputFile = path.join(tempDir, "output.json");
  try {
    const cli = await resolveCliInvocation(env);
    await writeFile(queryFile, query, "utf8");
    await writeFile(variableFile, JSON.stringify(variables), "utf8");
    const args = [...cli.prefix, "store", "execute", "--store", env.SHOPIFY_API_DOMAIN, "--query-file", queryFile, "--variable-file", variableFile, "--output-file", outputFile, "--json"];
    if (/(^|\n)\s*mutation\b/i.test(query)) args.push("--allow-mutations");
    const childEnv = { ...process.env };
    delete childEnv.NODE_TLS_REJECT_UNAUTHORIZED;
    try {
      await execFileAsync(cli.command, args, {
        timeout: 180000,
        maxBuffer: 1024 * 1024 * 20,
        windowsHide: true,
        env: childEnv,
      });
    } catch (error) {
      const detail = [error.stderr, error.stdout].filter(Boolean).map(String).join("\n");
      throw new Error(`${classifyCliError(error, detail)}: ${detail || error.message}`);
    }
    const raw = JSON.parse(await readFile(outputFile, "utf8"));
    const json = raw?.data ? raw : { data: raw, errors: raw?.errors };
    if (json.errors) throw new Error(`CLI_GRAPHQL_ERRORS: ${JSON.stringify(json.errors)}`);
    return json.data;
  } catch (error) {
    throw new Error(`${error.message}\nIf this is CLI_AUTH_REQUIRED, run: shopify store auth --store ${env.SHOPIFY_API_DOMAIN} --scopes read_products,read_content,write_content,read_files,write_files --json`);
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}
