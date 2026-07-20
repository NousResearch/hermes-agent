import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { assertRequiredScopes, devDashboardGraphql, getDevDashboardAccess, isDevDashboardMode, mergeRuntimeEnv } from './shopify-dev-dashboard-auth.mjs';

const REQUIRED_SCOPES = 'read_locales,write_locales,read_markets,write_markets,read_translations,write_translations';
const READ_SCOPES = 'read_locales,read_markets,read_translations';
let currentEnv = {};

export function loadSkillHubEnv(envPath) {
  const candidates = envPath ? [resolve(envPath)] : [resolve('skill-hub.env')];
  const env = {};
  for (const file of candidates) {
    if (!existsSync(file)) continue;
    for (const raw of readFileSync(file, 'utf8').split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      const separator = line.indexOf('=');
      if (separator >= 0) env[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
    }
    break;
  }
  const merged = mergeRuntimeEnv(env);
  if (!Object.keys(env).length && !merged.SKILL_HUB_SHOPIFY_STORE_DOMAIN) throw new Error('Missing skill-hub.env and Shopify runtime environment variables.');
  if (!['shopify_cli_oauth', 'dev_dashboard_client_credentials'].includes(merged.SKILL_HUB_SHOPIFY_ACCESS_METHOD || 'shopify_cli_oauth')) throw new Error('Unsupported SKILL_HUB_SHOPIFY_ACCESS_METHOD. Use shopify_cli_oauth or dev_dashboard_client_credentials.');
  currentEnv = merged;
  return merged;
}

export function resolveAdminHost(domain) {
  if (!domain) throw new Error('SKILL_HUB_SHOPIFY_STORE_DOMAIN is not set');
  const raw = String(domain).trim();
  const adminMatch = raw.match(/admin\.shopify\.com\/store\/([^/\s?#]+)/i);
  const candidate = (adminMatch ? `${adminMatch[1]}.myshopify.com` : raw)
    .replace(/^https?:\/\//i, '').replace(/\/.*$/, '').toLowerCase();
  const host = candidate.endsWith('.myshopify.com') ? candidate : `${candidate}.myshopify.com`;
  if (!/^[a-zA-Z0-9][-a-zA-Z0-9]*\.myshopify\.com$/.test(host)) {
    throw new Error('Invalid Shopify store domain. Provide a Shopify admin URL or .myshopify.com domain.');
  }
  return host;
}

function resolveCliJs() {
  const candidates = [
    process.env.SKILL_HUB_SHOPIFY_CLI_JS,
    process.env.APPDATA && join(process.env.APPDATA, 'npm', 'node_modules', '@shopify', 'cli', 'bin', 'run.js'),
  ].filter(Boolean);
  try {
    const globalRoot = execFileSync('npm', ['root', '-g'], { encoding: 'utf8', windowsHide: true }).trim();
    candidates.push(join(globalRoot, '@shopify', 'cli', 'bin', 'run.js'));
  } catch {}
  return candidates.find((file) => existsSync(file)) || null;
}

function runShopifyCli(args, options) {
  const cliJs = resolveCliJs();
  if (cliJs) return execFileSync(process.execPath, [cliJs, ...args], options);
  try {
    return execFileSync('shopify', args, options);
  } catch (error) {
    if (error.code === 'ENOENT') throw new Error('CLI_NOT_FOUND: Install Shopify CLI 3.93.0+ and ensure `shopify` is on PATH.');
    throw error;
  }
}

export async function shopifyGraphql(host, query, variables = {}) {
  if (isDevDashboardMode(currentEnv)) {
    const access = await getDevDashboardAccess(currentEnv, host);
    const requiredScopes = /(^|\n)\s*mutation\b/i.test(query) ? REQUIRED_SCOPES : READ_SCOPES;
    assertRequiredScopes(access.scopes, requiredScopes);
    const result = await devDashboardGraphql(currentEnv, host, query, variables);
    return result.data;
  }
  const directory = mkdtempSync(join(tmpdir(), 'skill-hub-shopify-cli-'));
  const queryFile = join(directory, 'query.graphql');
  const variableFile = join(directory, 'variables.json');
  const outputFile = join(directory, 'output.json');
  try {
    writeFileSync(queryFile, query, 'utf8');
    writeFileSync(variableFile, JSON.stringify(variables), 'utf8');
    const args = ['store', 'execute', '--store', host, '--query-file', queryFile, '--variable-file', variableFile, '--output-file', outputFile, '--json'];
    if (/(^|\n)\s*mutation\b/i.test(query)) args.push('--allow-mutations');
    runShopifyCli(args, { encoding: 'utf8', windowsHide: true, timeout: 180000 });
    const raw = JSON.parse(readFileSync(outputFile, 'utf8'));
    const json = raw.data ? raw : { data: raw };
    if (json.errors) throw new Error(JSON.stringify(json.errors));
    return json.data;
  } catch (error) {
    throw new Error(`CLI_GRAPHQL_FAILED: ${error.message}`);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}
