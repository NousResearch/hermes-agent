import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const PROXY_ENV_KEYS = [
  'HTTPS_PROXY',
  'https_proxy',
  'HTTP_PROXY',
  'http_proxy',
];

export function resolveProxyUrl(env = process.env) {
  for (const key of PROXY_ENV_KEYS) {
    const value = env?.[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return '';
}

function loadHttpsProxyAgent() {
  return require('https-proxy-agent').HttpsProxyAgent;
}

export function createProxyAgent(env = process.env, ProxyAgentCtor = undefined) {
  const proxyUrl = resolveProxyUrl(env);
  if (!proxyUrl) {
    return undefined;
  }
  const AgentCtor = ProxyAgentCtor || loadHttpsProxyAgent();
  return new AgentCtor(proxyUrl);
}

export function buildSocketProxyOptions(env = process.env, ProxyAgentCtor = undefined) {
  const agent = createProxyAgent(env, ProxyAgentCtor);
  return agent ? { agent } : {};
}
