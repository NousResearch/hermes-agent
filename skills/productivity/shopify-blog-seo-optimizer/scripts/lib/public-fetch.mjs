import dns from "node:dns/promises";
import net from "node:net";

const MAX_REDIRECTS = 5;

function ipv4ToNumber(value) {
  const parts = value.split(".").map(Number);
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return null;
  return (((parts[0] * 256 + parts[1]) * 256 + parts[2]) * 256 + parts[3]) >>> 0;
}

function inIpv4Range(value, start, end) {
  const number = ipv4ToNumber(value);
  return number !== null && number >= start && number <= end;
}

function isBlockedIp(value) {
  const ip = String(value || "").replace(/^\[|\]$/g, "").toLowerCase().split("%")[0];
  if (net.isIPv4(ip)) {
    return [
      [0, 0x00ffffff],
      [0x0a000000, 0x0affffff],
      [0x64400000, 0x647fffff],
      [0x7f000000, 0x7fffffff],
      [0xa9fe0000, 0xa9feffff],
      [0xac100000, 0xac1fffff],
      [0xc0a80000, 0xc0a8ffff],
      [0xc6120000, 0xc613ffff],
      [0xe0000000, 0xffffffff],
    ].some(([start, end]) => inIpv4Range(ip, start, end));
  }
  if (!net.isIPv6(ip)) return false;
  if (ip === "::1" || ip === "::") return true;
  if (ip.startsWith("fc") || ip.startsWith("fd") || ip.startsWith("fe8") || ip.startsWith("fe9") || ip.startsWith("fea") || ip.startsWith("feb")) return true;
  if (ip.startsWith("::ffff:")) return isBlockedIp(ip.slice(7));
  return false;
}

function hostMatches(host, pattern) {
  const normalized = String(pattern).toLowerCase().replace(/^\./, "");
  return host === normalized || host.endsWith(`.${normalized}`);
}

export function validatePublicUrl(raw, { allowedHosts = [] } = {}) {
  const parsed = new URL(String(raw));
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error(`Only HTTP and HTTPS URLs are allowed: ${raw}`);
  if (parsed.username || parsed.password) throw new Error(`Credential-bearing URLs are blocked: ${raw}`);
  const hostname = parsed.hostname.toLowerCase();
  if (!hostname || isBlockedIp(hostname)) throw new Error(`Private, local, or invalid destination is blocked: ${hostname}`);
  if (allowedHosts.length && !allowedHosts.some((pattern) => hostMatches(hostname, pattern))) {
    throw new Error(`Destination is outside the allowed host policy: ${hostname}`);
  }
  return parsed;
}

export async function assertPublicDestination(raw, options = {}) {
  const parsed = validatePublicUrl(raw, options);
  if (net.isIP(parsed.hostname)) return parsed;
  const addresses = await dns.lookup(parsed.hostname, { all: true, verbatim: true });
  if (!addresses.length || addresses.some(({ address }) => isBlockedIp(address))) {
    throw new Error(`DNS resolved to a private, local, or invalid destination: ${parsed.hostname}`);
  }
  return parsed;
}

export async function fetchPublic(raw, init = {}, options = {}) {
  let current = String(raw);
  const headers = { ...(init.headers || {}) };
  for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects += 1) {
    await assertPublicDestination(current, options);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 15000);
    let response;
    try {
      response = await fetch(current, { ...init, headers, redirect: "manual", signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }
    if (response.status < 300 || response.status >= 400) return response;
    const location = response.headers.get("location");
    if (!location) throw new Error(`Redirect response is missing Location: ${current}`);
    if (redirects === MAX_REDIRECTS) throw new Error(`Too many redirects while fetching ${raw}`);
    current = new URL(location, current).href;
  }
  throw new Error(`Unable to fetch ${raw}`);
}

export function isBlockedAddress(value) {
  return isBlockedIp(value);
}
