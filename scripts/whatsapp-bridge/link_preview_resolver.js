import { lookup } from 'dns/promises';

/**
 * Builds the `resolveDNSHost` callback link-preview-js expects: the bridge resolves the URL's
 * hostname to an address itself, and the library then rejects loopback/private results and pins the
 * connection to the validated address, so a later DNS answer cannot move the request elsewhere.
 *
 * Baileys forwards the `options` given to `makeWASocket` into the link-preview fetch options, so
 * supplying this is what turns that protection on. Without it the library logs "not resolving DNS
 * addresses … can cause loopback attacks" on every URL and lets the real request re-resolve DNS.
 *
 * `lookupHostname` is injectable so the behaviour is testable without touching the network.
 */
export function createResolveDNSHost(lookupHostname = lookup) {
  return async (url) => {
    // WHATWG keeps IPv6 hostnames bracketed; `dns.lookup` wants the bare address.
    const { hostname } = new URL(url);
    return (await lookupHostname(hostname.replace(/^\[|\]$/g, ''))).address;
  };
}

export const resolveDNSHost = createResolveDNSHost();
