import test from 'node:test';
import assert from 'node:assert/strict';

import { createResolveDNSHost } from './link_preview_resolver.js';

const stubLookup = (address, seen = []) =>
  Object.assign(async (hostname) => {
    seen.push(hostname);
    return { address, family: address.includes(':') ? 6 : 4 };
  }, { seen });

test('resolves the URL hostname, not its path or query', async () => {
  const seen = [];
  const resolve = createResolveDNSHost(stubLookup('93.184.216.34', seen));
  assert.equal(await resolve('https://example.com/a/b?c=d#e'), '93.184.216.34');
  assert.deepEqual(seen, ['example.com']);
});

test('drops the port before resolving', async () => {
  const seen = [];
  const resolve = createResolveDNSHost(stubLookup('93.184.216.34', seen));
  await resolve('https://example.com:8443/hook');
  assert.deepEqual(seen, ['example.com']);
});

test('unbrackets IPv6 hostnames, which dns.lookup cannot take bracketed', async () => {
  const seen = [];
  const address = '2606:2800:220:1:248:1893:25c8:1946';
  const resolve = createResolveDNSHost(stubLookup(address, seen));
  assert.equal(await resolve(`https://[${address}]/x`), address);
  assert.deepEqual(seen, [address]);
});

test('resolves an IP literal through the real resolver without a network round trip', async () => {
  assert.equal(await createResolveDNSHost()('http://127.0.0.1:3000/hook'), '127.0.0.1');
});

test('a lookup failure propagates so the caller drops the preview instead of fetching', async () => {
  const resolve = createResolveDNSHost(async () => {
    throw new Error('ENOTFOUND');
  });
  await assert.rejects(() => resolve('https://nope.invalid/x'), /ENOTFOUND/);
});

test('resolves a public hostname through the real resolver', async (t) => {
  try {
    assert.match(await createResolveDNSHost()('https://example.com'), /^[0-9a-fA-F:.]+$/);
  } catch (error) {
    t.skip(`no DNS in this environment: ${error.code ?? error.message}`);
  }
});
