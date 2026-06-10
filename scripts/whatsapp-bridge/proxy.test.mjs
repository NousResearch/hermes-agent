import test from 'node:test';
import assert from 'node:assert/strict';

import { buildSocketProxyOptions, createProxyAgent, resolveProxyUrl } from './proxy.js';

class FakeProxyAgent {
  constructor(proxyUrl) {
    this.proxyUrl = proxyUrl;
  }
}

test('resolveProxyUrl prefers HTTPS proxy variables first', () => {
  const env = {
    HTTP_PROXY: 'http://http-proxy.example:8080',
    HTTPS_PROXY: 'http://https-proxy.example:8443',
  };

  assert.equal(resolveProxyUrl(env), 'http://https-proxy.example:8443');
});

test('resolveProxyUrl falls back across lowercase and HTTP variants', () => {
  assert.equal(
    resolveProxyUrl({ http_proxy: 'http://lower-http.example:8080' }),
    'http://lower-http.example:8080',
  );
  assert.equal(
    resolveProxyUrl({ HTTP_PROXY: 'http://upper-http.example:8080' }),
    'http://upper-http.example:8080',
  );
  assert.equal(resolveProxyUrl({ HTTPS_PROXY: '   ' }), '');
});

test('createProxyAgent returns undefined when no proxy is configured', () => {
  assert.equal(createProxyAgent({}), undefined);
});

test('buildSocketProxyOptions wires an HttpsProxyAgent for Baileys', () => {
  const proxyUrl = 'http://proxy.example:3128';
  const options = buildSocketProxyOptions({ HTTPS_PROXY: proxyUrl }, FakeProxyAgent);

  assert.ok(options.agent instanceof FakeProxyAgent);
  assert.equal(options.agent.proxyUrl, proxyUrl);
});
