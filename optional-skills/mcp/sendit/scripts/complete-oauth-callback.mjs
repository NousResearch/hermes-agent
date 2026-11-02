#!/usr/bin/env node

const sensitiveKeys = new Set(['code', 'state']);
const allowedHosts = new Set(['127.0.0.1', 'localhost']);

function redactCallbackUrl(value) {
  try {
    const parsed = new URL(value);
    for (const key of sensitiveKeys) {
      if (parsed.searchParams.has(key)) {
        parsed.searchParams.set(key, 'REDACTED');
      }
    }
    return parsed.toString();
  } catch {
    return value
      .replace(/([?&](?:code|state)=)[^"'&<\s]+/g, '$1REDACTED')
      .replace(/((?:code|state)=)[^"'&<\s]+/g, '$1REDACTED');
  }
}

function parseAndValidateCallback(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error('Invalid URL.');
  }

  if (parsed.protocol !== 'http:' || !allowedHosts.has(parsed.hostname)) {
    throw new Error('Refusing to replay callback: URL must be http://127.0.0.1 or http://localhost.');
  }

  if (parsed.pathname !== '/callback') {
    throw new Error('Refusing to replay callback: path must be /callback.');
  }

  const port = Number(parsed.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('Refusing to replay callback: URL must include a valid port.');
  }

  if (!parsed.searchParams.has('code') && !parsed.searchParams.has('error')) {
    throw new Error('Refusing to replay callback: URL must include code or error.');
  }

  return parsed;
}

function assertSelfTest(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function runSelfTest() {
  const valid = parseAndValidateCallback(
    'http://127.0.0.1:43879/callback?code=secret-code&state=secret-state',
  );
  assertSelfTest(valid.port === '43879', 'Expected valid localhost callback to parse.');

  const localhost = parseAndValidateCallback('http://localhost:1/callback?error=access_denied');
  assertSelfTest(localhost.hostname === 'localhost', 'Expected localhost error callback to parse.');

  for (const badUrl of [
    'https://127.0.0.1:43879/callback?code=x',
    'http://example.com:43879/callback?code=x',
    'http://127.0.0.1:43879/not-callback?code=x',
    'http://127.0.0.1/callback?code=x',
    'not a url',
  ]) {
    let rejected = false;
    try {
      parseAndValidateCallback(badUrl);
    } catch {
      rejected = true;
    }
    assertSelfTest(rejected, `Expected rejection for ${badUrl}`);
  }

  const redacted = redactCallbackUrl(
    'http://127.0.0.1:43879/callback?code=secret-code&state=secret-state&scope=ok',
  );
  assertSelfTest(!redacted.includes('secret-code'), 'Expected code redaction.');
  assertSelfTest(!redacted.includes('secret-state'), 'Expected state redaction.');
  assertSelfTest(redacted.includes('scope=ok'), 'Expected non-sensitive query values to remain.');

  console.log('complete-oauth-callback self-test passed.');
}

if (process.argv.includes('--self-test')) {
  runSelfTest();
  process.exit(0);
}

let rawUrl = process.argv.slice(2).join(' ').trim();

if (!rawUrl && !process.stdin.isTTY) {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  rawUrl = Buffer.concat(chunks).toString('utf8').trim();
}

if (!rawUrl) {
  console.error('Usage: complete-oauth-callback.mjs <localhost-callback-url>');
  process.exit(2);
}

let parsed;
try {
  parsed = parseAndValidateCallback(rawUrl);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}

const replayUrl = new URL(parsed.toString());
replayUrl.hostname = '127.0.0.1';

const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 10000);

try {
  const response = await fetch(replayUrl, { signal: controller.signal });
  const text = await response.text();
  const status = `${response.status} ${response.statusText}`.trim();
  console.log(`Callback replayed to VPS localhost: ${status}`);

  const sanitized = redactCallbackUrl(text);
  if (sanitized.trim()) {
    console.log(sanitized.slice(0, 500));
  }
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Callback replay failed: ${message}`);
  console.error('Make sure the Hermes OAuth login process is still waiting on the VPS.');
  process.exit(1);
} finally {
  clearTimeout(timeout);
}
