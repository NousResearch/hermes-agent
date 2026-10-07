import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { buildRendererCsp, inlineScriptHash, inlineScriptHashes, injectCspMeta, rendererCsp } from './renderer-csp.mjs'

const directive = (policy, name) => {
  const found = policy.split('; ').find(part => part.startsWith(`${name} `))

  assert.ok(found, `policy is missing the ${name} directive`)

  return found
}

test('the policy denies everything by default', () => {
  assert.equal(directive(buildRendererCsp(), 'default-src'), "default-src 'none'")
})

// Each of these was a real breakage found by launching the app, not a guess.
// Deleting any one of them silently kills a shipped feature, so pin them.
test("script-src keeps 'wasm-unsafe-eval' for shiki's Oniguruma engine", () => {
  assert.match(directive(buildRendererCsp(), 'script-src'), /'wasm-unsafe-eval'/)
})

test('script-src keeps blob: for the desktop plugin runtime', () => {
  assert.match(directive(buildRendererCsp(), 'script-src'), /(^|\s)blob:/)
})

test('media-src and img-src keep the privileged hermes-media: scheme', () => {
  assert.match(directive(buildRendererCsp(), 'media-src'), /hermes-media:/)
  assert.match(directive(buildRendererCsp(), 'img-src'), /hermes-media:/)
})

test('the Google Fonts pair is granted together or not at all', () => {
  const policy = buildRendererCsp()

  assert.match(directive(policy, 'style-src'), /https:\/\/fonts\.googleapis\.com/)
  assert.match(directive(policy, 'font-src'), /https:\/\/fonts\.gstatic\.com/)
})

test('script-src keeps the three social embed provider origins', () => {
  const scriptSrc = directive(buildRendererCsp(), 'script-src')

  for (const origin of ['https://www.instagram.com', 'https://www.tiktok.com', 'https://platform.twitter.com']) {
    assert.match(scriptSrc, new RegExp(origin.replace(/[.]/g, '\\.')))
  }
})

test('connect-src stays permissive because the user types an arbitrary gateway URL', () => {
  const connectSrc = directive(buildRendererCsp(), 'connect-src')

  for (const scheme of ['http:', 'https:', 'ws:', 'wss:']) {
    assert.match(connectSrc, new RegExp(`(^|\\s)${scheme}`))
  }
})

// The whole point of the policy. If either of these regresses, injected script
// executes in a file:// renderer that holds the preload bridge.
test("script-src never permits 'unsafe-inline' or 'unsafe-eval'", () => {
  const scriptSrc = directive(buildRendererCsp(['\'sha256-abc\'']), 'script-src')

  assert.doesNotMatch(scriptSrc, /'unsafe-inline'/)
  assert.doesNotMatch(scriptSrc, /'unsafe-eval'/)
})

test("worker-src is explicitly 'none' so it cannot inherit blob: from script-src", () => {
  assert.equal(directive(buildRendererCsp(), 'worker-src'), "worker-src 'none'")
})

// Chromium ignores frame-ancestors in a <meta>-delivered policy. Including it
// would log a warning every boot and read as protection that is not there.
test('the policy omits directives that are ignored when delivered via meta', () => {
  const policy = buildRendererCsp()

  assert.doesNotMatch(policy, /frame-ancestors/)
  assert.doesNotMatch(policy, /report-uri/)
  assert.doesNotMatch(policy, /(^|; )sandbox /)
})

test('object-src, base-uri and form-action are locked off', () => {
  const policy = buildRendererCsp()

  assert.equal(directive(policy, 'object-src'), "object-src 'none'")
  assert.equal(directive(policy, 'base-uri'), "base-uri 'none'")
  assert.equal(directive(policy, 'form-action'), "form-action 'none'")
})

test('inline scripts are hashed over their exact bytes, and src-ed scripts are not', () => {
  const html = '<html><head><script>let a = 1</script><script src="./x.js"></script></head></html>'

  assert.deepEqual(inlineScriptHashes(html), [inlineScriptHash('let a = 1')])
})

test('a whitespace change to an inline script changes its hash', () => {
  assert.notEqual(inlineScriptHash('let a = 1'), inlineScriptHash('let a = 1 '))
})

test('the injected meta carries the hash of the document it was injected into', () => {
  const html = '<html><head><script>window.boot = 1</script></head><body></body></html>'

  const out = injectCspMeta(html)

  assert.match(out, new RegExp(inlineScriptHash('window.boot = 1').replace(/[+/]/g, char => `\\${char}`)))
})

test('the meta is spliced in ahead of the boot script it authorises', () => {
  const html = '<html><head><script>window.boot = 1</script></head><body></body></html>'

  const out = injectCspMeta(html)

  assert.ok(out.indexOf('http-equiv="Content-Security-Policy"') < out.indexOf('<script>'))
})

test('injection leaves the inline script body byte-identical', () => {
  const body = '\n  // comment with  awkward   spacing\n  let a = 1\n'
  const html = `<html><head><script>${body}</script></head></html>`

  assert.ok(injectCspMeta(html).includes(`<script>${body}</script>`))
})

test('a document with no head is refused rather than shipped unprotected', () => {
  assert.throws(() => injectCspMeta('<div>no head</div>'), /refusing to ship an unprotected renderer/)
})

test('the plugin runs on build only, never the dev server', () => {
  const plugin = rendererCsp()

  assert.equal(plugin.apply, 'build')
  assert.equal(plugin.transformIndexHtml.order, 'post')
})

// The policy protects nothing if vite.config.ts stops calling the plugin.
test('vite.config.ts registers the plugin', () => {
  const config = readFileSync(fileURLToPath(new URL('../vite.config.ts', import.meta.url)), 'utf8')

  assert.match(config, /rendererCsp/)
})
