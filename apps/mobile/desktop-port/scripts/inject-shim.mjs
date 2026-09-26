/*
 * Post-build injection: copy the browser shim into dist/ and load it BEFORE the
 * app bundle in dist/index.html.
 *
 * The shim is a classic (non-module) <script>, so it executes synchronously
 * during HTML parsing — before the deferred `<script type="module">` app bundle
 * runs — guaranteeing `window.hermesDesktop` exists by the time the renderer's
 * module graph (store/zoom.ts side effects, use-gateway-boot) executes.
 *
 * Idempotent: re-running (or a dist that already carries the tag) is a no-op.
 */
import { readFileSync, writeFileSync, copyFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dist = resolve(root, 'dist')

const shimSrc = readFileSync(resolve(root, 'shim/hermes-web-shim.js'))
copyFileSync(resolve(root, 'shim/hermes-web-shim.js'), resolve(dist, 'hermes-web-shim.js'))

// Cache buster: iOS' WKWebView stubbornly caches the bundled files across
// app updates. A content-based ?v=<hash> changes the URL as soon as the shim
// changes -> WKWebView is forced to fetch it anew instead of serving the old
// version (and with it the old layout CSS) from the cache.
const shimHash = createHash('sha256').update(shimSrc).digest('hex').slice(0, 12)
const shimHref = './hermes-web-shim.js?v=' + shimHash

const indexPath = resolve(dist, 'index.html')
let html = readFileSync(indexPath, 'utf8')

if (!html.includes('hermes-web-shim.js')) {
  const shimTag = '<script src="' + shimHref + '"></script>'
  const next = html.replace(/<script type="module"/, shimTag + '\n    <script type="module"')
  if (next === html) {
    throw new Error('inject-shim: no <script type="module"> found in dist/index.html — cannot inject the shim.')
  }
  html = next
  writeFileSync(indexPath, html)
  console.log('Shim injected (' + shimHref + '):', indexPath)
} else {
  console.log('Shim already present:', indexPath)
}
