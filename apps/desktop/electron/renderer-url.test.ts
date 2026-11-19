import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { buildBrowserWindowUrl } from './browser-windows'
import { buildHudWindowUrl } from './hud-url'
import { isTrustedRendererUrl, rendererBaseUrl, resolveRendererFileRequest } from './renderer-url'
import { buildInstanceWindowUrl, buildSessionWindowUrl } from './session-windows'

const rendererRoot = () => {
  const root = mkdtempSync(path.join(tmpdir(), 'unitalk-renderer-'))
  mkdirSync(path.join(root, 'assets'))
  writeFileSync(path.join(root, 'index.html'), '<div id="root"></div>')
  writeFileSync(path.join(root, 'assets', 'app.js'), 'console.log("ok")')

  return root
}

test('rendererBaseUrl uses the Clerk custom scheme when enabled', () => {
  assert.equal(rendererBaseUrl({ customScheme: true, devServer: null }), 'unitalk-app://renderer/')
})

test('rendererBaseUrl keeps the dev server and file fallback', () => {
  assert.equal(rendererBaseUrl({ customScheme: false, devServer: 'http://127.0.0.1:5174/' }), 'http://127.0.0.1:5174/')
  assert.match(
    rendererBaseUrl({ customScheme: false, devServer: null, rendererIndexPath: '/opt/app/index.html' }),
    /^file:\/\/.*index\.html$/
  )
})

test('isTrustedRendererUrl accepts the custom origin and rejects file URLs when enabled', () => {
  assert.equal(isTrustedRendererUrl('unitalk-app://renderer/?win=hud#/', { customScheme: true, devServer: null }), true)
  assert.equal(isTrustedRendererUrl('file:///opt/app/index.html', { customScheme: true, devServer: null }), false)
  assert.equal(isTrustedRendererUrl('https://example.com', { customScheme: true, devServer: null }), false)
})

test('isTrustedRendererUrl accepts the dev server without a trailing slash', () => {
  assert.equal(
    isTrustedRendererUrl('http://127.0.0.1:5174/?win=hud#/', {
      customScheme: false,
      devServer: 'http://127.0.0.1:5174'
    }),
    true
  )
  assert.equal(
    isTrustedRendererUrl('http://127.0.0.1:5174.evil.test/', {
      customScheme: false,
      devServer: 'http://127.0.0.1:5174'
    }),
    false
  )
})

test('resolveRendererFileRequest serves the index, assets and extensionless routes', () => {
  const root = rendererRoot()

  assert.equal(resolveRendererFileRequest('unitalk-app://renderer/', root), path.join(root, 'index.html'))
  assert.equal(
    resolveRendererFileRequest('unitalk-app://renderer/assets/app.js', root),
    path.join(root, 'assets', 'app.js')
  )
  assert.equal(resolveRendererFileRequest('unitalk-app://renderer/session/abc', root), path.join(root, 'index.html'))
})

test('resolveRendererFileRequest rejects traversal and missing assets', () => {
  const root = rendererRoot()

  assert.equal(resolveRendererFileRequest('unitalk-app://renderer/..%2Fsecret.txt', root), null)
  assert.equal(resolveRendererFileRequest('unitalk-app://renderer/assets/missing.js', root), null)
  assert.equal(resolveRendererFileRequest('https://renderer/assets/app.js', root), null)
})

test('window URL builders emit the custom renderer origin', () => {
  assert.equal(
    buildHudWindowUrl('abc', { customScheme: true, profile: 'work' }),
    'unitalk-app://renderer/?win=hud&profile=work#/abc'
  )
  assert.equal(
    buildBrowserWindowUrl('tab-1', { customScheme: true }),
    'unitalk-app://renderer/?win=browser&tab=tab-1#/'
  )
  assert.equal(
    buildSessionWindowUrl('abc', { customScheme: true, profile: 'work' }),
    'unitalk-app://renderer/?win=secondary&profile=work#/abc'
  )
  assert.equal(buildInstanceWindowUrl({ customScheme: true }), 'unitalk-app://renderer/?peer=1')
})
