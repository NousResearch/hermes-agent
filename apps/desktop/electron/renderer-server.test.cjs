'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { rendererRequestPath, startRendererServer } = require('./renderer-server.cjs')

test('serves the packaged renderer from a loopback HTTP origin', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-renderer-'))
  fs.writeFileSync(path.join(root, 'index.html'), '<main>Hermes</main>')
  fs.mkdirSync(path.join(root, 'assets'))
  fs.writeFileSync(path.join(root, 'assets', 'app.js'), 'globalThis.loaded = true')
  const server = await startRendererServer(root, { port: 0 })

  t.after(async () => {
    await server.close()
    fs.rmSync(root, { force: true, recursive: true })
  })

  assert.match(server.origin, /^http:\/\/127\.0\.0\.1:\d+$/)

  const index = await fetch(`${server.origin}/`)
  assert.equal(index.status, 200)
  assert.equal(index.headers.get('content-type'), 'text/html; charset=utf-8')
  assert.equal(await index.text(), '<main>Hermes</main>')

  const asset = await fetch(`${server.origin}/assets/app.js`)
  assert.equal(asset.status, 200)
  assert.equal(asset.headers.get('content-type'), 'text/javascript; charset=utf-8')
  assert.equal(await asset.text(), 'globalThis.loaded = true')
})

test('falls back to the SPA shell for extensionless paths', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-renderer-'))
  fs.writeFileSync(path.join(root, 'index.html'), '<main>Hermes</main>')
  const server = await startRendererServer(root, { port: 0 })

  t.after(async () => {
    await server.close()
    fs.rmSync(root, { force: true, recursive: true })
  })

  const response = await fetch(`${server.origin}/session/abc`)
  assert.equal(response.status, 200)
  assert.equal(await response.text(), '<main>Hermes</main>')
})

test('rejects paths outside the renderer root', () => {
  const root = path.resolve('/tmp/hermes-renderer')

  assert.equal(rendererRequestPath(root, '/%E0%A4%A'), null)
  assert.equal(rendererRequestPath(root, '/%2e%2e%2fetc/passwd'), null)
})
