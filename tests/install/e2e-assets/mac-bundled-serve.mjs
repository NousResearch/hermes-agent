#!/usr/bin/env node
'use strict'
// mac-bundled-serve.mjs — loopback-only static file server for the macOS
// bundled-update E2E feed. Serves the directory produced by
// mac-bundled-feed.mjs over 127.0.0.1, logs every request to a JSONL file
// (the proof the app really fetched THIS feed), and stays up until killed.
//
// --port 0 (the default) binds an ephemeral port; the ACTUAL port is
// written as JSON to --ready once listening, so the driver configures the
// app with the real URL instead of a hard-coded one.
//
// Usage: node mac-bundled-serve.mjs --dir <feed-dir> [--port 0] \
//          --log <requests.jsonl> [--health /__healthz] [--ready <json>]

import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

/** Resolve `pathname` inside `root`, rejecting any path that escapes it
 * (path traversal). Returns null when the target is outside the root. */
export function safeJoin(root, pathname) {
  const target = path.resolve(root, '.' + decodeURIComponent(pathname))
  const rel = path.relative(path.resolve(root), target)
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) return null
  return target
}

const invokedDirectly = process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href

if (invokedDirectly) {
  const { values } = parseArgs({
    options: {
      dir: { type: 'string' },
      port: { type: 'string', default: '0' },
      log: { type: 'string' },
      health: { type: 'string', default: '/__healthz' },
      ready: { type: 'string' },
    }
  })

  const root = path.resolve(values.dir)
  if (!fs.existsSync(root)) {
    console.error(`feed dir not found: ${root}`)
    process.exit(1)
  }
  const logStream = values.log ? fs.createWriteStream(values.log, { flags: 'a' }) : null

  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://127.0.0.1')
    if (logStream) {
      logStream.write(JSON.stringify({ t: Date.now(), path: url.pathname, ua: req.headers['user-agent'] || '' }) + '\n')
    }
    if (url.pathname === values.health) {
      res.writeHead(200, { 'content-type': 'text/plain' })
      res.end('ok\n')
      return
    }
    const target = safeJoin(root, url.pathname)
    if (!target || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
      res.writeHead(404, { 'content-type': 'text/plain' })
      res.end('not found\n')
      return
    }
    // no-store: the updater must always see the live feed, never a cache.
    res.writeHead(200, {
      'content-type': target.endsWith('.yml') ? 'text/yaml; charset=utf-8' : 'application/zip',
      'content-length': fs.statSync(target).size,
      'cache-control': 'no-store'
    })
    fs.createReadStream(target).pipe(res)
  })

  // Loopback only, by construction and by policy: the production updater
  // accepts loopback HTTP overrides exclusively (mac-client.ts).
  server.listen(Number(values.port), '127.0.0.1', () => {
    const port = server.address().port
    const receipt = { port, url: `http://127.0.0.1:${port}`, root }
    if (values.ready) fs.writeFileSync(values.ready, JSON.stringify(receipt) + '\n')
    console.log(`serving ${root} on http://127.0.0.1:${port}`)
  })
}
