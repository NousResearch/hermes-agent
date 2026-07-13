'use strict'

const fs = require('node:fs')
const http = require('node:http')
const path = require('node:path')

const DEFAULT_PORT = 47891
const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ttf': 'font/ttf',
  '.wasm': 'application/wasm',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
}

function rendererRequestPath(root, requestUrl) {
  let pathname

  try {
    pathname = decodeURIComponent(new URL(requestUrl, 'http://127.0.0.1').pathname)
  } catch {
    return null
  }

  const relative = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '')
  const target = path.resolve(root, relative)

  if (target !== root && !target.startsWith(`${root}${path.sep}`)) {
    return null
  }

  return target
}

function startRendererServer(rootDir, { port = DEFAULT_PORT } = {}) {
  const root = path.resolve(rootDir)
  const indexPath = path.join(root, 'index.html')
  const server = http.createServer(async (request, response) => {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' })
      response.end()
      return
    }

    const requested = rendererRequestPath(root, request.url || '/')

    if (!requested) {
      response.writeHead(400)
      response.end()
      return
    }

    let filePath = requested

    try {
      const stat = await fs.promises.stat(filePath)
      if (stat.isDirectory()) filePath = path.join(filePath, 'index.html')
    } catch {
      // HashRouter routes never reach the server, but an extensionless reload
      // should still receive the SPA shell.
      filePath = path.extname(filePath) ? filePath : indexPath
    }

    try {
      const body = await fs.promises.readFile(filePath)
      const extension = path.extname(filePath).toLowerCase()
      const isIndex = filePath === indexPath
      response.writeHead(200, {
        'Cache-Control': isIndex ? 'no-store' : 'public, max-age=31536000, immutable',
        'Content-Type': MIME_TYPES[extension] || 'application/octet-stream',
        'X-Content-Type-Options': 'nosniff'
      })
      response.end(request.method === 'HEAD' ? undefined : body)
    } catch {
      response.writeHead(404)
      response.end()
    }
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port, exclusive: true }, () => {
      server.removeListener('error', reject)
      const address = server.address()
      resolve({
        close: () => new Promise(done => server.close(done)),
        origin: `http://127.0.0.1:${address.port}`
      })
    })
  })
}

module.exports = { DEFAULT_PORT, rendererRequestPath, startRendererServer }
