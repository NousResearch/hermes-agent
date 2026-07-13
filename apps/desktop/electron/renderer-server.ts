import fs from 'node:fs'
import http from 'node:http'
import type { AddressInfo } from 'node:net'
import path from 'node:path'

const DEFAULT_PORT = 47891

const MIME_TYPES: Record<string, string> = {
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

interface RendererServer {
  close: () => Promise<void>
  origin: string
}

function rendererRequestPath(root: string, requestUrl: string): string | null {
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

function startRendererServer(
  rootDir: string,
  { port = DEFAULT_PORT }: { port?: number } = {}
): Promise<RendererServer> {
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

      if (stat.isDirectory()) {
        filePath = path.join(filePath, 'index.html')
      }
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

  return new Promise<RendererServer>((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port, exclusive: true }, () => {
      server.removeListener('error', reject)
      const address = server.address() as AddressInfo
      resolve({
        close: () => new Promise<void>(done => server.close(() => done())),
        origin: `http://127.0.0.1:${address.port}`
      })
    })
  })
}

export { DEFAULT_PORT, rendererRequestPath, startRendererServer }
export type { RendererServer }
