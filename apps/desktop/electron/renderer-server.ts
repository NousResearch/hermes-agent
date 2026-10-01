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

function listenOnce(server: http.Server, port: number): Promise<AddressInfo> {
  return new Promise<AddressInfo>((resolve, reject) => {
    const onError = (error: NodeJS.ErrnoException) => {
      server.removeListener('error', onError)
      reject(error)
    }

    server.once('error', onError)
    server.listen({ host: '127.0.0.1', port, exclusive: true }, () => {
      server.removeListener('error', onError)
      resolve(server.address() as AddressInfo)
    })
  })
}

// Bind failures that another free port would resolve. EADDRINUSE is the common
// one (a second Hermes instance, or any unrelated listener); EACCES shows up on
// Windows when a port sits in an excluded/reserved range.
function isPortCollision(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException | null)?.code

  return code === 'EADDRINUSE' || code === 'EACCES'
}

async function startRendererServer(
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

  // Prefer the stable default port: the renderer's origin keys its localStorage /
  // sessionStorage, so a port that changed between launches would silently orphan
  // persisted renderer state. But a taken port must never block startup — main.ts
  // awaits this before createWindow(), so rejecting here means no window at all.
  // Fall back to an OS-assigned ephemeral port instead.
  let address: AddressInfo

  try {
    address = await listenOnce(server, port)
  } catch (error) {
    if (port === 0 || !isPortCollision(error)) {
      throw error
    }

    address = await listenOnce(server, 0)
  }

  return {
    close: () => new Promise<void>(done => server.close(() => done())),
    origin: `http://127.0.0.1:${address.port}`
  }
}

export { DEFAULT_PORT, rendererRequestPath, startRendererServer }
export type { RendererServer }
