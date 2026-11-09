// E2E: the real preview-reach module against a real remote host.
//
// Not a unit test — this spawns actual `ssh -N -L` tunnels to a live box and
// fetches a dev server that is bound to THAT machine's loopback and is
// provably unreachable from here. Run manually:
//   node --experimental-strip-types electron/preview-reach.e2e.mts <user@host>
import http from 'node:http'
import { spawn } from 'node:child_process'
import net from 'node:net'

import { loopbackTarget, PreviewReachRegistry, rewriteToLocalPort } from './preview-reach.ts'

const HOST = process.argv[2] || 'root@5.161.224.47'
const REMOTE_PORT = 5173

function pickLocalPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

const tunnels = new Map<string, ReturnType<typeof spawn>>()

async function forward(localPort: number, remotePort: number, remoteHost = '127.0.0.1') {
  const spec = `127.0.0.1:${localPort}:${remoteHost}:${remotePort}`
  const child = spawn(
    'ssh',
    ['-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-v', '-N', '-L', spec, '--', HOST],
    { stdio: ['ignore', 'ignore', 'pipe'] }
  )

  tunnels.set(spec, child)

  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('forward timed out')), 15_000)
    child.stderr?.on('data', d => {
      if (new RegExp(`Local forwarding listening on .* port ${localPort}\\b`).test(String(d))) {
        clearTimeout(timer)
        resolve()
      }
    })
    child.on('exit', code => {
      clearTimeout(timer)
      reject(new Error(`ssh exited ${code}`))
    })
  })
}

async function cancel(localPort: number, remotePort: number) {
  const spec = `127.0.0.1:${localPort}:127.0.0.1:${remotePort}`
  tunnels.get(spec)?.kill()
  tunnels.delete(spec)
}

function get(url: string, timeoutMs = 6000): Promise<{ body: string; status: number }> {
  return new Promise((resolve, reject) => {
    const req = http.get(url, res => {
      let body = ''
      res.on('data', c => (body += c))
      res.on('end', () => resolve({ body, status: res.statusCode || 0 }))
    })
    req.setTimeout(timeoutMs, () => req.destroy(new Error('timeout')))
    req.on('error', reject)
  })
}

const results: string[] = []
const check = (name: string, ok: boolean, detail = '') => {
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`)
}

const registry = new PreviewReachRegistry()
const deps = { cancel, forward, isCurrent: () => true, pickLocalPort }
const REMOTE_URL = `http://localhost:${REMOTE_PORT}/`

try {
  // 0. The bug itself: the agent's URL must be dead on this machine.
  let unreachable = false
  try {
    await get(REMOTE_URL, 4000)
  } catch {
    unreachable = true
  }
  check('agent URL is unreachable locally (the bug)', unreachable)

  // 1. Reach it.
  const reached = await registry.resolve(REMOTE_URL, deps)
  check('resolve() returns a rewritten URL', Boolean(reached) && reached !== REMOTE_URL, String(reached))

  const page = await get(reached!)
  check('rewritten URL serves the REMOTE page', page.status === 200 && page.body.includes('REMOTE box'), `status=${page.status}`)

  // 2. Same remote port reuses one tunnel; the path must survive the rewrite.
  const second = await registry.resolve(`http://localhost:${REMOTE_PORT}/index.html?q=1#frag`, deps)
  check('same port reuses one lease', registry.size === 1, `size=${registry.size}`)
  check('path/query/hash preserved', second!.endsWith('/index.html?q=1#frag'), String(second))

  const deep = await get(second!.split('#')[0])
  check('deep URL still serves the remote page', deep.status === 200 && deep.body.includes('REMOTE box'))

  // 3. Non-loopback is left alone — no tunnel to a public host.
  const external = await registry.resolve('https://example.com/x', deps)
  check('non-loopback returns null', external === null, String(external))

  // 4. Teardown actually closes the socket.
  const port = Number(new URL(reached!).port)
  await registry.closeAll()
  await new Promise(r => setTimeout(r, 500))
  let closed = false
  try {
    await get(`http://127.0.0.1:${port}/`, 2500)
  } catch {
    closed = true
  }
  check('closeAll() tears the forward down', closed && registry.size === 0)

  // 5. Pure helpers agree with the live behavior.
  check('loopbackTarget parses ipv6 loopback', loopbackTarget('http://[::1]:5173/')?.port === 5173)
  check('loopbackTarget rejects public host', loopbackTarget('http://example.com:5173/') === null)
  check('rewriteToLocalPort forces http', rewriteToLocalPort('https://localhost:5173/a', 42).startsWith('http://127.0.0.1:42/a'))
} finally {
  for (const child of tunnels.values()) {
    child.kill()
  }
}

console.log(results.join('\n'))
console.log(results.some(r => r.startsWith('FAIL')) ? '\nE2E FAILED' : '\nE2E PASSED')
process.exit(results.some(r => r.startsWith('FAIL')) ? 1 : 0)
