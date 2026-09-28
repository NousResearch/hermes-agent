/**
 * The GitHub edge for the Desktop update suite: a local stand-in for the two
 * api.github.com resources the source update check reads, answered from the
 * local bare origin, behind an HTTPS forward proxy the install is pointed at
 * (HTTPS_PROXY + a CA bundle that trusts this run's own root, i.e. a
 * TLS-inspecting corporate proxy).
 *
 *   GET /repos/<owner>/<repo>/commits/<branch>   (Accept: application/vnd.github.sha)
 *   GET /repos/<owner>/<repo>/compare/<base>...<head>
 *
 * Every other CONNECT target is tunnelled through unchanged and recorded, so a
 * failing cell can list every host the product reached.
 */

import { execFileSync, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as http from 'node:http'
import * as net from 'node:net'
import * as path from 'node:path'
import * as tls from 'node:tls'

export interface GithubEdge {
  proxyUrl: string
  caBundle: string
  /** host:port of every CONNECT the proxy saw, in order. */
  connects: string[]
  /** method + path of every request the fake api.github.com answered. */
  apiHits: string[]
  env: Record<string, string>
  close: () => Promise<void>
}

function openssl(args: string[], cwd: string): void {
  execFileSync('openssl', args, { cwd, stdio: 'pipe' })
}

/** A throwaway root + an api.github.com leaf signed by it. */
function makeCerts(dir: string): { key: Buffer; cert: Buffer; caPem: string } {
  fs.mkdirSync(dir, { recursive: true })
  // Python >= 3.13 verifies with X509_STRICT: the root needs keyUsage, the leaf an AKI.
  openssl(
    [
      'req',
      '-x509',
      '-newkey',
      'rsa:2048',
      '-nodes',
      '-keyout',
      'ca.key',
      '-out',
      'ca.pem',
      '-days',
      '2',
      '-subj',
      '/CN=hermes-update-e2e-root',
      '-addext',
      'basicConstraints=critical,CA:TRUE',
      '-addext',
      'keyUsage=critical,keyCertSign,cRLSign',
      '-addext',
      'subjectKeyIdentifier=hash'
    ],
    dir
  )
  openssl(
    ['req', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'leaf.key', '-out', 'leaf.csr', '-subj', '/CN=api.github.com'],
    dir
  )
  fs.writeFileSync(
    path.join(dir, 'leaf.ext'),
    'basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n' +
      'subjectAltName=DNS:api.github.com\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n'
  )
  openssl(
    [
      'x509',
      '-req',
      '-in',
      'leaf.csr',
      '-CA',
      'ca.pem',
      '-CAkey',
      'ca.key',
      '-CAcreateserial',
      '-out',
      'leaf.pem',
      '-days',
      '2',
      '-extfile',
      'leaf.ext'
    ],
    dir
  )

  return {
    key: fs.readFileSync(path.join(dir, 'leaf.key')),
    cert: fs.readFileSync(path.join(dir, 'leaf.pem')),
    caPem: fs.readFileSync(path.join(dir, 'ca.pem'), 'utf8')
  }
}

function gitIn(repo: string, args: string[]): { ok: boolean; out: string } {
  const cp = spawnSync('git', args, { cwd: repo, encoding: 'utf8', env: { ...process.env, GIT_NO_LAZY_FETCH: '1' } })

  return { ok: cp.status === 0, out: (cp.stdout ?? '').trim() }
}

/** The GitHub REST shapes the check reads, computed from the bare origin at request time. */
function apiHandler(origin: string, hits: string[]) {
  return (req: http.IncomingMessage, res: http.ServerResponse) => {
    const url = new URL(req.url ?? '/', 'https://api.github.com')
    hits.push(`${req.method} ${url.pathname}`)

    const send = (status: number, body: string, type = 'application/json') => {
      res.writeHead(status, { 'content-type': type })
      res.end(body)
    }

    const commit = /^\/repos\/[^/]+\/[^/]+\/commits\/(.+)$/.exec(url.pathname)

    if (commit) {
      const tip = gitIn(origin, ['rev-parse', '--verify', `refs/heads/${decodeURIComponent(commit[1])}^{commit}`])

      if (!tip.ok) {
        return send(404, JSON.stringify({ message: 'No commit found for SHA' }))
      }

      return /vnd\.github\.sha/.test(String(req.headers.accept))
        ? send(200, tip.out, 'application/vnd.github.sha')
        : send(200, JSON.stringify({ sha: tip.out }))
    }

    const compare = /^\/repos\/[^/]+\/[^/]+\/compare\/([0-9a-f]{40})\.\.\.([0-9a-f]{40})$/.exec(url.pathname)

    if (compare) {
      const [, base, head] = compare
      const log = gitIn(origin, ['log', '--reverse', '--format=%H%x1f%an%x1f%cI%x1f%s', `${base}..${head}`])

      if (!log.ok) {
        return send(404, JSON.stringify({ message: 'Not Found' }))
      }

      const behind = gitIn(origin, ['rev-list', '--count', `${head}..${base}`])

      const commits = log.out
        ? log.out.split('\n').map(line => {
            const [sha, author, date, message] = line.split('\x1f')

            return { sha, commit: { message, author: { name: author, date }, committer: { name: author, date } } }
          })
        : []

      return send(
        200,
        JSON.stringify({
          status: commits.length ? 'ahead' : 'identical',
          ahead_by: commits.length,
          behind_by: Number(behind.out || 0),
          total_commits: commits.length,
          commits
        })
      )
    }

    send(404, JSON.stringify({ message: 'Not Found' }))
  }
}

export async function startGithubEdge(
  workDir: string,
  origin: string,
  baseCaBundle: string | undefined
): Promise<GithubEdge> {
  const { key, cert, caPem } = makeCerts(path.join(workDir, 'edge-ca'))
  const caBundle = path.join(workDir, 'edge-ca', 'bundle.pem')
  const systemBundle = [baseCaBundle, '/etc/ssl/certs/ca-certificates.crt'].find(p => p && fs.existsSync(p))
  fs.writeFileSync(caBundle, `${systemBundle ? fs.readFileSync(systemBundle, 'utf8') : ''}\n${caPem}`)

  const connects: string[] = []
  const apiHits: string[] = []
  const api = http.createServer(apiHandler(origin, apiHits))
  const sockets = new Set<net.Socket>()

  const proxy = http.createServer((_req, res) => {
    res.writeHead(405)
    res.end('CONNECT only')
  })

  proxy.on('connection', socket => {
    sockets.add(socket)
    socket.on('close', () => sockets.delete(socket))
  })

  proxy.on('connect', (req: http.IncomingMessage, client: net.Socket, head: Buffer) => {
    const target = req.url ?? ''
    connects.push(target)
    const [host, portText] = target.split(':')
    const port = Number(portText || 443)
    client.on('error', () => undefined)

    if (host === 'api.github.com') {
      client.write('HTTP/1.1 200 Connection Established\r\n\r\n')

      if (head.length) {
        client.unshift(head)
      }

      const secure = new tls.TLSSocket(client, { isServer: true, key, cert })
      secure.on('error', () => undefined)
      api.emit('connection', secure)

      return
    }

    const upstream = net.connect(port, host, () => {
      client.write('HTTP/1.1 200 Connection Established\r\n\r\n')

      if (head.length) {
        upstream.write(head)
      }

      upstream.pipe(client)
      client.pipe(upstream)
    })

    sockets.add(upstream)
    upstream.on('close', () => sockets.delete(upstream))
    upstream.on('error', () => client.destroy())
  })

  await new Promise<void>(resolve => proxy.listen(0, '127.0.0.1', resolve))
  const proxyUrl = `http://127.0.0.1:${(proxy.address() as net.AddressInfo).port}`

  return {
    proxyUrl,
    caBundle,
    connects,
    apiHits,
    env: {
      HTTPS_PROXY: proxyUrl,
      https_proxy: proxyUrl,
      NO_PROXY: '127.0.0.1,localhost,::1',
      no_proxy: '127.0.0.1,localhost,::1',
      SSL_CERT_FILE: caBundle
    },
    close: async () => {
      for (const socket of sockets) {
        socket.destroy()
      }

      await new Promise<void>(resolve => proxy.close(() => resolve()))
      api.close()
    }
  }
}
