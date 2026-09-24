/**
 * Remote-backend topology for the core suite: a real `hermes serve` the TEST
 * spawns (not the app) under its OWN sandbox HOME / HERMES_HOME / cwd, reached
 * by the Desktop only by URL + token (HERMES_DESKTOP_REMOTE_URL/_TOKEN). Paths
 * that exist on the client sandbox do not exist on the backend's filesystem
 * view of the world (different root), which is the shape of a Desktop talking
 * to a backend on another machine.
 */

import { type ChildProcess, spawn } from 'node:child_process'
import * as crypto from 'node:crypto'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import * as zlib from 'node:zlib'

import { type CoreSandbox, REPO_ROOT } from './harness'

export interface RemoteBackend {
  url: string
  token: string
  port: number
  pid: () => number
  /** SIGKILL the backend and wait for it to exit (a crash / host reboot). */
  kill: () => Promise<void>
  /** Start it again on the same port and home (the user's service restarting). */
  restart: () => Promise<void>
  logTail: () => string
}

function python(): string {
  for (const venv of ['.venv', 'venv']) {
    const candidate = path.join(REPO_ROOT, venv, 'bin', 'python')

    if (fs.existsSync(candidate)) {
      return candidate
    }
  }

  throw new Error(`no Python venv under ${REPO_ROOT} (run uv sync)`)
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as net.AddressInfo
      server.close(() => resolve(port))
    })
  })
}

async function waitReady(url: string, token: string, child: ChildProcess, log: string[]): Promise<void> {
  const deadline = Date.now() + 120_000

  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`remote backend exited ${child.exitCode} before ready:\n${log.slice(-40).join('\n')}`)
    }

    try {
      const res = await fetch(`${url}/api/status`, { headers: { Authorization: `Bearer ${token}` } })

      if (res.ok) {
        return
      }
    } catch {
      /* not listening yet */
    }

    await new Promise(resolve => setTimeout(resolve, 250))
  }

  throw new Error(`remote backend never answered /api/status:\n${log.slice(-40).join('\n')}`)
}

/** Spawn `hermes serve` for `sandbox` (its HOME/HERMES_HOME, cwd = its root). */
export async function startRemoteBackend(sandbox: CoreSandbox): Promise<RemoteBackend> {
  const port = await freePort()
  const token = crypto.randomBytes(24).toString('base64url')
  const url = `http://127.0.0.1:${port}`
  const log: string[] = []
  let child: ChildProcess

  const env: Record<string, string> = {}

  for (const [key, value] of Object.entries(process.env)) {
    if (
      value &&
      !/^_?HERMES_/.test(key) &&
      !/(_API_KEY|_TOKEN|_SECRET|_BASE_URL)$/.test(key) &&
      key !== 'VIRTUAL_ENV'
    ) {
      env[key] = value
    }
  }

  const launch = async () => {
    child = spawn(python(), ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', String(port)], {
      cwd: sandbox.root,
      env: {
        ...env,
        PATH: `${sandbox.bin}${path.delimiter}${env.PATH ?? ''}`,
        PYTHONPATH: REPO_ROOT,
        HOME: sandbox.home,
        HERMES_HOME: sandbox.hermesHome,
        HERMES_DASHBOARD_SESSION_TOKEN: token,
        GIT_NO_LAZY_FETCH: '1'
      },
      stdio: ['ignore', 'pipe', 'pipe']
    })

    const collect = (chunk: Buffer) => {
      log.push(...chunk.toString('utf8').split('\n').filter(Boolean))
      log.splice(0, Math.max(0, log.length - 200))
    }

    child.stdout?.on('data', collect)
    child.stderr?.on('data', collect)
    await waitReady(url, token, child, log)
  }

  await launch()

  const kill = async () => {
    if (child.exitCode !== null || child.signalCode !== null) {
      return
    }

    const exited = new Promise<void>(resolve => child.once('exit', () => resolve()))
    child.kill('SIGKILL')
    await exited
  }

  return {
    url,
    token,
    port,
    pid: () => child.pid ?? -1,
    kill,
    restart: async () => {
      await kill()
      await launch()
    },
    logTail: () => log.slice(-60).join('\n')
  }
}

/** Remote-mode env for the app: no local backend, attach by URL + token. */
export function remoteEnv(backend: RemoteBackend): Record<string, string> {
  return { HERMES_DESKTOP_REMOTE_URL: backend.url, HERMES_DESKTOP_REMOTE_TOKEN: backend.token }
}

function withDb<T>(dbPath: string, read: (db: DatabaseSync) => T): null | T {
  if (!fs.existsSync(dbPath)) {
    return null
  }

  const db = new DatabaseSync(dbPath, { readOnly: true })

  try {
    return read(db)
  } catch {
    // Mid-WAL-checkpoint reads can fail transiently; callers poll.
    return null
  } finally {
    db.close()
  }
}

export interface SessionRow {
  id: string
  title: null | string
  parent_session_id: null | string
  end_reason: null | string
}

/** Every session row of the default profile, read-only, straight from state.db. */
export function sessionRows(sandbox: CoreSandbox): SessionRow[] {
  return (
    withDb(
      path.join(sandbox.hermesHome, 'state.db'),
      db =>
        db
          .prepare('SELECT id, title, parent_session_id, end_reason FROM sessions ORDER BY started_at, id')
          .all() as unknown as SessionRow[]
    ) ?? []
  )
}

/** Persisted message rows (role, content) of one session, oldest first, active rows only. */
export function messageRows(sandbox: CoreSandbox, sessionId: string): { role: string; content: string }[] {
  return (
    withDb(
      path.join(sandbox.hermesHome, 'state.db'),
      db =>
        db
          .prepare(
            "SELECT role, COALESCE(content, '') AS content FROM messages WHERE session_id = ? AND COALESCE(active, 1) = 1 ORDER BY id"
          )
          .all(sessionId) as unknown as { role: string; content: string }[]
    ) ?? []
  )
}

/** A 16x16 PNG with a unique payload chunk, so the bytes can be traced end to end. */
export function uniquePng(tag: string): Buffer {
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n

    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    }

    return c >>> 0
  })

  const crc = (buf: Buffer) => {
    let c = 0xffffffff

    for (const byte of buf) {
      c = crcTable[(c ^ byte) & 0xff]! ^ (c >>> 8)
    }

    return (c ^ 0xffffffff) >>> 0
  }

  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4)
    len.writeUInt32BE(data.length)
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
    const sum = Buffer.alloc(4)
    sum.writeUInt32BE(crc(body))

    return Buffer.concat([len, body, sum])
  }

  const size = 16
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(size, 0)
  ihdr.writeUInt32BE(size, 4)
  ihdr[8] = 8
  ihdr[9] = 2 // truecolor RGB
  const raw = Buffer.alloc(size * (1 + size * 3))

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const at = y * (1 + size * 3) + 1 + x * 3
      raw[at] = (x * 16) & 0xff
      raw[at + 1] = (y * 16) & 0xff
      raw[at + 2] = 0x80
    }
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('tEXt', Buffer.from(`Comment\0${tag}`, 'latin1')),
    chunk('IDAT', zlib.deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0))
  ])
}
