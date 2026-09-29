/**
 * Tailcat connections: a Hermes backend shared with `hermes serve --share tailcat`.
 *
 * The other machine hands out one-time connection codes
 * (`hermes-tailcat:<address>:<port>:<secret>`). Desktop runs
 * `tailcat forward` to that address, redeems the code once for a device token,
 * and from then on talks to the share over the forward's loopback port with the
 * ordinary token-auth remote path. The tunnel carries bytes only; the share's
 * own gate decides who gets in, so a revoked device is refused even though the
 * tunnel still opens.
 */

import type { ChildProcess, SpawnOptions } from 'node:child_process'
import { createHash } from 'node:crypto'

export const TAILCAT_CODE_SCHEME = 'hermes-tailcat:'

const FORWARD_READY_TIMEOUT_MS = 30_000
const PAIR_TIMEOUT_MS = 30_000
const ADDRESS_IN_TEXT_RE = /\btc[A-Za-z0-9_-]{20,}/g
const FORWARDING_RE = /forwarding\s+127\.0\.0\.1:(\d+)\s+->/

export interface TailcatCode {
  address: string
  port: number
  secret: string
}

/** Inverse of the backend's `ConnectionCode.render`; null for anything malformed. */
export function parseTailcatCode(text: unknown): TailcatCode | null {
  const raw = String(text ?? '').trim()

  if (!raw.startsWith(TAILCAT_CODE_SCHEME)) {
    return null
  }

  const parts = raw.slice(TAILCAT_CODE_SCHEME.length).split(':')

  if (parts.length !== 3 || parts.some(part => !part)) {
    return null
  }

  const [address, portText, secret] = parts
  const port = Number(portText)

  if (!address.startsWith('tc') || !/^\d+$/.test(portText) || port <= 0 || port >= 65536) {
    return null
  }

  return { address, port, secret }
}

/**
 * Same label as the backend's `address_fingerprint`: addresses share a long
 * fixed prefix and suffix, so a truncated address would label them all alike.
 */
export function tailcatAddressFingerprint(address: string): string {
  return address ? createHash('sha256').update(address, 'utf8').digest('hex').slice(0, 8) : ''
}

/** Addresses reach anyone who holds them; keep them out of logs and error text. */
export function redactTailcat(text: unknown): string {
  return String(text ?? '').replace(ADDRESS_IN_TEXT_RE, 'tc…')
}

/** Local port from `tailcat forward`'s "forwarding 127.0.0.1:N -> …" line. */
export function forwardListenPort(line: string): null | number {
  const match = FORWARDING_RE.exec(line)

  return match ? Number(match[1]) : null
}

export interface TailcatBinaryDeps {
  /** A user-installed tailcat on PATH (wins over PM's copy). */
  findOnPath: (name: string) => null | string
  /** Run `hermes pm <args>` and return stdout; rejects on failure. */
  runPm: (args: string[], timeoutMs: number) => Promise<string>
  exists: (candidate: string) => boolean
  platform?: NodeJS.Platform
}

/** The pinned tailcat PM installed, read from `hermes pm env tailcat`'s PATH export. */
export function tailcatFromPmEnv(stdout: string, platform: NodeJS.Platform, exists: (p: string) => boolean): null | string {
  // The JSON is indented over several lines, and a first run prints
  // runtime-preparation progress before it: parse from the last line that
  // opens an object at column 0 to the end.
  const lines = String(stdout).split(/\r?\n/)
  const start = lines.findLastIndex(line => line.startsWith('{'))

  let env: Record<string, unknown>

  try {
    env = JSON.parse(start < 0 ? '' : lines.slice(start).join('\n'))
  } catch {
    return null
  }

  const separator = platform === 'win32' ? ';' : ':'
  const slash = platform === 'win32' ? '\\' : '/'
  const name = platform === 'win32' ? 'tailcat.exe' : 'tailcat'

  for (const dir of String(env?.PATH || '').split(separator).filter(Boolean)) {
    const candidate = `${dir.replace(/[\\/]+$/, '')}${slash}${name}`

    if (exists(candidate)) {
      return candidate
    }
  }

  return null
}

/**
 * PATH first, then PM's pinned copy; with `install`, ask PM to fetch it. PM
 * pins no macOS build (tailcat publishes none), so macOS relies on PATH.
 */
export async function resolveTailcatBinary(deps: TailcatBinaryDeps, { install = false } = {}): Promise<null | string> {
  const platform = deps.platform ?? process.platform
  const onPath = deps.findOnPath(platform === 'win32' ? 'tailcat.exe' : 'tailcat')

  if (onPath) {
    return onPath
  }

  const fromPm = async () => tailcatFromPmEnv(await deps.runPm(['env', 'tailcat'], 60_000).catch(() => '{}'), platform, deps.exists)
  const pinned = await fromPm()

  if (pinned || !install || platform === 'darwin') {
    return pinned
  }

  await deps.runPm(['install', 'tailcat'], 300_000)

  return fromPm()
}

export class TailcatError extends Error {
  readonly tailcat = true

  constructor(message: string, readonly code: 'code-rejected' | 'forward-failed' | 'missing-binary' | 'share-unreachable') {
    super(message)
  }
}

export interface TailcatForwarderDeps {
  resolveBinary: () => Promise<null | string>
  spawn: (command: string, args: string[], options: SpawnOptions) => ChildProcess
  log?: (line: string) => void
  readyTimeoutMs?: number
}

interface Forward {
  address: string
  child: ChildProcess
  port: number
  ready: Promise<number>
}

/**
 * One `tailcat forward` per connection id, reused across every profile and
 * reconnect while it lives. A forward that exited is replaced on next use.
 */
export function createTailcatForwarder(deps: TailcatForwarderDeps) {
  const forwards = new Map<string, Forward>()
  const log = deps.log ?? (() => {})
  const readyTimeoutMs = deps.readyTimeoutMs ?? FORWARD_READY_TIMEOUT_MS

  const start = async (key: string, address: string, port: number): Promise<Forward> => {
    const binary = await deps.resolveBinary()

    if (!binary) {
      throw new TailcatError(
        process.platform === 'darwin'
          ? 'Tailcat is not installed. Install it from https://github.com/tailscale/tailcat and make sure `tailcat` is on your PATH.'
          : 'Tailcat is not installed. Pair this connection again to let Hermes install it, or run `hermes pm install tailcat`.',
        'missing-binary'
      )
    }

    // An ephemeral client key: the share authenticates the device by its
    // Hermes token, never by the tailcat key.
    const child = deps.spawn(binary, ['forward', '--key=new', address, `0:${port}`], {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true
    })

    const ready = new Promise<number>((resolve, reject) => {
      const tail: string[] = []

      const timer = setTimeout(() => {
        child.kill()
        reject(new TailcatError('Tailcat did not open a local tunnel in time.', 'forward-failed'))
      }, readyTimeoutMs)

      const onData = (chunk: Buffer | string) => {
        for (const line of String(chunk).split(/\r?\n/)) {
          if (!line.trim()) {
            continue
          }

          tail.push(redactTailcat(line.trim()))
          tail.splice(0, Math.max(0, tail.length - 5))
          const local = forwardListenPort(line)

          if (local) {
            clearTimeout(timer)
            resolve(local)
          }
        }
      }

      child.stdout?.on('data', onData)
      child.stderr?.on('data', onData)
      child.once('error', error => {
        clearTimeout(timer)
        reject(new TailcatError(`Could not start tailcat: ${redactTailcat(error.message)}`, 'forward-failed'))
      })
      child.once('exit', code => {
        clearTimeout(timer)

        if (forwards.get(key)?.child === child) {
          forwards.delete(key)
        }

        log(`[tailcat] forward for ${key} exited (${code ?? 'signal'})${tail.length ? `: ${tail.at(-1)}` : ''}`)
        reject(
          new TailcatError(
            `Tailcat exited before the tunnel opened${tail.length ? `: ${tail.at(-1)}` : '.'}`,
            'forward-failed'
          )
        )
      })
    })

    const forward = { address, child, port, ready }
    forwards.set(key, forward)
    ready.catch(() => {
      if (forwards.get(key) === forward) {
        forwards.delete(key)
      }
    })

    return forward
  }

  const ensure = async (key: string, address: string, port: number): Promise<number> => {
    let forward = forwards.get(key)

    if (forward && (forward.address !== address || forward.port !== port || forward.child.exitCode !== null)) {
      stop(key)
      forward = undefined
    }

    const localPort = await (forward ?? (await start(key, address, port))).ready
    log(`[tailcat] ${key} tunnel on 127.0.0.1:${localPort}`)

    return localPort
  }

  const stop = (key: string): void => {
    const forward = forwards.get(key)
    forwards.delete(key)

    if (forward && forward.child.exitCode === null) {
      forward.child.kill()
    }
  }

  const stopAll = (): void => {
    for (const key of [...forwards.keys()]) {
      stop(key)
    }
  }

  return { ensure, stop, stopAll, has: (key: string) => forwards.has(key) }
}

export type TailcatForwarder = ReturnType<typeof createTailcatForwarder>

export interface PairResult {
  deviceId: string
  token: string
}

export interface PairedTailcat {
  address: string
  deviceId: string
  port: number
  token: string
}

/**
 * Turn a pasted connection code into the fields a tailcat registry entry
 * stores. The forward opened for pairing lives under `key`; the caller stops
 * it once the entry is saved.
 */
export async function pairTailcatCode(
  forwarder: TailcatForwarder,
  key: string,
  codeText: string,
  deviceName: string,
  fetchImpl: typeof fetch = fetch
): Promise<PairedTailcat> {
  const code = parseTailcatCode(codeText)

  if (!code) {
    throw new TailcatError(
      'That is not a Hermes connection code. Run `hermes share code` on the other machine and paste the whole line.',
      'code-rejected'
    )
  }

  const localPort = await forwarder.ensure(key, code.address, code.port)
  const paired = await pairTailcatDevice(localPort, String(codeText).trim(), deviceName, fetchImpl)

  return { address: code.address, deviceId: paired.deviceId, port: code.port, token: paired.token }
}

/** Redeem a connection code through an open forward. */
export async function pairTailcatDevice(
  localPort: number,
  code: string,
  name: string,
  fetchImpl: typeof fetch = fetch
): Promise<PairResult> {
  let response: Response

  try {
    response = await fetchImpl(`http://127.0.0.1:${localPort}/api/share/pair`, {
      body: JSON.stringify({ code, name }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
      signal: AbortSignal.timeout(PAIR_TIMEOUT_MS)
    })
  } catch (error) {
    throw new TailcatError(
      `The tunnel opened but the shared Hermes did not answer (${redactTailcat((error as Error)?.message || error)}). ` +
        'Is `hermes serve --share tailcat` still running on the other machine?',
      'share-unreachable'
    )
  }

  const body = (await response.json().catch(() => ({}))) as Record<string, any>

  if (!response.ok) {
    throw new TailcatError(
      String(body?.detail || `Pairing failed (HTTP ${response.status}).`),
      response.status === 403 ? 'code-rejected' : 'share-unreachable'
    )
  }

  const token = String(body?.token || '')
  const deviceId = String(body?.device?.id || '')

  if (!token || !deviceId) {
    throw new TailcatError('The shared Hermes returned an incomplete pairing reply.', 'share-unreachable')
  }

  return { deviceId, token }
}
