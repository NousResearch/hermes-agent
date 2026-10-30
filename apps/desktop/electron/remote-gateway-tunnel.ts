import { randomUUID } from 'node:crypto'
import net from 'node:net'

import { normalizeRemoteBaseUrl, normalizeSshConfig } from './connection-config'
import { markExpectedTransition } from './crash-forensics'
import { SshConnection, validateSshTarget } from './ssh-connection'

/**
 * Optional SSH transport for a Remote gateway. Desktop only forwards a local
 * loopback port to 127.0.0.1:<gateway URL port> on the SSH host; it never
 * inspects, starts, updates or stops the remote runtime.
 */
export interface RemoteGatewayTunnelConfig {
  /** user@host[:port] or a Host alias from ~/.ssh/config. */
  host: string
}

type SshTarget = NonNullable<ReturnType<typeof normalizeSshConfig>>

const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])
const BIND_COLLISION = /address already in use|cannot listen to port|bind.*failed/i

function sshTarget(host: unknown): SshTarget {
  const ssh = normalizeSshConfig({ mode: 'ssh', host: String(host || '').trim() })

  if (!ssh) {
    throw new Error('SSH tunnels need an SSH host.')
  }

  validateSshTarget(ssh.host, ssh.user || '', ssh.port || 22)

  return ssh
}

export function normalizeRemoteGatewayTunnel(value: unknown): RemoteGatewayTunnelConfig | undefined {
  if (value == null) {
    return undefined
  }

  if (typeof value !== 'object') {
    throw new Error('SSH tunnel configuration must contain an SSH host.')
  }

  const host = String((value as Record<string, unknown>).host || '').trim()
  sshTarget(host)

  return { host }
}

/** Identity of the SSH endpoint (not the dynamic local port). */
export function remoteGatewayTunnelKey(value?: RemoteGatewayTunnelConfig | null): string {
  if (!value) {
    return ''
  }

  const ssh = sshTarget(value.host)

  return JSON.stringify([ssh.user || '', ssh.host.toLowerCase(), ssh.port || 22])
}

/**
 * The gateway URL is the single source of the remote port. It must be plain
 * HTTP on the remote loopback: SSH already encrypts the hop, and the gateway
 * only has to listen where `ssh -L` can reach it. Returns the remote port.
 */
export function validateRemoteGatewayTunnel(url: unknown, tunnel: RemoteGatewayTunnelConfig): number {
  normalizeRemoteGatewayTunnel(tunnel)
  const parsed = new URL(normalizeRemoteBaseUrl(url))

  if (parsed.protocol !== 'http:' || !LOOPBACK_HOSTS.has(parsed.hostname) || parsed.username || parsed.password) {
    throw new Error('SSH tunnels need an http:// loopback gateway URL, e.g. http://127.0.0.1:9119.')
  }

  return Number(parsed.port || 80)
}

/** Registry fields for a saved or submitted tunnel: `{}` when there is none. */
export function remoteGatewayTunnelFields(
  kind: unknown,
  url: unknown,
  raw: unknown
): { sshTunnel?: RemoteGatewayTunnelConfig } {
  const sshTunnel = normalizeRemoteGatewayTunnel(raw)

  if (!sshTunnel) {
    return {}
  }

  if (kind !== 'remote') {
    throw new Error('SSH tunnels are only supported for Remote gateway connections.')
  }

  validateRemoteGatewayTunnel(url, sshTunnel)

  return { sshTunnel }
}

/**
 * The tunnel a sign-in rides: the editor draft's when it carries `sshTunnel`
 * (even null = direct), else the saved entry's by id. A URL alone never picks
 * a tunnel; a re-auth prompt for the active tunnelled connection passes its
 * live local URL, which `connectionIdForUrl` maps back to the id.
 */
export function remoteGatewayTunnelLoginSource(
  registry: { connections: TunnelSource[] },
  opts: { sshTunnel?: unknown } | null | undefined,
  connectionId: string,
  url: string
): TunnelSource {
  if (opts && 'sshTunnel' in opts) {
    return { id: connectionId, url, sshTunnel: normalizeRemoteGatewayTunnel(opts.sshTunnel) }
  }

  const entry = registry.connections.find(c => c.id === connectionId)

  return entry?.sshTunnel ? { ...entry, url } : { url }
}

/** Lease scope of the primary backend, or of a pooled one when `poolKey` is set. */
export function remoteGatewayTunnelScope(poolKey: string | undefined, profile: string): string {
  return poolKey || `primary:${profile}`
}

interface TunnelTransport {
  open(options?: { signal: AbortSignal }): Promise<unknown>
  forward(localPort: number, remotePort: number, remoteHost: string): Promise<unknown>
  close(): Promise<unknown>
  isAlive?(): Promise<boolean>
}

/** A registry entry (`id`) or a resolved route (`connectionId`). */
export interface TunnelSource {
  id?: string
  connectionId?: string
  url?: string
  sshTunnel?: RemoteGatewayTunnelConfig
}

interface TunnelEntry {
  identity: string
  controller: AbortController
  ssh: TunnelTransport
  /** Live acquisitions per scope; a scope is released when its last one ends. */
  leases: Map<string, Set<symbol>>
  ready: Promise<string>
  localUrl: string
  retired: boolean
}

/** One acquisition of a tunnel; `release()` drops only this hold. */
export interface TunnelLease {
  url: string
  release: () => Promise<void>
}

async function allocateLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const port = (server.address() as net.AddressInfo).port
      server.close(error => (error ? reject(error) : resolve(port)))
    })
  })
}

function originOf(url: string): string {
  try {
    return new URL(url).origin
  } catch {
    return ''
  }
}

function sourceId(source: TunnelSource): string {
  return source.connectionId || source.id || ''
}

/** One tunnel per registered connection, leased by every scope (profile, test, login) using it. */
export class RemoteGatewayTunnels {
  private entries = new Map<string, TunnelEntry>()
  /** Transports still shutting down: quit must wait for them too. */
  private closing = new Set<Promise<void>>()
  /** Per-connection queue so a replacement never races another acquisition or removal. */
  private turns = new Map<string, Promise<unknown>>()
  private shutDown = false

  constructor(
    private readonly deps: {
      createConnection?: (config: SshTarget, scope: string) => TunnelTransport
      allocatePort?: () => Promise<number>
      /** Same logger as the other SSH connections, so open/forward/close land in desktop.log. */
      rememberLog?: (chunk: string) => void
    } = {}
  ) {}

  /**
   * Holds the connection's tunnel (or its direct URL) under `scope` until
   * `release()`. Backend scopes keep it until the scope is released; one-shot
   * callers (Test, sign-in) should prefer `run`.
   */
  async acquire(source: TunnelSource, scope: string): Promise<TunnelLease> {
    if (!source.sshTunnel) {
      return { url: source.url || '', release: async () => {} }
    }

    const id = sourceId(source)

    if (!id) {
      throw new Error('No saved connection owns this SSH tunnel.')
    }

    const entry = await this.serialized(id, () => this.entryFor(id, source))
    const token = Symbol(scope)
    const held = entry.leases.get(scope) ?? new Set<symbol>()
    held.add(token)
    entry.leases.set(scope, held)
    const release = () => this.release(id, entry, scope, token)

    try {
      const url = await entry.ready
      this.assertCurrent(id, entry)

      return { url, release }
    } catch (error) {
      await release()

      throw error
    }
  }

  /** Dials a backend under `scope`: the hold outlives success (until the scope is released), not failure. */
  async connect<T>(source: TunnelSource, scope: string, dial: (baseUrl: string) => Promise<T>): Promise<T> {
    const lease = await this.acquire(source, scope)

    try {
      return await dial(lease.url)
    } catch (error) {
      await lease.release()

      throw error
    }
  }

  /** Runs `work` against the gateway and drops its hold afterwards, success or not. */
  async run<T>(source: TunnelSource, scope: string, work: (baseUrl: string) => Promise<T>): Promise<T> {
    const lease = await this.acquire(source, scope)

    try {
      return await work(lease.url)
    } finally {
      await lease.release()
    }
  }

  /**
   * Connection id behind a tunnel's local URL. Cookie jars are chosen by URL,
   * and the local port changes on every reconnect, so callers resolve the id
   * here instead of matching the saved URL.
   */
  connectionIdForUrl(url: unknown): string {
    const origin = originOf(String(url || ''))

    for (const [id, entry] of this.entries) {
      if (origin && originOf(entry.localUrl) === origin) {
        return id
      }
    }

    return ''
  }

  private serialized<T>(id: string, step: () => Promise<T>): Promise<T> {
    const turn = (this.turns.get(id) ?? Promise.resolve()).then(step, step)

    const settled = turn.then(
      () => {},
      () => {}
    )

    this.turns.set(id, settled)
    void settled.then(() => this.turns.get(id) === settled && this.turns.delete(id))

    return turn
  }

  private async entryFor(id: string, source: TunnelSource): Promise<TunnelEntry> {
    if (this.shutDown) {
      // Same sentinel as the local backend's quit, so desktop.log records a transition, not a crash.
      throw markExpectedTransition(new Error('Hermes Desktop is quitting.'))
    }

    const remotePort = validateRemoteGatewayTunnel(source.url, source.sshTunnel!)
    const identity = JSON.stringify([normalizeRemoteBaseUrl(source.url), remoteGatewayTunnelKey(source.sshTunnel)])
    const current = this.entries.get(id)

    if (current && current.identity === identity && !(current.localUrl && !(await this.alive(current)))) {
      return current
    }

    if (current) {
      await this.retire(id, current)
    }

    const config = sshTarget(source.sshTunnel!.host)
    const sshScope = `remote-gateway:${id}:${randomUUID()}`

    const ssh = this.deps.createConnection
      ? this.deps.createConnection(config, sshScope)
      : new SshConnection(config, { rememberLog: this.deps.rememberLog, scope: sshScope })

    const entry: TunnelEntry = {
      identity,
      controller: new AbortController(),
      ssh,
      leases: new Map(),
      ready: Promise.resolve(''),
      localUrl: '',
      retired: false
    }

    this.entries.set(id, entry)
    entry.ready = this.bootstrap(id, source, entry, remotePort)
    // Acquirers observe failures through `ready`; this only silences the unobserved rejection.
    entry.ready.catch(() => {})

    return entry
  }

  private async alive(entry: TunnelEntry): Promise<boolean> {
    return !entry.ssh.isAlive || (await entry.ssh.isAlive())
  }

  private assertCurrent(id: string, entry: TunnelEntry): void {
    if (entry.retired || this.entries.get(id) !== entry) {
      throw new Error('SSH tunnel was cancelled or retired.')
    }
  }

  private async bootstrap(id: string, source: TunnelSource, entry: TunnelEntry, remotePort: number): Promise<string> {
    try {
      await entry.ssh.open({ signal: entry.controller.signal })
      this.assertCurrent(id, entry)
      let localPort = 0

      for (let attempt = 0; attempt < 3; attempt++) {
        localPort = await (this.deps.allocatePort || allocateLoopbackPort)()
        this.assertCurrent(id, entry)

        try {
          await entry.ssh.forward(localPort, remotePort, '127.0.0.1')

          break
        } catch (error) {
          if (attempt === 2 || !BIND_COLLISION.test(String(error))) {
            throw error
          }
        }
      }

      this.assertCurrent(id, entry)
      const url = new URL(normalizeRemoteBaseUrl(source.url))
      url.hostname = '127.0.0.1'
      url.port = String(localPort)
      entry.localUrl = url.toString().replace(/\/+$/, '')

      return entry.localUrl
    } catch (error) {
      await this.retire(id, entry)

      throw error
    }
  }

  private async release(id: string, entry: TunnelEntry, scope: string, token: symbol): Promise<void> {
    const held = entry.leases.get(scope)

    if (!held?.delete(token)) {
      return
    }

    if (!held.size) {
      entry.leases.delete(scope)
    }

    if (!entry.leases.size) {
      await this.retire(id, entry)
    }
  }

  /** Stops `entry`, tracking the close so `hasOpen()`/`closeAll()` still see it. */
  private async retire(id: string, entry: TunnelEntry): Promise<void> {
    if (this.entries.get(id) === entry) {
      this.entries.delete(id)
    }

    if (entry.retired) {
      return
    }

    entry.retired = true
    entry.leases.clear()
    entry.controller.abort()

    const closing = entry.ssh.close().then(
      () => {},
      () => {}
    )

    this.closing.add(closing)
    await closing
    this.closing.delete(closing)
  }

  async closeOne(id: string): Promise<void> {
    await this.serialized(id, async () => {
      const entry = this.entries.get(id)

      if (entry) {
        await this.retire(id, entry)
      }
    })
  }

  /** Drops every hold of `scope`; a tunnel closes once nothing else holds it. */
  async releaseScope(scope: string): Promise<void> {
    await Promise.all(
      [...this.entries].map(async ([id, entry]) => {
        if (entry.leases.delete(scope) && !entry.leases.size) {
          await this.retire(id, entry)
        }
      })
    )
  }

  async releaseScopes(scopes: Iterable<string>): Promise<void> {
    await Promise.all([...scopes].map(scope => this.releaseScope(scope)))
  }

  /** Quit: refuse new tunnels, then close open ones and wait for those already closing. */
  async closeAll(): Promise<void> {
    this.shutDown = true
    await Promise.all([...this.entries.keys()].map(id => this.closeOne(id)))
    await Promise.all(this.closing)
  }

  hasOpen(): boolean {
    return this.entries.size > 0 || this.closing.size > 0
  }
}
