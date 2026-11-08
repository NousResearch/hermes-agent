/**
 * Loopback reach for the in-app browser against a REMOTE gateway.
 *
 * The `<webview>` renders on the user's machine. The agent runs on the gateway
 * host. So when the agent says "your dev server is at http://localhost:5173",
 * that address is true *there* and meaningless *here* — locally it is usually
 * nothing at all, and occasionally somebody else's service on the same port.
 *
 * The fix is to make the address true here too: open a local→remote forward
 * over the transport we are ALREADY authenticated on, and hand the renderer a
 * `127.0.0.1:<localPort>` URL that lands on the gateway's port. No new
 * credentials, no new listening surface beyond one loopback-bound port.
 *
 * Only SSH-backed remotes can do this today: a `url`/`cloud` gateway is an HTTP
 * endpoint with no tunnel to borrow, so there is nothing to forward through.
 * Those callers get `null` and the pane keeps explaining the mismatch instead
 * of failing silently — see `isRemoteLoopbackUrl` in preview-pane.tsx.
 *
 * The lease/capability shape here is adapted from tuancookiez-hub's #87243,
 * which solved the same problem for Windows SSH previews.
 */

/** Hosts that mean "the machine this resolved on" — the whole problem class. */
const LOOPBACK_HOSTS = new Set(['0.0.0.0', '127.0.0.1', '::1', 'localhost'])

/** A forward is a live socket to someone else's machine; it should not outlive
 *  the user's attention on it. Refreshed on every reuse. */
export const PREVIEW_REACH_LEASE_MS = 15 * 60 * 1000

export interface PreviewReachDeps {
  /** Tear the forward down. */
  cancel: (localPort: number, remotePort: number) => Promise<void>
  /** Open it. Mirrors `SshConnection.forward`. */
  forward: (localPort: number, remotePort: number, remoteHost?: string) => Promise<void>
  /** False once the connection that authorized this lease is gone. */
  isCurrent: () => boolean
  /** Kernel-assigned free loopback port. */
  pickLocalPort: () => Promise<number>
}

export interface PreviewReachLease {
  close: () => Promise<void>
  expiresAt: number
  localPort: number
  remoteHost: string
  remotePort: number
  /** The address to hand the renderer. */
  url: string
}

interface ParsedLoopback {
  host: string
  port: number
}

/**
 * The loopback target inside `rawUrl`, or null when the URL isn't one.
 *
 * Deliberately permissive about the port: a dev server is whatever the
 * framework picked (5173, 3000, 8080, 4321, …), and an allowlist just means
 * the next framework's default silently fails. The security boundary is the
 * transport — we can only ever reach a host we are already authenticated to —
 * not a guess about which ports are wholesome.
 */
export function loopbackTarget(rawUrl: string): null | ParsedLoopback {
  let parsed: URL

  try {
    parsed = new URL(rawUrl)
  } catch {
    return null
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return null
  }

  const host = parsed.hostname.replace(/^\[|\]$/g, '').toLowerCase()

  if (!LOOPBACK_HOSTS.has(host)) {
    return null
  }

  const port = Number(parsed.port || (parsed.protocol === 'https:' ? 443 : 80))

  return Number.isInteger(port) && port > 0 && port < 65_536 ? { host, port } : null
}

/** Swap the authority for the local end of a forward, preserving everything
 *  else. Path, query, and hash are what make the URL useful. */
export function rewriteToLocalPort(rawUrl: string, localPort: number): string {
  const parsed = new URL(rawUrl)

  parsed.hostname = '127.0.0.1'
  parsed.port = String(localPort)
  // The forward carries plain TCP to a dev server that is almost never
  // TLS-terminated; https would fail the handshake against it.
  parsed.protocol = 'http:'

  return parsed.toString()
}

/**
 * Open a forward for `rawUrl`, or null when it isn't a loopback address.
 *
 * Failure to forward is thrown, not swallowed: the caller decides whether a
 * dead tunnel is worth surfacing, and a silent null here would be
 * indistinguishable from "this URL was fine all along".
 */
export async function openPreviewReach(rawUrl: string, deps: PreviewReachDeps): Promise<null | PreviewReachLease> {
  const target = loopbackTarget(rawUrl)

  if (!target) {
    return null
  }

  const localPort = await deps.pickLocalPort()

  // The connection can die between picking a port and using it.
  if (!deps.isCurrent()) {
    return null
  }

  await deps.forward(localPort, target.port, '127.0.0.1')

  let closed = false
  let timer: null | ReturnType<typeof setTimeout> = null

  const close = async () => {
    if (closed) {
      return
    }

    closed = true

    if (timer) {
      clearTimeout(timer)
      timer = null
    }

    await deps.cancel(localPort, target.port)
  }

  timer = setTimeout(() => void close(), PREVIEW_REACH_LEASE_MS)
  // A pending lease timer must never hold the app open at quit. Node's timer
  // has unref; the DOM lib's number type (what tsc picks here) does not.
  ;(timer as unknown as { unref?: () => void }).unref?.()

  return {
    close,
    expiresAt: Date.now() + PREVIEW_REACH_LEASE_MS,
    localPort,
    remoteHost: target.host,
    remotePort: target.port,
    url: rewriteToLocalPort(rawUrl, localPort)
  }
}

/**
 * Forwards currently open, keyed by the remote port they reach.
 *
 * One lease per remote port, not per URL: navigating around a dev server
 * (`/`, `/about`, `?q=1`) is the same tunnel, and minting a fresh one per page
 * would leak a socket per click.
 */
export class PreviewReachRegistry {
  private leases = new Map<number, PreviewReachLease>()

  /** Reuse a live lease for this port, else open one. */
  async resolve(rawUrl: string, deps: PreviewReachDeps): Promise<null | string> {
    const target = loopbackTarget(rawUrl)

    if (!target) {
      return null
    }

    const existing = this.leases.get(target.port)

    if (existing && existing.expiresAt > Date.now()) {
      return rewriteToLocalPort(rawUrl, existing.localPort)
    }

    if (existing) {
      await existing.close().catch(() => {})
      this.leases.delete(target.port)
    }

    const lease = await openPreviewReach(rawUrl, deps)

    if (!lease) {
      return null
    }

    this.leases.set(target.port, lease)

    return lease.url
  }

  /** Drop every forward — the authorizing connection changed or went away. */
  async closeAll(): Promise<void> {
    const open = [...this.leases.values()]

    this.leases.clear()
    await Promise.allSettled(open.map(lease => lease.close()))
  }

  get size(): number {
    return this.leases.size
  }
}
