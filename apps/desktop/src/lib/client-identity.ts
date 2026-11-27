import type { CanonicalClientIdentity } from '@hermes/shared'

import { isBrowserHostedDesktop } from '@/lib/platform'

// The backend's rule for a label's version: 1-64 printable ASCII, no whitespace.
const VERSION = /^[!-~]{1,64}$/

// What the version readers answer when they know nothing.
const NO_VERSION = new Set(['0.0.0', 'unknown', 'v0.0.0'])

// How long the hello waits for the version. Reading it asks the backend and this machine's install, which
// takes many seconds on a cold start, and until the hello lands this connection's turns carry no client.
const VERSION_WAIT_MS = 1000

let reading: Promise<string> | null = null

function usable(candidate: unknown): candidate is string {
  return typeof candidate === 'string' && VERSION.test(candidate) && !NO_VERSION.has(candidate)
}

function readVersion(): Promise<string> {
  // Read once per window. This machine's own build comes first: `appVersion` is a display string that can
  // hold spaces a label may not, and on a source install it names the backend. Behind the browser bridge it
  // is the only one, and it is the serving checkout's version.
  reading ??= Promise.resolve(window.hermesDesktop?.getVersion?.())
    .then(info => [info?.clientVersion, info?.baseVersion, info?.appVersion].find(usable) ?? '')
    .catch(() => '')

  return reading
}

/** The version when it is known within `ms`, else none: the label matters more than its version. */
function versionWithin(ms: number): Promise<string> {
  return new Promise(resolve => {
    const timer = setTimeout(() => resolve(''), ms)

    void readVersion().then(version => {
      clearTimeout(timer)
      resolve(version)
    })
  })
}

/**
 * Which client this window is, for the per-connection hello (`client.capabilities {client}`): the Electron
 * app or the same renderer behind the browser bridge. A label for attribution only; it grants nothing.
 */
export async function desktopClientIdentity(): Promise<CanonicalClientIdentity> {
  const client: CanonicalClientIdentity = isBrowserHostedDesktop()
    ? { kind: 'webapp', name: 'hermes-webapp' }
    : { kind: 'desktop', name: 'hermes-desktop' }

  const version = await versionWithin(VERSION_WAIT_MS)

  return version ? { ...client, version } : client
}
