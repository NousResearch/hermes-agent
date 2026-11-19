import { isRecord } from '@hermes/shared/i18n'

const UNITALK_ORIGIN = 'https://unitalk.ai'
const REQUEST_TIMEOUT_MS = 30_000

interface UnitalkInstanceAccount {
  id: string
  ownerId: string
  userId: string
}

export interface ReadyUnitalkInstance extends UnitalkInstanceAccount {
  origin: string
  ready: true
}

interface InactiveUnitalkInstance extends UnitalkInstanceAccount {
  ready: false
}

export type UnitalkInstance = ReadyUnitalkInstance | InactiveUnitalkInstance

export class UnitalkResponseError extends Error {
  readonly status: number
  readonly endpoint: string

  constructor(message: string, status: number, endpoint: string) {
    super(message)
    this.name = 'UnitalkResponseError'
    this.status = status
    this.endpoint = endpoint
  }
}

async function requestJson(
  fetcher: typeof fetch,
  url: URL,
  signal?: AbortSignal
): Promise<Record<string, unknown>> {
  const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  const response = await fetcher(url, {
    cache: 'no-store',
    credentials: 'include',
    headers: { Accept: 'application/json' },
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout
  })

  const text = await response.text()

  let body: unknown

  try {
    body = JSON.parse(text)
  } catch {
    // auth-init also returns plain-text errors with HTTP 200.
    const fallback = response.ok ? 'Invalid Unitalk response' : `Unitalk request failed (${response.status})`
    const message = text && !text.trimStart().startsWith('<') ? text.slice(0, 300) : fallback

    throw new UnitalkResponseError(message, response.status, url.pathname)
  }

  if (isRecord(body) && typeof body.error === 'string') {
    throw new UnitalkResponseError(body.error, response.status, url.pathname)
  }

  if (!response.ok) {
    throw new UnitalkResponseError(`Unitalk request failed (${response.status})`, response.status, url.pathname)
  }

  if (!isRecord(body)) {
    throw new UnitalkResponseError('Invalid Unitalk response', response.status, url.pathname)
  }

  return body
}

function instanceOrigin(webDomain: unknown): string {
  if (typeof webDomain !== 'string') {
    throw new Error('Invalid Hermes web domain')
  }

  const destination = new URL(`https://${webDomain}`)

  if (
    destination.host !== webDomain.toLowerCase() ||
    !destination.hostname.endsWith('.unitalk.ai') ||
    destination.port ||
    destination.username ||
    destination.password ||
    destination.pathname !== '/' ||
    destination.search ||
    destination.hash
  ) {
    throw new Error('Invalid Hermes web domain')
  }

  return destination.origin
}

export async function getUnitalkInstance(fetcher: typeof fetch, signal: AbortSignal): Promise<UnitalkInstance | null> {
  const { instance } = await requestJson(fetcher, new URL('/api/instance', UNITALK_ORIGIN), signal)

  if (instance == null) {
    return null
  }

  if (
    !isRecord(instance) ||
    instance.app !== 'hermes' ||
    typeof instance._id !== 'string' ||
    !instance._id ||
    typeof instance.ownerId !== 'string' ||
    typeof instance.userId !== 'string' ||
    typeof instance.status !== 'string'
  ) {
    throw new Error('Invalid Hermes instance response')
  }

  const account = {
    id: instance._id,
    ownerId: instance.ownerId,
    userId: instance.userId
  }

  if (instance.status !== 'ready') {
    return { ...account, ready: false }
  }

  return { ...account, origin: instanceOrigin(instance.webDomain), ready: true }
}

export async function initializeUnitalkInstance(
  fetcher: typeof fetch,
  sessionId: string,
  instance: ReadyUnitalkInstance
): Promise<string> {
  const url = new URL('/api/auth-init', UNITALK_ORIGIN)
  url.search = new URLSearchParams({ instanceId: instance.id, session: sessionId }).toString()
  const body = await requestJson(fetcher, url)

  if (typeof body.url !== 'string') {
    throw new Error('Unitalk did not return an instance URL')
  }

  const destination = new URL(body.url)

  if (destination.origin !== instance.origin || destination.username || destination.password) {
    throw new Error('Unitalk returned a different instance URL')
  }

  return destination.href
}
