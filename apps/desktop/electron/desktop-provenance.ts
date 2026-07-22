import crypto from 'node:crypto'

export const DIRECT_ACTION_PROVENANCE_VERSION = 1
export const DIRECT_ACTION_GESTURE_TTL_MS = 5_000
export const DIRECT_ACTION_RECOVERY_TTL_MS = 120_000

export interface DesktopPromptPayload {
  version: 1
  event_id: string
  issued_at: string
  installation_id: string
  os_account: string
  app_identity: string
  app_instance_id: string
  profile: string
  window_id: string
  session_id: string
  text_hash: string
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalValue)
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
        .map(([key, child]) => [key, canonicalValue(child)])
    )
  }

  return value
}

export function canonicalDesktopPayload(payload: DesktopPromptPayload): Buffer {
  return Buffer.from(JSON.stringify(canonicalValue(payload)), 'utf8')
}

export function desktopTextHash(text: string): string {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex')
}

export function desktopPublicKeyFingerprint(publicKeyPem: string): string {
  const der = crypto.createPublicKey(publicKeyPem).export({ type: 'spki', format: 'der' })

  return crypto.createHash('sha256').update(der).digest('hex')
}

export function generateDesktopSigningIdentity(): { privateKeyPem: string; publicKeyPem: string } {
  const { privateKey, publicKey } = crypto.generateKeyPairSync('ed25519')

  return {
    privateKeyPem: privateKey.export({ type: 'pkcs8', format: 'pem' }).toString(),
    publicKeyPem: publicKey.export({ type: 'spki', format: 'pem' }).toString()
  }
}

export function signDesktopPayload(privateKeyPem: string, payload: DesktopPromptPayload): string {
  return crypto.sign(null, canonicalDesktopPayload(payload), privateKeyPem).toString('base64url')
}

export function verifyDesktopPayload(publicKeyPem: string, payload: DesktopPromptPayload, signature: string): boolean {
  try {
    return crypto.verify(null, canonicalDesktopPayload(payload), publicKeyPem, Buffer.from(signature, 'base64url'))
  } catch {
    return false
  }
}

interface RecoveryReceipt {
  eventId: string
  expiresAt: number
  profile: string
  textHash: string
}

interface TrustedGesture {
  expiresAt: number
  textHash: string
}

/**
 * A native trusted gesture mints one durable event identity. Transport/session
 * recovery may re-sign that same identity for the same text/profile, but can
 * never turn it into authority for different content.
 */
export class TrustedGestureLedger {
  private readonly gestures = new Map<number, TrustedGesture>()
  private readonly recoveries = new Map<number, RecoveryReceipt>()

  note(webContentsId: number, textHash: string, now = Date.now()): void {
    if (!textHash) {
      return
    }

    this.gestures.set(webContentsId, {
      expiresAt: now + DIRECT_ACTION_GESTURE_TTL_MS,
      textHash
    })
  }

  eventIdFor(webContentsId: number, textHash: string, profile: string, now = Date.now()): string | null {
    const gesture = this.gestures.get(webContentsId)

    if (gesture && gesture.expiresAt >= now && gesture.textHash === textHash) {
      this.gestures.delete(webContentsId)
      const eventId = crypto.randomUUID()

      this.recoveries.set(webContentsId, {
        eventId,
        expiresAt: now + DIRECT_ACTION_RECOVERY_TTL_MS,
        profile,
        textHash
      })

      return eventId
    }

    const recovery = this.recoveries.get(webContentsId)

    if (recovery && recovery.expiresAt >= now && recovery.profile === profile && recovery.textHash === textHash) {
      return recovery.eventId
    }

    return null
  }
}
