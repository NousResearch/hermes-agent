export interface SendEnvelope {
  readonly clientMessageId: string
  /** Unix seconds captured when Desktop accepted the send or local enqueue. */
  readonly submittedAt: number
}

const nextClientMessageId = (nowMs: number): string => {
  const uuid = globalThis.crypto?.randomUUID?.()

  return uuid ?? `message-${nowMs}-${Math.random().toString(36).slice(2, 10)}`
}

/** Immutable identity/time carried unchanged through every retry and queue hop. */
export function createSendEnvelope(nowMs = Date.now()): SendEnvelope {
  return Object.freeze({
    clientMessageId: nextClientMessageId(nowMs),
    submittedAt: nowMs / 1000
  })
}
