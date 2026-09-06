export type ServiceMutationConfirmation = 'RESTART' | 'UPDATE'

export interface ServiceMutationRequest {
  confirmation: ServiceMutationConfirmation
  idempotency_key: string
}

function randomUuid(): string {
  const cryptoObj = globalThis.crypto

  if (typeof cryptoObj?.randomUUID === 'function') {
    return cryptoObj.randomUUID()
  }

  const bytes = cryptoObj.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')

  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function serviceMutationRequest(confirmation: ServiceMutationConfirmation): ServiceMutationRequest {
  return {
    confirmation,
    idempotency_key: randomUuid()
  }
}
