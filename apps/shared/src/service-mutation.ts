export type ServiceMutationConfirmation = 'RESTART' | 'UPDATE'

export interface ServiceMutationRequest {
  confirmation: ServiceMutationConfirmation
  idempotency_key: string
}

export function serviceMutationRequest(confirmation: ServiceMutationConfirmation): ServiceMutationRequest {
  return {
    confirmation,
    idempotency_key: globalThis.crypto.randomUUID()
  }
}
