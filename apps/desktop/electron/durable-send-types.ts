// Native outbox contract. No filesystem paths, credentials or ambient UI destination.
export interface DurableSendOwner {
  connectionId: string
  gatewayId: string
  profile: string
  principalId: string
}

export type DurableSendTarget =
  | { sessionId: string }
  | { draftId: string; createRequestId: string; params: Record<string, string | number | boolean> }

export interface DurableSendFile {
  id: string
  name: string
  mediaType: string
  size: number
  sha256: string
}

export interface DurableSendIntent {
  id: string
  owner: DurableSendOwner
  target: DurableSendTarget
  text: string
  files: DurableSendFile[]
}

export type DurableSendState = 'staging' | 'queued' | 'sending' | 'waiting' | 'blocked' | 'confirmed' | 'cancelled'

export interface DurableSendRecord {
  version: 1
  intent: DurableSendIntent
  state: DurableSendState
  createdAt: number
  attempts: number
  retryAt: number
  sessionId?: string
  uploads: Record<string, { reference: string; sha256: string; size: number }>
  submitStarted: boolean
  cancelPending?: boolean
  admissionId?: string
  problem?: 'connection' | 'capability' | 'integrity' | 'uncertain' | 'refused'
}

export interface DurableTransportRequest {
  token: string
  leaseId?: string
  route: Pick<DurableSendOwner, 'connectionId' | 'profile'>
  method: string
  params: Record<string, unknown>
}

export interface DurableSendApi {
  negotiate: (route: Pick<DurableSendOwner, 'connectionId' | 'profile'>) => Promise<DurableSendOwner>
  pump: () => Promise<void>
  onChange?: (handler: () => void) => () => void
  onTransportRequest: (handler: (request: DurableTransportRequest) => Promise<unknown>) => () => void
  begin: (intent: DurableSendIntent) => Promise<DurableSendRecord>
  append: (id: string, fileId: string, offset: number, bytes: Uint8Array) => Promise<number>
  commit: (id: string) => Promise<DurableSendRecord>
  list: () => Promise<DurableSendRecord[]>
  retired?: (ids: string[]) => Promise<Record<string, 'confirmed' | 'cancelled'>>
  cancel: (id: string) => Promise<DurableSendRecord>
}
