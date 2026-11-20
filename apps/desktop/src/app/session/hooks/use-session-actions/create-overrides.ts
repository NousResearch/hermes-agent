import type { ResolvedOwner } from '@/api/client'

export interface VoiceDraftDestination {
  backendProfile: string
  owner: ResolvedOwner
  runtimeSessionId: string
  storedSessionId: string
}

export interface SessionCreateOverrides {
  /** Renderer-only handoff, fired at the stored-id assignment before navigation. */
  onComposerScopeAssigned?: (scope: string) => void
  voiceDraft?: {
    key: string
    isCurrent: () => boolean
    onAssigned: (destination: VoiceDraftDestination) => void
  }
}

export type CreateBackendSessionForSend = (
  preview?: string | null,
  createOverrides?: SessionCreateOverrides
) => Promise<string | null>
