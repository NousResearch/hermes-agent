import { createContext, useContext } from 'react'

import { useSessionView } from '@/app/chat/session-view'

export interface TranscriptIdentity {
  cwd: string
  runtimeId: null | string
}

interface TranscriptRuntimeExtras {
  transcriptIdentity?: TranscriptIdentity
}

export function transcriptIdentityFromRuntimeExtras(extras: unknown): TranscriptIdentity | null {
  if (!extras || typeof extras !== 'object') {
    return null
  }

  const identity = (extras as TranscriptRuntimeExtras).transcriptIdentity

  return identity &&
    typeof identity.cwd === 'string' &&
    (identity.runtimeId === null || typeof identity.runtimeId === 'string')
    ? identity
    : null
}

const TranscriptIdentityContext = createContext<TranscriptIdentity | null>(null)

export const TranscriptIdentityProvider = TranscriptIdentityContext.Provider

export function useTranscriptIdentity(): TranscriptIdentity {
  const identity = useContext(TranscriptIdentityContext)
  const view = useSessionView()

  return identity ?? { cwd: view.$cwd.get() || '', runtimeId: view.$runtimeId.get() }
}
