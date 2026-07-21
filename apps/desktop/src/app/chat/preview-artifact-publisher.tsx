import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { memo, useEffect, useMemo } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import {
  isPreviewableTarget,
  isSuccessfulToolPart,
  previewTargetFromToolPart,
  type ToolPart
} from '@/components/assistant-ui/tool/fallback-model'
import type { ChatMessage } from '@/lib/chat-messages'
import {
  clearPreviewArtifacts,
  previewArtifactOwnerId,
  previewArtifactPublicationKey,
  syncPreviewArtifacts
} from '@/store/preview-status'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'

export interface PreviewArtifactPublication {
  publicationId: string
  target: string
}

const publicationByToolPart = new WeakMap<object, PreviewArtifactPublication | null>()

const publicationFromToolPart = (part: ToolPart): PreviewArtifactPublication | null => {
  const cached = publicationByToolPart.get(part)

  if (cached !== undefined) {
    return cached
  }

  if (!part.toolCallId || !isSuccessfulToolPart(part)) {
    publicationByToolPart.set(part, null)

    return null
  }

  const target = previewTargetFromToolPart(part)
  const publication = isPreviewableTarget(target) ? { publicationId: part.toolCallId, target } : null
  publicationByToolPart.set(part, publication)

  return publication
}

export function previewArtifactPublications(messages: readonly ChatMessage[]): PreviewArtifactPublication[] {
  const publications: PreviewArtifactPublication[] = []

  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type !== 'tool-call') {
        continue
      }

      const publication = publicationFromToolPart(part)

      if (publication) {
        publications.push(publication)
      }
    }
  }

  return publications
}

const samePublications = (
  left: readonly PreviewArtifactPublication[],
  right: readonly PreviewArtifactPublication[]
): boolean =>
  left.length === right.length &&
  left.every(
    (publication, index) =>
      publication.publicationId === right[index]?.publicationId && publication.target === right[index]?.target
  )

function PreviewArtifactPublisherComponent({ disabled = false }: { disabled?: boolean }) {
  const sessionView = useSessionView()

  const publicationsStore = useMemo(() => {
    let previous: PreviewArtifactPublication[] = []

    return computed(sessionView.$messages, messages => {
      const next = previewArtifactPublications(messages)

      if (samePublications(previous, next)) {
        return previous
      }

      previous = next

      return next
    })
  }, [sessionView.$messages])

  const publications = useStore(publicationsStore)
  const runtimeId = useStore(sessionView.$runtimeId)
  const storedSessionId = useStore(sessionView.$storedId)
  const cwd = useStore(sessionView.$cwd)
  const sessions = useStore($sessions)
  const connection = useStore($connection)
  const profile = useStore($activeGatewayProfile)

  useEffect(() => {
    if (!runtimeId) {
      return
    }

    return () => clearPreviewArtifacts(runtimeId)
  }, [runtimeId])

  useEffect(() => {
    if (disabled || !runtimeId || !storedSessionId || publications.length === 0) {
      return
    }

    const ownerId = previewArtifactOwnerId(storedSessionId, sessions, profile)

    if (!ownerId) {
      return
    }

    syncPreviewArtifacts(
      runtimeId,
      publications.map(publication => ({
        cwd,
        publicationKey: previewArtifactPublicationKey({
          baseUrl: connection?.baseUrl,
          mode: connection?.mode ?? 'local',
          ownerId,
          profile,
          publicationId: publication.publicationId
        }),
        target: publication.target
      }))
    )
  }, [
    connection?.baseUrl,
    connection?.mode,
    cwd,
    disabled,
    profile,
    publications,
    runtimeId,
    sessions,
    storedSessionId
  ])

  return null
}

export const PreviewArtifactPublisher = memo(PreviewArtifactPublisherComponent)
