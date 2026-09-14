import { useEffect, useState } from 'react'

import { ToolImageGallery } from '@/components/assistant-ui/tool/image-previews'
import { useI18n } from '@/i18n'
import { type DesktopFileOrigin } from '@/lib/desktop-fs'
import { toolImageSources } from '@/lib/tool-images'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'

import { rejectUnownedSubagentRequest } from './use-subagent-snapshot'

interface Tail {
  available: boolean
  text: string
  truncated: boolean
  images?: string[]
  image_session_id?: string
  images_truncated?: boolean
  image_revision?: number
}

interface TailView {
  key: string
  owner: string | undefined
  available: boolean
  text: string
  truncated: boolean
  images: string[]
  imagesTruncated: boolean
  imageRevision: number
  origin?: DesktopFileOrigin
}

export function SubagentTranscript({ sessionId, subagentId }: { sessionId: string; subagentId: string }) {
  const { t } = useI18n()
  const [snapshot, setSnapshot] = useState<TailView | null>(null)
  const key = JSON.stringify([sessionId, subagentId])
  const currentOwner = JSON.stringify(knownOwnerForSession(sessionId))
  const tail = snapshot?.key === key && snapshot.owner === currentOwner ? snapshot : null

  useEffect(() => {
    let cancelled = false
    let pending = false

    const refresh = async () => {
      if (cancelled || pending || document.visibilityState === 'hidden') {
        return
      }

      pending = true
      const owner = knownOwnerForSession(sessionId)
      const ownerKey = JSON.stringify(owner)

      try {
        const result = await requestForOwnedSession<Tail>(sessionId, rejectUnownedSubagentRequest, 'subagent.tail', {
          session_id: sessionId,
          subagent_id: subagentId
        })

        if (cancelled) {
          return
        }

        if (ownerKey !== JSON.stringify(knownOwnerForSession(sessionId))) {
          setSnapshot(null)

          return
        }

        const childSessionId = typeof result.image_session_id === 'string' ? result.image_session_id.trim() : ''
        const profile = typeof owner === 'string' ? owner : owner?.targetProfile || owner?.profile

        // The authorized RPC supplies the child's file scope; the commissioning
        // session supplies the gateway route. Never substitute the parent's cwd.
        const origin =
          childSessionId && profile
            ? {
                sessionId: childSessionId,
                connectionId: typeof owner === 'string' ? undefined : owner?.connectionId,
                profile
              }
            : undefined

        const images = origin ? toolImageSources({}, { images: result.images }) : []
        setSnapshot(previous => ({
          key,
          owner: ownerKey,
          origin,
          available: Boolean(result.available),
          text: typeof result.text === 'string' ? result.text.slice(-16384) : '',
          truncated: Boolean(result.truncated),
          imagesTruncated: Boolean(result.images_truncated),
          imageRevision:
            Number.isSafeInteger(result.image_revision) && result.image_revision! >= 0 ? result.image_revision! : 0,
          // Text progresses every poll; unchanged images must not cancel and
          // restart in-flight reads or invalidate the gallery's page cache.
          images:
            previous?.key === key &&
            previous.owner === ownerKey &&
            previous.origin?.sessionId === childSessionId &&
            previous.images.length === images.length &&
            images.every((source, index) => source === previous.images[index])
              ? previous.images
              : images
        }))
      } catch {
        if (!cancelled) {
          setSnapshot({
            key,
            owner: ownerKey,
            available: false,
            text: '',
            truncated: false,
            images: [],
            imagesTruncated: false,
            imageRevision: 0
          })
        }
      } finally {
        pending = false
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), 2000)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [key, sessionId, subagentId])

  return (
    <section className="mt-2 text-xs" data-slot="subagent-transcript">
      {tail?.imagesTruncated && <p className="text-(--ui-text-tertiary)">{t.agents.imagePreviewsTruncated}</p>}
      {tail?.origin && tail.images.length > 0 && (
        <>
          <h4 className="text-(--ui-text-secondary)">{t.desktop.imageGallery}</h4>

          <ToolImageGallery
            active
            compact
            context={{
              sessionId: tail.origin.sessionId,
              runtimeId: sessionId,
              origin: tail.origin,
              revision: tail.imageRevision
            }}
            key={JSON.stringify([key, tail.owner, tail.origin])}
            sources={tail.images}
            toolCallId={`subagent:${subagentId}`}
          />
        </>
      )}
      <h4 className="text-(--ui-text-secondary)">{t.agents.extendedTranscript}</h4>
      {tail?.truncated && <p className="text-(--ui-text-tertiary)">{t.agents.transcriptTruncated}</p>}
      {tail?.available ? (
        <pre className="max-h-[30vh] overflow-auto whitespace-pre-wrap break-words font-mono text-[0.68rem]">
          {tail.text}
        </pre>
      ) : (
        <p className="text-(--ui-text-tertiary)">{tail ? t.agents.transcriptUnavailable : t.agents.waitingActivity}</p>
      )}
    </section>
  )
}
