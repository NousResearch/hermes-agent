import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ZoomableImage } from '@/components/chat/zoomable-image'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { isInlineMediaSrc, mediaName } from '@/lib/media'
import { openPreview } from '@/store/preview'
import { $sessions, knownSessionOwner } from '@/store/session'

interface ToolImagePreviewsProps {
  active: boolean
  sources: string[]
  toolCallId: string
}

export function ToolImagePreviews({ active, sources, toolCallId }: ToolImagePreviewsProps) {
  const { $storedId } = useSessionView()
  const sessionId = useStore($storedId)

  return (
    <div className={active ? 'grid gap-3 p-2' : 'hidden'} data-slot="tool-image-previews">
      {sources.map((source, index) => (
        <ToolImagePreview
          active={active}
          key={`${sessionId}:${source}`}
          previewId={`tool-image:${sessionId}:${toolCallId}:${index}`}
          sessionId={sessionId}
          source={source}
        />
      ))}
    </div>
  )
}

function ToolImagePreview({
  active,
  previewId,
  sessionId,
  source
}: {
  active: boolean
  previewId: string
  sessionId: string | null
  source: string
}) {
  const { t } = useI18n()
  const [src, setSrc] = useState('')
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const label = source.startsWith('data:') ? t.assistant.tool.outputAlt : mediaName(source)

  useEffect(() => {
    if (!active || src) {
      return
    }

    let current = true

    const load = async () => {
      if (isInlineMediaSrc(source)) {
        return source
      }

      const owner = knownSessionOwner($sessions.get(), sessionId)

      // Never reinterpret another session's path against the currently focused gateway.
      if (!sessionId || !owner) {
        throw new Error('Image source session is unavailable')
      }

      const dataUrl = await readDesktopFileDataUrl(source, {
        sessionId,
        connectionId: typeof owner === 'string' ? undefined : owner.connectionId,
        profile: typeof owner === 'string' ? owner : owner.targetProfile || owner.profile
      })

      if (!dataUrl.startsWith('data:image/')) {
        throw new Error('Not an image')
      }

      return dataUrl
    }

    void load()
      .then(value => {
        if (current) {
          setSrc(value)
        }
      })
      .catch(() => {
        if (current) {
          setFailed(true)
        }
      })

    return () => {
      current = false
    }
  }, [active, attempt, sessionId, source, src])

  return (
    <figure className="m-0 min-w-0 max-w-xl" data-slot="tool-image-preview">
      {src && !failed ? (
        <ZoomableImage
          alt={label}
          className="max-h-80 w-auto max-w-full cursor-zoom-in rounded-md object-contain"
          decoding="async"
          onError={() => setFailed(true)}
          src={src}
        />
      ) : (
        <div className="flex min-h-24 items-center gap-2 text-xs text-muted-foreground" role="status">
          <Codicon name="file-media" />
          {failed ? t.preview.unavailable : t.preview.loading}
          {failed && (
            <Button
              onClick={() => {
                setFailed(false)
                setSrc('')
                setAttempt(value => value + 1)
              }}
              size="xs"
              variant="text"
            >
              {t.common.retry}
            </Button>
          )}
        </div>
      )}
      <figcaption className="mt-1 flex min-w-0 items-center gap-3 text-xs text-muted-foreground">
        <span className="truncate">{label}</span>
        {src && !failed && (
          <Button
            onClick={() =>
              openPreview({
                kind: 'file',
                label,
                source: previewId,
                url: previewId,
                dataUrl: src,
                previewKind: 'image',
                transient: true
              })
            }
            size="xs"
            variant="text"
          >
            {t.preview.openPreview}
          </Button>
        )}
      </figcaption>
    </figure>
  )
}
