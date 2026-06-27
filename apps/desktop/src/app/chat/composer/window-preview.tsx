import { useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { AppWindow } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ComposerAttachment } from '@/store/composer'

const POLL_INTERVAL_MS = 1500

/**
 * Live preview of an attached desktop window. Polls the sidecar
 * (hermesDesktop.captureWindow) on an interval and renders the latest frame, so
 * the user can watch the agent drive the window without leaving Hermes. Polling
 * chains via setTimeout (not setInterval) so a slow capture never overlaps.
 */
export function WindowPreview({ attachment, onRemove }: { attachment: ComposerAttachment; onRemove?: (id: string) => void }) {
  const { t } = useI18n()
  const c = t.composer
  const title = attachment.label
  const [frame, setFrame] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const aliveRef = useRef(true)

  useEffect(() => {
    aliveRef.current = true
    const capture = window.hermesDesktop?.captureWindow

    if (!capture) {
      setFailed(true)

      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    const tick = async () => {
      try {
        const img = await capture(title)

        if (!aliveRef.current) {
          return
        }

        if (img) {
          setFrame(img)
          setFailed(false)
        } else {
          setFailed(true)
        }
      } catch {
        if (aliveRef.current) {
          setFailed(true)
        }
      } finally {
        if (aliveRef.current) {
          timer = setTimeout(() => void tick(), POLL_INTERVAL_MS)
        }
      }
    }

    void tick()

    return () => {
      aliveRef.current = false

      if (timer) {
        clearTimeout(timer)
      }
    }
  }, [title])

  return (
    <div className="relative mx-1 mt-1 overflow-hidden rounded-xl border border-border/60 bg-background/40">
      <div className="flex items-center gap-1.5 px-2.5 py-1.5 text-[0.68rem] text-(--ui-text-tertiary)">
        <AppWindow className="size-3" />
        <span className="min-w-0 flex-1 truncate font-medium text-foreground/80">{title}</span>
        <span className="flex items-center gap-1">
          <span
            className={cn('size-1.5 rounded-full', frame && !failed ? 'animate-pulse bg-emerald-500' : 'bg-muted-foreground/50')}
          />
          {c.windowPreviewLive}
        </span>
        {onRemove && (
          <button
            aria-label={c.removeAttachment(title)}
            className="ml-1 grid size-4 place-items-center rounded text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={() => onRemove(attachment.id)}
            type="button"
          >
            <Codicon name="close" size="0.7rem" />
          </button>
        )}
      </div>
      <div className="grid max-h-56 min-h-24 place-items-center bg-muted/20 p-1.5">
        {frame ? (
          <img alt={title} className="max-h-52 w-auto rounded-md object-contain" draggable={false} src={frame} />
        ) : (
          <span className="px-3 py-6 text-center text-[0.7rem] text-(--ui-text-tertiary)">
            {failed ? c.windowPreviewUnavailable : c.windowPickerLoading}
          </span>
        )}
      </div>
    </div>
  )
}

/** Render a live preview for each attached window (usually one). */
export function WindowPreviews({
  attachments,
  onRemove
}: {
  attachments: ComposerAttachment[]
  onRemove?: (id: string) => void
}) {
  const windows = attachments.filter(a => a.kind === 'window')

  if (windows.length === 0) {
    return null
  }

  return (
    <>
      {windows.map(a => (
        <WindowPreview attachment={a} key={a.id} onRemove={onRemove} />
      ))}
    </>
  )
}
