import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { AppWindow } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ComposerAttachment } from '@/store/composer'
import { $dockedWindow } from '@/store/dock'

const POLL_INTERVAL_MS = 1500
const POLL_MAX_INTERVAL_MS = 15000

/**
 * Live preview of an attached desktop window. Polls the sidecar
 * (hermesDesktop.captureWindow) on an interval and renders the latest frame, so
 * the user can watch the agent drive the window without leaving Hermes.
 *
 * Each poll spawns a sidecar capture, so cost matters: polling chains via
 * setTimeout (a slow capture never overlaps), PAUSES while the document is
 * hidden (no work for a backgrounded window), and BACKS OFF exponentially on
 * failure so a closed/missing target isn't hammered every 1.5s.
 */
export function WindowPreview({ attachment, onRemove }: { attachment: ComposerAttachment; onRemove?: (id: string) => void }) {
  const { t } = useI18n()
  const c = t.composer
  const title = attachment.label
  const hwnd = attachment.hwnd
  const [frame, setFrame] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const aliveRef = useRef(true)

  useEffect(() => {
    aliveRef.current = true
    const capture = window.hermesDesktop?.captureWindow

    if (!capture || !hwnd) {
      setFailed(true)

      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined
    let delay = POLL_INTERVAL_MS

    const schedule = () => {
      if (aliveRef.current) {
        timer = setTimeout(() => void tick(), delay)
      }
    }

    const tick = async () => {
      // Don't spend a sidecar process on a backgrounded window — recheck cheaply.
      if (typeof document !== 'undefined' && document.hidden) {
        delay = POLL_INTERVAL_MS
        schedule()

        return
      }

      try {
        const img = await capture(hwnd)

        if (!aliveRef.current) {
          return
        }

        if (img) {
          setFrame(img)
          setFailed(false)
          delay = POLL_INTERVAL_MS
        } else {
          setFailed(true)
          delay = Math.min(delay * 2, POLL_MAX_INTERVAL_MS)
        }
      } catch {
        if (aliveRef.current) {
          setFailed(true)
          delay = Math.min(delay * 2, POLL_MAX_INTERVAL_MS)
        }
      } finally {
        schedule()
      }
    }

    void tick()

    return () => {
      aliveRef.current = false

      if (timer) {
        clearTimeout(timer)
      }
    }
  }, [hwnd])

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

/** Render a live preview for each attached window (usually one). The currently
 *  docked window is skipped — it's already tiled beside Hermes, so a thumbnail
 *  would be redundant. */
export function WindowPreviews({
  attachments,
  onRemove
}: {
  attachments: ComposerAttachment[]
  onRemove?: (id: string) => void
}) {
  const docked = useStore($dockedWindow)
  const windows = attachments.filter(a => a.kind === 'window' && a.label !== docked)

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
