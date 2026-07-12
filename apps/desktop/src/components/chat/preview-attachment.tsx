import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { MonitorPlay } from '@/lib/icons'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { previewName } from '@/lib/preview-targets'
import { notifyError } from '@/store/notifications'
import {
  $previewTarget,
  dismissPreviewTarget,
  type PreviewRecordSource,
  setCurrentSessionPreviewTarget
} from '@/store/preview'
import { $currentCwd } from '@/store/session'

export function PreviewAttachment({ source = 'manual', target }: { source?: PreviewRecordSource; target: string }) {
  const { t } = useI18n()
  const cwd = useStore($currentCwd)
  const activePreview = useStore($previewTarget)
  const [opening, setOpening] = useState(false)
  const activePreviewRef = useRef(activePreview)
  const cwdRef = useRef(cwd)
  const mountedRef = useRef(false)
  const requestTokenRef = useRef(0)
  const targetRef = useRef(target)
  const name = previewName(target)
  const isActive = activePreview?.source === target

  activePreviewRef.current = activePreview
  cwdRef.current = cwd
  targetRef.current = target

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false
      requestTokenRef.current += 1
    }
  }, [])

  useEffect(() => {
    requestTokenRef.current += 1
    setOpening(false)
  }, [cwd, target])

  async function togglePreview() {
    if (opening) {
      return
    }

    if (isActive) {
      dismissPreviewTarget()

      return
    }

    const requestToken = ++requestTokenRef.current
    const requestTarget = target
    const requestCwd = cwd

    setOpening(true)

    try {
      const preview = await normalizeOrLocalPreviewTarget(requestTarget, requestCwd || undefined)

      if (
        !mountedRef.current ||
        requestTokenRef.current !== requestToken ||
        targetRef.current !== requestTarget ||
        cwdRef.current !== requestCwd
      ) {
        return
      }

      if (!preview) {
        throw new Error(`Could not open preview target: ${requestTarget}`)
      }

      const currentPreview = activePreviewRef.current

      if (currentPreview?.source === preview.source && currentPreview.url === preview.url) {
        return
      }

      setCurrentSessionPreviewTarget(preview, source, requestTarget)
    } catch (error) {
      if (
        !mountedRef.current ||
        requestTokenRef.current !== requestToken ||
        targetRef.current !== requestTarget ||
        cwdRef.current !== requestCwd
      ) {
        return
      }

      notifyError(error, t.preview.unavailable)
    } finally {
      if (mountedRef.current && requestTokenRef.current === requestToken) {
        setOpening(false)
      }
    }
  }

  return (
    <div className="flex w-full max-w-160 items-center gap-2 rounded-lg border border-border/55 bg-card/55 px-2.5 py-1.5 text-sm">
      <span className="grid size-6 shrink-0 place-items-center rounded-md bg-muted/55 text-muted-foreground/85">
        <MonitorPlay className="size-3.5" />
      </span>
      <span className="min-w-0 flex-1 truncate text-[0.78rem] font-medium text-foreground/90" title={target}>
        {name}
      </span>
      <button
        className="shrink-0 rounded-md border border-border/55 bg-background/40 px-2 py-1 text-[0.7rem] font-medium text-muted-foreground transition-colors hover:bg-accent/55 hover:text-foreground disabled:opacity-50"
        disabled={opening}
        onClick={() => void togglePreview()}
        type="button"
      >
        {opening ? t.preview.opening : isActive ? t.preview.hide : t.preview.openPreview}
      </button>
    </div>
  )
}
