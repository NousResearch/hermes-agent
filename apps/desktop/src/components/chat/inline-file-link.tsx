import { useStore } from '@nanostores/react'
import { type ReactNode, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useI18n } from '@/i18n'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { previewName } from '@/lib/preview-targets'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { openPreview } from '@/store/preview'

/**
 * A file the agent named mid-sentence, as link text rather than a card: a
 * block card inside a paragraph splits the sentence in two. Click opens the
 * preview pane, resolved at CLICK time against this session's cwd/backend
 * (local reads the file; remote goes over the /api/fs bridge) — the same
 * resolution PreviewAttachment uses, so the transcript works from any machine.
 * Download stays one click away inside the preview pane.
 */
export function InlineFileLink({
  children,
  className,
  path
}: {
  children?: ReactNode
  className?: string
  path: string
}) {
  const { t } = useI18n()
  const cwd = useStore(useSessionView().$cwd)
  const [opening, setOpening] = useState(false)
  const pendingRef = useRef(false)

  async function open() {
    if (pendingRef.current) {
      return
    }

    pendingRef.current = true
    setOpening(true)

    try {
      const preview = await normalizeOrLocalPreviewTarget(path, cwd || undefined)

      if (!preview) {
        throw new Error(`Could not open preview target: ${path}`)
      }

      openPreview(preview)
    } catch (error) {
      notifyError(error, t.preview.unavailable)
    } finally {
      pendingRef.current = false
      setOpening(false)
    }
  }

  return (
    <a
      aria-busy={opening || undefined}
      className={cn(
        'wrap-anywhere text-(--dt-primary) underline-offset-2 hover:underline',
        opening && 'opacity-70',
        className
      )}
      data-inline-file-link=""
      href={path}
      onClick={event => {
        event.preventDefault()
        event.stopPropagation()
        void open()
      }}
      title={path}
    >
      {children ?? previewName(path)}
    </a>
  )
}
