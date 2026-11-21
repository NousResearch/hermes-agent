import { useStore } from '@nanostores/react'
import { type ComponentProps, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useI18n } from '@/i18n'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { CHAT_LINK_ATTR, shouldOpenChatLink } from '@/store/chat-link-open'
import { notifyError } from '@/store/notifications'
import { openPreview } from '@/store/preview'

/**
 * A bare path the agent wrote in prose or a code span, made clickable (#48313).
 * Inline rather than a PreviewAttachment card so the sentence still reads as a
 * sentence. It is a real `<a href>`, so the app context menu resolves a
 * right-click to the link menu (copy) for free.
 *
 * The path resolves at CLICK time through the same pipeline as
 * PreviewAttachment (`normalizeOrLocalPreviewTarget`, the session's cwd and
 * backend) and only ever opens the preview pane; nothing is launched.
 */
export function InlinePathLink({
  children,
  className,
  path,
  ...props
}: Omit<ComponentProps<'a'>, 'href' | 'onClick'> & { path: string }) {
  const { t } = useI18n()
  const view = useSessionView()
  const cwd = useStore(view.$cwd)
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const [opening, setOpening] = useState(false)
  const pendingRef = useRef(false)

  async function open() {
    if (pendingRef.current) {
      return
    }

    pendingRef.current = true
    setOpening(true)

    try {
      const preview = await normalizeOrLocalPreviewTarget(path, cwd || undefined, {
        sessionId: storedId || runtimeId || ''
      })

      if (!preview) {
        throw new Error(`${path}: ${t.preview.missingTarget}`)
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
      className={cn('ref wrap-anywhere', opening && 'opacity-70', className)}
      data-ref="file"
      href={path}
      onClick={event => {
        event.preventDefault()
        event.stopPropagation()

        if (shouldOpenChatLink(event)) {
          void open()
        }
      }}
      title={path}
      {...props}
      {...{ [CHAT_LINK_ATTR]: '' }}
    >
      {children ?? path}
    </a>
  )
}
