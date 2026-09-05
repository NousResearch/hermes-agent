import { useStore } from '@nanostores/react'
import { type ComponentProps, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useI18n } from '@/i18n'
import { openLink } from '@/lib/external-link'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { cn } from '@/lib/utils'
import { $requireModifierToOpenInlineLinks, INLINE_LINK_GATED_ATTR, shouldOpenInlineLink } from '@/store/inline-link-open'
import { notifyError } from '@/store/notifications'
import { openPreview } from '@/store/preview'

/**
 * A bare path the agent mentioned in prose or dropped in a code span, made
 * clickable. Reads as the text the agent wrote (the path), opens the preview
 * pane on click, and — because it is a real `<a href>` — the app context menu
 * resolves a right-click on it to the link menu, so Copy comes for free.
 *
 * The href is the raw path: `resolveDomTarget` reads it back as the menu's
 * `linkUrl`, and `PreviewAttachment`-style resolution happens at CLICK time
 * against the session's cwd/backend, never at render.
 */
export function InlinePathLink({
  children,
  className,
  path,
  ...props
}: Omit<ComponentProps<'a'>, 'href' | 'onClick'> & { path: string }) {
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

      openPreview(preview, 'explicit-link')
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

        if (!shouldOpenInlineLink(event)) {
          return
        }

        void open()
      }}
      title={path}
      {...props}
      {...{ [INLINE_LINK_GATED_ATTR]: '' }}
    >
      {children ?? path}
    </a>
  )
}

/**
 * A URL that filled a code span (`http://localhost:8931/`). Same routing as
 * a prose chat URL: in-app browser when the click is allowed to open,
 * middle-click for the OS browser, right-click for the link menu. When the
 * appearance preference is on, a regular mouse click does not open; ⌘/Ctrl
 * opens in-app rather than the OS browser.
 */
export function InlineUrlLink({
  children,
  className,
  url,
  ...props
}: Omit<ComponentProps<'a'>, 'href' | 'onClick'> & { url: string }) {
  return (
    <a
      className={cn('ref wrap-anywhere', className)}
      data-ref="url"
      href={url}
      onAuxClick={event => {
        if (event.button !== 1) {
          return
        }

        event.preventDefault()
        event.stopPropagation()
        openLink(url, { native: true })
      }}
      onClick={event => {
        event.preventDefault()
        event.stopPropagation()

        if (!shouldOpenInlineLink(event)) {
          return
        }

        openLink(url, {
          native: !$requireModifierToOpenInlineLinks.get() && (event.metaKey || event.ctrlKey)
        })
      }}
      rel="noopener noreferrer"
      target="_blank"
      title={url}
      {...props}
      {...{ [INLINE_LINK_GATED_ATTR]: '' }}
    >
      {children ?? url}
    </a>
  )
}
