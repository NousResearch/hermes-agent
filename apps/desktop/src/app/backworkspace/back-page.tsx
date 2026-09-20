import { useStore } from '@nanostores/react'
import { useEffect, useLayoutEffect, useRef } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { type Translations, useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'

import { BackworkspaceEditor } from './editor'
import { imagePreviews, imageTheme } from './image-previews'
import { liveEditor } from './live-editor'
import {
  $backworkspacePage,
  type BackworkspacePageState,
  editBackworkspacePage,
  flushBackworkspacePage,
  loadBackworkspacePage
} from './page'
import { quoteDecorationPlugin, quoteTheme } from './quote-decorations'
import { $backworkspaceOpen, toggleBackworkspace } from './store'
import { useApprovalCard } from './use-approval-card'
import { useAskAgent } from './use-ask-agent'
import { useMentionPopup } from './use-mention-popup'
import { usePasteImage } from './use-paste-image'

// The notice sits under the same centered column the editor paints (see editor.tsx).
const TEXT_COLUMN_CLASS = 'w-full px-[max(2rem,calc((100%_-_var(--bw-measure))/2))]'

function noticeFor(page: BackworkspacePageState | null, copy: Translations['backworkspace']): null | string {
  if (page?.status === 'unsupported') {
    return copy.unsupported
  }

  if (page?.status === 'error') {
    return copy.loadFailed
  }

  return page?.saveFailed ? copy.saveFailed : null
}

/** The page on the back of the window. Mounted only while the window is turned over. */
export function BackworkspacePage() {
  const open = useStore($backworkspaceOpen)

  return open ? <BackworkspaceSheet /> : null
}

function BackworkspaceSheet() {
  const { t } = useI18n()
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const page = useStore($backworkspacePage)
  const sheetRef = useRef<HTMLElement>(null)
  const ready = page?.status === 'ready'
  const mention = useMentionPopup()
  const ask = useAskAgent(page?.path ?? null)
  const paste = usePasteImage()
  const approval = useApprovalCard()
  // The approval speaks first: it is the only line that is waiting on the
  // reader rather than reporting on the page.
  // Only while there is an editor to press the chord on.
  const notice = (ready ? approval.notice : null) ?? ask.notice ?? paste.notice ?? noticeFor(page, t.backworkspace)

  // While this is mounted styles.css hides the shell. Tying the attribute to
  // this component's life means anything that unmounts it — turning back, or
  // an error boundary replacing the tree — shows the shell again. Focus moves
  // here at once so no key reaches the hidden composer, even while loading.
  useLayoutEffect(() => {
    document.documentElement.setAttribute('data-backworkspace', '')
    sheetRef.current?.focus({ preventScroll: true })

    return () => document.documentElement.removeAttribute('data-backworkspace')
  }, [])

  // The page belongs to the profile this window shows; switching profile or
  // connection while turned over shows that owner's page instead.
  useEffect(() => {
    void loadBackworkspacePage({ connectionId, profile })
  }, [connectionId, profile])

  // Closing the window inside the save debounce still sends the last keystrokes.
  useEffect(() => {
    const flush = () => void flushBackworkspacePage()

    window.addEventListener('pagehide', flush)

    return () => window.removeEventListener('pagehide', flush)
  }, [])

  return (
    <section
      aria-label={t.backworkspace.label}
      className="visible fixed inset-0 z-(--z-backworkspace) flex flex-col bg-(--bw-paper) text-(--bw-ink) outline-none"
      data-backworkspace-page=""
      // Paints its own field under window glass, so the shell's rail seam never
      // shows through, and owns the keyboard like any overlay: the hidden
      // composer's type-to-focus and Esc-cancel stand down while it is here.
      data-glass-opaque=""
      data-overlay-surface=""
      onKeyDown={event => {
        // A surface inside the page may own the key first — the mention list
        // closes on Escape and marks it handled, and the window stays put.
        if (event.key === 'Escape' && !event.defaultPrevented && !event.nativeEvent.isComposing) {
          event.preventDefault()
          void toggleBackworkspace()
        }
      }}
      ref={sheetRef}
      tabIndex={-1}
    >
      {/* No titlebar on this side: the band only keeps the window draggable. */}
      <div aria-hidden className="shrink-0 [-webkit-app-region:drag]" style={{ height: TITLEBAR_HEIGHT }} />
      {ready ? (
        <BackworkspaceEditor
          ariaLabel={t.backworkspace.label}
          autoFocus
          extensions={[
            liveEditor(page.key),
            mention.extension,
            approval.extension,
            ask.extension,
            paste.extension,
            imagePreviews(page.path),
            imageTheme,
            quoteDecorationPlugin,
            quoteTheme
          ]}
          initialValue={page.content}
          key={page.key}
          onChange={editBackworkspacePage}
        />
      ) : (
        <div className="min-h-0 flex-1" />
      )}
      {mention.popover}
      {approval.card}
      {notice && (
        <p className={cn(TEXT_COLUMN_CLASS, 'pb-4 text-xs text-(--bw-ink-quiet)')} role="status">
          {notice}
        </p>
      )}
    </section>
  )
}
