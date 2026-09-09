import { type ReactNode, type RefObject, useEffect } from 'react'

import { publishThreadMessagesBelow } from '@/store/thread-scroll'

const MESSAGE_ROOTS =
  '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"], [data-slot="aui_system-message-root"]'

interface MessagesBelowOptions {
  contentRef: RefObject<HTMLElement | null>
  scrollRef: RefObject<HTMLElement | null>
  isAtBottom: boolean
  paneVisible: boolean
  rows: ReactNode
  sessionKey: string | null | undefined
}

/** Only measure inside the viewport's turn; leave skipped off-screen content asleep. */
export function countMessagesBelow(viewport: HTMLElement, content: HTMLElement): number {
  const bottom = viewport.getBoundingClientRect().bottom
  let count = 0

  for (const group of content.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')) {
    const rect = group.getBoundingClientRect()

    if (rect.bottom <= bottom + 1) {
      continue
    }

    const messages = group.querySelectorAll<HTMLElement>(MESSAGE_ROOTS)

    if (rect.top >= bottom) {
      count += messages.length

      continue
    }

    for (const message of messages) {
      const messageRect = message.getBoundingClientRect()

      if (messageRect.height > 0 && messageRect.bottom > bottom + 1) {
        count++
      }
    }
  }

  return count
}

export function useMessagesBelow({
  contentRef,
  scrollRef,
  isAtBottom,
  paneVisible,
  rows,
  sessionKey
}: MessagesBelowOptions) {
  useEffect(() => {
    if (!paneVisible) {
      return
    }

    if (isAtBottom) {
      publishThreadMessagesBelow(0, { paneVisible })

      return
    }

    const viewport = scrollRef.current
    const content = contentRef.current

    if (!viewport || !content) {
      return
    }

    let frame = 0

    const measure = () => {
      frame = 0
      publishThreadMessagesBelow(countMessagesBelow(viewport, content), { paneVisible })
    }

    const schedule = () => {
      if (!frame) {
        frame = requestAnimationFrame(measure)
      }
    }

    schedule()
    viewport.addEventListener('scroll', schedule, { passive: true })
    const observer = new ResizeObserver(schedule)
    observer.observe(viewport)
    observer.observe(content)

    return () => {
      cancelAnimationFrame(frame)
      viewport.removeEventListener('scroll', schedule)
      observer.disconnect()
    }
  }, [contentRef, scrollRef, isAtBottom, paneVisible, rows, sessionKey])
}
