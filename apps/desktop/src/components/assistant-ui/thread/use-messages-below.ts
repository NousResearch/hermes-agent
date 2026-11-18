import { type ReactNode, type RefObject, useEffect } from 'react'

import { publishThreadMessagesBelow } from '@/store/thread-scroll'

const GROUP_ROOT = '[data-slot="aui_message-group"]'

const MESSAGE_ROOTS =
  '[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"], [data-slot="aui_system-message-root"]'

const COUNTED_STRUCTURE = `${GROUP_ROOT}, ${MESSAGE_ROOTS}`

interface MessagesBelowOptions {
  contentRef: RefObject<HTMLElement | null>
  scrollRef: RefObject<HTMLElement | null>
  isAtBottom: boolean
  paneVisible: boolean
  rows: ReactNode
  sessionKey: string | null | undefined
  sessionId: string | null
}

interface GroupSummary {
  element: HTMLElement
  messageCount: number
  messagesAfter: number
}

const containsCountedStructure = (node: Node): boolean =>
  node instanceof Element && (node.matches(COUNTED_STRUCTURE) || node.querySelector(COUNTED_STRUCTURE) !== null)

/**
 * Cache membership/counts only. Geometry stays live so images, expanded tools
 * and content-visibility can change heights without invalidating an offset
 * table. The transcript's outer message groups are in normal-flow DOM order,
 * so their bottoms are monotonic and the fold can be found with binary search.
 */
export function createMessagesBelowReader(viewport: HTMLElement, content: HTMLElement) {
  let groups: GroupSummary[] | null = null

  const snapshot = (): GroupSummary[] => {
    if (groups) {
      return groups
    }

    const elements = [...content.querySelectorAll<HTMLElement>(GROUP_ROOT)]
    let messagesAfter = 0
    const reversed: GroupSummary[] = []

    for (let index = elements.length - 1; index >= 0; index--) {
      const element = elements[index]
      const messageCount = element.querySelectorAll(MESSAGE_ROOTS).length
      reversed.push({ element, messageCount, messagesAfter })
      messagesAfter += messageCount
    }

    groups = reversed.reverse()

    return groups
  }

  return {
    invalidate(records: readonly MutationRecord[]): boolean {
      const changed = records.some(record => {
        if (record.type === 'attributes') {
          const oldSlot = record.oldValue

          return (
            (record.target instanceof Element && record.target.matches(COUNTED_STRUCTURE)) ||
            oldSlot === 'aui_message-group' ||
            oldSlot === 'aui_user-message-root' ||
            oldSlot === 'aui_assistant-message-root' ||
            oldSlot === 'aui_system-message-root'
          )
        }

        return [...record.addedNodes, ...record.removedNodes].some(containsCountedStructure)
      })

      if (changed) {
        groups = null
      }

      return changed
    },

    read(): { count: number; settled: boolean } {
      const bottom = viewport.getBoundingClientRect().bottom
      const current = snapshot()
      const measured = new Map<number, DOMRect>()

      const rectAt = (index: number) => {
        const cached = measured.get(index)

        if (cached) {
          return cached
        }

        const rect = current[index].element.getBoundingClientRect()
        measured.set(index, rect)

        return rect
      }

      let before = 0
      let after = current.length

      while (before < after) {
        const middle = Math.floor((before + after) / 2)

        if (rectAt(middle).bottom <= bottom + 1) {
          before = middle + 1
        } else {
          after = middle
        }
      }

      if (before >= current.length) {
        return { count: 0, settled: true }
      }

      let count = 0
      let settled = true

      // Normal-flow transcript groups do not overlap, so this loop normally
      // measures exactly one straddler plus the next group's outer box. Keep
      // walking if a test/transition temporarily overlaps groups; cost is
      // O(log n + k) where k is the number intersecting the fold.
      for (let index = before; index < current.length; index++) {
        const group = current[index]
        const rect = rectAt(index)

        if (rect.top >= bottom) {
          count += group.messageCount + group.messagesAfter

          break
        }

        for (const message of group.element.querySelectorAll<HTMLElement>(MESSAGE_ROOTS)) {
          const messageRect = message.getBoundingClientRect()

          if (messageRect.height === 0 && messageRect.width === 0) {
            settled = false
          } else if (messageRect.height > 0 && messageRect.bottom > bottom + 1) {
            count++
          }
        }
      }

      return { count, settled }
    }
  }
}

/**
 * Stateless helper retained for focused callers/tests. The hook below keeps one
 * reader for the mounted viewport so repeated scroll frames reuse membership.
 */
export function countMessagesBelow(viewport: HTMLElement, content: HTMLElement): { count: number; settled: boolean } {
  return createMessagesBelowReader(viewport, content).read()
}

export function useMessagesBelow({
  contentRef,
  scrollRef,
  isAtBottom,
  paneVisible,
  rows,
  sessionKey,
  sessionId
}: MessagesBelowOptions) {
  useEffect(() => {
    if (!paneVisible) {
      return
    }

    if (isAtBottom) {
      publishThreadMessagesBelow(0, { paneVisible, sessionId })

      return
    }

    const viewport = scrollRef.current
    const content = contentRef.current

    if (!viewport || !content) {
      return
    }

    let frame = 0
    let retried = false
    const reader = createMessagesBelowReader(viewport, content)

    const measure = () => {
      frame = 0
      const { count, settled } = reader.read()

      // One extra frame lets the skipped turn gain boxes; then publish what is
      // there so an empty turn can never stall the count.
      if (!settled && !retried) {
        retried = true
        schedule()

        return
      }

      retried = false
      publishThreadMessagesBelow(count, { paneVisible, sessionId })
    }

    const schedule = () => {
      if (!frame) {
        frame = requestAnimationFrame(measure)
      }
    }

    schedule()
    viewport.addEventListener('scroll', schedule, { passive: true })
    const resize = new ResizeObserver(schedule)
    resize.observe(viewport)
    resize.observe(content)

    const mutations = new MutationObserver(records => {
      if (reader.invalidate(records)) {
        schedule()
      }
    })

    mutations.observe(content, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['data-slot'],
      attributeOldValue: true
    })

    return () => {
      cancelAnimationFrame(frame)
      viewport.removeEventListener('scroll', schedule)
      resize.disconnect()
      mutations.disconnect()
    }
  }, [contentRef, scrollRef, isAtBottom, paneVisible, rows, sessionKey, sessionId])
}
