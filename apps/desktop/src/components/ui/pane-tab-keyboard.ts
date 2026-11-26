import { type FocusEvent, type KeyboardEvent, type RefObject, useLayoutEffect, useRef, useState } from 'react'

/** Focus belongs to the strip, not its active pane. Keep it through tab closes
 * and the vertical-rail → horizontal-strip remount when a zone is restored. */
export function usePaneTabKeyboard(ids: readonly string[], activeId: string, root: RefObject<HTMLElement | null>) {
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const focusedElement = useRef<HTMLElement | null>(null)
  const previousIds = useRef(ids)
  const entryId = focusedId && ids.includes(focusedId) ? focusedId : ids.includes(activeId) ? activeId : ids[0]

  useLayoutEffect(() => {
    const oldElement = focusedElement.current

    // A detached focused tab leaves focus on body. Never reclaim focus that
    // the user moved to a menu, another control, or another pane.
    if (oldElement && !oldElement.isConnected && document.activeElement === document.body) {
      const oldId = oldElement.dataset.paneTabId ?? ''

      const nextId = ids.includes(oldId)
        ? oldId
        : ids[Math.min(Math.max(previousIds.current.indexOf(oldId), 0), ids.length - 1)]

      const next = [...(root.current?.querySelectorAll<HTMLElement>('[data-pane-tab-id]') ?? [])].find(
        element => element.dataset.paneTabId === nextId
      )

      focusedElement.current = null
      next?.focus({ preventScroll: true })
    }

    previousIds.current = ids
  })

  return (id: string) => ({
    'data-pane-tab-id': id,
    tabIndex: id === entryId ? 0 : -1,
    onFocus: (event: FocusEvent<HTMLDivElement>) => {
      if (event.target !== event.currentTarget) {
        return
      }

      focusedElement.current = event.currentTarget
      setFocusedId(id)
    },
    onBlur: (event: FocusEvent<HTMLDivElement>) => {
      const list = event.currentTarget.closest('[role="tablist"]')
      const nextList = event.relatedTarget?.closest('[role="tab"]')?.closest('[role="tablist"]')

      if (event.relatedTarget && nextList !== list) {
        focusedElement.current = null
        setFocusedId(null)
      }
    },
    onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
      if (
        event.defaultPrevented ||
        event.target !== event.currentTarget ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey
      ) {
        return
      }

      const list = event.currentTarget.closest('[role="tablist"]')
      const vertical = list?.getAttribute('aria-orientation') === 'vertical'
      const index = ids.indexOf(id)
      let nextIndex: number

      switch (event.key) {
        case 'Home':
          nextIndex = 0

          break

        case 'End':
          nextIndex = ids.length - 1

          break

        case vertical ? 'ArrowUp' : 'ArrowLeft':
          nextIndex = (index - 1 + ids.length) % ids.length

          break

        case vertical ? 'ArrowDown' : 'ArrowRight':
          nextIndex = (index + 1) % ids.length

          break

        default:
          return
      }

      const next = [...(list?.querySelectorAll<HTMLElement>('[data-pane-tab-id]') ?? [])].find(
        element => element.dataset.paneTabId === ids[nextIndex] && element.closest('[role="tablist"]') === list
      )

      if (!next) {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      next.focus({ preventScroll: true })
      next.scrollIntoView?.({ block: 'nearest', inline: 'nearest' })
    }
  })
}
