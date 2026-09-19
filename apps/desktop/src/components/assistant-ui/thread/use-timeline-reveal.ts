import { type RefObject, useEffect, useRef } from 'react'

import { useSessionView } from '@/app/chat/session-view'

import { EARLIER_TIMELINE_ID, TIMELINE_REVEAL_EVENT, type TimelineRevealRequest } from './timeline-data'
import { useTranscriptWindow } from './transcript-window'

interface TimelineRevealOptions {
  viewport: RefObject<HTMLElement | null>
  groups: readonly { id: string; weight: number }[]
  hiddenCount: number
  renderBudget: number
  olderAvailable: boolean
  revealBudget: (budget: number) => void
  expandWindow: (beforePrepend?: () => void) => void | Promise<boolean>
  prepare: () => void
  sessionKey?: string | null
}

/** Direct archive jumps select one bounded page, never every intervening turn. */
export function useTimelineReveal(options: TimelineRevealOptions) {
  const view = useSessionView()
  const history = useTranscriptWindow()
  const latest = useRef({ ...options, history })
  latest.current = { ...options, history }

  useEffect(() => {
    const viewport = options.viewport.current

    if (!viewport) {
      return
    }
    let current: AbortController | null = null

    const onReveal = (event: Event) => {
      const request = (event as CustomEvent<TimelineRevealRequest>).detail
      current?.abort()
      const controller = new AbortController()
      current = controller
      const abort = () => controller.abort()
      request.signal.addEventListener('abort', abort, { once: true })

      if (request.signal.aborted) {
        controller.abort()
      }
      const timeout = window.setTimeout(abort, 15000)
      const valid = () => !controller.signal.aborted
      const find = (id: string) => viewport.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(id)}"]`)

      void (async () => {
        latest.current.prepare()
        let id = request.id
        const rowId = request.rowId ?? (id.startsWith('history:') ? Number(id.slice(8)) : undefined)
        const loaded =
          rowId === undefined
            ? undefined
            : [...(latest.current.history.currentMessages ?? []), ...view.$messages.get()].find(
                message => message.rowId === rowId
              )
        id = loaded?.id ?? id

        if (find(id)) {
          return id
        }

        if (
          rowId !== undefined &&
          !latest.current.groups.some(group => group.id === id) &&
          latest.current.history.revealRow
        ) {
          const target = await latest.current.history.revealRow(rowId, controller.signal)

          if (!target || !valid()) {
            return false
          }
          id = target
        } else if (id === EARLIER_TIMELINE_ID) {
          const state = latest.current

          if (!state.olderAvailable && !state.hiddenCount) {
            return false
          }
          state.revealBudget(state.renderBudget + 600)

          if (!state.hiddenCount) {
            await state.expandWindow()
          }
        }

        if (!valid()) {
          return false
        }

        // React commits and late row mounts wake this observer; no polling loop.
        return await new Promise<string | false>(resolve => {
          const first = viewport.querySelector('[data-message-id]')?.getAttribute('data-message-id')

          const finish = (result: string | false) => {
            observer.disconnect()
            controller.signal.removeEventListener('abort', cancelled)
            resolve(result)
          }

          const cancelled = () => finish(false)

          const check = () => {
            if (!valid()) {
              return finish(false)
            }
            const state = latest.current
            const index =
              id === EARLIER_TIMELINE_ID
                ? Math.max(0, state.hiddenCount - 1)
                : state.groups.findIndex(group => group.id === id)

            if (index >= 0) {
              const budget = state.groups.slice(index).reduce((sum, group) => sum + group.weight, 0) + 1

              if (budget > state.renderBudget) {
                state.revealBudget(budget)
              }
            }

            const node =
              id === EARLIER_TIMELINE_ID ? viewport.querySelector<HTMLElement>('[data-message-id]') : find(id)

            if (node && (id !== EARLIER_TIMELINE_ID || node.dataset.messageId !== first)) {
              latest.current.prepare()
              finish(node.dataset.messageId!)
            }
          }

          const observer = new MutationObserver(check)
          observer.observe(viewport, {
            subtree: true,
            childList: true,
            attributes: true,
            attributeFilter: ['data-message-id']
          })
          controller.signal.addEventListener('abort', cancelled, { once: true })
          check()
        })
      })()
        .then(
          result => request.complete(result),
          () => request.complete(false)
        )
        .finally(() => {
          clearTimeout(timeout)
          request.signal.removeEventListener('abort', abort)
        })
    }

    viewport.addEventListener(TIMELINE_REVEAL_EVENT, onReveal)

    return () => {
      current?.abort()
      viewport.removeEventListener(TIMELINE_REVEAL_EVENT, onReveal)
    }
  }, [options.viewport, options.sessionKey, view])
}
