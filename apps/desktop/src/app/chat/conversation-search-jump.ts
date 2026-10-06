import { TIMELINE_REVEAL_EVENT, type TimelineRevealRequest } from '@/components/assistant-ui/thread/timeline-data'

export async function revealConversationMatch(surface: HTMLElement, rowId: number, signal: AbortSignal) {
  const viewport = surface.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')

  if (!viewport || signal.aborted) {
    return false
  }

  const id = await new Promise<string | false>(resolve => {
    const finish = (value: string | false) => {
      clearTimeout(timeout)
      signal.removeEventListener('abort', abort)
      resolve(value)
    }

    const abort = () => finish(false)
    const timeout = window.setTimeout(abort, 15000)
    signal.addEventListener('abort', abort, { once: true })
    const detail: TimelineRevealRequest = { id: `history:${rowId}`, rowId, kind: 'match', signal, complete: finish }
    viewport.dispatchEvent(new CustomEvent(TIMELINE_REVEAL_EVENT, { detail }))
  })

  if (!id || signal.aborted) {
    return false
  }

  const node = viewport.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(id)}"]`)

  if (!node) {
    return false
  }

  clearConversationHighlight(surface)
  node.dataset.conversationMatch = ''
  node.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' })

  return true
}

export function clearConversationHighlight(surface: HTMLElement | null) {
  surface?.querySelectorAll<HTMLElement>('[data-conversation-match]').forEach(node => {
    delete node.dataset.conversationMatch
  })
}
