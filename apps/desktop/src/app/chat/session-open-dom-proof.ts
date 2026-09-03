import type { ChatMessage } from '@/lib/chat-messages'

export interface SessionTranscriptDomProofOptions {
  expectedMessageId: string
  expectedStoredSessionId: string
  expectedText: string
  root?: ParentNode
  timeoutMs?: number
}

export type FirstPaintTranscriptProof = Pick<
  SessionTranscriptDomProofOptions,
  'expectedMessageId' | 'expectedText'
>

/**
 * The thread list walks whole turns newest-first until FIRST_PAINT_BUDGET is
 * met, so its newest user-led group is always mounted even when that one group
 * alone exceeds the budget. Select that group's user row, whose
 * `data-message-id` is the stable DOM proof used by the perf fixture.
 */
export function selectFirstPaintTranscriptProof(
  messages: readonly Pick<ChatMessage, 'id' | 'parts' | 'role'>[]
): FirstPaintTranscriptProof | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]

    if (message.role !== 'user' || !message.id) {
      continue
    }

    const text = message.parts
      .filter(part => part.type === 'text')
      .map(part => part.text)
      .join('\n')
      .trim()

    if (text) {
      return { expectedMessageId: message.id, expectedText: text }
    }
  }

  return null
}

/**
 * Perf-fixture-only proof that the selected session's transcript reached the
 * DOM. A performance mark or animation frame alone cannot distinguish a real
 * commit from a blank/wrong-session paint.
 */
export function waitForSessionTranscriptDom({
  expectedMessageId,
  expectedStoredSessionId,
  expectedText,
  root = document,
  timeoutMs = 5_000
}: SessionTranscriptDomProofOptions): Promise<number> {
  return new Promise((resolve, reject) => {
    let observer: MutationObserver | null = null
    let timeout: ReturnType<typeof setTimeout> | null = null

    const cleanup = () => {
      observer?.disconnect()

      if (timeout !== null) {
        clearTimeout(timeout)
      }
    }

    const findProof = (): boolean => {
      return Array.from(root.querySelectorAll<HTMLElement>('[data-hermes-perf-session]'))
        .filter(candidate => candidate.dataset.hermesPerfSession === expectedStoredSessionId)
        .some(sessionRoot => {
          const message = Array.from(sessionRoot.querySelectorAll<HTMLElement>('[data-message-id]')).find(
            candidate => candidate.dataset.messageId === expectedMessageId
          )

          return Boolean(message?.textContent?.includes(expectedText))
        })
    }

    const inspect = () => {
      if (!findProof()) {
        return
      }

      cleanup()
      resolve(performance.now())
    }

    observer = new MutationObserver(inspect)
    observer.observe(root, { attributes: true, characterData: true, childList: true, subtree: true })
    timeout = setTimeout(() => {
      cleanup()
      const sessionRoots = Array.from(root.querySelectorAll<HTMLElement>('[data-hermes-perf-session]'))
      const sessions = sessionRoots
        .map(candidate => candidate.dataset.hermesPerfSession)
        .filter((value): value is string => Boolean(value))
        .slice(0, 20)
      const matchingMessageIds = sessionRoots
        .filter(candidate => candidate.dataset.hermesPerfSession === expectedStoredSessionId)
        .flatMap(candidate =>
          Array.from(candidate.querySelectorAll<HTMLElement>('[data-message-id]'))
            .map(message => message.dataset.messageId)
            .filter((value): value is string => Boolean(value))
        )
        .slice(0, 20)

      reject(
        new Error(
          `session-switch fixture did not observe transcript DOM for ${expectedStoredSessionId}/${expectedMessageId} ` +
            `(sessions=${sessions.join(',') || 'none'}; matchingMessageIds=${matchingMessageIds.join(',') || 'none'})`
        )
      )
    }, timeoutMs)

    inspect()
  })
}
