import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'

import { AgentTypingIndicator } from './agent-huddle'
import { rejectUnownedSubagentRequest } from './use-subagent-snapshot'

interface Tail {
  available: boolean
}

/**
 * What this panel shows for a live worker: not the child's transcript, but the
 * fact that it is producing one. The raw tail used to be printed here as a
 * `<pre>` above the composer, which is exactly the "I can read what Hermes
 * answered" the user asked to stop seeing. The child's output still exists on
 * the backend (and reaches the Live agent as a spoken report) — this panel only
 * ever renders the animated stand-in.
 */
export function SubagentTranscript({ sessionId, subagentId }: { sessionId: string; subagentId: string }) {
  const { t } = useI18n()
  const [tail, setTail] = useState<Tail | null>(null)
  useEffect(() => {
    let cancelled = false
    let pending = false
    const owner = JSON.stringify(knownOwnerForSession(sessionId))

    const refresh = async () => {
      if (pending || document.visibilityState === 'hidden') {
        return
      }

      pending = true

      try {
        const result = await requestForOwnedSession<{ available?: boolean }>(
          sessionId,
          rejectUnownedSubagentRequest,
          'subagent.tail',
          {
            session_id: sessionId,
            subagent_id: subagentId
          }
        )

        if (!cancelled && owner === JSON.stringify(knownOwnerForSession(sessionId))) {
          setTail({ available: result.available === true })
        }
      } catch {
        if (!cancelled) {
          setTail({ available: false })
        }
      } finally {
        pending = false
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), 2000)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [sessionId, subagentId])

  return (
    <section className="mt-2 text-xs" data-slot="subagent-transcript">
      <h4 className="text-(--ui-text-secondary)">{t.agents.extendedTranscript}</h4>
      <div className="mt-1 flex items-center gap-2">
        <AgentTypingIndicator active={tail?.available === true} />
        <span className="text-(--ui-text-tertiary)">
          {!tail
            ? t.agents.waitingActivity
            : tail.available
              ? t.agents.running
              : t.agents.transcriptUnavailable}
        </span>
      </div>
    </section>
  )
}
