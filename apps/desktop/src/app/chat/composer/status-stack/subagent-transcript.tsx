import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'

import { rejectUnownedSubagentRequest } from './use-subagent-snapshot'

interface Tail {
  available: boolean
  text: string
  truncated: boolean
}

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
        const result = await requestForOwnedSession<Tail>(sessionId, rejectUnownedSubagentRequest, 'subagent.tail', {
          session_id: sessionId,
          subagent_id: subagentId
        })

        if (!cancelled && owner === JSON.stringify(knownOwnerForSession(sessionId))) {
          setTail({
            available: result.available,
            text: typeof result.text === 'string' ? result.text.slice(-16384) : '',
            truncated: result.truncated
          })
        }
      } catch {
        if (!cancelled) {
          setTail({ available: false, text: '', truncated: false })
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
      {tail?.truncated && <p className="text-(--ui-text-tertiary)">{t.agents.transcriptTruncated}</p>}
      {tail?.available ? (
        <pre className="max-h-[30vh] overflow-auto whitespace-pre-wrap break-words font-mono text-[0.68rem]">
          {tail.text}
        </pre>
      ) : (
        <p className="text-(--ui-text-tertiary)">{tail ? t.agents.transcriptUnavailable : t.agents.waitingActivity}</p>
      )}
    </section>
  )
}
