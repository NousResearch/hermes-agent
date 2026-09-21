import { useEffect, useState } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
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
  const paneVisible = usePaneVisible()
  const [tail, setTail] = useState<Tail | null>(null)
  useEffect(() => {
    if (!paneVisible) {
      // Keep-alive tiles stay mounted; only poll while this pane is the visible tab (reveal re-runs the effect).
      return
    }

    let cancelled = false
    let pending = false

    const refresh = async () => {
      if (pending || document.visibilityState === 'hidden') {
        return
      }

      pending = true
      const owner = JSON.stringify(knownOwnerForSession(sessionId))

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
        if (!cancelled && owner === JSON.stringify(knownOwnerForSession(sessionId))) {
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
  }, [sessionId, subagentId, paneVisible])

  return (
    <section className="mt-2 text-xs" data-slot="subagent-transcript">
      <h4 className="text-(--ui-text-secondary)">{t.agents.extendedTranscript}</h4>
      {tail?.truncated && <p className="text-(--ui-text-tertiary)">{t.agents.transcriptTruncated}</p>}
      {tail?.available ? (
        <pre className="max-h-[30vh] overflow-auto overscroll-y-auto whitespace-pre-wrap break-words font-mono text-[0.68rem]">
          {tail.text}
        </pre>
      ) : (
        <p className="text-(--ui-text-tertiary)">{tail ? t.agents.transcriptUnavailable : t.agents.waitingActivity}</p>
      )}
    </section>
  )
}
