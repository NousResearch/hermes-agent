import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { useMemo } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { isFirstBuildSession } from '@/app/contrib/handoff-receipt'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { latestConnectorPart } from '@/lib/connector-tools'
import { canStartWithConnections } from '@/lib/first-build-start'
import { $firstBuildConnections, startFirstBuild } from '@/store/first-build-connectors'

export function OnboardingStart() {
  const { t } = useI18n()
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const $latest = useMemo(() => computed(view.$messages, latestConnectorPart), [view.$messages])
  const latest = useStore($latest)
  const connections = useStore($firstBuildConnections, { keys: [storedId ?? ''] })
  const state = storedId ? connections[storedId] : undefined

  if (
    !storedId ||
    !isFirstBuildSession(storedId) ||
    !state ||
    state.started ||
    latest?.type !== 'tool-call' ||
    state.toolCallId !== latest.toolCallId ||
    !canStartWithConnections(latest)
  ) {
    return null
  }

  const count = state.rows.filter(row => row.phase === 'connected').length

  return (
    <Button
      onClick={() =>
        startFirstBuild(storedId, text =>
          requestComposerSubmit(text, {
            target: view.kind === 'tile' ? `tile:${storedId}` : 'main'
          })
        )
      }
      size="xs"
      variant="secondary"
    >
      {count ? t.connectors.startWith(count) : t.connectors.startWithout}
    </Button>
  )
}
