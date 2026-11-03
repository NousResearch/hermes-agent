import { useStore } from '@nanostores/react'
import { type ComponentProps, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { RowButton } from '@/components/ui/row-button'
import { useI18n } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'

import { sessionRowKey } from './simplified-session-identity'
import { SimplifiedSessionList } from './simplified-session-list'

export function SimplifiedAgentSessions({
  searching,
  revealedSessionKey,
  ...props
}: ComponentProps<typeof SimplifiedSessionList> & {
  searching: boolean
  revealedSessionKey: string | null
}) {
  const { t } = useI18n()
  const profile = useStore($activeGatewayProfile)
  const scope = useStore($apiRequestScope)
  const [expanded, setExpanded] = useState(false)

  const preview = props.sessions.filter(
    (row, index) =>
      index < 5 ||
      sessionRowKey(row) === revealedSessionKey ||
      (row.id === props.activeSessionId &&
        (row.profile || 'default') === profile &&
        (!row.connection_id || row.connection_id === scope.connectionId))
  )

  const hidden = props.sessions.length - preview.length

  return (
    <>
      <SimplifiedSessionList {...props} sessions={expanded || searching ? props.sessions : preview} />
      {!searching && hidden > 0 && (
        <RowButton
          aria-expanded={expanded}
          className="min-h-7 w-full rounded-sm px-2 text-left text-xs text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover)"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? t.common.collapse : t.sidebar.showMoreIn(hidden, props.sessions[0]?.profile || 'default')}
        </RowButton>
      )}
    </>
  )
}
