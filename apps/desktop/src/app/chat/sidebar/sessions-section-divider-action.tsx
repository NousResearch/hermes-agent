import { type NewSessionSplitHandler, startNewSessionDrag } from '@/app/chat/new-session-drag'
import { useI18n } from '@/i18n'

import { WorkspaceAddButton } from './projects/workspace-header'

interface SessionsDividerActionProps {
  /** A flat list's grouping; only 'date' dividers take the "+". */
  grouping: 'date' | 'none' | 'status'
  onNewSessionInWorkspace?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
}

/**
 * Date dividers head a group the same way a repo header does, so they carry the
 * same hover-revealed "+". Only for dates: "new session in WORKING" is not a
 * thing. The divider "+" is a drag source too (the same gesture as the nav's
 * "New session" row): drag it onto a chat zone to create the session exactly
 * there; a sub-threshold release stays the ordinary click.
 */
export function SessionsDividerAction({
  grouping,
  onNewSessionInWorkspace,
  onNewSessionSplit
}: SessionsDividerActionProps) {
  const { t } = useI18n()

  if (grouping !== 'date' || !onNewSessionInWorkspace) {
    return null
  }

  return (
    <WorkspaceAddButton
      label={t.sidebar.nav['new-session']}
      onClick={() => onNewSessionInWorkspace(null)}
      onPointerDown={
        onNewSessionSplit
          ? event => {
              startNewSessionDrag(placement => {
                onNewSessionSplit(placement.dir, {
                  anchor: placement.anchor,
                  before: placement.before
                })
              }, event)
            }
          : undefined
      }
    />
  )
}
