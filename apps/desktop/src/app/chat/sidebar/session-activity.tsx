import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { summarizeShellCommand } from '@/lib/summarize-command'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $sidebarRowMeta } from '@/store/layout'
import { $sessionDotStateById } from '@/store/session-dot-state'
import { $sidebarActivityById, type SidebarActivity } from '@/store/sidebar-activity'

/** The task label explains the monitoring state; it never determines it. */
function activityLabel(activity: SidebarActivity): string {
  return activity.title.startsWith('# ') ? activity.title.slice(2).trim() : summarizeShellCommand(activity.title)
}

export function SidebarSessionActivity({ sessionId }: { sessionId: string }) {
  const enabled = useStore($sidebarRowMeta).includes('activity')

  // Opting out also avoids subscribing to this row's background activity.
  return enabled ? <SessionActivityLabel sessionId={sessionId} /> : null
}

function SessionActivityLabel({ sessionId }: { sessionId: string }) {
  const { t } = useI18n()
  const activity = useStoreSelector($sidebarActivityById, all => all[sessionId])
  const status = useStoreSelector($sessionDotStateById, all => all[sessionId])
  // The selectors above only repaint for this row's activity, not text deltas
  // in unrelated sessions. Formatting here also follows locale changes.
  const detail = activity ? activityLabel(activity) : ''
  const r = t.sidebar.row

  // The existing arc already reports active work. This extra cue is only
  // for work that remains after the agent has yielded, never another spinner.
  const label = status === 'background' ? detail || r.activity : ''

  if (!label) {
    return null
  }

  return (
    <Tip label={label}>
      <span
        aria-label={label}
        className="inline-flex size-3.5 shrink-0 items-center justify-center text-(--ui-text-tertiary) focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-sidebar-ring"
        data-session-activity=""
        role="img"
        tabIndex={0}
      >
        <Codicon name="pulse" size="0.875rem" />
      </span>
    </Tip>
  )
}
