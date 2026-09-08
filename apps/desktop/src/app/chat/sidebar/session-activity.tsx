import { useStore } from '@nanostores/react'

import { buildToolView } from '@/components/assistant-ui/tool/fallback-model'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { summarizeShellCommand } from '@/lib/summarize-command'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $sidebarRowMeta } from '@/store/layout'
import { $sessionDotStateById } from '@/store/session-dot-state'
import { $sidebarActivityById, type SidebarActivity } from '@/store/sidebar-activity'

/** Display reported work, never classify a command or infer intent from prose. */
function activityLabel(activity: SidebarActivity): string {
  if (activity.type === 'tool-call') {
    // Shell/code calls may carry an explicit human-readable first-line comment.
    // Prefer that label over the gateway's flattened command preview.
    if (['terminal', 'execute_code', 'browser_exec'].includes(activity.toolName)) {
      const code = activity.args?.command ?? activity.args?.code
      const comment = typeof code === 'string' ? /^[ \t]*#[ \t]+([^\r\n]+)/u.exec(code)?.[1]?.trim() : undefined

      if (comment) {
        return comment
      }
    }

    const context = activity.args?.context

    return typeof context === 'string' && context.trim() ? context.trim() : buildToolView(activity, '').title
  }

  if (activity.type === 'background') {
    // The registry keeps the first command line. A leading shell comment is
    // already the agent's human label, not a command to show with a '#' prefix.
    return activity.title.startsWith('# ') ? activity.title.slice(2).trim() : summarizeShellCommand(activity.title)
  }

  return activity.title
}

export function SidebarSessionActivity({ sessionId }: { sessionId: string }) {
  const enabled = useStore($sidebarRowMeta).includes('activity')

  // Opting out must also opt out of projecting the live transcript stream.
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

  const label =
    status === 'needs-input'
      ? r.waitingForAnswer
      : detail ||
        (status === 'background'
          ? r.backgroundRunning
          : status === 'working' || status === 'stalled'
            ? r.sessionRunning
            : '')

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
