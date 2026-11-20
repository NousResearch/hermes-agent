/**
 * One door for "open this subagent" (delegate card row, status-stack row,
 * completion card). `desktop.subagent_open` picks the destination: `window`
 * (default) pops the child into its own watch window (a new browser tab in the
 * Webapp); `tab` opens it as an in-app session tab beside the parent chat,
 * which stays open.
 *
 * A row that only knows the child's `subagent_id` (a finished async delegation,
 * a reloaded transcript: the live roster forgot it) resolves the stored child
 * session through the parent first.
 */
import { getSubagentChildSession } from '@/api/sessions'
import { translateNow } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $subagentOpenMode } from '@/store/subagent-display'
import { openSessionInNewWindow } from '@/store/windows'

import { openSessionFromPicker } from './open-session'

export interface SubagentTarget {
  childSessionId?: null | string
  parentSessionId?: null | string
  subagentId?: null | string
}

export const canOpenSubagent = ({ childSessionId, parentSessionId, subagentId }: SubagentTarget): boolean =>
  Boolean(childSessionId || (subagentId && parentSessionId))

async function resolveChild({ childSessionId, parentSessionId, subagentId }: SubagentTarget): Promise<string> {
  if (childSessionId) {
    return childSessionId
  }

  if (!subagentId || !parentSessionId) {
    return ''
  }

  const found = await getSubagentChildSession(parentSessionId, subagentId)

  return found?.child_session_id || ''
}

export async function openSubagentSession(target: SubagentTarget): Promise<void> {
  let child = ''

  try {
    child = await resolveChild(target)
  } catch (err) {
    notifyError(err, translateNow('assistant.tool.openSubagentFailed'))

    return
  }

  if (!child) {
    return
  }

  if ($subagentOpenMode.get() === 'tab') {
    // 'tab': front it when already open, else stack a new session tab beside
    // the current one. Never spends the parent's tab.
    openSessionFromPicker(child, to => void (window.location.hash = to), 'tab')

    return
  }

  await openSessionInNewWindow(child, { watch: true, parentSessionId: target.parentSessionId ?? null })
}
