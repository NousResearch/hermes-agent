import { useNavigate } from 'react-router'

import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify } from '@/store/notifications'
import { requestFreshSession } from '@/store/profile'

import { requestComposerInsert } from '../chat/composer/focus'
import { CRON_ROUTE, MESSAGING_ROUTE } from '../routes'

import { CronView } from '.'

// The Cron route as the shell mounts it: CronView plus the two places its
// editor can send the user. CronView parks the unsaved form before either
// hand-off and reopens it on the next visit.
export function CronOverlay({
  onClose,
  onOpenSession
}: {
  onClose: () => void
  onOpenSession: (sessionId: string, session?: SessionInfo) => void
}) {
  const navigate = useNavigate()
  const { t } = useI18n()
  const c = t.cron

  // A fresh chat with the prompt typed in but unsent: the user reads it, then
  // presses Enter. Same order as the composer's "start work in a worktree"
  // hand-off: open the draft, then let the deferred insert bus fill it.
  const testPrompt = (prompt: string) => {
    requestFreshSession()
    requestComposerInsert(prompt, { target: 'main' })
    // A fresh draft clears the toast stack, so raise this a tick later — the
    // same deferral that lets the insert above land in the new composer.
    window.setTimeout(() =>
      notify({
        action: { label: c.backToDraft, onClick: () => navigate(CRON_ROUTE) },
        kind: 'info',
        message: c.testDraftKeptDesc,
        title: c.testDraftKept
      })
    )
  }

  return (
    <CronView
      onClose={onClose}
      onOpenMessaging={() => navigate(MESSAGING_ROUTE)}
      onOpenSession={onOpenSession}
      onTestPrompt={testPrompt}
    />
  )
}
