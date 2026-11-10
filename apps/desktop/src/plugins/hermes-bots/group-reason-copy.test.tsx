import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const selection = vi.hoisted(() => ({ locale: 'en' }))
vi.mock('@hermes/plugin-sdk', async () => ({
  ...await (await import('./group-test-utils')).pluginSdkMock({}),
  Tip: ({ children, label }: { children: ReactNode; label: string }) => <div title={label}>{children}</div>
}))
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => CANONICAL_GROUP_LOCALES[selection.locale as keyof typeof CANONICAL_GROUP_LOCALES] }
})

import { CanonicalGroupHistory } from './canonical-group-history'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { groupActivityLabel } from './group-activity'
import { memberOutcomes, memberStatus } from './group-room-status'
import { MemberStatusChip, memberStatusText } from './group-room-ui'
import { translateBotsIn } from './i18n-test-helper'
import { setPluginCtx } from './shared'

afterEach(() => { cleanup(); setPluginCtx(null) })
const binding = { connectionId: 'local', profile: 'team', roomId: 'copy' }

const knownReasons = [
  'runtime_offline', 'queued_expired', 'delivery_timeout', 'agent_blocked', 'cancelled',
  'provider_auth_or_access', 'provider_quota_limit', 'provider_rate_limit', 'provider_server_error',
  'context_overflow', 'missing_config', 'model_unavailable', 'target_scope_unresolved',
  'prepare_failed', 'slot_wait_timeout', 'target_busy', 'rate_limited', 'member gateway restarted'
]

it.each(Object.keys(CANONICAL_GROUP_LOCALES) as (keyof typeof CANONICAL_GROUP_LOCALES)[])('renders known reasons and hides unknown codes in %s', locale => {
  selection.locale = locale
  setPluginCtx({ i18n: { t: translateBotsIn(locale) } } as never)
  const labels = CANONICAL_GROUP_LOCALES[locale as keyof typeof CANONICAL_GROUP_LOCALES]
  const unavailable = labels.reasonMemberUnavailable
  const superseded = labels.reasonSuperseded
  const deadline = labels.reasonDeadline

  const cases = [
    ['member_unavailable', unavailable], ['superseded_by_newer_user_event', superseded], ['turn_deadline_exceeded', deadline],
    ['unknown', ''], ['future_reason', ''], ['future.reason:v2', ''], ['Provider timed out', 'Provider timed out']
  ]

  for (const [reason, text] of cases) {
    const events = [
      { seq: 1, kind: 'turn.deferred', payload: { reason } },
      { seq: 2, kind: 'turn.failed', payload: { error: reason } },
      { seq: 3, kind: 'turn.failed', payload: { reason_code: reason } },
      { seq: 4, kind: 'turn.cancelled', payload: { reason } }
    ]

    const view = render(<CanonicalGroupHistory binding={binding} events={events} />)
    // A member that did not answer in time reads as that in the log; its status chip keeps the reason.
    const logged = reason === 'member_unavailable' ? labels.turnNoAnswer : text
    expect([...view.container.querySelectorAll('[data-room-notice] p')].map(node => node.textContent)).toEqual(
      [labels.turnDeferred, labels.turnFailed, labels.turnFailed, labels.turnCancelled].map(word => word + (logged ? ': ' + logged : '')))
    view.unmount()

    const activity = groupActivityLabel({ at: 1, epoch: 0, kind: 'failed', member: 'Helper', reason })
    expect(activity).toBe('Helper ' + translateBotsIn(locale)('group.activityFailed') + (text ? ' — ' + text : ''))
  }

  // Exercise the durable outcome -> status -> chip path, including unknown classification with useful error text.
  for (const [reason, error, expected] of [
    ['member_unavailable', '', unavailable], ['unknown', '', ''], ['unknown', 'Provider timed out', 'Provider timed out'],
    ['future_reason', 'unknown', '']
  ]) {
    const events = [
      { seq: 1, event_id: 'discussion', kind: 'message.user', payload: {} },
      { seq: 2, kind: 'turn.failed', payload: { member_id: 'helper', discussion_event_id: 'discussion', reason_code: reason, error } }
    ]

    const status = memberStatus('helper', [], memberOutcomes(events), 1)
    const text = labels.statusFailed + (expected ? ' · ' + expected : '')
    expect(memberStatusText(status, labels)).toBe(text)
    const view = render(<MemberStatusChip avatar={null} labels={labels} name="Helper" status={status} />)
    expect(view.container.textContent).toBe('Helper·' + text)
    expect(view.container.querySelector('[title]')?.getAttribute('title')).not.toMatch(/unknown|future_reason|member_unavailable/)
    view.unmount()
  }

  for (const reason of knownReasons) {
    const text = memberStatusText({ kind: 'failed', reason }, labels)
    expect(text).toContain(labels.statusFailed + ' · ')
    expect(text).not.toBe(labels.statusFailed + ' · ' + reason)

    if (locale !== 'en') {
      expect(text.slice(labels.statusFailed.length)).not.toBe(
        memberStatusText({ kind: 'failed', reason }, CANONICAL_GROUP_LOCALES.en).slice(CANONICAL_GROUP_LOCALES.en.statusFailed.length))
    }
  }
})

