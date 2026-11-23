import { cleanup, render, renderHook, screen } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { en } = await import('@/i18n/en')
  const { relativeTime } = await import('@/lib/time')

  return { ...await (await import('./group-test-utils')).pluginSdkMock({}), relativeTime, en,
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => CANONICAL_GROUP_LOCALES.en }
})

import { useCanonicalRoomLive } from './canonical-group-activity'
import { CanonicalGroupHistory } from './canonical-group-history'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

afterEach(cleanup)

const binding = { connectionId: 'local', profile: 'default', roomId: 'room' }
const members = [{ member_id: 'peer', profile: 'default', handle: 'peer', display_name: 'Hermes · Homelab' }]

it('says what a deferred turn means instead of printing its reason code', () => {
  const events = [{ seq: 1, kind: 'turn.deferred', payload: { member_id: 'peer', reason: 'member_unavailable' } }]
  const { container } = render(<CanonicalGroupHistory binding={binding} events={events} members={members} />)
  expect(container.textContent).not.toContain('member_unavailable')
  expect(container.textContent).toContain(CANONICAL_GROUP_LOCALES.en.turnNoAnswer)
  expect(screen.getByText('Turn deferred')).toBeTruthy()
})

it.each([[[{ action: 'retry' }], null], [[], { kind: 'settled' }]])('calls a discussion settled only when no turn waits on a person: %j', (pending, cycle) => {
  const events = [
    { seq: 1, kind: 'message.user', event_id: 'user:d', payload: { thread_id: 'd' } },
    { seq: 2, kind: 'room.activity', payload: { status: 'settled', discussion_event_id: 'user:d', thread_id: 'd' } }
  ]

  const { result } = renderHook(() => useCanonicalRoomLive(events as never, { pending_actions: pending } as never, members))
  expect(result.current.cycle).toEqual(cycle)
})
