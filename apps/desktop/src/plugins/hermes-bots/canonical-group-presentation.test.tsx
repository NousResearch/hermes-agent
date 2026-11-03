import { cleanup, render, screen } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { Streamdown } = await import('streamdown')
  const { en } = await import('@/i18n/en')
  const { relativeTime } = await import('@/lib/time')

  // The shell's shared message renderer; media marks whether MEDIA: paths would resolve locally.
  const MessageTextContent = ({ media = true, text }: { media?: boolean; text: string }) =>
    <div data-media={String(media)} data-testid="message-text"><Streamdown>{text}</Streamdown></div>

  return { MessageTextContent, Streamdown, relativeTime, useI18n: () => ({ locale: 'en', t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, download: 'Download' }) }
})

import { CanonicalGroupHistory } from './canonical-group-history'
import type { CanonicalGroupEvent } from './canonical-group-history'

const binding = { connectionId: 'local', profile: 'reviewer', roomId: 'room-one' }

afterEach(cleanup)

// groups.log returns actor.kind/id and Unix-second created_at, including on replay.
it('renders the canonical log actors, timestamps and Markdown without transport markers', () => {
  const events = [
    { seq: 1, kind: 'message.user', actor: { kind: 'user', id: 'uid:42', display_name: 'Morgan' },
      created_at: 1792152000, payload: { text: 'Please **review** this.' } },
    { seq: 2, kind: 'message.member', actor: { kind: 'member', id: 'helper', display_name: 'Historical Helper' },
      created_at: 1792152001, payload: { text: '**Done**' } },
    { seq: 3, kind: 'message.member', actor: { kind: 'member', id: 'reviewer' },
      created_at: 1792152002, payload: { text: 'Checked.' } },
    { seq: 4, kind: 'message.member', actor: { kind: 'member', id: 'departed-bot' },
      created_at: 1792152003, payload: { text: 'Earlier reply.' } },
    { seq: 5, kind: 'turn.settled', actor: { kind: 'gateway', id: 'owner' },
      created_at: 1792152004, payload: {} }
  ]

  const members = [
    { member_id: 'helper', profile: 'helper', handle: 'helper', display_name: 'Renamed Helper' },
    { member_id: 'reviewer', profile: 'reviewer', handle: 'review', display_name: 'Code Reviewer' }
  ]

  const { container } = render(<CanonicalGroupHistory binding={binding} events={events} members={members} />)

  expect(screen.getByText('Morgan')).toBeTruthy()
  expect(screen.getByText('Historical Helper')).toBeTruthy()
  expect(screen.getByText('Code Reviewer')).toBeTruthy()
  expect(screen.getByText('departed-bot')).toBeTruthy()
  expect(screen.queryByText('Renamed Helper')).toBeNull()
  expect(screen.getByText('Done').getAttribute('data-streamdown')).toBe('strong')
  expect(screen.getByText('review').getAttribute('data-streamdown')).toBe('strong')
  expect(screen.queryByText('turn.settled')).toBeNull()
  // Room text goes through the shared renderer, never resolving MEDIA: paths on this gateway.
  expect(screen.getAllByTestId('message-text').map(node => node.dataset.media)).toEqual(['false', 'false', 'false', 'false'])
  expect([...container.querySelectorAll('time')].map(node => node.dateTime)).toEqual(
    events.slice(0, 4).map(event => new Date(event.created_at * 1000).toISOString())
  )
  expect([...container.querySelectorAll('time')].every(node => node.textContent)).toBe(true)
})

it('keeps attachment-only messages and rows without optional display metadata readable', () => {
  const events = [
    { seq: 1, room_id: binding.roomId, event_id: 'file-event', kind: 'message.member',
      actor: { kind: 'member', id: 'bare-bot' }, created_at: 0,
      payload: { attachments: [{ attachment_id: 'att_00000000000000000000000000000001', kind: 'file', name: 'notes.txt', mime: 'text/plain' }] } },
    { seq: 2, kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'Older message' } },
    { seq: 3, kind: 'message.member', actor: { kind: 'member', id: 'helper' }, created_at: Number.NaN,
      payload: { text: 'Missing clock' } },
    { seq: 4, kind: 'message.user', actor: { kind: 'user', id: 'helper' }, payload: { text: 'Different principal' } }
  ]

  const members = [{ member_id: 'helper', profile: 'helper', handle: 'bot-handle' }]
  const { container } = render(<CanonicalGroupHistory binding={binding} events={events} members={members} />)

  expect(screen.getByText('bare-bot')).toBeTruthy()
  expect(screen.getByText('notes.txt')).toBeTruthy()
  expect(screen.getByText('Older message')).toBeTruthy()
  expect(screen.getByText('Missing clock')).toBeTruthy()
  expect(screen.getByText('bot-handle')).toBeTruthy()
  expect(screen.getByText('helper')).toBeTruthy()
  expect(container.querySelector('time')?.dateTime).toBe(new Date(0).toISOString())
  expect(container.querySelectorAll('time')).toHaveLength(1)
  expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
})

it('keeps failed, cancelled, deferred and unavailable work visible without printing transport markers', () => {
  const events = [
    { seq: 1, kind: 'turn.failed', payload: { member_id: 'helper', error: 'Provider timed out' } },
    { seq: 2, kind: 'turn.cancelled', payload: { member_id: 'helper', reason: 'Stopped by the operator' } },
    { seq: 3, kind: 'turn.deferred', payload: { member_id: 'helper', reason: 'Waiting for the owner' } },
    { seq: 4, kind: 'member.unavailable', payload: { member_id: 'helper', reason: 'Connection unavailable' } },
    { seq: 5, kind: 'turn.settled', payload: {} },
    { seq: 6, kind: 'turn.started', payload: {} },
    { seq: 7, kind: 'constructor', payload: {} },
    { seq: 8, kind: 'room.stop_requested', payload: {} },
    { seq: 9, kind: 'room.disbanded', payload: {} },
    { seq: 10, kind: 'authority.lost', payload: {} },
    { seq: 11, kind: 'room.activity', payload: { status: 'bounded', reason_code: 'max_rounds' } },
    { seq: 12, kind: 'room.activity', payload: { status: 'settled', reason_code: 'settled' } }
  ]

  const members = [{ member_id: 'helper', profile: 'helper', handle: 'helper', display_name: 'Code Helper' }]
  const { container } = render(<CanonicalGroupHistory binding={binding} events={events} members={members} />)

  for (const label of ['Turn failed', 'Turn cancelled', 'Turn deferred', 'Member unavailable',
    'Group stopped', 'Group disbanded', 'Group ownership changed', 'Group work limit reached']) {
    expect(screen.getByText(label)).toBeTruthy()
  }

  for (const detail of ['Provider timed out', 'Stopped by the operator', 'Waiting for the owner', 'Connection unavailable']) {
    expect(container.textContent).toContain(detail)
  }

  expect(screen.getAllByText('Code Helper')).toHaveLength(4)
  expect(screen.getByText('Group stopped').parentElement?.textContent).toBe('Group stopped')
  expect(container.textContent).not.toMatch(/turn\.(settled|started)|constructor/)
})

it.each([null, false, '', '1792152000', Number.NaN, Number.POSITIVE_INFINITY, 1e20])('omits an invalid event timestamp (%s) without losing its message', createdAt => {
  const event = { seq: 1, kind: 'message.user', created_at: createdAt, payload: { text: 'Still readable' } } as unknown as CanonicalGroupEvent
  const { container } = render(<CanonicalGroupHistory binding={binding} events={[event]} />)

  expect(screen.getByText('Still readable')).toBeTruthy()
  expect(container.querySelector('time')).toBeNull()
})
