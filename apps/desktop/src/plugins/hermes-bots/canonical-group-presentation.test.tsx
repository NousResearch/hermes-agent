import { cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ request: vi.fn(), locale: 'en' as 'en' | 'ja' }))
vi.mock('@hermes/plugin-sdk', async () => ({
  host: { requestProfile: mocks.request },
  ...await import('@/components/ui/button'),
  ...await import('@/components/ui/textarea'),
  ...await import('@/components/ui/codicon'),
  Tip: ({ children }: { children: ReactNode }) => <>{children}</>
}))
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES[mocks.locale],
    back: 'Back', refresh: 'Refresh', retry: 'Retry', send: 'Send', stop: 'Stop',
    download: 'Download', discard: 'Discard', cancel: 'Cancel' }) }
})

import { CanonicalGroupHistory } from './canonical-group-history'
import type { CanonicalGroupEvent } from './canonical-group-history'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const binding = { connectionId: 'original-owner', profile: 'reviewer', roomId: 'room-one' }

const event = (seq: number, kind: string, payload: CanonicalGroupEvent['payload'] = {}): CanonicalGroupEvent => ({
  seq, kind, payload, room_id: binding.roomId, event_id: `event-${seq}`, actor: { kind: 'gateway', id: 'install:private-id' }
})

afterEach(() => { cleanup(); mocks.request.mockReset(); mocks.locale = 'en'; localStorage.clear() })

it('labels transport users and system events without exposing their internal speaker IDs', () => {
  const view = render(<CanonicalGroupHistory binding={binding} events={[
    { ...event(1, 'message.user', { text: 'Hello' }), actor: { kind: 'user', id: 'desktop' } },
    event(2, 'turn.failed', { text: 'Please retry later' }),
    { ...event(3, 'message.member', { text: 'Reply' }), actor: { kind: 'member', id: 'desktop', display_name: 'Original writer' } },
    { ...event(4, 'message.user', { text: 'Follow-up' }), actor: { kind: 'user', id: 'native-user', display_name: 'Room participant' } }
  ]} />)

  expect([...view.container.querySelectorAll('strong')].map(node => node.textContent)).toEqual([
    'User: ', 'System: ', 'Original writer: ', 'Room participant: '
  ])
  expect(view.container.textContent).not.toContain('install:private-id')
  expect(mocks.request).not.toHaveBeenCalled()
})

it('hides only routine bookkeeping, keeping failures, retry, approval and unknown events', () => {
  render(<CanonicalGroupHistory binding={binding} events={[
    event(1, 'turn.settled'), event(2, 'room.activity'),
    event(3, 'turn.failed', { text: 'Failed to produce a reply' }),
    event(4, 'turn.deferred', { text: 'Retry is available' }),
    event(5, 'approval.requested', { text: 'Approval needed' }),
    event(6, 'future.notice'), event(7, 'turn.cancelled')
  ]} />)
  expect(screen.queryByText('turn.settled')).toBeNull()
  expect(screen.queryByText('room.activity')).toBeNull()

  for (const text of ['Failed to produce a reply', 'Retry is available', 'Approval needed', 'future.notice', 'turn.cancelled']) {
    expect(screen.getByText(text, { exact: false })).toBeTruthy()
  }

  expect(mocks.request).not.toHaveBeenCalled()
})

it('does not hide content or event-scoped files on a bookkeeping-kind event', () => {
  render(<CanonicalGroupHistory binding={binding} events={[
    event(1, 'turn.settled', { text: 'An explicit notice' }),
    event(2, 'room.activity', { attachments: [{ attachment_id: 'att_00000000000000000000000000000001',
      kind: 'file', name: 'result.txt', mime: 'text/plain', size: 1 }] }),
    event(3, 'room.activity', { content: 'Another explicit notice' })
  ]} />)
  expect(screen.getByText('An explicit notice', { exact: false })).toBeTruthy()
  expect(screen.getByText('Another explicit notice', { exact: false })).toBeTruthy()
  expect(screen.getByText('result.txt')).toBeTruthy()
  expect((screen.getByRole('button', { name: 'Download' }) as HTMLButtonElement).disabled).toBe(false)
  expect(mocks.request).not.toHaveBeenCalled()
})

it.each([binding.roomId, 'another-room'])('uses only same-room member labels from snapshot %s', async roomId => {
  mocks.request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { room_id: roomId, name: 'Original room', members: [
      { member_id: 'default', profile: 'default', handle: 'hermes' }
    ] }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [
      { ...event(1, 'message.bot', { text: 'Reply from current room' }), actor: { kind: 'bot', id: 'default' } },
      { ...event(2, 'message.bot', { text: 'Different event room' }), room_id: 'another-room', actor: { kind: 'bot', id: 'default' } },
      { ...event(3, 'message.bot', { text: 'Original speaker' }), actor: { kind: 'bot', id: 'default', display_name: 'Recorded name' } }
    ] }}

    throw new Error(`Unexpected mutation: ${method}`)
  })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('Reply from current room', { exact: false })
  expect([...view.container.querySelectorAll('strong')].map(node => node.textContent)).toEqual([
    roomId === binding.roomId ? '@hermes: ' : 'default: ', 'default: ', 'Recorded name: '
  ])
  expect(mocks.request.mock.calls.every(call => ['groups.state', 'groups.log'].includes(call[1]))).toBe(true)
})

it.each([['en', 'Message the group…', 'User', 'System'], ['ja', 'グループにメッセージを送信…', 'ユーザー', 'システム']] as const)(
  'uses the shared visible composer and localized transcript labels in %s', async (locale, placeholder, user, system) => {
    mocks.locale = locale
    mocks.request.mockImplementation(async (_route, method) => {
      if (method === 'groups.state') {return { room: { name: 'Original room' }, driver_status: {} }}

      if (method === 'groups.log') {return { events: [
        { ...event(1, 'message.user', { text: 'Question' }), actor: { kind: 'user', id: 'desktop' } },
        event(2, 'future.notice')
      ] }}

      throw new Error(`Unexpected mutation: ${method}`)
    })
    render(<CanonicalGroupWorkspace binding={binding} onBack={vi.fn()} />)
    const composer = screen.getByRole('textbox') as HTMLTextAreaElement
    await waitFor(() => expect(composer.disabled).toBe(false))
    expect(composer.getAttribute('aria-label')).toBeTruthy()
    expect(composer.placeholder).toBe(placeholder)
    expect(composer.dataset.slot).toBe('textarea')
    expect(composer.classList.contains('desktop-input-chrome')).toBe(true)
    expect(screen.getByText(`${user}:`)).toBeTruthy()
    expect(screen.getByText(`${system}:`)).toBeTruthy()

    for (const [name, icon] of [['Back', 'arrow-left'], ['Stop', 'debug-stop']] as const) {
      const button = screen.getByRole('button', { name })
      expect(button.querySelector(`.codicon-${icon}`)).toBeTruthy()
      expect(button.dataset.variant).toBe('ghost')
    }

    const attach = screen.getByRole('button', { name: locale === 'en' ? 'Attach files' : 'ファイルを添付' })
    expect(attach.querySelector('.codicon-attach')).toBeTruthy()
    expect(attach.dataset.variant).toBe('ghost')
    expect(attach.getAttribute('type')).toBe('button')
    expect(mocks.request.mock.calls.every(call => ['groups.state', 'groups.log'].includes(call[1]))).toBe(true)
  }
)
