import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// Room bodies go through the shell's message renderer (the 1:1 chat's code
// card + `MEDIA:` transform) when the SDK exports it. The stub records what the
// room handed it so the test asserts the wiring, not the renderer's output.
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const base = await pluginSdkMock(createGroupGateway().host)

  const Button = ({ children, onClick, title }: { children?: ReactNode; onClick?: () => void; title?: string }) => (
    <button onClick={onClick} title={title}>
      {children}
    </button>
  )

  return {
    ...base,
    Button,
    RowButton: Button,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: () => null,
    CopyButton: () => null,
    ConfirmDialog: () => null,
    Dialog: () => null,
    DialogContent: () => null,
    DialogDescription: () => null,
    DialogFooter: () => null,
    DialogHeader: () => null,
    DialogTitle: () => null,
    Input: () => null,
    MessageTextContent: ({ media = true, text }: { media?: boolean; text: string }) => (
      <span data-media={String(media)} data-testid="message-text-content">
        {text}
      </span>
    ),
    ToggleRow: () => null,
    Tip: ({ children }: { children: ReactNode }) => children,
    relativeTime: () => 'now',
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./avatar', () => ({ avatarColor: () => '#888', botAppearance: () => ({}), BotFace: () => null }))
vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: () => null
}))
afterEach(cleanup)

it('pins group comments to the durable room and hosting route across an ambient connection switch', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { host } = await import('@hermes/plugin-sdk')
  const connection = vi.spyOn(host.state.connectionId!, 'get').mockReturnValue('room-host-B')
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  const { capturePreviewAnnotateDestination } = await import('@/lib/preview-annotate/handoff')
  const listeners = new Set<(packet: unknown) => void>()
  const send = vi.fn()
  Object.assign(window, { hermesDesktop: { windowRelay: {
    send,
    onMessage: (listener: (packet: unknown) => void) => {
      listeners.add(listener)

      return () => listeners.delete(listener)
    }
  } } })

  try {
    $groupChats.set({ Bound: { roomId: 'original-room', epoch: 1, log: [], sessions: {}, watermarks: {} } })
    const view = render(<GroupChatWorkspace group="Bound" members={[]} />)
    const destination = capturePreviewAnnotateDestination()!
    expect(destination.conversation).toMatchObject({ kind: 'group', connectionId: 'room-host-B', profile: 'default' })

    const deliver = (owner = destination) => act(() => {
      for (const listener of listeners) {
        listener({ type: 'preview-annotate-handoff', requestId: crypto.randomUUID(), destination: owner, prompt: 'Bound comment', images: [], count: 1 })
      }
    })

    connection.mockReturnValue('ambient-A')
    view.rerender(<GroupChatWorkspace group="Bound" members={[]} />)
    deliver({ ...destination, conversation: { ...destination.conversation!, connectionId: 'ambient-A' } })
    expect(send).not.toHaveBeenCalled()
    deliver()
    expect(send).toHaveBeenCalledWith(expect.objectContaining({ ok: true }))
    send.mockClear()
    act(() => $groupChats.set({ Bound: { roomId: 'replacement-room', epoch: 2, log: [], sessions: {}, watermarks: {} } }))
    deliver()
    expect(send).not.toHaveBeenCalled()
    act(() => $groupChats.set({}))
    deliver()
    expect(send).not.toHaveBeenCalled()
  } finally {
    cleanup()
    connection.mockRestore()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  }
})

it('renders member replies through the shell message renderer, resolving media only for members on this gateway', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  const log = [
    { id: 'u1', thread: 'a', from: { kind: 'user' as const, name: 'You' }, text: 'Show me', at: 1 },
    { id: 'm1', thread: 'a', from: { kind: 'member' as const, name: 'builder' }, text: 'MEDIA:/tmp/local.png', at: 2 },
    {
      id: 'm2',
      thread: 'a',
      from: { kind: 'member' as const, name: 'builder', source: 'mini' },
      text: 'MEDIA:/tmp/remote.png',
      at: 3
    }
  ]

  const members = [
    { name: 'builder' },
    { connectionId: 'mini', connectionLabel: 'mini', name: 'builder', remoteSource: true, sourceScoped: true }
  ] as never

  $groupChats.set({ Room: { log, watermarks: {}, sessions: {} } })
  const { getAllByTestId } = render(<GroupChatWorkspace group="Room" members={members} />)
  const bodies = getAllByTestId('message-text-content').map(el => [el.textContent, el.dataset.media])

  expect(bodies).toEqual([
    ['Show me', 'true'],
    ['MEDIA:/tmp/local.png', 'true'],
    ['MEDIA:/tmp/remote.png', 'false']
  ])
})

it('removes Stop controls from historical working rows after the room settles', async () => {
  Element.prototype.scrollIntoView = vi.fn()

  const [{ $groupChats }, activity, { GroupChatWorkspace }] = await Promise.all([
    import('./group-chat'),
    import('./group-activity'),
    import('./group-chat-view')
  ])

  $groupChats.set({
    Settled: {
      epoch: 1,
      log: [],
      members: [{ name: 'builder' }],
      running: false,
      sessions: {},
      watermarks: {}
    }
  })
  activity.recordGroupActivity('Settled', { kind: 'working', member: 'builder' })
  activity.recordGroupActivity('Settled', { kind: 'replied', member: 'builder' })
  activity.recordGroupActivity('Settled', { kind: 'settled', member: null })

  render(<GroupChatWorkspace group="Settled" members={[{ name: 'builder' }]} />)
  fireEvent.click(screen.getByRole('button', { name: /^Activity/ }))

  expect(screen.getByText('builder is working…')).toBeTruthy()
  expect(screen.getByText('builder replied')).toBeTruthy()
  expect(screen.getByText('turn settled')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
})
