import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const base = await pluginSdkMock(createGroupGateway().host)
  const { useStore } = await import('@nanostores/react')

  const Button = ({ size: _size, variant: _variant, ...props }: ComponentProps<'button'> & { size?: string; variant?: string }) => (
    <button {...props} />
  )

  return {
    ...base,
    useValue: useStore,
    Button,
    RowButton: Button,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: () => null,
    CopyButton: () => null,
    ToggleRow: () => null,
    ConfirmDialog: () => null,
    Dialog: () => null,
    DialogContent: () => null,
    DialogDescription: () => null,
    DialogFooter: () => null,
    DialogHeader: () => null,
    DialogTitle: () => null,
    Input: (props: ComponentProps<'input'>) => <input {...props} />,
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

it('keeps an astral character intact when saving and projecting a 120-code-point title', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats, groupChatSyncSnapshot, durableGroupChatRooms } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  const title = 'a'.repeat(119) + '😀'

  $groupChats.set({ Room: { log: [{ id: 'root', thread: 'topic', at: 1, from: { kind: 'user', name: 'You' }, text: 'Root' }], watermarks: {} } })
  const view = render(<GroupChatWorkspace group="Room" members={[]} />)

  fireEvent.click(await view.findByRole('button', { name: 'Rename…: Root' }))
  fireEvent.change(view.getByRole('textbox', { name: 'Rename…' }), { target: { value: title } })
  fireEvent.click(view.getByRole('button', { name: 'Save' }))
  expect(durableGroupChatRooms().Room.threadTitles?.topic).toBe(title)
  expect(groupChatSyncSnapshot($groupChats.get()).rooms['name:Room'].threadTitles?.topic).toBe(title)
  expect(view.getByRole('button', { name: `Collapse this thread: ${title}` })).toBeDefined()
})

it('names and explicitly collapses one thread without hiding other topics or changing arrival order', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  const log = [
    { id: 'u1', thread: 'a', from: { kind: 'user' as const, name: 'You' }, text: '**Build** request', at: 1 },
    { id: 'u2', thread: 'b', from: { kind: 'user' as const, name: 'You' }, text: '', at: 2 },
    { id: 'm1', thread: 'a', from: { kind: 'member' as const, name: 'Builder' }, text: 'Build complete', at: 3 }
  ]

  $groupChats.set({ Room: { log, watermarks: {}, sessions: {} } })
  const view = render(<GroupChatWorkspace group="Room" members={[]} />)
  const toggle = await view.findByRole('button', { name: 'Collapse this thread: Build request' })

  expect(toggle.getAttribute('aria-expanded')).toBe('true')
  expect(view.getByRole('button', { name: 'Collapse this thread: Thread' }).getAttribute('aria-expanded')).toBe('true')
  expect(view.getByText('Build complete').closest('[hidden]')).toBeNull()
  fireEvent.click(toggle)
  expect(toggle.getAttribute('aria-expanded')).toBe('false')
  expect(view.getByText('Build complete').closest('[hidden]')).not.toBeNull()
  expect(view.getByRole('button', { name: 'Collapse this thread: Thread' }).closest('[hidden]')).toBeNull()
  expect($groupChats.get().Room.log).toEqual(log)

  act(() => $groupChats.set({ Room: { ...$groupChats.get().Room, log: [...log,
    { id: 'm2', thread: 'b', from: { kind: 'member', name: 'Reviewer' }, text: 'Other reply', at: 4 }
  ] } }))
  expect(view.getByText('Other reply').closest('[hidden]')).toBeNull()
  fireEvent.click(toggle)
  expect(view.getByText('Build complete').closest('[hidden]')).toBeNull()
  const messages = [...view.container.querySelectorAll('[data-group-entry]')].map(node => node.textContent)

  expect(messages[0]).toContain('Build')
  expect(messages[2]).toContain('Build complete')
  expect(messages[3]).toContain('Other reply')
  fireEvent.click(view.getByRole('button', { name: 'Rename…: Build request' }))
  fireEvent.change(view.getByRole('textbox', { name: 'Rename…' }), { target: { value: 'Release checklist' } })
  fireEvent.click(view.getByRole('button', { name: 'Save' }))
  expect($groupChats.get().Room.threadTitles).toEqual({ a: 'Release checklist' })
  expect(view.getByRole('button', { name: 'Collapse this thread: Release checklist' }).closest('[hidden]')).toBeNull()
  view.unmount()
  const reopened = render(<GroupChatWorkspace group="Room" members={[]} />)

  expect((await reopened.findByRole('button', { name: 'Collapse this thread: Release checklist' })).getAttribute('aria-expanded')).toBe('true')
})

it('keeps every public member reply readable in room arrival order across interleaved threads', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  const log = [
    { id: 'u1', thread: 'a', from: { kind: 'user' as const, name: 'You' }, text: 'Build request', at: 1 },
    { id: 'm1', thread: 'a', from: { kind: 'member' as const, name: 'Builder' }, text: 'Build complete', at: 2 },
    { id: 'u2', thread: 'b', from: { kind: 'user' as const, name: 'You' }, text: 'Release request', at: 3 },
    { id: 'm2', thread: 'a', from: { kind: 'member' as const, name: 'Reviewer' }, text: 'QA complete', at: 4 },
    { id: 'm3', thread: 'b', from: { kind: 'member' as const, name: 'Lead' }, text: 'Release ready', at: 5 }
  ]

  $groupChats.set({ Room: { log, watermarks: {}, sessions: {} } })
  const { container } = render(<GroupChatWorkspace group="Room" members={[]} />)
  await waitFor(() => expect(container.textContent).toContain('Build request'))
  const text = [...container.querySelectorAll('[data-group-entry]')].map(node => node.textContent).join('\n')
  let previous = -1

  for (const entry of log) {
    const index = text.indexOf(entry.text)
    const node = container.querySelector(`[data-group-entry="${entry.id}"]`)

    expect(node).not.toBeNull()
    expect(node!.closest('[hidden]')).toBeNull()
    expect(index, `${entry.from.name}: ${entry.text} must remain visible in arrival order`).toBeGreaterThan(previous)
    expect(text.split(entry.text)).toHaveLength(2)
    previous = index
  }

  for (const name of ['Builder', 'Reviewer', 'Lead']) {
    expect(text).toContain(name)
  }
})
