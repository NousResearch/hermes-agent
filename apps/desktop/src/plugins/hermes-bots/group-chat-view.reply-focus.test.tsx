import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const { en } = await import('@/i18n/en')

  return {
    ...await pluginSdkMock(createGroupGateway().host),
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button {...p} />,
    Tip: ({ children }: {children: ReactNode}) => children,
    CopyButton: () => null,
    ConfirmDialog: () => null,
    Dialog: () => null,
    DialogContent: () => null,
    DialogDescription: () => null,
    DialogFooter: () => null,
    DialogHeader: () => null,
    DialogTitle: () => null,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    ToggleRow: () => null,
    MessageTextContent: ({text}: {text: string}) => <span>{text}</span>,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./avatar', () => ({ avatarColor: () => '#888', botAppearance: () => ({}), BotBadge: () => null }))
afterEach(cleanup)

it('legacy Reply focuses the seeded main or matching thread input without disturbing another draft', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  $groupChats.set({ Review: {log: [
    {id: 'a', thread: 'a', from: {kind: 'member', name: 'builder'}, text: 'thread A', at: 1},
    {id: 'b', thread: 'b', from: {kind: 'member', name: 'ops'}, text: 'thread B', at: 2}
  ], watermarks: {}, sessions: {} } })
  render(<GroupChatWorkspace group="Review" members={[{name: 'builder'}, {name:'ops'}]} />)
  const reply = await screen.findByRole('button', {name: 'Reply to Builder'})
  const main = screen.getByRole('textbox') as HTMLTextAreaElement
  fireEvent.change(main, {target: {value: 'existing draft'}})
  reply.focus()
  fireEvent.click(reply)
  await waitFor(() => expect(window.document.activeElement).toBe(main))
  expect(main.value).toBe('@builder existing draft')
  expect(main.selectionStart).toBe(main.value.length)
  expect(main.selectionEnd).toBe(main.value.length)
  fireEvent.click(screen.getAllByRole('button', {name: 'Reply in thread'})[0])
  const thread = screen.getByRole('textbox', {name: 'Reply in thread'}) as HTMLTextAreaElement
  fireEvent.change(thread, {target: {value: 'thread draft'}})
  reply.focus()
  fireEvent.click(reply)
  await waitFor(() => expect(window.document.activeElement).toBe(thread))
  expect(thread.value).toBe('@builder thread draft')
  expect(thread.selectionStart).toBe(thread.value.length)
  expect(thread.selectionEnd).toBe(thread.value.length)
  const other = screen.getByRole('button', {name:'Reply to Ops'})
  other.focus()
  fireEvent.click(other)
  await waitFor(() => expect(window.document.activeElement).toBe(main))
  expect(main.value).toBe('@ops @builder existing draft')
  expect(thread.value).toBe('@builder thread draft')
})
