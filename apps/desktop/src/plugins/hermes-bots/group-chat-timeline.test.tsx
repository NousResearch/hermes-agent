import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

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
  const text = container.textContent || ''
  let previous = -1

  for (const entry of log) {
    const index = text.indexOf(entry.text)
    expect(index, `${entry.from.name}: ${entry.text} must remain visible in arrival order`).toBeGreaterThan(previous)
    expect(text.split(entry.text)).toHaveLength(2)
    previous = index
  }

  for (const name of ['Builder', 'Reviewer', 'Lead']) {
    expect(text).toContain(name)
  }
})
