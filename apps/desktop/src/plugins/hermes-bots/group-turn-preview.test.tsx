/**
 * The room's working line read "X is thinking…" for the whole length of a
 * member turn, even while that member was visibly deep in a tool call — the
 * "is it stuck?" read the activity rows were added for. The line now names the
 * tool the CURRENT turn is running (runtime-only, cleared with the turn) and
 * still falls back to the thinking line when the turn ran none.
 */

import { cleanup, render, screen } from '@testing-library/react'
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

it('names a tool from this turn only, and the latest one it ran', async () => {
  const { pickCurrentTool } = await import('./group-turns')

  const thisTurn = [
    { role: 'user', text: 'go' },
    { role: 'tool', tool_name: 'read_file' },
    { role: 'assistant', text: 'streaming…' },
    { role: 'tool', tool_name: 'patch_file' }
  ]

  // Newest-first, so a turn that used several tools reports the latest.
  expect(pickCurrentTool(thisTurn, 0)).toBe('patch_file')
  expect(pickCurrentTool(thisTurn, 3)).toBe('patch_file')

  // Rows before the turn's own prompt belong to an earlier turn: naming their
  // tool would report work this turn never ran.
  const quietTurn = [
    { role: 'tool', tool_name: 'read_file' },
    { role: 'user', text: 'go' },
    { role: 'assistant', text: 'thinking…' }
  ]

  expect(pickCurrentTool(quietTurn, 0)).toBe('read_file')
  expect(pickCurrentTool(quietTurn, 1)).toBeNull()
})

it('shows the running tool on the working line, and thinking when the turn ran none', async () => {
  Element.prototype.scrollIntoView = vi.fn()

  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  const members = [{ name: 'builder' }] as never
  const room = { log: [], running: true, sessions: {}, turn: { name: 'builder' }, watermarks: {} }

  $groupChats.set({ Room: { ...room, turnPreview: 'patch_file.py' } })
  render(<GroupChatWorkspace group="Room" members={members} />)

  // displayName() title-cases the roster name.
  expect(screen.getByText('Builder is working: patch_file.py')).toBeTruthy()
  expect(screen.queryByText('Builder is thinking…')).toBeNull()

  cleanup()
  $groupChats.set({ Room: room })
  render(<GroupChatWorkspace group="Room" members={members} />)

  expect(screen.getByText('Builder is thinking…')).toBeTruthy()
})
