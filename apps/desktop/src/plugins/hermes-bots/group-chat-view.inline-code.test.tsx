import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// Without a room-owned hook the room's `<code>` falls through to Tailwind
// Typography's fixed near-black ink, which is invisible on every dark theme
// (#114086). The renderer stub emits a bare `<code>` with NO `.aui-md`
// wrapper, so only the room's own `[data-slot='group-chat-message-content']`
// rule in the real stylesheet can theme it: the cascade decides, not a regex
// over the source text.
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
    Switch: () => null,
    ConfirmDialog: () => null,
    Dialog: () => null,
    DialogContent: () => null,
    DialogDescription: () => null,
    DialogFooter: () => null,
    DialogHeader: () => null,
    DialogTitle: () => null,
    Input: () => null,
    MessageTextContent: ({ text }: { text: string }) => (
      <p>
        set <code data-testid="renderer-code">{text}</code> first
      </p>
    ),
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

const STYLES = resolve(dirname(fileURLToPath(import.meta.url)), '../../styles.css')

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  const sheet = document.createElement('style')
  sheet.textContent = readFileSync(STYLES, 'utf8')
  document.head.appendChild(sheet)
})
afterEach(cleanup)

it('themes inline code in room message bodies with the chat inline-code tokens', async () => {
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  const log = [
    { id: 'm1', thread: 'a', from: { kind: 'member' as const, name: 'builder' }, text: 'discover_models', at: 1 }
  ]

  $groupChats.set({ Room: { log, watermarks: {}, sessions: {} } })

  const { getByTestId } = render(<GroupChatWorkspace group="Room" members={[{ name: 'builder' }] as never} />)
  const style = getComputedStyle(getByTestId('renderer-code'))

  expect(style.color).toBe('var(--ui-inline-code-foreground)')
  expect(style.background).toBe('var(--ui-inline-code-background)')
})
