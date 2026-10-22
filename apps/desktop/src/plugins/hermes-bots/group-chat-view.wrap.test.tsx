import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { cleanup, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// #70451: the group room renders bot replies through the raw Streamdown path
// (no code card), so a wide fenced block used to inherit `[&_pre]:overflow-x-auto`
// on the room's message-body wrapper — growing a scrollbar under the room log,
// or worse, clipping with no scroll affordance (#91706). The wrapper now
// soft-wraps `pre` content; the room body itself never scrolls sideways.
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
    ToggleRow: () => null,
    ConfirmDialog: () => null,
    Dialog: () => null,
    DialogContent: () => null,
    DialogDescription: () => null,
    DialogFooter: () => null,
    DialogHeader: () => null,
    DialogTitle: () => null,
    Input: () => null,
    MessageTextContent: undefined,
    // The room's Streamdown fallback renders a bare <pre> for fenced code —
    // wide content must wrap via the room wrapper, not scroll sideways.
    Streamdown: ({ children }: { children?: ReactNode }) => <pre>{children}</pre>,
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

it('soft-wraps fenced code in room message bodies instead of scrolling sideways', async () => {
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  const log = [
    {
      id: 'm1',
      thread: 'a',
      from: { kind: 'member' as const, name: 'builder' },
      text: '```js\nconst x = 1\n```',
      at: 1
    }
  ]

  $groupChats.set({ Room: { log, watermarks: {}, sessions: {} } })

  const { container } = render(<GroupChatWorkspace group="Room" members={[{ name: 'builder' }] as never} />)

  // The room paints once the canonical `groups.capabilities` gate resolves.
  const body = await waitFor(() => {
    const found = container.querySelector('[data-slot="group-chat-message-content"]')
    expect(found).toBeTruthy()

    return found!
  })

  // The room body wrapper wraps pre content rather than growing an X scrollbar.
  expect(body.className).toContain('[&_pre]:whitespace-pre-wrap')
  expect(body.className).toContain('[&_pre]:wrap-anywhere')
  expect(body.className).not.toContain('[&_pre]:overflow-x-auto')

  // The real Streamdown fallback path rendered the fence as a <pre>.
  const pre = body.querySelector('pre')
  expect(pre?.textContent).toContain('const x = 1')
})
