import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { cleanup, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// Both room implementations share the conversation typography scope.
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
    MessageTextContent: ({ text }: { text: string }) => (
      <p>
        <span data-testid="renderer-text">{text}</span>
      </p>
    ),
    Tip: ({ children }: { children: ReactNode }) => children,
    relativeTime: () => 'now',
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' }, composer: {}, fileMenu: {} } }),
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
afterEach(() => {
  cleanup()
  document.documentElement.style.removeProperty('--chat-text-scale')
})

it.each([110, 150])('shares conversation typography at %s percent in canonical and legacy rooms', async scale => {
  const { setChatTextScale } = await import('@/store/chat-text-scale')
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  const { CanonicalGroupHistory } = await import('./canonical-group-history')

  setChatTextScale(scale)
  $groupChats.set({
    Room: {
      log: [
        { id: 'u1', thread: 'a', from: { kind: 'user', name: 'You' }, text: 'Legacy user', at: 1 },
        { id: 'm1', thread: 'a', from: { kind: 'member', name: 'builder' }, text: 'Legacy member', at: 2 }
      ],
      watermarks: {},
      sessions: {}
    }
  })

  const { container, getAllByTestId } = render(
    <>
      <GroupChatWorkspace group="Room" members={[{ name: 'builder' }]} />
      <CanonicalGroupHistory
        binding={{ connectionId: 'local', profile: 'default', roomId: 'room' }}
        events={[
          { seq: 1, kind: 'message.user', payload: { text: 'Canonical user' } },
          { seq: 2, kind: 'message.member', payload: { text: 'Canonical member' } }
        ]}
      />
    </>
  )
  await waitFor(() => expect(getAllByTestId('renderer-text')).toHaveLength(4))

  expect(document.documentElement.style.getPropertyValue('--chat-text-scale')).toBe(String(scale / 100))
  const bodies = container.querySelectorAll('[data-slot="group-chat-message-content"]')
  expect(bodies).toHaveLength(4)

  for (const body of bodies) {
    const style = getComputedStyle(body)
    // jsdom preserves var()/calc(); Chromium verification covers pixel sizes.
    expect(style.fontSize).toBe('var(--conversation-text-font-size)')
    expect(style.getPropertyValue('--conversation-text-scale')).toBe('var(--chat-text-scale, 1)')
    expect(style.getPropertyValue('--conversation-text-font-size')).toBe(
      'calc(var(--conversation-text-base-size) * var(--chat-text-scale, 1))'
    )
  }
})
