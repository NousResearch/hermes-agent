import { cleanup, fireEvent, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// Room bodies go through the shell's message renderer (the 1:1 chat's code
// card + `MEDIA:` transform) when the SDK exports it. The stub records what the
// room handed it so the test asserts the wiring, not the renderer's output.
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const base = await pluginSdkMock(createGroupGateway().host)

  const Button = ({
    'aria-label': ariaLabel,
    children,
    onClick,
    title
  }: {
    'aria-label'?: string
    children?: ReactNode
    onClick?: () => void
    title?: string
  }) => (
    <button aria-label={ariaLabel} onClick={onClick} title={title}>
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
    Switch: () => null,
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
  // A real input so a test can type into a composer and submit it; the shipped
  // component is the mention-aware editor.
  GroupMentionInput: ({ onChange, value }: { onChange?: (text: string) => void; value?: string }) => (
    <input onChange={event => onChange?.(event.target.value)} value={value ?? ''} />
  )
}))
afterEach(cleanup)

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


it('quotes a message into the composer, and renders the quoted line on the reply', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  $groupChats.set({
    Room: {
      log: [
        { from: { kind: 'member' as const, name: 'builder' }, id: 'm1', text: 'totals are 5', thread: 'a', at: 2 },
        {
          from: { kind: 'user' as const, name: 'You' },
          id: 'u1',
          replyTo: { at: 2, from: 'Builder', text: 'totals are 5' },
          text: 'no — the second number',
          thread: 'a',
          at: 3
        }
      ],
      sessions: {},
      watermarks: {}
    }
  })

  const { container, getByLabelText, getByText } = render(
    <GroupChatWorkspace group="Room" members={[{ name: 'builder' }] as never} />
  )

  // The reference renders above the message that answers it.
  const quote = container.querySelector('[data-slot="group-quote"]')
  expect(quote?.textContent).toContain('totals are 5')

  // Quoting a message arms the composer with the same line.
  fireEvent.click(getByLabelText('Quote Builder'))
  expect(container.querySelector('[data-slot="group-quote-draft"]')?.textContent).toContain('totals are 5')

  // …and the composer can drop it again.
  fireEvent.click(getByLabelText('Clear quote'))
  expect(container.querySelector('[data-slot="group-quote-draft"]')).toBeNull()
  expect(getByText('no — the second number')).toBeTruthy()
})


it('moves the armed quote into a thread reply box and consumes it on send', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  $groupChats.set({
    Room: {
      log: [
        { from: { kind: 'member' as const, name: 'builder' }, id: 'm1', text: 'totals are 5', thread: 'a', at: 2 }
      ],
      sessions: {},
      watermarks: {}
    }
  })

  const { container, getByLabelText, getByText } = render(
    <GroupChatWorkspace group="Room" members={[{ name: 'builder' }] as never} />
  )

  fireEvent.click(getByLabelText('Quote Builder'))
  expect(container.querySelectorAll('[data-slot="group-quote-draft"]')).toHaveLength(1)

  // The reply box takes the quote over — and still shows it while typing.
  fireEvent.click(getByLabelText('Reply to Builder'))
  const strip = container.querySelector('[data-slot="group-quote-draft"]')
  expect(strip).toBeTruthy()
  expect(strip?.closest('form')).toBeTruthy()

  const replyForm = strip!.closest('form') as HTMLFormElement
  fireEvent.change(replyForm.querySelector('input') as HTMLElement, { target: { value: 'agreed' } })
  fireEvent.submit(replyForm)

  // Consumed, not leaked: the reply carries the quote, and nothing stays armed
  // for the next main-composer send.
  expect(container.querySelector('[data-slot="group-quote-draft"]')).toBeNull()
  expect($groupChats.get().Room.log.at(-1)?.replyTo?.text).toBe('totals are 5')
  expect(getByText('agreed')).toBeTruthy()
})
