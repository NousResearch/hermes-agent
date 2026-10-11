/**
 * The room composer's two submit sites share the app composer's middleware
 * seam: handlers registered on `composer.middleware` see the room's drafts,
 * may rewrite them (the rewrite is what ships), may cancel (null — nothing
 * sends, and the draft stays put), and receive a `context` saying which room
 * (and thread, for replies) the draft came from. With no handler registered
 * the room sends exactly as before — the seam is default-off.
 *
 * The chain itself is unit-tested in core (`contrib.test.ts`); what this file
 * pins is the ROOM WIRING, so the stubs below expose the real chain + real
 * registry while recording what the send actually received.
 */

// Test harness reaches the real core chain + registry so the wiring test pins
// production behaviour. Plugins can't reach those at runtime.
 
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Test harness only: the room-wiring test needs the real core types; production
// plugin code gets them through @hermes/plugin-sdk.
// eslint-disable-next-line no-restricted-imports
import type { ComposerDraft, ComposerMiddleware } from '@/app/chat/composer/contrib'
// eslint-disable-next-line no-restricted-imports
import { registry } from '@/contrib/registry'

import { translateBots } from './i18n-test-helper'

const { sent } = vi.hoisted(() => ({ sent: vi.fn((_group: string, _text: string, _thread: null | string) => true) }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { createGroupGateway, pluginSdkMock } = await import('./group-test-utils')
  // Test harness only — the SDK re-exports these for real plugins; the mock
  // factory needs the genuine chain to stub what production would use.
   
  const { COMPOSER_AREAS, runComposerMiddleware } = await import('@/app/chat/composer/contrib')
  const base = await pluginSdkMock(createGroupGateway().host)

  const Button = ({ children, onClick, title }: { children?: ReactNode; onClick?: () => void; title?: string }) => (
    <button onClick={onClick} title={title}>
      {children}
    </button>
  )

  return {
    ...base,
    Button,
    // The real area ids + the real chain: the view consumes the seam for
    // real; only the SDK plumbing is mocked.
    COMPOSER_AREAS,
    runComposerMiddleware,
    // The chip row: a stub that records the area id the view asked to host.
    ComposerSlot: ({ area }: { area: string }) => <span data-testid="composer-slot">{area}</span>,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: () => null,
    ConfirmDialog: () => null,
    CopyButton: () => null,
    Dialog: () => null,
    DialogContent: () => null,
    DialogDescription: () => null,
    DialogFooter: () => null,
    DialogHeader: () => null,
    DialogTitle: () => null,
    Input: () => null,
    MessageTextContent: ({ text }: { text: string }) => <span data-testid="message-text-content">{text}</span>,
    RowButton: Button,
    Tip: ({ children }: { children: ReactNode }) => children,
    ToggleRow: () => null,
    relativeTime: () => 'now',
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})

vi.mock('./avatar', () => ({ avatarColor: () => '#888', botAppearance: () => ({}), BotFace: () => null }))

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  // The input the room owns: enough surface to type and Enter, no popover.
  GroupMentionInput: (props: {
    'aria-label'?: string
    onChange: (value: string) => void
    onSubmitDraft?: () => void
    value: string
  }) => (
    <textarea
      aria-label={props['aria-label']}
      onChange={event => props.onChange(event.target.value)}
      onKeyDown={event => {
        if (event.key === 'Enter') {
          props.onSubmitDraft?.()
        }
      }}
      value={props.value}
    />
  )
}))

vi.mock('./group-rounds', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  sendToGroupChat: (group: string, _members: unknown, text: string, thread: null | string) => sent(group, text, thread)
}))

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  sent.mockClear()

  for (const dispose of disposers.splice(0)) {
    dispose()
  }
})

function addMiddleware(id: string, handler: ComposerMiddleware['handler']) {
  disposers.push(registry.register({ area: 'composer.middleware', data: { handler }, id }))
}

async function mountRoom(log: unknown[] = []) {
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')

  Element.prototype.scrollIntoView = vi.fn()

  $groupChats.set({
    Room: { epoch: 1, log: log as never, members: [{ name: 'builder' }], running: false, sessions: {}, watermarks: {} }
  } as never)

  const members = [{ name: 'builder' }] as never[]

  render(<GroupChatWorkspace group="Room" members={members} />)
}

function roomInput() {
  return screen.getByRole('textbox', { name: 'Message Room' }) as HTMLTextAreaElement
}

function typeAndEnter(input: HTMLTextAreaElement, text: string) {
  fireEvent.change(input, { target: { value: text } })
  fireEvent.keyDown(input, { key: 'Enter' })
}

describe('room composer × composer.middleware', () => {
  it('sends unchanged through an empty chain and clears the draft', async () => {
    await mountRoom()

    typeAndEnter(roomInput(), 'ship it')

    await waitFor(() => expect(sent).toHaveBeenCalledWith('Room', 'ship it', null))

    expect(roomInput().value).toBe('')
  })

  it('a null handler cancels the send and leaves the draft in place', async () => {
    addMiddleware('gate', () => null)
    await mountRoom()

    typeAndEnter(roomInput(), 'do not send')

    await new Promise(resolve => setTimeout(resolve, 0))

    expect(sent).not.toHaveBeenCalled()
    expect(roomInput().value).toBe('do not send')
  })

  it('a rewrite is what ships, and the handler sees the room context', async () => {
    const seen: ComposerDraft[] = []

    addMiddleware('mark', draft => {
      seen.push(draft)

      return { ...draft, text: `[guarded] ${draft.text}` }
    })
    await mountRoom()

    typeAndEnter(roomInput(), 'as written')

    await waitFor(() => expect(sent).toHaveBeenCalledWith('Room', '[guarded] as written', null))
    expect(seen[0]?.context).toEqual({ kind: 'group-room', roomId: 'Room' })
    expect(seen[0]?.context?.threadId).toBeUndefined()
  })

  it('a reply send carries the thread in the context', async () => {
    const seen: ComposerDraft[] = []

    addMiddleware('reader', draft => {
      seen.push(draft)

      return draft
    })
    await mountRoom([
      { at: 1, from: { kind: 'user', name: 'You' }, id: 'u1', text: 'thread opener', thread: 'a' }
    ])

    fireEvent.click(screen.getByRole('button', { name: 'Reply in thread' }))

    const reply = screen.getByRole('textbox', { name: 'Reply in thread' }) as HTMLTextAreaElement

    typeAndEnter(reply, 'me too')

    await waitFor(() => expect(sent).toHaveBeenCalledWith('Room', 'me too', 'a'))
    expect(seen[0]?.context).toEqual({ kind: 'group-room', roomId: 'Room', threadId: 'a' })
  })

  it('Enter-mashing while a chain is in flight sends only once', async () => {
    let release: ((value: void) => void) | null = null

    const gate = new Promise<void>(resolve => {
      release = resolve as () => void
    })

    addMiddleware('slow', async draft => {
      await gate

      return draft
    })
    await mountRoom()

    const input = roomInput()

    fireEvent.change(input, { target: { value: 'once' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    fireEvent.keyDown(input, { key: 'Enter' })

    release!()

    await waitFor(() => expect(sent).toHaveBeenCalledTimes(1))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(sent).toHaveBeenCalledTimes(1)
  })

  it('hosts the room chip area under the input', async () => {
    await mountRoom()

    const slots = screen.getAllByTestId('composer-slot')

    expect(slots[0]?.textContent).toBe('composer.roomBottom')
  })
})
