import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import type { Attachment, GroupMember } from './types'

const { host, sendToGroupChatDurably } = vi.hoisted(() => ({
  host: {} as Record<string, unknown>,
  sendToGroupChatDurably: vi.fn(
    async (_group: string, _members: GroupMember[], _text: string, _thread: null | string, images: Attachment[]) => {
      images[0].uploadId = 'stable-upload-id'
      throw new Error('The connected device rejected this action.')
    }
  )
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)

  return {
    ...base,
    Button: (props: React.ComponentProps<'button'>) => <button type={props.type || 'button'} {...props} />,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: ({ name }: { name: string }) => <span aria-hidden data-icon={name} />,
    ConfirmDialog: () => null,
    CopyButton: () => null,
    Dialog: ({ children, open }: { children: React.ReactNode; open: boolean }) =>
      open ? <div role="dialog">{children}</div> : null,
    DialogContent: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogDescription: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogFooter: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogHeader: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    DialogTitle: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
    relativeTime: () => 'now',
    RowButton: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    Tip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    useI18n: () => ({ t: { common: { cancel: 'Cancel', save: 'Save' } } }),
    usePluginI18n: () => translateBots
  }
})

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: (props: { 'aria-label'?: string; value?: string }) => (
    <textarea aria-label={props['aria-label']} readOnly value={props.value} />
  )
}))

vi.mock('./group-rounds', () => ({
  sendToGroupChatDurably,
  stopGroupThread: vi.fn()
}))

const MEMBERS: GroupMember[] = [
  { connectionId: 'gateway-a', name: 'research', sourceScoped: true, targetProfile: 'research' },
  { connectionId: 'gateway-a', name: 'builder', sourceScoped: true, targetProfile: 'builder' }
]

beforeEach(() => {
  vi.resetModules()
  Object.assign(host, { notify: vi.fn() })
  Object.defineProperty(Element.prototype, 'scrollIntoView', {
    configurable: true,
    value: vi.fn()
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('hosted Group Chat composer durability', () => {
  it.each(['groups.state', 'groups.log'])(
    'keeps settings read-only after a revoked %s response and preserves recovery',
    async window => {
      const { scriptedStorage } = await import('./group-test-utils')
      const { GroupChatWorkspace } = await import('./group-chat-view')
      const chat = await import('./group-chat')
      const runtime = await import('./hosted-room-runtime')
      const shared = await import('./shared')
      let persistent = false
      let hold = false
      let entered!: () => void
      let release!: (value: unknown) => void

      const started = new Promise<void>(resolve => {
        entered = resolve
      })

      const held = new Promise<unknown>(resolve => {
        release = resolve
      })

      const room = {
        room_id: 'room-1',
        name: 'Core',
        authority_gateway_id: 'install:home',
        authority_epoch: 1,
        latest_seq: 0,
        members: MEMBERS.map(member => ({ member_id: member.name, profile: member.name }))
      }

      const state = { room }
      const log = { events: [], latest_seq: 0, has_more: false }
      Object.assign(host, {
        activeConnectionId: () => 'gateway-a',
        profileRoutes: async () => [
          { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' }
        ],
        requestProfile: async (_route: unknown, method: string) => {
          if (method === 'groups.capabilities') {
            return { driver: true, persistent_process: persistent, authority_gateway_id: 'install:home' }
          }

          if (method === 'groups.list') {
            return { rooms: [room], next_offset: null }
          }

          if (hold && method === window) {
            entered()

            return held
          }

          if (method === 'groups.state') {
            return state
          }

          if (method === 'groups.log') {
            return log
          }

          throw new Error(`Unexpected ${method}`)
        },
        state: Object.fromEntries(
          [
            ['connectionId', 'gateway-a'],
            ['gateway', 'open'],
            ['profile', 'default']
          ].map(([key, value]) => [key, { get: () => value, listen: () => () => undefined }])
        )
      })
      const ctx = scriptedStorage(new Map())
      shared.setPluginCtx(ctx)
      chat.$groupChats.set({ Core: { roomId: 'room-1', members: MEMBERS, log: [], watermarks: {} } })

      try {
        await runtime.startHostedRoomRuntime(ctx.storage)
        const view = render(<GroupChatWorkspace group="Core" members={MEMBERS} />)
        fireEvent.click(screen.getByRole('button', { name: 'Group settings for Core' }))
        expect(screen.getByRole('dialog').textContent).toContain('Read-only history')
        hold = true
        persistent = true
        const refresh = runtime.refreshHostedRooms()
        await started
        persistent = false
        await runtime.probeHostedRoomMembers(MEMBERS)
        release(window === 'groups.state' ? state : log)
        await refresh
        view.rerender(<GroupChatWorkspace group="Core" members={MEMBERS} />)
        expect(screen.getByRole('dialog').textContent).toContain('Read-only history')
        expect(screen.getByRole('dialog').textContent).not.toContain('Bots can continue while Desktop is closed.')
        expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true)
        hold = false
        persistent = true
        await runtime.refreshHostedRooms()
        view.rerender(<GroupChatWorkspace group="Core" members={MEMBERS} />)
        expect(screen.getByRole('dialog').textContent).toContain('Works without Desktop')
        expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
      } finally {
        release({})
        runtime.stopHostedRoomRuntime()
        shared.setPluginCtx(null)
      }
    }
  )

  it.each([
    ['desktop', undefined, 'Keep Desktop open'],
    ['gateway', 'ready', 'Works without Desktop'],
    ['gateway', 'read-only', 'Read-only history']
  ] as const)('shows accurate continuity for %s/%s', async (continuityMode, state, title) => {
    const [{ GroupChatWorkspace }, chat] = await Promise.all([import('./group-chat-view'), import('./group-chat')])
    chat.$groupChats.set({
      Core: {
        continuityMode,
        hosted: continuityMode === 'gateway' ? 'install:home' : null,
        hostedStatus: state ? { state, label: title } : null,
        roomId: 'room-1',
        members: MEMBERS,
        log: [],
        watermarks: {}
      }
    })
    render(<GroupChatWorkspace group="Core" members={MEMBERS} />)
    fireEvent.click(screen.getByRole('button', { name: 'Group settings for Core' }))
    expect(screen.getByRole('dialog').textContent).toContain(title)

    if (state === 'read-only') {
      expect(screen.getByRole('dialog').textContent).not.toContain('Bots can continue while Desktop is closed.')
      expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true)
    }
  })

  it('restores text, attachments, and retry upload identity after terminal rejection', async () => {
    const [{ GroupChatWorkspace }, chat, panes] = await Promise.all([
      import('./group-chat-view'),
      import('./group-chat'),
      import('./group-panes')
    ])

    const room = {
      continuityMode: 'gateway' as const,
      hosted: 'install:home',
      hostedConnectionId: 'gateway-a',
      hostedEpoch: 1,
      hostedSeq: 0,
      hostedStatus: { label: 'Ready', state: 'ready' },
      log: [],
      members: MEMBERS,
      roomId: 'room-1',
      watermarks: {}
    }

    const key = panes.groupComposerDraftKey('Core', room)

    chat.$groupChats.set({ Core: room })
    panes.updateGroupComposerDraft(key, draft => ({
      ...draft,
      main: 'Review this file',
      pendingAttachments: {
        ...draft.pendingAttachments,
        main: [{ data: 'data:image/png;base64,YQ==', kind: 'image', name: 'proof.png' }]
      }
    }))
    render(<GroupChatWorkspace group="Core" members={MEMBERS} />)

    fireEvent.click(screen.getByRole('button', { name: 'New Thread' }))
    await waitFor(() => expect(sendToGroupChatDurably).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(panes.groupComposerDraftSnapshot(key).main).toBe('Review this file'))

    expect(panes.groupComposerDraftSnapshot(key).pendingAttachments.main).toEqual([
      expect.objectContaining({ name: 'proof.png', uploadId: 'stable-upload-id' })
    ])
    expect(chat.$groupChats.get().Core.log).toEqual([])
    expect(host.notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'error', message: 'The connected device rejected this action.' })
    )
  })
})
