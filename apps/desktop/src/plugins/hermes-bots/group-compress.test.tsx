import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createGroupGateway } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
import { translateBots } from './i18n-test-helper'

// #102291: room member sessions are hidden plumbing nothing else can compress.
// The room settings dialog owns a per-member "Compress history" that resumes
// every session the room holds for that member and runs session.compress on
// the LIVE runtime id — never minting a new session as a side effect.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
const state = { gateway: null as null | ScriptedGateway }

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)
  const Passthrough = ({ children }: { children?: ReactNode }) => <>{children}</>

  const Button = ({
    children,
    disabled,
    onClick,
    'aria-label': label
  }: {
    'aria-label'?: string
    children?: ReactNode
    disabled?: boolean
    onClick?: () => void
  }) => (
    <button aria-label={label} disabled={disabled} onClick={onClick}>
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
    Dialog: Passthrough,
    DialogContent: Passthrough,
    DialogDescription: Passthrough,
    DialogFooter: Passthrough,
    DialogHeader: Passthrough,
    DialogTitle: Passthrough,
    Input: () => null,
    Tip: Passthrough,
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

beforeEach(() => {
  vi.resetModules()
  state.gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  // The scripted gateway answers unknown methods with `{}`; give
  // session.compress the in-process gateway's real result shape.
  const inner = state.gateway.host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>

  state.gateway.host.request = async (method: string, params: Record<string, unknown> = {}) => {
    if (method === 'session.compress') {
      state.gateway!.rpc.push({ method, params, refcountAfter: 0 })

      return { status: 'compressed', before_messages: 1588, after_messages: 134, summary: { headline: '1588 → 134' } }
    }

    return inner(method, params)
  }

  Object.assign(host, state.gateway.host)
})
afterEach(cleanup)

async function seedMemberSession(profile: string, title: string) {
  const request = state.gateway!.host.request as (method: string, params: Record<string, unknown>) => Promise<unknown>
  const created = (await request('session.create', { profile, title, hidden: true })) as { stored_session_id: string }
  state.gateway!.rpc.length = 0

  return created.stored_session_id
}

it('compressGroupMemberHistory resumes each stored session for the member and compresses the live runtime id', async () => {
  const { $groupChats } = await import('./group-chat')
  const { compressGroupMemberHistory } = await import('./group-compress')
  const mason = await seedMemberSession('mason', 'Group: room-1 · t1')
  const critic = await seedMemberSession('critic', 'Group: room-1 · t1')
  $groupChats.set({
    Build: {
      log: [],
      roomId: 'room-1',
      sessions: { 'thread:t1::mason': mason, 'thread:t1::critic': critic },
      watermarks: {}
    }
  })

  const outcome = await compressGroupMemberHistory('Build', { name: 'mason' } as never)

  expect(outcome).toMatchObject({ compressed: 1, pending: 0, skipped: 0, lines: ['1588 → 134'] })
  const methods = state.gateway!.rpc.map(call => call.method)
  expect(methods).toEqual(['session.resume', 'session.compress'])
  const [resume, compress] = state.gateway!.rpc
  expect(resume.params).toMatchObject({ session_id: mason, profile: 'mason', omit_messages: true })
  // The compress targets the runtime id the resume minted, not the stored id.
  expect(compress.params.session_id).toBe(state.gateway!.sessions.get(mason)!.runtime)
  expect(compress.params.session_id).not.toBe(mason)
  expect(methods).not.toContain('session.create')
})

it('the room settings dialog offers Compress history per member and reports the result', async () => {
  Element.prototype.scrollIntoView = vi.fn()
  const { $groupChats } = await import('./group-chat')
  const { GroupChatWorkspace } = await import('./group-chat-view')
  const mason = await seedMemberSession('mason', 'Group: room-1 · t1')
  $groupChats.set({ Build: { log: [], roomId: 'room-1', sessions: { 'thread:t1::mason': mason }, watermarks: {} } })
  const members = [{ name: 'mason' }, { name: 'critic' }] as never

  const { getByLabelText } = render(<GroupChatWorkspace group="Build" members={members} />)
  fireEvent.click(getByLabelText('Group settings for Build'))
  fireEvent.click(getByLabelText('Compress history: mason'))

  await waitFor(() => expect(state.gateway!.rpcFor('session.compress')).toHaveLength(1))
  const notify = state.gateway!.host.notify as ReturnType<typeof vi.fn>
  await waitFor(() =>
    expect(notify.mock.calls.map(([toast]) => (toast as { message: string }).message)).toContain(
      'Compressed 1 room session for Mason — 1588 → 134'
    )
  )
  // Only mason's session was touched — critic never had one and none was created.
  expect(state.gateway!.rpcFor('session.create')).toHaveLength(0)
})
