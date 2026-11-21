import { host } from '@hermes/plugin-sdk'
import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  answerGroupRequests,
  CANONICAL_GROUP_CAPABILITIES,
  PEER_HOME as HOME,
  PEER_MINI as MINI,
  peerRosterRow
} from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { PeerJoinError, peerJoinText, peerSetupText, peerTargetState } from './canonical-group-peers'
import { $canonicalGroupBindings } from './canonical-group-registry'
import { CreateGroupChatDialog } from './create-dialog'
import { $botMeta } from './data'
import { $groupChats } from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { request } = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  (await import('./canonical-group-contract-test-utils')).localGatewaySdkMock(
    await importOriginal<typeof HermesSdk>(),
    request,
    () => translateBots
  )
)
vi.mock('./group-chat-parts', async importOriginal => ({
  ...(await importOriginal<typeof GroupChatParts>()),
  GroupImageControls: () => null
}))

const roster: RosterRow[] = [{ name: 'default', connectionId: 'local' }, peerRosterRow('mini', 'Mac Mini')]
const link = (reason: string) => ({ ...MINI, room_link: { enabled: false, reason } })
const catalog = (change: Record<string, unknown>) => ({
  ...MINI,
  room_link: { ...MINI.room_link, catalog: { ...MINI.room_link.catalog, ...change } }
})

/** Every refusal a lender's `groups.capabilities` can produce, as its gateway sends it: `room_link.reason`
 *  from gateway/session_group_peers.py `room_link` and tui_gateway/methods_groups.py, and the catalog
 *  and method checks of `peerTargetState`; then one this Desktop does not know yet. */
const REFUSALS = [
  ['api_server_required', link('api_server_required'), 'peerSetupApiServer'],
  ['durable_run_storage_required', link('durable_run_storage_required'), 'peerSetupRunStorage'],
  ['gateway_roomlink_secret_unavailable', link('gateway_roomlink_secret_unavailable'), 'peerSetupRoomKey'],
  ['execution_policy_unsupported', link('execution_policy_unsupported'), 'peerSetupApprovals'],
  ['endpoint_required', link('endpoint_required'), 'peerSetupAddress'],
  ['endpoint_required', catalog({ endpoint: { available: false, reason: 'not_configured' } }), 'peerSetupAddress'],
  ['persistent_process_required', catalog({ persistent_process: false }), 'peerSetupBackground'],
  ['peer_invite_unsupported', CANONICAL_GROUP_CAPABILITIES, 'peerSetupUpdate'],
  ['peer_target_unsupported', catalog({ protocol_versions: [1] }), 'peerSetupUpdate'],
  ['text_only_required', catalog({ attachments: true }), 'peerSetupUpdate'],
  ['room_link_unavailable', { ...MINI, room_link: { enabled: false } }, 'peerSetupUpdate'],
  ['room_link_unavailable', catalog({ installation_id: 'install:other' }), 'peerSetupUpdate'],
  ['a_future_reason', link('a_future_reason'), 'peerSetupGeneric']
] as const

const CODES = [...new Set([...REFUSALS.map(([code]) => code), 'canonical_driver_required', 'unreachable'])]
const rawCode = new RegExp(CODES.join('|'))

beforeEach(() => {
  ;(host.state.gateway as ReturnType<typeof HermesSdk.atom<string>>).set('open')
  $canonicalGroupBindings.set({})
  $botMeta.set({})
  $groupChats.set({})
  request.mockReset()
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  localStorage.clear()
})

const box = (row: Element | undefined) =>
  row?.closest('label')?.querySelector('button[role="checkbox"]') as HTMLButtonElement

it.each(REFUSALS)(
  'explains a lender refusing with %s in words, not its code, and never falls back to Desktop',
  async (code, capabilities, key) => {
    expect((peerTargetState(capabilities, 'default') as { reason?: string }).reason).toBe(code)
    answerGroupRequests(request, {
      local: method => (method === 'groups.capabilities' ? { ...HOME, room_link: { enabled: false } } : undefined),
      mini: method => (method === 'groups.capabilities' ? capabilities : undefined)
    })
    const { baseElement } = render(<CreateGroupChatDialog onClose={vi.fn()} open roster={roster} />)
    await waitFor(() => expect(request.mock.calls.some(call => call[0]?.connectionId === 'mini')).toBe(true))
    await act(async () => {})
    const rows = [...baseElement.querySelectorAll('[data-bot-face="default"]')]
    fireEvent.click(box(rows[0]))
    fireEvent.click(box(rows[1]))
    const note = await waitFor(() => baseElement.querySelector('[data-group-engine="unavailable"]')!)
    const words = CANONICAL_GROUP_LOCALES.en
    expect(baseElement.querySelector('[data-group-engine="desktop"]')).toBeNull()
    expect(note.textContent).toContain(
      words.peerTargetSetup.replace('{connection}', 'Mac Mini').replace('{reason}', words[key])
    )
    expect(note.textContent).not.toMatch(rawCode)
    expect(note.textContent).not.toMatch(/[{}]/)
  }
)

it.each(Object.keys(CANONICAL_GROUP_LOCALES) as Array<keyof typeof CANONICAL_GROUP_LOCALES>)(
  'gives each lender refusal its own advice, without codes, in %s',
  locale => {
    const words = CANONICAL_GROUP_LOCALES[locale]
    const texts = new Map<string, string>()

    for (const [code, , key] of REFUSALS) {
      const text = peerSetupText(code, 'Mac Mini', words)
      expect(text, `${locale} ${code}`).toContain('Mac Mini')
      expect(text, `${locale} ${code}`).toContain(words[key])
      expect(text, `${locale} ${code}`).not.toMatch(rawCode)
      expect(text, `${locale} ${code}`).not.toMatch(/\{(?:connection|reason)\}/)
      texts.set(key, text)
      // A Retry that meets the same known refusal gives the same advice, in the room (a pinned member
      // whose computer no longer speaks its protocol keeps the room's own "settings changed" copy).
      const joined =
        code === 'peer_target_unsupported'
          ? words.peerJoinChanged
          : words.peerJoinFailed.replace('{reason}', words[key])

      if (key !== 'peerSetupGeneric') {
        expect(peerJoinText(new PeerJoinError(code), 'Hermes · Mac Mini', words), `${locale} ${code}`).toBe(
          joined.replace('{name}', 'Hermes · Mac Mini')
        )
      }
    }

    expect(new Set(texts.values()).size, locale).toBe(texts.size)
  }
)
