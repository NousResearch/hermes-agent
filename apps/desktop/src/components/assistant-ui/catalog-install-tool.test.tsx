import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import type { ConnectionOperationTarget, ConnectionRequestPayload } from '@hermes/shared'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { CatalogInstallTool } from '@/components/assistant-ui/catalog-install-tool'
import { I18nProvider } from '@/i18n'
import {
  $connectionRequests,
  applyOperationStatus,
  normalizeConnectionRequest,
  setConnectionRequest
} from '@/store/connection-request'
import { $gateway, setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { setSessionOwnerHint } from '@/store/session'

const SESSION_ID = 'session-1'
const SHA = '5023f3a4ea5023f3a4ea5023f3a4ea5023f3a4ea'

const NVIDIA_APP: ConnectionOperationTarget = {
  action: 'install',
  description: 'Read driver and overlay state, optimise games, capture the overlay.',
  display: 'NVIDIA App',
  has_desktop_half: false,
  kind: 'plugin',
  name: 'nvidia-app',
  platforms: ['windows'],
  repo: 'https://github.com/NousResearch/hermes-nvidia',
  sha: SHA,
  state: 'pending',
  subdir: 'nvidia-app',
  target_profile: 'default',
  tier: 'official'
}

const OBSIDIAN: ConnectionOperationTarget = {
  action: 'install',
  description: 'Read, search and write notes in your vault.',
  display: 'Obsidian notes',
  kind: 'skill',
  name: 'obsidian-notes',
  state: 'pending',
  target_profile: 'default',
  tier: 'community'
}

const PAYLOAD: ConnectionRequestPayload = {
  deadline_at: 1_800_000_000,
  op_id: 'operation-1',
  seq: 1,
  targets: [NVIDIA_APP, OBSIDIAN],
  timeout_seconds: 600,
  tool_call_id: 'catalog-call-1'
}

const ARGS = {
  action: 'install',
  items: [
    { id: 'nvidia-app', kind: 'plugin' },
    { id: 'obsidian-notes', kind: 'skill' }
  ]
}

const PROPS: ToolCallMessagePartProps = {
  addResult: vi.fn(),
  args: ARGS,
  argsText: JSON.stringify(ARGS),
  isError: false,
  respondToApproval: vi.fn(),
  resume: vi.fn(),
  status: { type: 'running' },
  toolCallId: 'catalog-call-1',
  toolName: 'manage_catalog',
  type: 'tool-call'
}

function view(sessionId: string): SessionView {
  return {
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom(''),
    $fast: atom(false),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(false),
    $model: atom(''),
    $provider: atom(''),
    $reasoningEffort: atom(''),
    $reasoningEffortPending: atom(false),
    $reasoningEffortWire: atom(''),
    $runtimeId: atom(sessionId),
    $storedId: atom(sessionId),
    $turnStartedAt: atom(null),
    kind: 'primary'
  }
}

/** Open the card from a wire payload, as the `connection.request` handler does. */
function openCard(payload: ConnectionRequestPayload = PAYLOAD) {
  const request = normalizeConnectionRequest(payload, SESSION_ID)

  if (!request) {
    throw new Error('fixture payload did not normalize')
  }

  setConnectionRequest(request)

  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionViewProvider value={view(SESSION_ID)}>
        <div data-testid="catalog">
          <CatalogInstallTool {...PROPS} />
        </div>
      </SessionViewProvider>
    </I18nProvider>
  )
}

/** Move the open operation to a newer snapshot, as `connection.update` does. */
function pushFrame(targets: ConnectionOperationTarget[], seq: number) {
  const current = $connectionRequests.get()[SESSION_ID]
  const status = { deadline_at: 1_800_000_000, op_id: 'operation-1', seq, settled: false, targets }

  act(() => setConnectionRequest(applyOperationStatus(current, status)))
}

const row = (name: string) => {
  const node = screen.getByTestId('catalog').querySelector<HTMLElement>(`[data-connector-row="${name}"]`)

  if (!node) {
    throw new Error(`no row ${name}`)
  }

  return within(node)
}

let rpc: ReturnType<typeof vi.fn>

beforeEach(() => {
  // Radix Select scrolls the chosen option into view; jsdom has no layout.
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  rpc = vi.fn().mockResolvedValue({ settled: false, status: 'ok' })
  setSessionOwnerHint(SESSION_ID, { connectionId: 'local', profile: 'default' })
  // SAFETY: the card calls only `request`; the rest of the client is never touched in this test.
  setPrimaryGateway({ request: rpc } as never)
  setPrimaryGatewayConnectionId('local')
  $profiles.set([
    {
      has_env: false,
      is_default: true,
      model: null,
      name: 'default',
      path: '/p/default',
      provider: null,
      skill_count: 0
    },
    { has_env: false, is_default: false, model: null, name: 'work', path: '/p/work', provider: null, skill_count: 0 }
  ])
})

afterEach(() => {
  cleanup()
  $connectionRequests.set({})
  $gateway.set(null)
  setPrimaryGateway(null)
  vi.clearAllMocks()
})

const respondedWith = (targets: unknown[]) => [
  'connection.respond',
  { op_id: 'operation-1', owner: { session_id: SESSION_ID, type: 'session' }, result: { targets } }
]

describe('the catalog install card', () => {
  it('draws plugin and skill rows as catalog items with their purpose, and a platform only when restricted', () => {
    openCard()

    expect(screen.getByTestId('catalog').querySelectorAll('[data-connector-row]')).toHaveLength(2)
    expect(row('nvidia-app').getByText('plugin')).toBeTruthy()
    expect(row('nvidia-app').getByText(NVIDIA_APP.description!)).toBeTruthy()
    expect(row('nvidia-app').getByText('Windows')).toBeTruthy()
    expect(row('obsidian-notes').getByText('skill')).toBeTruthy()
    expect(row('obsidian-notes').getByText(OBSIDIAN.description!)).toBeTruthy()
    expect(row('obsidian-notes').queryByText('Windows')).toBeNull()
  })

  it('sends defaults from the row, the modal values from Advanced, and nothing on Cancel', async () => {
    openCard()

    fireEvent.click(row('obsidian-notes').getByRole('button', { name: 'Install' }))
    await waitFor(() => expect(rpc).toHaveBeenCalledTimes(1))
    expect(rpc).toHaveBeenLastCalledWith(...respondedWith([{ env: null, name: 'obsidian-notes', status: 'approved' }]))

    fireEvent.click(row('nvidia-app').getByRole('button', { name: 'Advanced' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(rpc).toHaveBeenCalledTimes(1)

    fireEvent.click(row('nvidia-app').getByRole('button', { name: 'Advanced' }))
    const dialog = within(screen.getByRole('dialog'))
    fireEvent.click(dialog.getByRole('combobox', { name: 'Install for profile' }))
    fireEvent.click(await screen.findByRole('option', { name: 'work' }))
    fireEvent.click(dialog.getByRole('switch', { name: /Force reinstall/ }))
    fireEvent.click(dialog.getByRole('button', { name: 'Install' }))

    await waitFor(() => expect(rpc).toHaveBeenCalledTimes(2))
    expect(rpc).toHaveBeenLastCalledWith(
      ...respondedWith([
        {
          env: { agent_half: '1', enable: '1', force: '1', ref: SHA, target_profile: 'work' },
          name: 'nvidia-app',
          status: 'approved'
        }
      ])
    )
  })

  it('settles each row to its outcome and retries only the failed one', async () => {
    const tools = Array.from({ length: 12 }, (_, index) => `nvapp_client_tool_${index}`)

    openCard({
      ...PAYLOAD,
      targets: [NVIDIA_APP, { ...NVIDIA_APP, display: 'NVIDIA Broadcast', name: 'nvidia-broadcast' }, OBSIDIAN]
    })
    pushFrame(
      [
        { ...NVIDIA_APP, state: 'connected', tools },
        {
          ...NVIDIA_APP,
          detail: 'NVIDIA Broadcast is not running',
          display: 'NVIDIA Broadcast',
          name: 'nvidia-broadcast',
          state: 'failed'
        },
        { ...OBSIDIAN, state: 'skipped' }
      ],
      2
    )

    expect(row('nvidia-app').getByText(/Installed · 12 tools/)).toBeTruthy()
    expect(row('nvidia-app').queryByText('nvapp_client_tool_0')).toBeNull()
    fireEvent.click(row('nvidia-app').getByRole('button', { name: 'show names' }))
    expect(row('nvidia-app').getByText('nvapp_client_tool_0')).toBeTruthy()
    expect(row('nvidia-broadcast').getByText(/NVIDIA Broadcast is not running/)).toBeTruthy()
    expect(row('obsidian-notes').getByText('Skipped')).toBeTruthy()
    expect(row('obsidian-notes').queryByRole('button')).toBeNull()

    fireEvent.click(row('nvidia-broadcast').getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(rpc).toHaveBeenCalledTimes(1))
    expect(rpc).toHaveBeenLastCalledWith(
      ...respondedWith([{ env: null, name: 'nvidia-broadcast', status: 'approved' }])
    )
  })
})
