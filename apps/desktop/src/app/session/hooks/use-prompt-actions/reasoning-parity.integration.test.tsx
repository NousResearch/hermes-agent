import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { type MutableRefObject, useEffect } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSub,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { en } from '@/i18n/en'
import { formatModelStatusLabel } from '@/lib/model-status-label'
import { REASONING_COMMAND_HELP } from '@/lib/reasoning-effort'
import { $modelPresets, getModelPreset, setModelPreset } from '@/store/model-presets'
import { $activeSessionId, $currentReasoningEffort, setCurrentReasoningEffort } from '@/store/session'

import { ModelEditSubmenu } from '../../../shell/model-edit-submenu'

import { useSlashCommand } from './slash'

const SESSION_ID = 'rt-reasoning'
const PROVIDER = 'nous'
const MODEL = 'hermes-4'
const appendSessionTextMessage = vi.fn()

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  window.localStorage.clear()
  $activeSessionId.set(SESSION_ID)
  $currentReasoningEffort.set('medium')
  $modelPresets.set({})
})

afterEach(() => {
  vi.clearAllMocks()
})

function renderPicker(requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>) {
  return render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <DropdownMenuSub open>
          <DropdownMenuSubTrigger>edit</DropdownMenuSubTrigger>
          <ModelEditSubmenu
            effort={$currentReasoningEffort.get()}
            fastControl={{ kind: 'none' }}
            isActive
            model={MODEL}
            onSelectModel={vi.fn()}
            provider={PROVIDER}
            reasoning
            requestGateway={requestGateway}
          />
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function SlashHarness({
  onReady,
  requestGateway
}: {
  onReady: (runSlash: (command: string) => Promise<void>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const activeSessionIdRef: MutableRefObject<string | null> = { current: SESSION_ID }

  const runSlash = useSlashCommand({
    activeSessionIdRef,
    appendSessionTextMessage,
    branchCurrentSession: async () => true,
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: async () => SESSION_ID,
    handleSkinCommand: () => '',
    handoffSession: async () => ({ ok: true }),
    openMemoryGraph: vi.fn(),
    refreshSessions: async () => undefined,
    requestGateway,
    resumeStoredSession: vi.fn(),
    startFreshSessionDraft: vi.fn(),
    submitPromptText: async () => true
  })

  useEffect(() => {
    onReady(runSlash)
  }, [onReady, runSlash])

  return null
}

function gatewayHarness() {
  let value = 'medium'

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'config.set') {
      value = String(params?.value ?? '')

      return { value } as never
    }

    if (method === 'config.get') {
      return { display: 'show', value } as never
    }

    return {} as never
  })

  return requestGateway
}

describe('desktop reasoning picker and slash parity', () => {
  it.each([
    ['Extra High', 'xhigh', 'XHigh'],
    ['Max', 'max', 'Max'],
    ['Ultra', 'ultra', 'Ultra']
  ])('%s and /reasoning %s converge on live state while preserving preset scope', async (label, effort, compact) => {
    const pickerGateway = gatewayHarness()
    renderPicker(pickerGateway)

    fireEvent.click(screen.getByRole('menuitemradio', { name: label }))

    await waitFor(() => expect($currentReasoningEffort.get()).toBe(effort))
    expect(pickerGateway).toHaveBeenCalledWith('config.set', {
      key: 'reasoning',
      session_id: SESSION_ID,
      value: effort
    })
    expect(formatModelStatusLabel(MODEL, { reasoningEffort: $currentReasoningEffort.get() })).toContain(compact)
    expect(getModelPreset(PROVIDER, MODEL).effort).toBe(effort)

    setCurrentReasoningEffort('medium')
    setModelPreset(PROVIDER, MODEL, { effort: 'medium' })

    const slashGateway = gatewayHarness()
    let runSlash: ((command: string) => Promise<void>) | null = null
    render(<SlashHarness onReady={fn => (runSlash = fn)} requestGateway={slashGateway} />)

    await waitFor(() => expect(runSlash).not.toBeNull())
    await runSlash!(`/reasoning ${effort}`)

    expect(slashGateway).toHaveBeenCalledWith('config.set', {
      key: 'reasoning',
      session_id: SESSION_ID,
      value: effort
    })
    expect($currentReasoningEffort.get()).toBe(effort)
    expect(formatModelStatusLabel(MODEL, { reasoningEffort: $currentReasoningEffort.get() })).toContain(compact)
    expect(getModelPreset(PROVIDER, MODEL).effort).toBe('medium')
  })

  it('keeps display commands separate from the live effort', async () => {
    setCurrentReasoningEffort('ultra')

    const requestGateway = vi.fn(async () => ({ value: 'hide' }) as never)
    let runSlash: ((command: string) => Promise<void>) | null = null
    render(<SlashHarness onReady={fn => (runSlash = fn)} requestGateway={requestGateway} />)

    await waitFor(() => expect(runSlash).not.toBeNull())
    await runSlash!('/reasoning hide')

    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      key: 'reasoning',
      session_id: SESSION_ID,
      value: 'hide'
    })
    expect($currentReasoningEffort.get()).toBe('ultra')
    expect(appendSessionTextMessage.mock.calls.at(-1)?.[2]).toContain('Reasoning display: hide')
  })

  it('renders current status with the shared command vocabulary', async () => {
    const requestGateway = vi.fn(async () => ({ display: 'full', value: 'max' }) as never)
    let runSlash: ((command: string) => Promise<void>) | null = null
    render(<SlashHarness onReady={fn => (runSlash = fn)} requestGateway={requestGateway} />)

    await waitFor(() => expect(runSlash).not.toBeNull())
    await runSlash!('/reasoning')

    expect(requestGateway).toHaveBeenCalledWith('config.get', {
      key: 'reasoning',
      session_id: SESSION_ID
    })
    expect(appendSessionTextMessage.mock.calls.at(-1)?.[2]).toContain(REASONING_COMMAND_HELP)
  })

  it('preserves the live effort and renders a failure when the gateway rejects the change', async () => {
    setCurrentReasoningEffort('high')

    const requestGateway = vi.fn(async () => {
      throw new Error('unsupported')
    })

    let runSlash: ((command: string) => Promise<void>) | null = null
    render(<SlashHarness onReady={fn => (runSlash = fn)} requestGateway={requestGateway} />)

    await waitFor(() => expect(runSlash).not.toBeNull())
    await runSlash!('/reasoning ultra')

    expect($currentReasoningEffort.get()).toBe('high')
    expect(appendSessionTextMessage.mock.calls.at(-1)?.[2]).toContain('Could not update reasoning: unsupported')
  })
})
