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
import { $modelPresets, getModelPreset, setModelPreset } from '@/store/model-presets'
import { $activeSessionId, $currentReasoningEffort, setCurrentReasoningEffort } from '@/store/session'

import { ModelEditSubmenu } from '../../../shell/model-edit-submenu'

import { useSlashCommand } from './slash'

const SESSION_ID = 'rt-reasoning'
const PROVIDER = 'nous'
const MODEL = 'hermes-4'

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
    appendSessionTextMessage: vi.fn(),
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
  const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    calls.push({ method, params })

    if (method === 'config.set') {
      value = String(params?.value ?? '')

      return { value } as never
    }

    if (method === 'config.get') {
      return { display: 'show', value } as never
    }

    return {} as never
  })

  return { calls, requestGateway }
}

describe('desktop reasoning picker and slash parity', () => {
  it.each([
    ['Extra High', 'xhigh', 'XHigh'],
    ['Max', 'max', 'Max']
  ])('%s and /reasoning %s converge on live state while preserving preset scope', async (label, effort, compact) => {
    const pickerGateway = gatewayHarness()
    renderPicker(pickerGateway.requestGateway)

    fireEvent.click(screen.getByRole('menuitemradio', { name: label }))

    await waitFor(() => expect($currentReasoningEffort.get()).toBe(effort))

    const pickerReadback = (await pickerGateway.requestGateway('config.get', {
      key: 'reasoning',
      session_id: SESSION_ID
    })) as { value?: string }

    expect(pickerReadback.value).toBe(effort)
    expect(formatModelStatusLabel(MODEL, { reasoningEffort: $currentReasoningEffort.get() })).toContain(compact)
    expect(getModelPreset(PROVIDER, MODEL).effort).toBe(effort)

    setCurrentReasoningEffort('medium')
    setModelPreset(PROVIDER, MODEL, { effort: 'medium' })

    const slashGateway = gatewayHarness()
    let runSlash: ((command: string) => Promise<void>) | null = null
    render(<SlashHarness onReady={fn => (runSlash = fn)} requestGateway={slashGateway.requestGateway} />)

    await waitFor(() => expect(runSlash).not.toBeNull())
    await runSlash!(`/reasoning ${effort}`)

    expect($currentReasoningEffort.get()).toBe(effort)

    const slashReadback = (await slashGateway.requestGateway('config.get', {
      key: 'reasoning',
      session_id: SESSION_ID
    })) as { value?: string }

    expect(slashReadback.value).toBe(effort)
    expect(formatModelStatusLabel(MODEL, { reasoningEffort: $currentReasoningEffort.get() })).toContain(compact)
    expect(getModelPreset(PROVIDER, MODEL).effort).toBe('medium')
  })
})
