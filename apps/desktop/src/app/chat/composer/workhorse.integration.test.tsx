import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'

import { ComposerControls } from './controls'

// Dry-run integration: mount the REAL ComposerControls with the REAL workhorse
// pills (not mocked) and verify they render beside the orchestrator pills in
// the correct order. The config query is the only seam stubbed.
const mocks = vi.hoisted(() => ({
  config: { delegation: { model: 'deepseek/deepseek-v4-flash', provider: 'deepseek', reasoning_effort: 'high' } }
}))

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual }
})

vi.mock('@/app/hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: mocks.config })
}))

vi.mock('@/components/ui/tooltip', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
  }
})

// The orchestrator model pill stays mocked (it needs the whole session/menu
// stack); the workhorse pills and the reasoning pill render for real.
vi.mock('./model-pill', () => ({ ModelPill: () => <span data-testid="orchestrator-model-pill">orch</span> }))

const state: ChatBarState = {
  model: { canSwitch: true, model: 'gpt-6', provider: 'openai' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: false }
}

afterEach(() => {
  cleanup()
  $activeGatewayProfile.set('default')
})

function wrap(ui: React.ReactNode) {
  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        {ui}
      </I18nProvider>
    </QueryClientProvider>
  )
}

describe('ComposerControls workhorse dry run', () => {
  it('renders both workhorse pills left of the orchestrator pill', () => {
    wrap(
      <ComposerControls
        autoSpeak={false}
        busy={false}
        busyAction="stop"
        canSubmit={true}
        conversation={{ active: false, level: 0, muted: false, onEnd: vi.fn(), onStart: vi.fn(), onStopTurn: vi.fn(), onToggleMute: vi.fn(), status: 'idle' }}
        disabled={false}
        hasComposerPayload={true}
        onDictate={vi.fn()}
        onQueue={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={state}
        voiceStatus="idle"
      />
    )

    expect(screen.getByTestId('workhorse-model-pill')).toBeTruthy()
    expect(screen.getByTestId('workhorse-reasoning-pill')).toBeTruthy()
    expect(screen.getByTestId('orchestrator-model-pill')).toBeTruthy()

    // Order: workhorse model pill must sit LEFT of the orchestrator pill.
    const wh = screen.getByTestId('workhorse-model-pill')
    const whEffort = screen.getByTestId('workhorse-reasoning-pill')
    const orch = screen.getByTestId('orchestrator-model-pill')

    expect(wh.compareDocumentPosition(orch) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(whEffort.compareDocumentPosition(orch) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('renders the pinned workhorse model + effort labels', () => {
    wrap(
      <ComposerControls
        autoSpeak={false}
        busy={false}
        busyAction="stop"
        canSubmit={true}
        conversation={{ active: false, level: 0, muted: false, onEnd: vi.fn(), onStart: vi.fn(), onStopTurn: vi.fn(), onToggleMute: vi.fn(), status: 'idle' }}
        disabled={false}
        hasComposerPayload={true}
        onDictate={vi.fn()}
        onQueue={vi.fn()}
        onToggleAutoSpeak={vi.fn()}
        state={state}
        voiceStatus="idle"
      />
    )

    expect(screen.getByTestId('workhorse-model-pill').textContent).toContain('Flash')
    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('High')
  })
})
