import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'

import { ComposerControls } from './controls'

vi.mock('@/app/hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({
    data: {
      delegation: { model: 'deepseek/deepseek-v4-flash', provider: 'deepseek', reasoning_effort: 'high' }
    }
  })
}))
vi.mock('./model-pill', () => ({ ModelPill: () => <span data-testid="orchestrator-model-pill">orch</span> }))

const state: ChatBarState = {
  model: { canSwitch: true, model: 'gpt-6', provider: 'openai' },
  tools: { enabled: false, label: '' },
  voice: { active: false, enabled: false }
}

afterEach(cleanup)

function mount(props: Partial<React.ComponentProps<typeof ComposerControls>> = {}) {
  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerControls
          autoSpeak={false}
          busy={false}
          busyAction="stop"
          canSubmit
          conversation={{
            active: false,
            level: 0,
            muted: false,
            onEnd: vi.fn(),
            onStart: vi.fn(),
            onStopTurn: vi.fn(),
            onToggleMute: vi.fn(),
            status: 'idle'
          }}
          disabled={false}
          hasComposerPayload
          onDictate={vi.fn()}
          onQueue={vi.fn()}
          onToggleAutoSpeak={vi.fn()}
          state={state}
          voiceStatus="idle"
          {...props}
        />
      </I18nProvider>
    </QueryClientProvider>
  )
}

describe('ComposerControls subagents', () => {
  it.each([false, true])(
    'keeps one icon left of the parent and no permanent model/effort text (compact=%s)',
    compactModelPill => {
      mount({ compactModelPill })
      const child = screen.getByRole('button', { name: 'Subagents' })
      const parent = screen.getByTestId('orchestrator-model-pill')
      expect(child.compareDocumentPosition(parent) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      expect(child.textContent).toBe('')
      expect(screen.queryByTestId('workhorse-reasoning-pill')).toBeNull()
      expect(screen.queryByText(/Flash|High/)).toBeNull()
      expect(screen.queryByRole('menu')).toBeNull()
    }
  )
  it.each([{ minimal: true }, { hideModelPill: true }])('hides the icon together with model controls for %s', props => {
    mount(props)
    expect(screen.queryByTestId('workhorse-model-pill')).toBeNull()
    expect(screen.queryByTestId('workhorse-reasoning-pill')).toBeNull()
  })
})
