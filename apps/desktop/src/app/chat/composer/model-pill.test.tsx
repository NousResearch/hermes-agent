import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  quotaCard: vi.fn(),
  setModelPickerOpen: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      shell: {
        statusbar: {
          modelNone: 'No model',
          modelTitle: (provider: string, model: string) => `${provider} · ${model}`,
          openModelPicker: 'Open model picker',
          switchModel: 'Switch model'
        }
      }
    }
  })
}))

vi.mock('./codex-quota-card', () => ({
  CodexQuotaCard: (props: { enabled: boolean; profile: string; provider: string }) => {
    mocks.quotaCard(props)

    return (
      <div data-testid="quota-card">
        {String(props.enabled)}|{props.profile}|{props.provider}
      </div>
    )
  }
}))

vi.mock('@/store/session', async () => {
  const actual = await vi.importActual('@/store/session')

  return {
    ...(actual as object),
    setModelPickerOpen: (open: boolean) => mocks.setModelPickerOpen(open)
  }
})

import { $activeGatewayProfile } from '@/store/profile'
import { $currentFastMode, $currentModel, $currentProvider, $currentReasoningEffort } from '@/store/session'

import { ModelPill } from './model-pill'

function resetStores({ profile = 'acewill-dev', provider = 'openai-codex' } = {}) {
  $activeGatewayProfile.set(profile)
  $currentModel.set('gpt-5.5')
  $currentProvider.set(provider)
  $currentFastMode.set(false)
  $currentReasoningEffort.set('')
}

describe('ModelPill Codex quota hover', () => {
  beforeEach(() => {
    resetStores()
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('shows the Codex quota card on hover and passes the active gateway profile', async () => {
    render(
      <ModelPill
        disabled={false}
        model={{ canSwitch: true, model: 'gpt-5.5', modelMenuContent: <div>models</div>, provider: 'openai-codex' }}
      />
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'openai-codex · gpt-5.5' }))

    await waitFor(() => expect(screen.queryByTestId('quota-card')).toBeTruthy())
    expect(screen.getByTestId('quota-card').textContent).toBe('true|acewill-dev|openai-codex')
  })

  it('does not show Codex quota for non-Codex providers', () => {
    resetStores({ profile: 'default', provider: 'anthropic' })
    render(
      <ModelPill
        disabled={false}
        model={{ canSwitch: true, model: 'gpt-5.5', modelMenuContent: <div>models</div>, provider: 'openai-codex' }}
      />
    )

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'anthropic · gpt-5.5' }))

    expect(screen.queryByTestId('quota-card')).toBeNull()
  })
})
