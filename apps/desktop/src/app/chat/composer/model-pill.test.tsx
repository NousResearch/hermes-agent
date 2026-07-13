// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = {
  currentFastMode: true,
  currentModel: 'gpt-5.6-sol-with-an-intentionally-long-display-name',
  currentProvider: 'github-copilot',
  currentReasoningEffort: 'ultra'
}

vi.mock('@nanostores/react', () => ({ useStore: (store: keyof typeof state) => state[store] }))
vi.mock('@/store/session', () => ({
  $currentFastMode: 'currentFastMode',
  $currentModel: 'currentModel',
  $currentProvider: 'currentProvider',
  $currentReasoningEffort: 'currentReasoningEffort',
  setModelPickerOpen: vi.fn()
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      shell: {
        statusbar: {
          modelNone: 'No model',
          modelTitle: (provider: string, model: string) => `${provider}: ${model}`,
          openModelPicker: 'Open model picker',
          switchModel: 'Switch model'
        }
      }
    }
  })
}))

afterEach(cleanup)

describe('ModelPill', () => {
  it('truncates a long status label on one line', async () => {
    const { ModelPill } = await import('./model-pill')
    render(<ModelPill disabled={false} model={{ canSwitch: true, model: state.currentModel, modelMenuContent: null, provider: state.currentProvider }} />)

    const label = screen.getByText(/gpt-5\.6-sol-with-an-intentionally-long-display-name/i)
    expect(label.className.split(' ')).toEqual(expect.arrayContaining(['min-w-0', 'truncate', 'whitespace-nowrap']))

    const button = screen.getByRole('button', { name: 'Open model picker' })
    expect(button.className.split(' ')).toEqual(expect.arrayContaining(['max-w-40', 'overflow-hidden', 'whitespace-nowrap']))
  })
})
