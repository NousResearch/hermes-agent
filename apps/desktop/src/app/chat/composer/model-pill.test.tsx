import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  setActiveSessionId,
  setCurrentModel,
  setCurrentProvider,
  setProfileDefaultModel,
  setProfileDefaultProvider
} from '@/store/session'

import { ModelPill } from './model-pill'

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: ReactNode }) => children
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      shell: {
        statusbar: {
          modelNone: 'none',
          modelOverrideTitle: (provider: string, model: string, defaultProvider: string, defaultModel: string) =>
            `Composer override · ${provider}: ${model}. Default: ${defaultProvider}: ${defaultModel}`,
          modelTitle: (provider: string, model: string) => `Model · ${provider}: ${model}`,
          openModelPicker: 'Open model picker',
          switchModel: 'Switch model',
          unknown: 'unknown'
        }
      }
    }
  })
}))

const model = {
  canSwitch: true,
  model: 'deepseek/deepseek-v4-flash',
  provider: 'deepseek'
}

describe('ModelPill composer override indicator', () => {
  beforeEach(() => {
    setActiveSessionId(null)
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentProvider('deepseek')
    setProfileDefaultModel('google/gemma-4-26b-a4b-it:free')
    setProfileDefaultProvider('openrouter')
  })

  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    setCurrentModel('')
    setCurrentProvider('')
    setProfileDefaultModel('')
    setProfileDefaultProvider('')
  })

  it('discloses a sticky new-chat selection that differs from the profile default', () => {
    const { container } = render(<ModelPill disabled={false} model={model} />)

    const button = screen.getByRole('button', { name: /Composer override/ })

    expect(button.getAttribute('data-model-override')).toBe('true')
    expect(container.querySelector('[data-slot="model-override-indicator"]')).not.toBeNull()
  })

  it('does not label a live session model as a composer override', () => {
    setActiveSessionId('session-1')
    const { container } = render(<ModelPill disabled={false} model={model} />)

    const button = screen.getByRole('button', { name: 'Open model picker' })

    expect(button.getAttribute('data-model-override')).toBeNull()
    expect(container.querySelector('[data-slot="model-override-indicator"]')).toBeNull()
  })
})
