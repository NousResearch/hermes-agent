import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import {
  setCurrentFastMode,
  setCurrentModel,
  setCurrentProvider,
  setCurrentReasoningEffort
} from '@/store/session'

import { ModelPill } from './model-pill'

function renderModelPill(modelMenuContent: ReactNode = null) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ModelPill
        disabled={false}
        model={{
          canSwitch: true,
          model: 'grok/composer-2.5-fast',
          modelMenuContent,
          provider: 'xai'
        }}
      />
    </I18nProvider>
  )
}

describe('ModelPill', () => {
  beforeEach(() => {
    setCurrentModel('grok/composer-2.5-fast')
    setCurrentProvider('xai')
    setCurrentFastMode(false)
    setCurrentReasoningEffort('medium')
  })

  afterEach(() => {
    cleanup()
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentFastMode(false)
    setCurrentReasoningEffort('')
  })

  it('shows the full current model in the title when the live menu is unavailable', () => {
    renderModelPill()

    expect(screen.getByRole('button', { name: 'Open model picker' }).getAttribute('title')).toBe(
      'Model · xai: grok/composer-2.5-fast'
    )
  })

  it('keeps the full current model title on the live menu trigger', () => {
    renderModelPill(<div>menu</div>)

    expect(screen.getByRole('button', { name: 'Open model picker' }).getAttribute('title')).toBe(
      'Model · xai: grok/composer-2.5-fast'
    )
  })

  it('keeps the current model visible while provider state is still loading', () => {
    setCurrentProvider('')

    renderModelPill()

    expect(screen.getByRole('button', { name: 'Open model picker' }).getAttribute('title')).toBe(
      'Model · none: grok/composer-2.5-fast'
    )
  })
})
