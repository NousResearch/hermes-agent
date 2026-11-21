import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { createPreviewConsoleState } from './preview-console-state'
import { PreviewLoadError } from './preview-pane'

afterEach(cleanup)

const renderError = (consoleOpen: boolean) => {
  const consoleState = createPreviewConsoleState()
  consoleState.setHeight(240)
  consoleState.setOpen(consoleOpen)

  const { container } = render(
    <I18nProvider configClient={null} initialLocale="en">
      <PreviewLoadError
        consoleOpen={consoleOpen}
        consoleState={consoleState}
        error={{ code: 0, description: 'offline', url: 'http://127.0.0.1:9/' }}
        onRetry={() => {}}
      />
    </I18nProvider>
  )

  const panel = container.querySelector('[style*="--preview-error-bottom"]') as HTMLElement | null

  return panel?.style.getPropertyValue('--preview-error-bottom') ?? null
}

describe('preview load-error console inset', () => {
  it('does not reserve console height while the console is closed', () => {
    expect(renderError(false)).toBe('0px')
  })

  it('reserves the console height while the console is open', () => {
    expect(renderError(true)).toBe('240px')
  })
})
