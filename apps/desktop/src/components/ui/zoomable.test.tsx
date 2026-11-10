import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { Zoomable } from './zoomable'

afterEach(cleanup)

describe('Zoomable', () => {
  it('localizes the default trigger and shared viewer controls in Japanese', () => {
    render(
      <I18nProvider configClient={null} initialLocale="ja">
        <Zoomable>
          <div>رسم</div>
        </Zoomable>
      </I18nProvider>
    )
    fireEvent.click(screen.getByRole('button', { name: '展開' }))
    expect(screen.getByRole('button', { name: '拡大' })).toBeTruthy()
    expect(screen.getByRole('button', { name: '縮小' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'ズームをリセット' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '閉じる' }))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('opens the full-view overlay when the trigger is clicked', () => {
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">Expanded diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    expect(screen.queryByTestId('overlay')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Open diagram' }))
    expect(screen.getByTestId('overlay')).toBeTruthy()
  })
})
