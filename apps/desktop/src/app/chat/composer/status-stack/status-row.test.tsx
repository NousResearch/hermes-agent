import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { ComposerStatusItem } from '@/store/composer-status'

import { StatusItemRow } from './status-row'

const backgroundItem = (overrides: Partial<ComposerStatusItem> = {}): ComposerStatusItem => ({
  id: 'proc_abc',
  state: 'running',
  title: 'npm run dev',
  type: 'background',
  ...overrides
})

function renderRow(item: ComposerStatusItem, onStop = vi.fn()) {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <StatusItemRow item={item} onStop={onStop} />
    </I18nProvider>
  )

  return onStop
}

describe('StatusItemRow background actions', () => {
  afterEach(cleanup)

  it('offers Stop on a running process this gateway owns', () => {
    renderRow(backgroundItem())

    expect(screen.getByRole('button', { name: /stop/i })).toBeTruthy()
  })

  // A peer row is mirrored from another gateway process's registry: this
  // backend holds no handle on that PID and refuses to signal it, so the
  // control would be a button that can only ever fail.
  it('offers no Stop on a row mirrored from a peer gateway', () => {
    renderRow(backgroundItem({ peer: true }))

    expect(screen.queryByRole('button', { name: /stop/i })).toBeNull()
    expect(screen.getByText('npm run dev')).toBeTruthy()
  })
})
