import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import {
  $sessionRestorePromptVisible,
  $sessionRestoreSnapshot
} from '@/store/session-restore'

import { SessionRestorePrompt } from './session-restore-prompt'

function mockSnapshot(entries = 1, createdAt = Date.now() - 60000) {
  return {
    schemaVersion: 1,
    createdAt,
    entries: Array.from({ length: entries }, (_, i) => ({
      sessionId: `session-${i + 1}`,
      watch: false,
      bounds: { x: 0, y: 0, width: 1200, height: 800 }
    }))
  }
}

function resetStores() {
  $sessionRestoreSnapshot.set(null)
  $sessionRestorePromptVisible.set(false)
}

beforeEach(resetStores)
afterEach(() => {
  resetStores()
  cleanup()
})

function renderPrompt(onRestore = vi.fn(), onDiscard = vi.fn()) {
  return render(
    <I18nProvider configClient={null}>
      <SessionRestorePrompt onDiscard={onDiscard} onRestore={onRestore} />
    </I18nProvider>
  )
}

describe('SessionRestorePrompt', () => {
  it('renders nothing when prompt is not visible', () => {
    $sessionRestorePromptVisible.set(false)
    $sessionRestoreSnapshot.set(mockSnapshot(3))

    const { container } = renderPrompt()

    expect(container.textContent).toBe('')
  })

  it('renders nothing when snapshot is null', () => {
    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(null)

    const { container } = renderPrompt()

    expect(container.textContent).toBe('')
  })

  it('renders nothing when snapshot has zero entries', () => {
    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(mockSnapshot(0))

    const { container } = renderPrompt()

    expect(container.textContent).toBe('')
  })

  it('shows title and session count', () => {
    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(mockSnapshot(3))

    renderPrompt()

    expect(screen.getByText('Restore previous sessions?')).toBeTruthy()
    expect(screen.getByText(/Hermes found 3 session/)).toBeTruthy()
  })

  it('shows restore and discard buttons', () => {
    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(mockSnapshot(1))

    renderPrompt()

    expect(screen.getByRole('button', { name: 'Restore' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Start fresh' })).toBeTruthy()
  })

  it('calls onRestore when Restore button is clicked', async () => {
    const onRestore = vi.fn()
    const onDiscard = vi.fn()

    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(mockSnapshot(1))

    renderPrompt(onRestore, onDiscard)

    fireEvent.click(screen.getByRole('button', { name: 'Restore' }))

    expect(onRestore).toHaveBeenCalledTimes(1)
    expect(onDiscard).not.toHaveBeenCalled()
  })

  it('calls onDiscard when Start fresh button is clicked', async () => {
    const onRestore = vi.fn()
    const onDiscard = vi.fn()

    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(mockSnapshot(1))

    renderPrompt(onRestore, onDiscard)

    fireEvent.click(screen.getByRole('button', { name: 'Start fresh' }))

    expect(onDiscard).toHaveBeenCalledTimes(1)
    expect(onRestore).not.toHaveBeenCalled()
  })

  it('shows timestamp for a recent snapshot', () => {
    $sessionRestorePromptVisible.set(true)
    $sessionRestoreSnapshot.set(mockSnapshot(1, Date.now() - 120000))

    renderPrompt()

    expect(screen.getByText(/Saved 2 minutes ago/)).toBeTruthy()
  })
})
