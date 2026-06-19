import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $backgroundStatusBySession, type ComposerStatusItem } from '@/store/composer-status'
import { $activeSessionId } from '@/store/session'

import { ProcessOutputModal } from './process-output-modal'

const SID = 'sess-1'

function seed(items: ComposerStatusItem[]) {
  $activeSessionId.set(SID)
  $backgroundStatusBySession.set({ [SID]: items })
}

function bg(overrides: Partial<ComposerStatusItem> = {}): ComposerStatusItem {
  return {
    id: 'bg-1',
    type: 'background',
    title: 'npm run build',
    state: 'running',
    ...overrides
  } as ComposerStatusItem
}

function renderModal(itemId: string, onClose = () => {}) {
  return render(
    <I18nProvider configClient={null}>
      <ProcessOutputModal itemId={itemId} onClose={onClose} />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  $backgroundStatusBySession.set({})
})

describe('ProcessOutputModal', () => {
  it('renders the running task title and its live output', () => {
    seed([bg({ output: 'building…\nstep 2\n' })])
    renderModal('bg-1')

    expect(screen.getByText('npm run build')).toBeTruthy()
    expect(screen.getByText(/building…/)).toBeTruthy()
  })

  it('shows the waiting placeholder for a running task with no output yet', () => {
    seed([bg({ output: undefined, state: 'running' })])
    renderModal('bg-1')

    expect(screen.getByText('Waiting for output…')).toBeTruthy()
  })

  it('shows the no-output placeholder for a finished task that captured nothing', () => {
    seed([bg({ output: undefined, state: 'done' })])
    renderModal('bg-1')

    expect(screen.getByText('No output captured.')).toBeTruthy()
  })

  it('reflects the latest store snapshot (stays live via id lookup, not a snapshot)', () => {
    seed([bg({ output: 'first\n' })])
    const { rerender } = renderModal('bg-1')

    expect(screen.getByText(/first/)).toBeTruthy()

    // Simulate the 5s process.list poll updating the task's output_tail.
    $backgroundStatusBySession.set({ [SID]: [bg({ output: 'first\nsecond\n' })] })
    rerender(
      <I18nProvider configClient={null}>
        <ProcessOutputModal itemId="bg-1" onClose={() => {}} />
      </I18nProvider>
    )

    expect(screen.getByText(/second/)).toBeTruthy()
  })

  it('renders nothing when the task id is no longer in the registry', () => {
    seed([bg({ id: 'bg-1' })])
    const { container } = renderModal('bg-GONE')

    // No task → null render (the overlay is not mounted).
    expect(container.querySelector('[data-slot]')).toBeNull()
    expect(screen.queryByText('npm run build')).toBeNull()
  })

  it('portals the overlay to document.body so fixed-positioning escapes the composer containing block', () => {
    seed([bg({ output: 'building…\n' })])
    const { container } = renderModal('bg-1')

    // The overlay content must NOT live inside the in-place render container
    // (which sits in the composer subtree, trapped by backdrop-filter +
    // contain:paint). It is portaled to body, so the title resolves via the
    // document but is absent from `container`.
    expect(container.textContent).not.toContain('npm run build')
    expect(screen.getByText('npm run build')).toBeTruthy()
    // The fixed full-screen scrim is a direct child of body, not of container.
    const scrim = document.body.querySelector('.fixed.inset-0')
    expect(scrim).toBeTruthy()
    expect(container.contains(scrim)).toBe(false)
  })
})
