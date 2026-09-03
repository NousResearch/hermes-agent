import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type TriageActionLabels, TriageActions } from './triage-actions'

const labels: TriageActionLabels = {
  specify: 'Specify',
  specifying: 'Specifying…',
  specified: 'Specified',
  specifiedRetitled: title => `Specified — retitled: ${title}`,
  decompose: 'Decompose',
  decomposing: 'Decomposing…',
  decomposed: count => `Decomposed into ${count} children`,
  decomposedSingle: 'Single task (no fanout)',
  failed: (action, reason) => `${action} failed: ${reason}`,
  unknownError: 'unknown error'
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('TriageActions', () => {
  it('is hidden outside the Triage status', () => {
    render(
      <TriageActions labels={labels} onDecompose={vi.fn()} onRefresh={vi.fn()} onSpecify={vi.fn()} status="ready" />
    )

    expect(screen.queryByRole('button', { name: 'Specify' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Decompose' })).toBeNull()
  })

  it('prevents duplicate actions while Specify is pending and refreshes on success', async () => {
    let resolve!: (value: { ok: boolean; task_id: string; reason: null; new_title: string }) => void
    const onSpecify = vi.fn(
      () => new Promise<{ ok: boolean; task_id: string; reason: null; new_title: string }>(done => (resolve = done))
    )
    const onRefresh = vi.fn()

    render(
      <TriageActions
        labels={labels}
        onDecompose={vi.fn()}
        onRefresh={onRefresh}
        onSpecify={onSpecify}
        status="triage"
      />
    )

    const specify = screen.getByRole('button', { name: 'Specify' })
    fireEvent.click(specify)
    fireEvent.click(specify)

    expect(onSpecify).toHaveBeenCalledTimes(1)
    expect((screen.getByRole('button', { name: 'Specifying…' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Decompose' }) as HTMLButtonElement).disabled).toBe(true)

    resolve({ ok: true, task_id: 't_1', reason: null, new_title: 'Concrete task' })

    expect(await screen.findByText('Specified — retitled: Concrete task')).not.toBeNull()
    expect(onRefresh).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole('button', { name: 'Specify' }))

    expect(onSpecify).toHaveBeenCalledTimes(1)
    expect((screen.getByRole('button', { name: 'Specify' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('shows a backend failure reason without refreshing', async () => {
    const onRefresh = vi.fn()

    render(
      <TriageActions
        labels={labels}
        onDecompose={vi.fn().mockResolvedValue({
          ok: false,
          task_id: 't_1',
          reason: 'no auxiliary client configured',
          fanout: false,
          child_ids: [],
          new_title: null
        })}
        onRefresh={onRefresh}
        onSpecify={vi.fn()}
        status="triage"
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Decompose' }))

    expect(await screen.findByText('Decompose failed: no auxiliary client configured')).not.toBeNull()
    await waitFor(() =>
      expect((screen.getByRole('button', { name: 'Decompose' }) as HTMLButtonElement).disabled).toBe(false)
    )
    expect(onRefresh).not.toHaveBeenCalled()
  })
})
