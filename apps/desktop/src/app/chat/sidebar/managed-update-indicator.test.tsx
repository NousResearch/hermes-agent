import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $managedUpdate } from '@/store/updates'

import { ManagedUpdateIndicator } from './managed-update-indicator'

beforeEach(() => $managedUpdate.set(null))
afterEach(cleanup)

describe('ManagedUpdateIndicator', () => {
  it('stays out of the profile rail while idle or unsupported', () => {
    const { rerender } = render(<ManagedUpdateIndicator />)
    expect(screen.queryByRole('status')).toBeNull()

    act(() => $managedUpdate.set({ percent: null, stage: 'disabled' }))
    rerender(<ManagedUpdateIndicator />)
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('shows automatic download progress without an action button', () => {
    $managedUpdate.set({ percent: 42.3, stage: 'downloading', version: '0.18.0' })
    render(<ManagedUpdateIndicator />)

    expect(screen.getByRole('status').getAttribute('aria-label')).toBe('Downloading 42%')
    expect(screen.getByText('Downloading 42%')).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('shows that the verified package will be picked up automatically', () => {
    $managedUpdate.set({ percent: 100, stage: 'downloaded', version: '0.18.0' })
    render(<ManagedUpdateIndicator />)

    expect(screen.getByRole('status').getAttribute('aria-label')).toBe('Update downloaded')
    expect(screen.getByText('Update downloaded')).toBeTruthy()
  })

  it('surfaces a compact failure state', () => {
    $managedUpdate.set({ error: 'network', percent: null, stage: 'error' })
    render(<ManagedUpdateIndicator />)

    expect(screen.getByRole('status').getAttribute('aria-label')).toBe('Update download failed')
  })
})
