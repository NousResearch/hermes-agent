import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $workstreamMetadata } from '@/store/workstream-metadata'

import { SessionActionsMenu } from './session-actions-menu'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  $workstreamMetadata.set({})
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderMenu() {
  render(
    <SessionActionsMenu sessionId="stored-1" title="Lifecycle test">
      <button type="button">Actions</button>
    </SessionActionsMenu>
  )

  fireEvent.pointerDown(screen.getByRole('button', { name: 'Actions' }))
}

describe('SessionActionsMenu workstream lifecycle actions', () => {
  it('closes a workstream', async () => {
    // Arrange
    renderMenu()

    // Act
    fireEvent.click(await screen.findByText('Close workstream'))

    // Assert
    expect($workstreamMetadata.get()['stored-1']?.lifecycle).toBe('closed')
  })

  it('marks a workstream safe to delete', async () => {
    // Arrange
    renderMenu()

    // Act
    fireEvent.click(await screen.findByText('Mark safe to delete'))

    // Assert
    expect($workstreamMetadata.get()['stored-1']?.lifecycle).toBe('safe_delete')
  })

  it('marks a workstream restart required', async () => {
    // Arrange
    renderMenu()

    // Act
    fireEvent.click(await screen.findByText('Mark restart required'))

    // Assert
    expect($workstreamMetadata.get()['stored-1']?.lifecycle).toBe('restart_required')
  })

  it('reopens a metadata-backed workstream', async () => {
    // Arrange
    $workstreamMetadata.set({ 'stored-1': { lifecycle: 'closed', updatedAt: 123 } })
    renderMenu()

    // Act
    fireEvent.click(await screen.findByText('Reopen workstream'))

    // Assert
    expect($workstreamMetadata.get()).toEqual({})
  })
})
