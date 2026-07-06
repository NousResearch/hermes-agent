import { beforeEach, describe, expect, it } from 'vitest'

import {
  $workstreamMetadata,
  explicitStateForLifecycle,
  setWorkstreamLifecycle,
  workstreamLifecycle
} from './workstream-metadata'

describe('workstream metadata', () => {
  beforeEach(() => {
    $workstreamMetadata.set({})
  })

  it('sets lifecycle metadata by stored session id', () => {
    // Arrange & Act
    setWorkstreamLifecycle('stored-1', 'restart_required', 123)

    // Assert
    expect($workstreamMetadata.get()).toEqual({
      'stored-1': { lifecycle: 'restart_required', updatedAt: 123 }
    })
    expect(workstreamLifecycle('stored-1')).toBe('restart_required')
    expect(explicitStateForLifecycle('restart_required')).toBe('restart')
  })

  it('maps closed and safe-delete lifecycles to the existing close workstream state', () => {
    // Arrange & Act / Assert
    expect(explicitStateForLifecycle('closed')).toBe('close')
    expect(explicitStateForLifecycle('safe_delete')).toBe('close')
    expect(explicitStateForLifecycle('active')).toBeNull()
  })

  it('preserves closed and safe-delete as distinct stored lifecycle metadata', () => {
    // Arrange & Act
    setWorkstreamLifecycle('closed-session', 'closed', 123)
    setWorkstreamLifecycle('safe-delete-session', 'safe_delete', 456)

    // Assert
    expect($workstreamMetadata.get()).toEqual({
      'closed-session': { lifecycle: 'closed', updatedAt: 123 },
      'safe-delete-session': { lifecycle: 'safe_delete', updatedAt: 456 }
    })
    expect(workstreamLifecycle('closed-session')).toBe('closed')
    expect(workstreamLifecycle('safe-delete-session')).toBe('safe_delete')
  })

  it('clears lifecycle metadata when reopened', () => {
    // Arrange
    setWorkstreamLifecycle('stored-1', 'closed', 123)

    // Act
    setWorkstreamLifecycle('stored-1', 'active', 456)

    // Assert
    expect($workstreamMetadata.get()).toEqual({})
    expect(workstreamLifecycle('stored-1')).toBe('active')
  })

  it('ignores empty session ids', () => {
    // Arrange & Act
    setWorkstreamLifecycle('', 'safe_delete', 123)

    // Assert
    expect($workstreamMetadata.get()).toEqual({})
  })
})
