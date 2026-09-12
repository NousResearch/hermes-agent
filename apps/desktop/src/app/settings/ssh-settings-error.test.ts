import { describe, expect, it } from 'vitest'

import { presentSshSettingsError } from './ssh-settings-error'

const mapped = {
  'auth-failed': 'Auth failed.',
  'hermes-not-found': 'Not installed.',
  'host-key-changed': 'Host key changed.',
  timeout: 'Timed out.',
  unreachable: 'Unreachable.',
  'unsupported-platform': 'Unsupported.',
  'update-required': 'Update required.'
}

const unknown = 'SSH connection failed.'

describe('presentSshSettingsError', () => {
  it('keeps generic copy and surfaces a sanitized spawn-log excerpt', () => {
    const presented = presentSshSettingsError({
      sshError: 'spawn-failed',
      mapped,
      unknown,
      detail: 'PermissionError: denied'
    })

    expect(presented.message).toBe(unknown)
    expect(presented.detail).toBe('PermissionError: denied')
  })

  it('surfaces ready-timeout detail the same way', () => {
    const presented = presentSshSettingsError({
      sshError: 'ready-timeout',
      mapped,
      unknown,
      detail: 'OOM killed'
    })

    expect(presented.message).toBe(unknown)
    expect(presented.detail).toBe('OOM killed')
  })

  it('keeps generic copy only when the excerpt is empty', () => {
    expect(
      presentSshSettingsError({
        sshError: 'spawn-failed',
        mapped,
        unknown,
        detail: '   '
      })
    ).toEqual({ message: unknown })
  })

  it('does not attach spawn-log detail to already-mapped SSH kinds', () => {
    expect(
      presentSshSettingsError({
        sshError: 'auth-failed',
        mapped,
        unknown,
        detail: 'PermissionError: denied'
      })
    ).toEqual({ message: 'Auth failed.' })
  })
})
