import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $approvalModes, approvalModeForProfile } from '@/store/approval-mode'
import * as gateway from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

import { applyRuntimeInfo } from './utils'

describe('applyRuntimeInfo approval mode owner scoping', () => {
  beforeEach(() => {
    $approvalModes.set({})
    $activeGatewayProfile.set('work')
  })

  it('never lets another backend runtime set the active profile approval chip', () => {
    applyRuntimeInfo({ approval_mode: 'manual' })

    // A bot tile, a branch of a bot chat, an All-profiles resume, and a
    // same-named profile on another connection all report THEIR config.
    applyRuntimeInfo({ approval_mode: 'off' }, { foreground: false, owner: { connectionId: 'local', profile: 'bot' } })
    applyRuntimeInfo({ approval_mode: 'off' }, { owner: 'bot' })
    applyRuntimeInfo({ approval_mode: 'off' }, { owner: { connectionId: 'ssh-box', profile: 'work' } })

    expect(approvalModeForProfile('work')).toBe('manual')

    applyRuntimeInfo({ approval_mode: 'off' }, { owner: { connectionId: 'local', profile: 'work' } })

    expect(approvalModeForProfile('work')).toBe('off')

    applyRuntimeInfo({ approval_mode: 'manual' })
    // The active socket is (ssh-box, work); a bare 'work' owner dialled the local profile door.
    const primary = vi.spyOn(gateway, 'isActivePrimary').mockReturnValue(false)
    const connection = vi.spyOn(gateway, 'activeGatewayConnectionId').mockReturnValue('ssh-box')

    try {
      applyRuntimeInfo({ approval_mode: 'off' }, { owner: 'work' })
      expect(approvalModeForProfile('work')).toBe('manual')

      applyRuntimeInfo({ approval_mode: 'off' }, { owner: { connectionId: 'ssh-box', profile: 'work' } })
      expect(approvalModeForProfile('work')).toBe('off')
    } finally {
      primary.mockRestore()
      connection.mockRestore()
    }
  })
})
