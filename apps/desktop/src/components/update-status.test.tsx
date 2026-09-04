import { describe, expect, it } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'
import { en } from '@/i18n/en'
import type { UpdateApplyState } from '@/store/updates'

import { deriveUpdateStatus } from './update-status'

const IDLE_APPLY: UpdateApplyState = {
  applying: false,
  stage: 'idle',
  message: '',
  percent: null,
  error: null,
  command: null,
  log: []
}

function derive(status: DesktopUpdateStatus | null, apply: UpdateApplyState = IDLE_APPLY, checking = false) {
  return deriveUpdateStatus({ apply, checking, status, target: 'client', u: en.updates })
}

describe('deriveUpdateStatus', () => {
  it('never-checked: idle tone, invites a check', () => {
    const view = derive(null)

    expect(view.tone).toBe('idle')
    expect(view.updateAvailable).toBe(false)
    expect(view.line).toBe(en.updates.tapCheck)
  })

  it('unsupported wins over everything else and keeps the backend message', () => {
    const view = derive({ supported: false, message: 'managed install', behind: 5, updateAvailable: true })

    expect(view.tone).toBe('unsupported')
    expect(view.line).toBe('managed install')
    expect(view.supported).toBe(false)
  })

  it('check error surfaces the tone and the transport message separately', () => {
    const view = derive({ supported: true, error: 'check-failed', message: 'ECONNREFUSED' })

    expect(view.tone).toBe('error')
    expect(view.line).toBe(en.updates.cantReach)
    expect(view.error).toBe('ECONNREFUSED')
  })

  it('applying beats available so the card cannot offer a second install', () => {
    const view = derive({ supported: true, behind: 3 }, { ...IDLE_APPLY, applying: true, stage: 'update' })

    expect(view.applying).toBe(true)
    expect(view.tone).toBe('available')
    expect(view.line).toBe(en.updates.installing)
  })

  it('restart stage counts as applying even when the applying flag already dropped', () => {
    const view = derive({ supported: true }, { ...IDLE_APPLY, applying: false, stage: 'restart' })

    expect(view.applying).toBe(true)
  })

  it('behind count renders counted copy; count-free updateAvailable falls back', () => {
    expect(derive({ supported: true, behind: 4 }).line).toBe(en.updates.updateReady(4))

    // Shallow clone: exact count unknowable, flagged via updateAvailable.
    const unknown = derive({ supported: true, behind: 0, updateAvailable: true })

    expect(unknown.line).toBe(en.updates.updateReadyUnknown)
    expect(unknown.updateAvailable).toBe(true)
  })

  it('up to date: idle tone with the latest-version line', () => {
    const view = derive({ supported: true, behind: 0 })

    expect(view.tone).toBe('idle')
    expect(view.updateAvailable).toBe(false)
    expect(view.line).toBe(en.updates.latestBody)
  })

  it('backend target says the backend is current, not "you"', () => {
    const view = deriveUpdateStatus({
      apply: IDLE_APPLY,
      checking: false,
      status: { supported: true, behind: 0 },
      target: 'backend',
      u: en.updates
    })

    expect(view.line).toBe(en.updates.latestBodyBackend)
  })
})
