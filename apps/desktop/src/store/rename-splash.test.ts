import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { existingUserVerdict } from '@/lib/existing-user'

import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests
} from './desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from './desktop-metrics-test-locks'
import { $guidanceLedger, GUIDANCE_LEDGER_KEY } from './guidance-ledger'
import {
  $renameSplash,
  $renameSplashReopenable,
  considerRenameSplash,
  dismissRenameSplash,
  noteRenameSplashRow,
  reopenRenameSplash
} from './rename-splash'
import { $shellChoice } from './shell-variant'

const evidence = { sessionsSettled: true, setupRunning: false, chats: 0, agents: 1, routines: 0 }

function recorded(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set({})
  $renameSplash.set(null)
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.unstubAllGlobals()
})

describe('existing-user verdict', () => {
  it('decides nothing until the session list has answered and first-run setup is off screen', () => {
    expect(existingUserVerdict({ ...evidence, sessionsSettled: false, chats: 4 })).toBe('unknown')
    expect(existingUserVerdict({ ...evidence, setupRunning: true, chats: 4 })).toBe('unknown')
  })

  it('reads a chat, a second agent or a routine as prior use, and an empty machine as a new install', () => {
    expect(existingUserVerdict({ ...evidence, chats: 1 })).toBe('existing')
    expect(existingUserVerdict({ ...evidence, agents: 2 })).toBe('existing')
    expect(existingUserVerdict({ ...evidence, routines: 1 })).toBe('existing')
    expect(existingUserVerdict(evidence)).toBe('fresh')
  })
})

describe('rename splash', () => {
  it('shows once to an existing user and stays seen after a nav click dismisses it', () => {
    considerRenameSplash('existing')
    expect($renameSplash.get()).toBe('shown')

    dismissRenameSplash('nav')
    expect($renameSplash.get()).toBeNull()
    expect(JSON.parse(window.localStorage.getItem(GUIDANCE_LEDGER_KEY) ?? '{}')).toEqual({ renameSplash: 'seen' })

    considerRenameSplash('existing')
    expect($renameSplash.get()).toBeNull()
  })

  it('never shows to a new install, even after it later gains chats', () => {
    considerRenameSplash('fresh')
    considerRenameSplash('existing')

    expect($renameSplash.get()).toBeNull()
    expect($renameSplashReopenable.get()).toBe(false)
  })

  it('shows again on the next visit when the page was left without dismissing it', () => {
    considerRenameSplash('existing')
    $renameSplash.set(null) // the page unloads; nothing was acknowledged

    considerRenameSplash('existing')
    expect($renameSplash.get()).toBe('shown')
  })

  it('reopens from the tray only once seen, and a reopened dismissal keeps it seen', () => {
    reopenRenameSplash()
    expect($renameSplash.get()).toBeNull()

    considerRenameSplash('existing')
    dismissRenameSplash('button')
    expect($renameSplashReopenable.get()).toBe(true)

    reopenRenameSplash()
    expect($renameSplash.get()).toBe('reopened')
    dismissRenameSplash('escape')
    expect($renameSplash.get()).toBeNull()
    expect($guidanceLedger.get().renameSplash).toBe('seen')
  })

  it('records closed codes only behind consent: shown, each row once per showing, how it closed, reopened', async () => {
    considerRenameSplash('existing')
    dismissRenameSplash('nav')
    expect(recorded()).toBe('')

    await setMetricsGate(captureDesktopMetricsGate, 'on')
    reopenRenameSplash()
    noteRenameSplashRow('library')
    noteRenameSplashRow('library')
    noteRenameSplashRow('machines')
    dismissRenameSplash('escape')

    const state = recorded()
    expect(state).toContain('ia.splash.rename_reopened|state')
    expect(state).toContain('"ia.splash_row.rename_library|state":1')
    expect(state).toContain('ia.splash_row.rename_machines|state')
    expect(state).toContain('ia.splash.rename_dismissed_escape|shortcut')
    expect(state).not.toContain('"other')
  })
})
