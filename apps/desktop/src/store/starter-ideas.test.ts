import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $apiRequestScope } from '@/api/client'

import {
  bindDesktopMetrics,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests,
  setDesktopMetricsGate
} from './desktop-metrics'
import { $guidanceLedger, GUIDANCE_LEDGER_KEY } from './guidance-ledger'
import { $sessions } from './session'
import { $shellChoice } from './shell-variant'
import {
  $landingMoment,
  AWAY_AFTER_MS,
  $starterIdeas,
  hideStarterIdeas,
  noteStarterIdeaPicked,
  noteTaskFinished,
  showStarterIdeas
} from './starter-ideas'

function recorded(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

const NEWCOMER = { renameSplash: 'not_applicable' } as const

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
  $guidanceLedger.set(NEWCOMER)
})

afterEach(() => resetDesktopMetricsForTests())

describe('a newcomer’s starter ideas', () => {
  it('show until two chats have had a task finish, then leave on their own', () => {
    expect($starterIdeas.get()).toBe('ideas')
    expect($landingMoment.get()).toBe('first')

    noteTaskFinished('chat-a')
    noteTaskFinished('chat-a') // the same chat finishing again is still one chat
    expect($starterIdeas.get()).toBe('ideas')
    // C4: one task done, the landing says welcome back with the ideas still out.
    expect($landingMoment.get()).toBe('back')

    noteTaskFinished('chat-b')
    expect($starterIdeas.get()).toBe('retired')
    expect($landingMoment.get()).toBe('back')
  })

  it('leave sooner with Hide, come back with Show ideas, and leave again with Hide', () => {
    hideStarterIdeas()
    expect($starterIdeas.get()).toBe('retired')

    showStarterIdeas()
    expect($starterIdeas.get()).toBe('ideas')

    // Brought back on purpose: finishing more tasks does not take them away again.
    noteTaskFinished('chat-a')
    noteTaskFinished('chat-b')
    expect($starterIdeas.get()).toBe('ideas')

    hideStarterIdeas()
    expect($starterIdeas.get()).toBe('retired')
  })

  it('are not there for anyone who used Hermes before, whose finished tasks are not counted', () => {
    $guidanceLedger.set({ renameSplash: 'seen' })

    noteTaskFinished('chat-a')

    expect($starterIdeas.get()).toBe('none')
    expect($guidanceLedger.get().finishedTaskChats).toBeUndefined()
  })

  it('keep their count and marks in the guidance ledger, the seam that moves to the account', () => {
    noteTaskFinished('chat-a')
    hideStarterIdeas()

    expect(JSON.parse(window.localStorage.getItem(GUIDANCE_LEDGER_KEY) ?? '{}')).toEqual({
      ...NEWCOMER,
      starterIdeas: 'hidden',
      finishedTaskChats: ['chat-a']
    })
  })

  it('record content-free codes only behind consent: the idea picked, leaving on their own once, Hide, Show', () => {
    noteStarterIdeaPicked('clean_inbox')
    expect(recorded()).toBe('')

    setDesktopMetricsGate('on')
    noteStarterIdeaPicked('research_options')
    noteTaskFinished('chat-a')
    noteTaskFinished('chat-b')
    noteTaskFinished('chat-c')
    showStarterIdeas()
    hideStarterIdeas()

    const state = recorded()
    expect(state).toContain('ia.starter_chip.research_options|click')
    expect(state).toContain('"ia.starter_chips.hidden_auto|state":1')
    expect(state).toContain('ia.starter_chips.shown_again|click')
    expect(state).toContain('ia.starter_chips.hidden_user|click')
    expect(state).not.toContain('"other')
  })
})

describe('a newcomer’s landing after a month away (C8)', () => {
  const chat = (id: string, lastActiveMs: number, extra: Partial<SessionInfo> = {}) =>
    ({ id, last_active: lastActiveMs / 1000, ...extra }) as SessionInfo

  afterEach(() => {
    $sessions.set([])
    vi.useRealTimers()
  })

  it('says hello again once the newest chat has been still for thirty days, and settles when a task finishes', () => {
    vi.useFakeTimers({ now: new Date('2026-11-19T12:00:00Z') })
    const now = Date.now()
    noteTaskFinished('chat-a')

    $sessions.set([chat('chat-a', now - AWAY_AFTER_MS - 1000), chat('chat-b', now - AWAY_AFTER_MS - 5000)])
    expect($landingMoment.get()).toBe('away')

    // A task finishes: that chat is the newest now, and the landing is the settled one again.
    $sessions.set([chat('chat-a', now), chat('chat-b', now - AWAY_AFTER_MS - 5000)])
    expect($landingMoment.get()).toBe('back')
  })

  it('never counts a routine’s own runs as the person being here', () => {
    vi.useFakeTimers({ now: new Date('2026-11-19T12:00:00Z') })
    const now = Date.now()
    noteTaskFinished('chat-a')

    $sessions.set([chat('chat-a', now - AWAY_AFTER_MS - 1000), chat('cron-run', now, { scheduler_owned: true })])
    expect($landingMoment.get()).toBe('away')
  })

  it('is not for someone who has not finished a task yet', () => {
    vi.useFakeTimers({ now: new Date('2026-11-19T12:00:00Z') })
    $sessions.set([chat('chat-a', Date.now() - AWAY_AFTER_MS * 2)])

    expect($landingMoment.get()).toBe('first')
  })
})
