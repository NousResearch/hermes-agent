import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import {
  bindDesktopMetrics,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests,
  setDesktopMetricsGate
} from './desktop-metrics'
import { $guidanceLedger, GUIDANCE_LEDGER_KEY } from './guidance-ledger'
import { $shellChoice } from './shell-variant'
import {
  $newcomerGreeting,
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

    noteTaskFinished('chat-a')
    noteTaskFinished('chat-a') // the same chat finishing again is still one chat
    expect($starterIdeas.get()).toBe('ideas')
    expect($newcomerGreeting.get()).toBe(true)

    noteTaskFinished('chat-b')
    expect($starterIdeas.get()).toBe('retired')
    expect($newcomerGreeting.get()).toBe(false)
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
