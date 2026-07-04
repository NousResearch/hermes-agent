import { afterEach, describe, expect, it } from 'vitest'

import {
  $activeSessionId,
  $busy,
  $messages,
  $selectedStoredSessionId,
  setActiveSessionId,
  setBusy,
  setMessages,
  setSelectedStoredSessionId
} from '@/store/session'
import { $activePaneId } from '@/store/split'

import { createSplitPaneView, MAIN_PANE_VIEW } from './pane-view'
import { mirrorPaneToGlobals } from './split-mirror'

// The activation republish (design §3.6): while the split pane is focused the
// IDENTITY singletons carry its session, and switching back restores the main
// pane's — without ever touching the atoms the main pane renders from.

// Main-pane identity as production writes it: atom + cache ref in lockstep
// (use-session-actions pairs every id write; the refs are the UNMIRRORED
// truth the teardown restores from).
function setMainIdentity(runtimeId: null | string, storedId: null | string) {
  setActiveSessionId(runtimeId)
  MAIN_PANE_VIEW.activeSessionIdRef.current = runtimeId
  setSelectedStoredSessionId(storedId)
  MAIN_PANE_VIEW.selectedStoredSessionIdRef.current = storedId
}

function resetGlobals() {
  setMainIdentity(null, null)
  setMessages([])
  setBusy(false)
  $activePaneId.set('main')
}

describe('mirrorPaneToGlobals', () => {
  afterEach(resetGlobals)

  it('activation copy: the identity globals equal the split view the instant the mirror installs', () => {
    setMainIdentity('rt-main', 'stored-main')

    const view = createSplitPaneView()
    view.setActiveSessionId('rt-split')
    view.setSelectedStoredSessionId('stored-split')

    const off = mirrorPaneToGlobals(view)

    expect($activeSessionId.get()).toBe('rt-split')
    expect($selectedStoredSessionId.get()).toBe('stored-split')

    off()
  })

  it('tracks split view identity writes while installed (pane-local navigation)', () => {
    const view = createSplitPaneView()
    const off = mirrorPaneToGlobals(view)

    view.setActiveSessionId('rt-a')
    view.setSelectedStoredSessionId('stored-a')

    expect($activeSessionId.get()).toBe('rt-a')
    expect($selectedStoredSessionId.get()).toBe('stored-a')

    view.setSelectedStoredSessionId('stored-b')

    expect($selectedStoredSessionId.get()).toBe('stored-b')

    off()
  })

  it('teardown restores the main pane identity from its cache refs', () => {
    setMainIdentity('rt-main', 'stored-main')

    const view = createSplitPaneView()
    view.setActiveSessionId('rt-split')
    view.setSelectedStoredSessionId('stored-split')

    const off = mirrorPaneToGlobals(view)

    expect($activeSessionId.get()).toBe('rt-split')

    off()

    expect($activeSessionId.get()).toBe('rt-main')
    expect($selectedStoredSessionId.get()).toBe('stored-main')
  })

  it('honors a BACKGROUND main-pane resume that rebound the ids mid-mirror', () => {
    // The reconnect route-resume / bounded auto-retry keep running for the
    // main session while the split is focused. They rewrite atom+ref in
    // lockstep; an install-time snapshot would restore the DEAD pre-reconnect
    // runtime id on switch-back — the refs are the live truth.
    setMainIdentity('rt-main-old', 'stored-main')

    const view = createSplitPaneView()
    view.setActiveSessionId('rt-split')
    view.setSelectedStoredSessionId('stored-split')

    const off = mirrorPaneToGlobals(view)

    // Background resume rebinds the main runtime id while the mirror is up.
    setMainIdentity('rt-main-rebound', 'stored-main')

    // A later split-view write re-asserts the mirrored identity…
    view.setActiveSessionId('rt-split-2')
    expect($activeSessionId.get()).toBe('rt-split-2')

    off()

    // …and switch-back lands on the REBOUND main id, not the stale snapshot.
    expect($activeSessionId.get()).toBe('rt-main-rebound')
    expect($selectedStoredSessionId.get()).toBe('stored-main')
  })

  it('stops tracking after teardown — later split writes stay pane-local', () => {
    const view = createSplitPaneView()
    const off = mirrorPaneToGlobals(view)
    off()

    view.setActiveSessionId('rt-late')

    expect($activeSessionId.get()).toBeNull()
  })

  it('never touches the atoms the main pane renders from ($messages / $busy)', () => {
    setMessages([{ id: 'm-main', parts: [{ text: 'main transcript', type: 'text' }], role: 'user' }])
    setBusy(false)

    const view = createSplitPaneView()
    view.setMessages([{ id: 'm-split', parts: [{ text: 'split transcript', type: 'text' }], role: 'user' }])
    view.setBusy(true)

    const off = mirrorPaneToGlobals(view)

    // The split streams while focused — its transcript/busy stay in ITS
    // bundle; the main pane keeps rendering its own.
    view.setMessages([
      { id: 'm-split', parts: [{ text: 'split transcript', type: 'text' }], role: 'user' },
      { id: 'a-split', parts: [{ text: 'reply', type: 'text' }], role: 'assistant' }
    ])

    expect($messages.get().map(message => message.id)).toEqual(['m-main'])
    expect($busy.get()).toBe(false)

    off()
  })
})
