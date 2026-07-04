import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { setSelectedStoredSessionId, setSessions } from '@/store/session'

import { $paneStates, getPaneStateSnapshot } from './panes'
import {
  $activePaneId,
  $splitFreshDraft,
  $splitOpen,
  $splitPaneSession,
  closeSplitPane,
  openSplitPane,
  restoreSplitPaneSession,
  setActivePane,
  setSplitPaneFreshDraft,
  setSplitPaneSession,
  SPLIT_PANE_ID,
  toggleSplitPane
} from './split'

const SPLIT_SESSION_KEY = 'hermes.desktop.splitPane.v1'

function resetSplitState() {
  $paneStates.set({})
  $splitPaneSession.set(null)
  $splitFreshDraft.set(false)
  $activePaneId.set('main')
  setSelectedStoredSessionId(null)
  setSessions([])
  window.localStorage.clear()
}

function sessionRow(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-row',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: 'row',
    tool_call_count: 0,
    ...overrides
  }
}

describe('split store', () => {
  beforeEach(resetSplitState)

  afterEach(() => {
    resetSplitState()
    vi.restoreAllMocks()
  })

  describe('openSplitPane / closeSplitPane', () => {
    it('opens on a session: pane snapshot open, session set, split focused', () => {
      const opened = openSplitPane({ storedId: 'stored-2', profile: 'coder' })

      expect(opened).toBe(true)
      expect(getPaneStateSnapshot(SPLIT_PANE_ID)?.open).toBe(true)
      expect($splitPaneSession.get()).toEqual({ storedId: 'stored-2', profile: 'coder' })
      expect($activePaneId.get()).toBe('split')
      expect($splitOpen.get()).toBe(true)
    })

    it('opens without a session as a fresh draft', () => {
      expect(openSplitPane()).toBe(true)

      expect($splitPaneSession.get()).toBeNull()
      expect($splitFreshDraft.get()).toBe(true)
      expect($splitOpen.get()).toBe(true)
    })

    it('re-opening without a session keeps the split session it already holds', () => {
      openSplitPane({ storedId: 'stored-2', profile: null })
      closeSplitPane()
      setSplitPaneSession('stored-3')

      openSplitPane()

      expect($splitPaneSession.get()).toEqual({ storedId: 'stored-3', profile: null })
      expect($splitFreshDraft.get()).toBe(false)
    })

    it('same-session guard: asking for the main pane’s session focuses main instead', () => {
      setSelectedStoredSessionId('stored-dup')
      setActivePane('split')

      const opened = openSplitPane({ storedId: 'stored-dup', profile: null })

      expect(opened).toBe(false)
      expect($splitOpen.get()).toBe(false)
      expect($splitPaneSession.get()).toBeNull()
      expect($activePaneId.get()).toBe('main')
    })

    it('close clears the session + fresh flag and hands focus back to main', () => {
      openSplitPane({ storedId: 'stored-2', profile: null })
      closeSplitPane()

      expect(getPaneStateSnapshot(SPLIT_PANE_ID)?.open).toBe(false)
      expect($splitPaneSession.get()).toBeNull()
      expect($splitFreshDraft.get()).toBe(false)
      expect($activePaneId.get()).toBe('main')
      expect($splitOpen.get()).toBe(false)
    })

    it('toggleSplitPane round-trips open and closed', () => {
      toggleSplitPane()
      expect($splitOpen.get()).toBe(true)

      toggleSplitPane()
      expect($splitOpen.get()).toBe(false)
    })

    it('a stale open snapshot without a session or draft does NOT count as open', () => {
      // e.g. relaunch after the split session was deleted: the pane snapshot
      // still says open, but there is nothing to mount.
      openSplitPane({ storedId: 'stored-2', profile: null })
      $splitPaneSession.set(null)

      expect(getPaneStateSnapshot(SPLIT_PANE_ID)?.open).toBe(true)
      expect($splitOpen.get()).toBe(false)
    })
  })

  describe('setSplitPaneSession', () => {
    it('keeps the profile tag when re-setting the same stored id without one', () => {
      setSplitPaneSession('stored-5', 'coder')
      setSplitPaneSession('stored-5')

      expect($splitPaneSession.get()).toEqual({ storedId: 'stored-5', profile: 'coder' })
    })

    it('re-resolves the profile tag from $sessions when pointing at a different session', () => {
      // Pane-local navigation (fresh-draft send / branch in the split) never
      // carries a profile — the optimistic row upserted just before it does.
      // Carrying the OLD tag or nulling it would unpin the pane's request
      // wrapper and drop the profile from the gateway keep-set.
      setSessions([sessionRow({ id: 'stored-6', profile: 'analyst' })])
      setSplitPaneSession('stored-5', 'coder')
      setSplitPaneSession('stored-6')

      expect($splitPaneSession.get()).toEqual({ storedId: 'stored-6', profile: 'analyst' })
    })

    it('resolves via the lineage-root id and falls back to null for an unknown row', () => {
      setSessions([sessionRow({ id: 'tip-7', _lineage_root_id: 'stored-7', profile: 'coder' })])

      setSplitPaneSession('stored-7')
      expect($splitPaneSession.get()).toEqual({ storedId: 'stored-7', profile: 'coder' })

      setSplitPaneSession('stored-unknown')
      expect($splitPaneSession.get()).toEqual({ storedId: 'stored-unknown', profile: null })
    })

    it('clears the fresh-draft flag when a session takes the pane', () => {
      setSplitPaneFreshDraft()
      setSplitPaneSession('stored-5')

      expect($splitFreshDraft.get()).toBe(false)
    })
  })

  describe('persistence', () => {
    it('persists the split session (with profile) and clears the key on close', () => {
      openSplitPane({ storedId: 'stored-7', profile: 'analyst' })

      expect(JSON.parse(window.localStorage.getItem(SPLIT_SESSION_KEY) ?? 'null')).toEqual({
        storedId: 'stored-7',
        profile: 'analyst'
      })

      closeSplitPane()

      expect(window.localStorage.getItem(SPLIT_SESSION_KEY)).toBeNull()
    })

    it('restores a persisted split session on module load', async () => {
      window.localStorage.setItem(SPLIT_SESSION_KEY, JSON.stringify({ storedId: 'stored-9', profile: 'coder' }))
      vi.resetModules()

      const fresh = await import('./split')

      expect(fresh.$splitPaneSession.get()).toEqual({ storedId: 'stored-9', profile: 'coder' })
    })

    it('treats malformed / storedId-less persisted state as closed', async () => {
      window.localStorage.setItem(SPLIT_SESSION_KEY, '{not json')
      vi.resetModules()
      expect((await import('./split')).$splitPaneSession.get()).toBeNull()

      window.localStorage.setItem(SPLIT_SESSION_KEY, JSON.stringify({ profile: 'coder' }))
      vi.resetModules()
      expect((await import('./split')).$splitPaneSession.get()).toBeNull()
    })
  })

  describe('restoreSplitPaneSession', () => {
    it('resolves null without probing when the split holds no session', async () => {
      const resolve = vi.fn(async () => ({ id: 'x' }))

      expect(await restoreSplitPaneSession(resolve)).toBeNull()
      expect(resolve).not.toHaveBeenCalled()
    })

    it('keeps the split when the probe resolves the session', async () => {
      openSplitPane({ storedId: 'stored-2', profile: null })

      const restored = await restoreSplitPaneSession(async () => ({ id: 'stored-2' }))

      expect(restored).toEqual({ storedId: 'stored-2', profile: null })
      expect($splitOpen.get()).toBe(true)
    })

    it('closes the split on a probe miss (deleted/archived session)', async () => {
      openSplitPane({ storedId: 'stored-2', profile: null })

      expect(await restoreSplitPaneSession(async () => undefined)).toBeNull()
      expect($splitOpen.get()).toBe(false)
      expect($splitPaneSession.get()).toBeNull()
    })

    it('closes the split when the probe throws (backend 404 path)', async () => {
      openSplitPane({ storedId: 'stored-2', profile: null })

      expect(
        await restoreSplitPaneSession(async () => {
          throw new Error('404: Session not found')
        })
      ).toBeNull()

      expect($splitOpen.get()).toBe(false)
    })
  })
})
