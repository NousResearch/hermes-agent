import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ArtifactDetection } from '@/lib/artifact-detect'

import {
  $artifactRegistry,
  $artifactTabs,
  $artifactVersionSelection,
  artifactsForSession,
  artifactTabId,
  clearArtifactRegistry,
  closeArtifactTab,
  getArtifact,
  openArtifactTab,
  selectArtifactVersion,
  upsertArtifact
} from './artifacts'
import { $rightRailActiveTabId, PREVIEW_PANE_ID, RIGHT_RAIL_PREVIEW_TAB_ID } from './layout'
import { $paneOpen } from './panes'
import { $activeSessionId, $selectedStoredSessionId } from './session'

const HTML_DETECTION: ArtifactDetection = { kind: 'html', language: 'html', title: 'Pomodoro Timer' }

describe('artifacts store', () => {
  beforeEach(() => {
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearArtifactRegistry()
    $rightRailActiveTabId.set(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  afterEach(() => {
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    clearArtifactRegistry()
    window.localStorage.clear()
  })

  it('registers a new artifact with one version', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')

    expect(result?.versionAdded).toBe(true)
    expect(artifactsForSession('session-1')).toHaveLength(1)
    expect(getArtifact(result!.artifactId)?.versions).toHaveLength(1)
  })

  it('dedupes identical content by hash (streaming replays are no-ops)', () => {
    const first = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')
    const replay = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')

    expect(replay?.versionAdded).toBe(false)
    expect(replay?.artifactId).toBe(first?.artifactId)
    expect(getArtifact(first!.artifactId)?.versions).toHaveLength(1)
  })

  it('appends a version when the same artifact regenerates with new content', () => {
    const first = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')
    const second = upsertArtifact('session-1', HTML_DETECTION, '<html>v2</html>')

    expect(second?.versionAdded).toBe(true)
    expect(second?.artifactId).toBe(first?.artifactId)

    const record = getArtifact(first!.artifactId)

    expect(record?.versions).toHaveLength(2)
    expect(record?.versions.at(-1)?.content).toBe('<html>v2</html>')
    expect(artifactsForSession('session-1')).toHaveLength(1)
  })

  it('keeps different titles as separate artifacts', () => {
    upsertArtifact('session-1', HTML_DETECTION, '<html>timer</html>')
    upsertArtifact('session-1', { ...HTML_DETECTION, title: 'Budget Dashboard' }, '<html>budget</html>')

    expect(artifactsForSession('session-1')).toHaveLength(2)
  })

  it('scopes artifacts per session', () => {
    upsertArtifact('session-1', HTML_DETECTION, '<html>a</html>')
    upsertArtifact('session-2', HTML_DETECTION, '<html>b</html>')

    expect(artifactsForSession('session-1')).toHaveLength(1)
    expect(artifactsForSession('session-2')).toHaveLength(1)
  })

  it('persists the registry to localStorage', () => {
    upsertArtifact('session-1', HTML_DETECTION, '<html>persisted</html>')

    expect(window.localStorage.getItem('hermes.desktop.artifacts.v1')).toContain('persisted')
  })

  it('rejects empty sessions and empty content', () => {
    expect(upsertArtifact('', HTML_DETECTION, '<html>x</html>')).toBeNull()
    expect(upsertArtifact('session-1', HTML_DETECTION, '   ')).toBeNull()
  })

  it('opens and closes artifact tabs, driving the rail selection', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!
    const tabId = artifactTabId(result.artifactId)

    openArtifactTab(result.artifactId)

    expect($artifactTabs.get()).toEqual([tabId])
    expect($rightRailActiveTabId.get()).toBe(tabId)
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)

    closeArtifactTab(tabId)

    expect($artifactTabs.get()).toEqual([])
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  it('does not duplicate a tab when the same artifact opens twice', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    openArtifactTab(result.artifactId)
    openArtifactTab(result.artifactId)

    expect($artifactTabs.get()).toHaveLength(1)
  })

  it('tracks version selection and snaps back to latest', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    upsertArtifact('session-1', HTML_DETECTION, '<html>v2</html>')
    upsertArtifact('session-1', HTML_DETECTION, '<html>v3</html>')

    selectArtifactVersion(result.artifactId, 0)

    expect($artifactVersionSelection.get()[result.artifactId]).toBe(0)

    // Selecting the newest version clears the pin (absent = newest).
    selectArtifactVersion(result.artifactId, 2)

    expect(result.artifactId in $artifactVersionSelection.get()).toBe(false)

    // Out-of-range clamps.
    selectArtifactVersion(result.artifactId, -5)

    expect($artifactVersionSelection.get()[result.artifactId]).toBe(0)
  })

  it('reopening an artifact lands on the newest version', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    upsertArtifact('session-1', HTML_DETECTION, '<html>v2</html>')
    selectArtifactVersion(result.artifactId, 0)
    openArtifactTab(result.artifactId)

    expect(result.artifactId in $artifactVersionSelection.get()).toBe(false)
  })

  it('survives a registry reload round-trip', () => {
    upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')
    upsertArtifact('session-1', HTML_DETECTION, '<html>v2</html>')

    const raw = window.localStorage.getItem('hermes.desktop.artifacts.v1')!
    const parsed = JSON.parse(raw) as Record<string, unknown[]>

    expect(parsed['session-1']).toHaveLength(1)

    // Simulate a fresh boot: hydrate a clean registry from the persisted JSON.
    $artifactRegistry.set(JSON.parse(raw))

    const record = artifactsForSession('session-1')[0]

    expect(record?.versions).toHaveLength(2)
    expect(record?.versions.at(-1)?.content).toBe('<html>v2</html>')
  })
})
