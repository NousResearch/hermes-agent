import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ArtifactDetection } from '@/lib/artifact-detect'

import {
  $artifactVersionSelection,
  artifactsForSession,
  clearArtifactRegistry,
  getArtifact,
  openArtifact,
  selectArtifactVersion,
  upsertArtifact
} from './artifacts'
import { $rightRailActiveTabId } from './layout'
import { $previewTabs, closeRightRail, closeRightRailTab } from './preview'
import { $activeSessionId, $selectedStoredSessionId } from './session'

const HTML_DETECTION: ArtifactDetection = { kind: 'html', language: 'html', title: 'Pomodoro Timer' }

describe('artifacts store', () => {
  beforeEach(() => {
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearArtifactRegistry()
    closeRightRail()
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

  it('rejects empty sessions and empty content', () => {
    expect(upsertArtifact('', HTML_DETECTION, '<html>x</html>')).toBeNull()
    expect(upsertArtifact('session-1', HTML_DETECTION, '   ')).toBeNull()
  })

  it('opens an artifact as a real rail tab that references the registry', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    openArtifact(result.artifactId)

    const tab = $previewTabs.get()[0]!

    expect(tab.target).toMatchObject({ kind: 'artifact', label: 'Pomodoro Timer', url: result.artifactId })
    expect($rightRailActiveTabId.get()).toBe(tab.id)

    closeRightRailTab(tab.id)

    expect($previewTabs.get()).toEqual([])
    expect($rightRailActiveTabId.get()).toBeNull()
  })

  it('does not duplicate a tab when the same artifact opens twice', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    openArtifact(result.artifactId)
    openArtifact(result.artifactId)

    expect($previewTabs.get()).toHaveLength(1)
  })

  it('keeps artifact tabs out of the persisted tab list', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    openArtifact(result.artifactId)

    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBe('[]')
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

  it('opens at the newest version by default and at a pinned one on request', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    upsertArtifact('session-1', HTML_DETECTION, '<html>v2</html>')

    openArtifact(result.artifactId, 0)

    expect($artifactVersionSelection.get()[result.artifactId]).toBe(0)

    openArtifact(result.artifactId)

    expect(result.artifactId in $artifactVersionSelection.get()).toBe(false)
  })

  it('clearing the registry closes the tabs pointing into it', () => {
    const result = upsertArtifact('session-1', HTML_DETECTION, '<html>v1</html>')!

    openArtifact(result.artifactId)
    clearArtifactRegistry()

    expect($previewTabs.get()).toEqual([])
    expect(artifactsForSession('session-1')).toEqual([])
  })
})
