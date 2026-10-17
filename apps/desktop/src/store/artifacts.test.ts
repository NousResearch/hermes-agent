import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ArtifactDetection } from '@/lib/artifact-detect'

import {
  $artifactRegistry,
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

    // Artifact tabs are never persistable, so the profile's bucket stays empty
    // and the key is removed rather than stored as an empty list.
    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBeNull()
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

  it('evicts the oldest historical version when versions exceed the content budget (#108172)', () => {
    // One unit of headroom under the 4 Mi budget: latest versions are always
    // retained, so a single artifact with 5 oversized versions must drop its
    // oldest history while keeping every version's content byte-exact.
    const chunk = 'x'.repeat(1024 * 1024)
    const result = upsertArtifact('session-1', HTML_DETECTION, `${chunk} v1`)!

    for (let n = 2; n <= 5; n += 1) {
      upsertArtifact('session-1', HTML_DETECTION, `${chunk} v${n}`)
    }

    const record = getArtifact(result.artifactId)!

    // 5 Mi total exceeded the budget; the oldest historical version was
    // evicted, the newest version survives byte-exact.
    expect(record.versions.length).toBeLessThan(5)
    expect(record.versions.at(-1)!.content).toBe(`${chunk} v5`)

    const totalUnits = Object.values($artifactRegistry.get())
      .flatMap(records => records)
      .flatMap(record => record.versions)
      .reduce((sum, version) => sum + version.content.length, 0)

    expect(totalUnits).toBeLessThanOrEqual(4 * 1024 * 1024)
  })

  it('evicts oldest whole artifact records when latest-only records exceed the budget (#108172)', () => {
    const chunk = 'y'.repeat(1024 * 1024)

    // Six separate artifacts, each one Mi: cumulative latest-only content
    // (6 Mi) exceeds the 4 Mi budget, so the oldest whole records go.
    for (let n = 1; n <= 6; n += 1) {
      upsertArtifact(`session-${n}`, HTML_DETECTION, `${chunk} a${n}`)
    }

    const retained = Object.values($artifactRegistry.get()).flatMap(records => records)

    expect(retained.length).toBeLessThan(6)

    const totalUnits = retained
      .flatMap(record => record.versions)
      .reduce((sum, version) => sum + version.content.length, 0)

    expect(totalUnits).toBeLessThanOrEqual(4 * 1024 * 1024)

    // The newest record always survives, even when the registry is over budget.
    expect(retained.some(record => record.versions[0]!.content === `${chunk} a6`)).toBe(true)
  })

  it('keeps the single newest record even when it alone exceeds the budget (#108172)', () => {
    const huge = 'z'.repeat(5 * 1024 * 1024)
    const result = upsertArtifact('session-1', HTML_DETECTION, huge)!

    const record = getArtifact(result.artifactId)!

    expect(record).not.toBeNull()
    expect(record.versions.at(-1)!.content).toBe(huge)
  })

  it('follows a pinned selection by hash when pruning shifts version indexes (#108172)', () => {
    // One artifact with four versions: the FIRST is oversized, the second is
    // pinned. Going over budget evicts the oldest historical version (v1),
    // shifting the pinned v2 from index 1 to index 0 while keeping its content.
    const result = upsertArtifact('session-1', HTML_DETECTION, 'h'.repeat(3 * 1024 * 1024))!

    upsertArtifact('session-1', HTML_DETECTION, 'small-v2')
    upsertArtifact('session-1', HTML_DETECTION, 'small-v3')
    upsertArtifact('session-1', HTML_DETECTION, 'small-v4')

    // Pin the SECOND version (index 1) and capture ITS hash.
    selectArtifactVersion(result.artifactId, 1)
    const pinnedHash = getArtifact(result.artifactId)!.versions[1]!.hash
    expect(pinnedHash).toBeTruthy()

    // Push the registry over the 4 Mi budget: the oversized v1 is the oldest
    // historical version, so pruning evicts exactly it.
    upsertArtifact('session-2', { ...HTML_DETECTION, title: 'Huge Report' }, 'x'.repeat(1024 * 1024))

    const record = getArtifact(result.artifactId)!
    const selectionIndex = $artifactVersionSelection.get()[result.artifactId]

    // v1 was evicted; the pinned version survived at a shifted index.
    expect(record.versions).toHaveLength(3)
    expect(record.versions.some(version => version.hash === pinnedHash)).toBe(true)
    expect(selectionIndex).toBe(0)
    expect(record.versions[selectionIndex!]!.hash).toBe(pinnedHash)
  })
})
