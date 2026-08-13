/**
 * Unit tests for the renderer-side find walker.
 *
 * These tests plant DOM by hand and drive the walker directly so the
 * behavior is auditable without spinning up the React store. The end-to-
 * end behavior (open / setFindQuery / step) is covered by find-bar.test.tsx.
 */

import { afterEach, describe, expect, it } from 'vitest'

import {
  captureFindScope,
  currentFindScope,
  performScopedFind,
  releaseFindScope,
  resolveCurrentFindScope
} from './find-in-page-scope'

function plantSurface(id: string, html: string, hidden = false): HTMLElement {
  const root = document.createElement('div')

  root.id = id
  root.setAttribute('data-chat-surface', '')

  if (hidden) {
    root.setAttribute('data-pane-hidden', '')
  }

  root.innerHTML = html
  document.body.appendChild(root)

  return root
}

afterEach(() => {
  document.body.innerHTML = ''
})

describe('resolveCurrentFindScope', () => {
  it('returns the foreground chat surface and skips hidden ones', () => {
    plantSurface('background', 'hidden chat', true)
    const foreground = plantSurface('foreground', 'visible chat')

    expect(resolveCurrentFindScope()?.id).toBe('foreground')
    // Marking it lets currentFindScope find it without re-resolving.
    captureFindScope()
    expect(currentFindScope()?.id).toBe('foreground')
  })

  it('returns null when no chat surface is mounted', () => {
    expect(resolveCurrentFindScope()).toBeNull()
  })

  it('returns the FIRST visible chat surface in document order', () => {
    // queryVisible follows document order — the first matching surface
    // wins. Visibility (hidden panes are skipped) is the only filter.
    plantSurface('first', 'one')
    plantSurface('second', 'two')

    expect(resolveCurrentFindScope()?.id).toBe('first')
  })
})

describe('performScopedFind', () => {
  it('wraps every (case-insensitive) match in a <mark> and counts them', () => {
    const surface = plantSurface('surface', '<p>NEEDLE needle Needle</p>')

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(3)
    expect(result.activeOrdinal).toBe(1)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(3)
    // The first <mark> carries the active marker.
    expect(surface.querySelectorAll('mark.find-hit[data-find-active]').length).toBe(1)
  })

  it('returns zero counts when the query has no matches and leaves the DOM untouched', () => {
    const surface = plantSurface('surface', '<p>hello world</p>')

    const result = performScopedFind(surface, 'nothing', { forward: true, findNext: false })

    expect(result).toEqual({ count: 0, activeOrdinal: 0 })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(surface.querySelector('p')?.textContent).toBe('hello world')
  })

  it('clears highlights and zeroes the counter when the query is empty', () => {
    const surface = plantSurface('surface', '<p>needle in a haystack</p>')
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    const result = performScopedFind(surface, '', { forward: true, findNext: false })

    expect(result).toEqual({ count: 0, activeOrdinal: 0 })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    // Original text restored.
    expect(surface.querySelector('p')?.textContent).toBe('needle in a haystack')
  })

  it('advances the active mark on findNext and walks back on findPrevious', () => {
    const surface = plantSurface('surface', '<p>a a a a</p>')
    performScopedFind(surface, 'a', { forward: true, findNext: false })

    const first = surface.querySelectorAll('mark.find-hit')
    expect(first.length).toBe(4)
    expect(first[0].hasAttribute('data-find-active')).toBe(true)

    // Step forward twice; the active ordinal cycles 1 → 2 → 3.
    let result = performScopedFind(surface, 'a', { forward: true, findNext: true })
    expect(result.activeOrdinal).toBe(2)
    result = performScopedFind(surface, 'a', { forward: true, findNext: true })
    expect(result.activeOrdinal).toBe(3)

    // Step backward once → 2.
    result = performScopedFind(surface, 'a', { forward: false, findNext: true })
    expect(result.activeOrdinal).toBe(2)

    // Wrap: from the last match forward wraps to 1.
    performScopedFind(surface, 'a', { forward: true, findNext: true })
    performScopedFind(surface, 'a', { forward: true, findNext: true }) // → 4
    result = performScopedFind(surface, 'a', { forward: true, findNext: true }) // → wraps to 1
    expect(result.activeOrdinal).toBe(1)

    // Backward wrap: from 1 → 4.
    result = performScopedFind(surface, 'a', { forward: false, findNext: true })
    expect(result.activeOrdinal).toBe(4)
  })

  it('lands on the LAST match when entering find mode backwards', () => {
    const surface = plantSurface('surface', '<p>a a a</p>')
    const result = performScopedFind(surface, 'a', { forward: false, findNext: false })

    expect(result.activeOrdinal).toBe(3)
  })

  it('does not re-wrap when stepping with the same query (findNext fast path)', () => {
    const surface = plantSurface('surface', '<p>a a a a</p>')
    performScopedFind(surface, 'a', { forward: true, findNext: false })

    const before = [...surface.querySelectorAll('mark.find-hit')]
    const result = performScopedFind(surface, 'a', { forward: true, findNext: true })

    // Same elements — the walker advanced the active marker rather than
    // tearing the DOM down and rebuilding it.
    const after = [...surface.querySelectorAll('mark.find-hit')]
    expect(after.length).toBe(before.length)
    expect(after.every(el => before.includes(el))).toBe(true)
    expect(result.count).toBe(4)
    expect(result.activeOrdinal).toBe(2)
  })

  it('does not re-wrap when the query differs only in case (fast path, triage #81778)', () => {
    // The marks store the ORIGINAL-CASE source slice, so a byte-equality
    // fast-path check fails on the first differently-cased match and every
    // Enter re-wraps, losing `data-find-active` and pinning the ordinal to
    // 1 forever. Case-insensitive comparison keeps stepping.
    const surface = plantSurface('surface', '<p>Hermes Hermes</p>')
    performScopedFind(surface, 'hermes', { forward: true, findNext: false })

    const before = [...surface.querySelectorAll('mark.find-hit')]
    const result = performScopedFind(surface, 'hermes', { forward: true, findNext: true })

    const after = [...surface.querySelectorAll('mark.find-hit')]
    expect(after.length).toBe(before.length)
    expect(after.every(el => before.includes(el))).toBe(true)
    expect(result.count).toBe(2)
    expect(result.activeOrdinal).toBe(2)
  })

  it('re-wraps when a stale mark from an earlier query survives (fast path, #81778 review)', () => {
    // Regression: a mark left over from an earlier query can match the
    // current query case-insensitively even though the mark set no longer
    // covers every occurrence — the genuine mark was destroyed by a
    // re-render, the stale one survived, and the current query's live
    // matches arrived unwrapped. The fast path must not step on the stale
    // set — it must drop it and re-wrap, or the live match stays dark.
    const surface = plantSurface('surface', '<p>hermes</p>')
    performScopedFind(surface, 'hermes', { forward: true, findNext: false })
    // Re-render replaces the paragraph (destroying the genuine mark)…
    surface.querySelector('p')!.innerHTML = 'beta'
    // …but a stale mark survives the edit (external write / raced cleanup),
    // and the live content that matches the current query arrives unwrapped.
    surface.insertAdjacentHTML('beforeend', '<mark class="find-hit">Hermes</mark>')
    surface.insertAdjacentHTML('beforeend', '<p>hermes</p>')

    const result = performScopedFind(surface, 'hermes', { forward: true, findNext: true })

    // Re-wrapped: the stale mark's text is re-wrapped as a live occurrence
    // and the previously unwrapped match is wrapped too.
    expect(result.count).toBe(2)
    const marks = [...surface.querySelectorAll('mark.find-hit')]
    expect(marks.length).toBe(2)
    expect(marks.every(mark => mark.textContent.toLowerCase() === 'hermes')).toBe(true)
  })

  it('re-wraps when new unmarked content contains the query (fast path, #81778 review)', () => {
    // Regression: content appended after the last wrap (streamed tokens,
    // external writes) carries no mark, so the existing marks no longer
    // cover every occurrence. Stepping must re-wrap instead of walking the
    // stale set, or the new match would never be highlighted.
    const surface = plantSurface('surface', '<p>needle</p>')
    performScopedFind(surface, 'needle', { forward: true, findNext: false })

    surface.insertAdjacentHTML('beforeend', '<p>needle</p>')

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: true })

    expect(result.count).toBe(2)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)
  })

  it('re-wraps when the query changes (marks are not reused across queries)', () => {
    const surface = plantSurface('surface', '<p>alpha beta alpha</p>')
    performScopedFind(surface, 'alpha', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)

    performScopedFind(surface, 'beta', { forward: true, findNext: false })

    // The alpha marks are gone; the beta mark is wrapped.
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)
    expect(surface.querySelector('mark.find-hit')?.textContent).toBe('beta')
    expect(surface.querySelector('p')?.textContent).toBe('alpha beta alpha')
  })

  it('ignores text inside the FindBar search input itself (no self-match)', () => {
    // The walker skips any subtree that carries role="search" so a user
    // typing "needle" never matches the placeholder text in the input.
    const surface = plantSurface('surface', '')
    surface.innerHTML = `
      <div role="search"><input placeholder="needle placeholder" /></div>
      <p>needle haystack</p>
    `

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(1)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)
    // The mark lives inside the <p>, not inside the search overlay.
    expect(surface.querySelector('mark.find-hit')?.closest('p')).toBeTruthy()
  })

  it('does not match inside <script> / <style> nodes', () => {
    const surface = plantSurface('surface', '')
    surface.innerHTML = `
      <script>const needle = 'literal'</script>
      <style>.needle { color: red }</style>
      <p>needle visible</p>
    `

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(1)
    expect(surface.querySelector('mark.find-hit')?.textContent).toBe('needle')
  })

  it('does not double-match when re-running the SAME query in findNext mode', () => {
    const surface = plantSurface('surface', '<p>needle needle</p>')
    performScopedFind(surface, 'needle', { forward: true, findNext: false })

    // The walker must recognize that the existing marks already represent
    // this query and advance the active marker without rebuilding.
    const result = performScopedFind(surface, 'needle', { forward: true, findNext: true })

    expect(result.count).toBe(2)
    expect(result.activeOrdinal).toBe(2)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)
  })

  it('keeps searching sibling subtrees after a fully-consumed text node', () => {
    // Regression (#81778 review): a text node that was entirely consumed by
    // a match used to null the walker's `current`, terminating the sibling
    // traversal — `<div>needle<span>needle</span></div>` only matched the
    // first occurrence. The sibling subtree must still be searched.
    const surface = plantSurface(
      'surface',
      '<div>needle<span>needle</span></div><p>needle</p>'
    )
    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(3)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(3)
  })
})

describe('scope lifecycle', () => {
  it('captureFindScope marks the foreground surface for the walker', () => {
    const surface = plantSurface('surface', 'needle')
    captureFindScope()

    expect(currentFindScope()).toBe(surface)
    expect(surface.hasAttribute('data-find-root')).toBe(true)
  })

  it('releaseFindScope unwraps every highlight and clears the scope marker', () => {
    const surface = plantSurface('surface', '<p>needle in a haystack</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    releaseFindScope()

    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(surface.querySelector('[data-find-root]')).toBeNull()
    expect(currentFindScope()).toBeNull()
    // Original text restored.
    expect(surface.querySelector('p')?.textContent).toBe('needle in a haystack')
  })

  it('keeps the scope targeting the FOREGROUND surface even when a hidden one matches the selector first', () => {
    // Regression (#81726): the visible chat is the SECOND surface in
    // document order, but it must still win the scope resolution.
    plantSurface('background', '<p>background needle</p>', true)
    const foreground = plantSurface('foreground', '<p>foreground needle</p>')

    captureFindScope()
    expect(currentFindScope()).toBe(foreground)

    // The walker only wraps the foreground's match.
    const result = performScopedFind(foreground, 'needle', { forward: true, findNext: false })
    expect(result.count).toBe(1)
  })
})

describe('scoped find survives React re-render', () => {
  // The transcript is React-owned: assistant answers stream, and a new message
  // is appended whenever the assistant replies. React doesn't know about the
  // `<mark>`s we insert, so a render that re-allocates a region detaches its
  // marks. The scope watcher (MutationObserver) must re-wrap after the render
  // commits. jsdom delivers observer callbacks + our queueMicrotask re-apply
  // as microtasks, so awaiting a macrotask drains them all.
  async function flushAll(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 0))
  }

  it('re-wraps content React re-allocated, keeping the active position', async () => {
    const surface = plantSurface('surface', '<p>needle hay needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)

    // Step onto match #2, then mock a React re-render that rebuilds the
    // paragraph (destroying our marks) while the matching text remains.
    performScopedFind(surface, 'needle', { forward: true, findNext: true })
    expect(surface.querySelectorAll('mark.find-hit[data-find-active]')[0]).toBe(
      surface.querySelectorAll('mark.find-hit')[1]
    )

    surface.querySelector('p')!.innerHTML = 'needle hay needle'
    await flushAll()

    // Marks restored, and the active mark is still the second match.
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)
    expect(surface.querySelectorAll('mark.find-hit[data-find-active]').length).toBe(1)
    expect(surface.querySelectorAll('mark.find-hit[data-find-active]')[0]).toBe(
      surface.querySelectorAll('mark.find-hit')[1]
    )
    expect(surface.querySelector('p')!.textContent).toBe('needle hay needle')
  })

  it('keeps existing highlights and wraps a newly appended message', async () => {
    // The task's target scenario: the bar is open with highlights, then the
    // assistant appends a NEW message while the list stays mounted. Existing
    // highlights must persist and the appended content must not be broken.
    const surface = plantSurface('surface', '<p>needle in the first message</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    // React appends a new message block.
    const next = document.createElement('div')
    next.innerHTML = 'second message needle'
    surface.appendChild(next)
    await flushAll()

    // The first highlight survived the append AND the new match was wrapped
    // too — nothing clobbered, nothing lost.
    const marks = surface.querySelectorAll('mark.find-hit')
    expect(marks.length).toBe(2)
    expect(surface.querySelector('p')!.textContent).toBe('needle in the first message')
    expect(next.textContent).toBe('second message needle')
  })

  it('does NOT re-wrap when React touched a region with no unmarked match', async () => {
    const surface = plantSurface('surface', '<p>needle needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)

    const before = [...surface.querySelectorAll('mark.find-hit')]
    // React re-renders a paragraph that doesn't match the query — the marks
    // are untouched and must not be rebuilt.
    surface.insertAdjacentHTML('beforeend', '<p>irrelevant</p>')
    await flushAll()

    const after = [...surface.querySelectorAll('mark.find-hit')]
    expect(after.length).toBe(before.length)
    expect(after.every(mark => before.includes(mark))).toBe(true)
  })

  it('searches a typeahead that clears and re-runs still tears the watcher down', async () => {
    const surface = plantSurface('surface', '<p>needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    // Clearing the query tears down the watcher; a later React write that
    // would normally re-wrap must NOT resurrect highlights (the bar closed).
    performScopedFind(surface, '', { forward: true, findNext: false })
    surface.querySelector('p')!.innerHTML = 'needle again'
    await flushAll()

    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(surface.querySelector('p')!.textContent).toBe('needle again')
  })

  it('releaseFindScope detaches the watcher so no write can re-highlight', async () => {
    const surface = plantSurface('surface', '<p>needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    releaseFindScope()
    surface.querySelector('p')!.innerHTML = 'needle needle'
    await flushAll()

    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
  })
})
