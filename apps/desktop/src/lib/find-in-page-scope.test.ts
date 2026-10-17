/**
 * Unit tests for the renderer-side find engine.
 *
 * These tests plant DOM by hand and drive the engine directly so the behavior
 * is auditable without spinning up the React store. The end-to-end behavior
 * (open / setFindQuery / step) is covered by find-bar.test.tsx.
 *
 * Matches are painted with the CSS Custom Highlight API, so the observable
 * result is what the walker registered in `CSS.highlights` — `findHitTexts()`
 * and `activeFindHitText()` read it (the jsdom stand-in lives in
 * src/test/jsdom.ts). The other half of the contract is what the walker did
 * NOT do: the transcript DOM must come back byte-identical.
 */

import { afterEach, describe, expect, it } from 'vitest'

import { activeFindHitText, findHitTexts } from '@/test/jsdom'

import {
  captureFindScope,
  currentFindScope,
  performScopedFind,
  releaseFindScope,
  resolveCurrentFindScope
} from './find-in-page-scope'

/** A re-scan triggered by a mutation is throttled; wait past one interval. */
const flushRescan = () => new Promise(resolve => setTimeout(resolve, 260))

const registeredHits = (): Range[] => [...((CSS.highlights.get('hermes-find') ?? new Set()) as Set<Range>)]

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
  releaseFindScope()
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
  it('paints every (case-insensitive) match and counts them', () => {
    const surface = plantSurface('surface', '<p>NEEDLE needle Needle</p>')

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(3)
    expect(result.activeOrdinal).toBe(1)
    // The ranges carry the ORIGINAL-CASE source slice.
    expect(findHitTexts()).toEqual(['NEEDLE', 'needle', 'Needle'])
    // Exactly one match is the active one.
    expect(activeFindHitText()).toBe('NEEDLE')
  })

  it('never touches the transcript DOM it searches', () => {
    // The whole point of the highlight-api engine: the walker is a pure read.
    // The old node-wrapping engine split every matching text node, inserted a
    // <mark>, and re-normalized on clear.
    const surface = plantSurface('surface', '<p>needle <b>needle</b> hay</p><div>needle</div>')
    const before = surface.innerHTML

    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toHaveLength(3)
    expect(surface.innerHTML).toBe(before)

    performScopedFind(surface, '', { forward: true, findNext: false })
    expect(surface.innerHTML).toBe(before)
  })

  it('returns zero counts when the query has no matches and paints nothing', () => {
    const surface = plantSurface('surface', '<p>hello world</p>')

    const result = performScopedFind(surface, 'nothing', { forward: true, findNext: false })

    expect(result).toEqual({ count: 0, activeOrdinal: 0 })
    expect(findHitTexts()).toEqual([])
    expect(activeFindHitText()).toBeNull()
    expect(surface.querySelector('p')?.textContent).toBe('hello world')
  })

  it('clears highlights and zeroes the counter when the query is empty', () => {
    const surface = plantSurface('surface', '<p>needle in a haystack</p>')
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toHaveLength(1)

    const result = performScopedFind(surface, '', { forward: true, findNext: false })

    expect(result).toEqual({ count: 0, activeOrdinal: 0 })
    expect(findHitTexts()).toEqual([])
    expect(surface.querySelector('p')?.textContent).toBe('needle in a haystack')
  })

  it('advances the active match on findNext and walks back on findPrevious', () => {
    const surface = plantSurface('surface', '<p>a a a a</p>')
    performScopedFind(surface, 'a', { forward: true, findNext: false })

    expect(registeredHits().length).toBe(4)
    expect(activeFindHitText()).toBe('a')

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

  it('steps without re-collecting the matches (findNext fast path)', () => {
    const surface = plantSurface('surface', '<p>a a a a</p>')
    performScopedFind(surface, 'a', { forward: true, findNext: false })

    const before = registeredHits()
    const result = performScopedFind(surface, 'a', { forward: true, findNext: true })

    // Same Range objects — stepping moved the active pointer rather than
    // re-scanning the scope.
    const after = registeredHits()
    expect(after.length).toBe(before.length)
    expect(after.every(range => before.includes(range))).toBe(true)
    expect(result.count).toBe(4)
    expect(result.activeOrdinal).toBe(2)
  })

  it('steps a differently-cased query without re-collecting (fast path, triage #81778)', () => {
    // The query is compared as typed, so a byte-equality fast path would fail
    // on the first differently-cased match and re-collect on every Enter —
    // resetting the active ordinal to 1 forever.
    const surface = plantSurface('surface', '<p>Hermes Hermes</p>')
    performScopedFind(surface, 'hermes', { forward: true, findNext: false })

    const before = registeredHits()
    const result = performScopedFind(surface, 'hermes', { forward: true, findNext: true })

    const after = registeredHits()
    expect(after.length).toBe(before.length)
    expect(after.every(range => before.includes(range))).toBe(true)
    expect(result.count).toBe(2)
    expect(result.activeOrdinal).toBe(2)
  })

  it('re-collects when content the query matches was added after the last scan', async () => {
    // Regression (#81778 review): content appended after the scan (streamed
    // tokens, external writes) carries no match, so stepping a stale set would
    // walk past the live occurrence. A dirty scope must re-collect first.
    const surface = plantSurface('surface', '<p>needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toHaveLength(1)

    surface.insertAdjacentHTML('beforeend', '<p>needle</p>')
    // A real Enter arrives in a later task than the render's DOM write, so the
    // observer has delivered by then — a same-tick step is not a user path.
    await new Promise(resolve => setTimeout(resolve, 0))

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: true })

    expect(result.count).toBe(2)
    expect(findHitTexts()).toEqual(['needle', 'needle'])
  })

  it('re-collects when the query changes (matches are not reused across queries)', () => {
    const surface = plantSurface('surface', '<p>alpha beta alpha</p>')
    performScopedFind(surface, 'alpha', { forward: true, findNext: false })
    expect(findHitTexts()).toEqual(['alpha', 'alpha'])

    performScopedFind(surface, 'beta', { forward: true, findNext: false })

    // The alpha hits are gone; the beta hit is painted.
    expect(findHitTexts()).toEqual(['beta'])
    expect(surface.querySelector('p')?.textContent).toBe('alpha beta alpha')
  })

  it('ignores text inside the FindBar search input itself (no self-match)', () => {
    // The walker rejects any subtree that carries role="search" so a user
    // typing "needle" never matches the placeholder text in the input.
    const surface = plantSurface('surface', '')

    surface.innerHTML = `
      <div role="search"><input placeholder="needle placeholder" /></div>
      <p>needle haystack</p>
    `

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(1)
    expect(findHitTexts()).toEqual(['needle'])
  })

  it('ignores the composer and form fields, so the count matches what is on screen (#134070)', () => {
    // The composer is a live `contentEditable`: its draft is not part of the
    // transcript the user is reading. Counting it reported matches with nothing
    // behind them in the viewport, and prev/next could not show them.
    const surface = plantSurface('surface', '')

    surface.innerHTML = `
      <div class="markdown"><p>needle in the transcript</p></div>
      <div id="composer" contenteditable="true">draft with needle in it</div>
      <textarea>needle in a textarea</textarea>
      <div role="textbox">needle in a textbox role</div>
      <div contenteditable="false">needle in a non-editable shell</div>
    `

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    // Only the transcript occurrence: the composer, the textarea, the textbox
    // role. A `contenteditable="false"` wrapper is NOT an editor and is still
    // searched.
    expect(result.count).toBe(2)
    expect(findHitTexts()).toEqual(['needle', 'needle'])
    expect(findHitTexts()).not.toContain('needle in it')
  })

  it('leaves the composer caret and writing direction untouched (#134070)', () => {
    // The base engine wrapped matches by splitting the text node it found them
    // in — inside a live contentEditable that detaches the node the caret
    // points into, which is how the reported direction flip and caret jump
    // happened. The engine is a pure read now, and the editor is out of scope
    // on top of that; this pins both halves.
    const surface = plantSurface('surface', '<p>needle in the transcript</p>')
    const composer = document.createElement('div')

    composer.setAttribute('contenteditable', 'true')
    composer.textContent = 'draft needling along'
    surface.appendChild(composer)
    document.body.appendChild(surface)

    const draftNode = composer.firstChild as Text
    const selection = window.getSelection()!
    const caret = document.createRange()

    caret.setStart(draftNode, 6)
    caret.collapse(true)
    selection.removeAllRanges()
    selection.addRange(caret)

    performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(selection.anchorNode).toBe(draftNode)
    expect(selection.anchorOffset).toBe(6)
    expect(composer.textContent).toBe('draft needling along')
    expect(findHitTexts()).toEqual(['needle'])
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
    expect(findHitTexts()).toEqual(['needle'])
  })

  it('finds every occurrence in one text node and in sibling subtrees', () => {
    // Regression (#81778 review): a text node entirely consumed by a match used
    // to null the walker's cursor, terminating the sibling traversal —
    // `<div>needle<span>needle</span></div>` matched only the first occurrence.
    const surface = plantSurface('surface', '<div>needle<span>needle</span></div><p>needle</p>')

    const result = performScopedFind(surface, 'needle', { forward: true, findNext: false })

    expect(result.count).toBe(3)
    expect(findHitTexts()).toEqual(['needle', 'needle', 'needle'])
  })
})

describe('scope lifecycle', () => {
  it('captureFindScope marks the foreground surface for the walker', () => {
    const surface = plantSurface('surface', 'needle')
    captureFindScope()

    expect(currentFindScope()).toBe(surface)
    expect(surface.hasAttribute('data-find-root')).toBe(true)
  })

  it('releaseFindScope drops every highlight and clears the scope marker', () => {
    const surface = plantSurface('surface', '<p>needle in a haystack</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toHaveLength(1)

    releaseFindScope()

    expect(findHitTexts()).toEqual([])
    expect(activeFindHitText()).toBeNull()
    expect(surface.querySelector('[data-find-root]')).toBeNull()
    expect(currentFindScope()).toBeNull()
    // Original text untouched.
    expect(surface.querySelector('p')?.textContent).toBe('needle in a haystack')
  })

  it('keeps the scope targeting the FOREGROUND surface even when a hidden one matches the selector first', () => {
    // Regression (#81726): the visible chat is the SECOND surface in
    // document order, but it must still win the scope resolution.
    plantSurface('background', '<p>background needle</p>', true)
    const foreground = plantSurface('foreground', '<p>foreground needle</p>')

    captureFindScope()
    expect(currentFindScope()).toBe(foreground)

    // The walker only matches the foreground's occurrence.
    const result = performScopedFind(foreground, 'needle', { forward: true, findNext: false })
    expect(result.count).toBe(1)
    expect(findHitTexts()).toEqual(['needle'])
  })

  it('re-targets the scope to the foreground surface after a keep-alive flip hides the captured one', () => {
    // Regression (#81726): the bar stays open across a tab flip (the route
    // doesn't change, so the FindBar's pathname cleanup never runs). The
    // captured surface's pane layer goes `data-pane-hidden` and the flipped-
    // to surface is left unmarked — currentFindScope must re-resolve to the
    // new foreground surface instead of going dead (0/0 forever).
    const a = plantSurface('a', '<p>needle in A</p>')
    const b = plantSurface('b', '<p>needle in B</p>')

    captureFindScope()
    expect(currentFindScope()).toBe(a)
    expect(a.hasAttribute('data-find-root')).toBe(true)

    // Keep-alive tab flip: A's pane layer hides, B becomes the foreground.
    a.setAttribute('data-pane-hidden', '')

    const scope = currentFindScope()

    // The scope re-targeted to B and the marker moved with it.
    expect(scope).toBe(b)
    expect(b.hasAttribute('data-find-root')).toBe(true)
    expect(a.hasAttribute('data-find-root')).toBe(false)
    // The stale surface's matches are dropped with it — they belong to a
    // subtree the user can no longer see.
    expect(findHitTexts()).toEqual([])

    // The walker searches the new surface.
    const result = performScopedFind(scope!, 'needle', { forward: true, findNext: false })
    expect(result.count).toBe(1)
    expect(findHitTexts()).toEqual(['needle'])
  })

  it('re-targeting to a non-chat pane resolves to no scope (nothing on screen is a view)', () => {
    // The user flipped to a pane that isn't a chat surface — nothing to
    // search, matching the existing "no chat surface" contract. A later flip
    // back to a visible surface re-targets again.
    const a = plantSurface('a', 'needle')
    captureFindScope()

    a.setAttribute('data-pane-hidden', '')

    expect(currentFindScope()).toBeNull()
  })
})

describe('scoped find survives React re-render', () => {
  it('re-collects content React re-allocated, keeping the active position', async () => {
    const surface = plantSurface('surface', '<p>needle hay needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toEqual(['needle', 'needle'])

    // Step onto match #2, then mock a React re-render that rebuilds the
    // paragraph (detaching our ranges) while the matching text remains.
    performScopedFind(surface, 'needle', { forward: true, findNext: true })
    const steppedOnto = activeFindHitText()

    surface.querySelector('p')!.innerHTML = 'needle hay needle'
    await flushRescan()

    // Matches restored, and the active match is still the second one.
    expect(findHitTexts()).toEqual(['needle', 'needle'])
    expect(activeFindHitText()).toBe(steppedOnto)
    expect(activeFindHitText()).toBe(findHitTexts()[1])
    expect(surface.querySelector('p')!.textContent).toBe('needle hay needle')
  })

  it('keeps existing highlights and paints a newly appended message', async () => {
    // The task's target scenario: the bar is open with matches painted, then
    // the assistant appends a NEW message while the list stays mounted. The
    // existing match must persist and the appended content must be searched.
    const surface = plantSurface('surface', '<p>needle in the first message</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toEqual(['needle'])

    // React appends a new message block.
    const next = document.createElement('div')

    next.innerHTML = 'second message needle'
    surface.appendChild(next)
    await flushRescan()

    expect(findHitTexts()).toHaveLength(2)
    expect(surface.querySelector('p')!.textContent).toBe('needle in the first message')
    expect(next.textContent).toBe('second message needle')
    expect(activeFindHitText()).toBe('needle')
  })

  it('a re-render with no new match leaves the highlight set and position alone', async () => {
    const surface = plantSurface('surface', '<p>needle needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    performScopedFind(surface, 'needle', { forward: true, findNext: true })

    surface.insertAdjacentHTML('beforeend', '<p>irrelevant</p>')
    await flushRescan()

    expect(findHitTexts()).toEqual(['needle', 'needle'])
    // Still on the second match — a background render must not move the reader.
    expect(activeFindHitText()).toBe(findHitTexts()[1])
  })

  it('clearing the query tears the watcher down so a later write cannot resurrect hits', async () => {
    const surface = plantSurface('surface', '<p>needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toEqual(['needle'])

    // Clearing the query tears down the watcher; a later React write that would
    // normally re-scan must NOT resurrect highlights (the bar closed).
    performScopedFind(surface, '', { forward: true, findNext: false })
    surface.querySelector('p')!.innerHTML = 'needle again'
    await flushRescan()

    expect(findHitTexts()).toEqual([])
    expect(activeFindHitText()).toBeNull()
    expect(surface.querySelector('p')!.textContent).toBe('needle again')
  })

  it('releaseFindScope detaches the watcher so no write can re-highlight', async () => {
    const surface = plantSurface('surface', '<p>needle</p>')
    captureFindScope()
    performScopedFind(surface, 'needle', { forward: true, findNext: false })
    expect(findHitTexts()).toEqual(['needle'])

    releaseFindScope()
    surface.querySelector('p')!.innerHTML = 'needle needle'
    await flushRescan()

    expect(findHitTexts()).toEqual([])
  })
})
