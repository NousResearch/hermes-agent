import { describe, expect, it } from 'vitest'

import { HERMES_PATHS_MIME } from '../hooks/use-composer-actions'

import { createDragLifecycleGate, dragHasAttachments, isSparseExternalFileDrag } from './inline-refs'

/**
 * Minimal stand-in for `DataTransfer`. jsdom ships a `DataTransfer` shim
 * whose `types`/`items`/`files`/`setData`/`getData` behave well enough for
 * the lifecycle tests below, but it cannot model a Windows Explorer drag
 * whose `types` AND `items` are *both* empty during `dragenter` (the
 * constructors always set `types: ['Files']`). We use this local shim so
 * the regression test is hermetic and never depends on jsdom internals.
 */
function fakeTransfer(types: string[] = [], items: { kind: string }[] = []): DataTransfer {
  return {
    types: Object.freeze([...types]),
    items: Object.freeze([...items].map(item => Object.freeze(item))) as unknown as DataTransferItemList,
    files: Object.freeze([]) as unknown as FileList
  } as unknown as DataTransfer
}

describe('dragHasAttachments', () => {
  it('accepts the custom HERMES_PATHS_MIME drag', () => {
    const t = fakeTransfer([HERMES_PATHS_MIME])
    expect(dragHasAttachments(t, HERMES_PATHS_MIME)).toBe(true)
  })

  it('accepts a typed "Files" drag (Chromium, macOS)', () => {
    const t = fakeTransfer(['Files'])
    expect(dragHasAttachments(t, HERMES_PATHS_MIME)).toBe(true)
  })

  it('accepts a drag whose items include a file kind', () => {
    const t = fakeTransfer([], [{ kind: 'file' }])
    expect(dragHasAttachments(t, HERMES_PATHS_MIME)).toBe(true)
  })

  it('rejects a fully-empty transfer (Windows sparse dragenter is not yet a file)', () => {
    // REGRESSION PROOF for the pre-fix behaviour: a sparse transfer must NOT
    // be reported as "has attachments" by this helper, because the file list
    // is only materialised at `drop`. The stateful gate is what handles the
    // dragenter/dragover path, and the drop handler still has to verify
    // `extractDroppedFiles(...).length > 0`.
    const t = fakeTransfer()
    expect(dragHasAttachments(t, HERMES_PATHS_MIME)).toBe(false)
  })

  it('rejects a text drag', () => {
    const t = fakeTransfer(['text/plain'])
    expect(dragHasAttachments(t, HERMES_PATHS_MIME)).toBe(false)
  })

  it('rejects null', () => {
    expect(dragHasAttachments(null, HERMES_PATHS_MIME)).toBe(false)
  })
})

describe('isSparseExternalFileDrag', () => {
  it('flags a fully-empty transfer (Windows Explorer / Chromium sparse case)', () => {
    const t = fakeTransfer()
    expect(isSparseExternalFileDrag(t)).toBe(true)
  })

  it('does NOT flag a transfer with even a single type (text/plain, Files, etc.)', () => {
    // Any non-empty `types` array is an explicit signal from the source —
    // either a real file drag (`Files`), an in-app drag (`HERMES_PATHS_MIME`),
    // or a text drag. Those are NOT the Windows sparse case and are routed
    // through the existing `dragHasAttachments` path.
    expect(isSparseExternalFileDrag(fakeTransfer(['text/plain']))).toBe(false)
    expect(isSparseExternalFileDrag(fakeTransfer(['Files']))).toBe(false)
  })

  it('does NOT flag a transfer with non-empty items', () => {
    expect(isSparseExternalFileDrag(fakeTransfer([], [{ kind: 'string' }]))).toBe(false)
    expect(isSparseExternalFileDrag(fakeTransfer([], [{ kind: 'file' }]))).toBe(false)
  })

  it('rejects null', () => {
    expect(isSparseExternalFileDrag(null)).toBe(false)
  })
})

describe('createDragLifecycleGate (#97702)', () => {
  it('arms on a typed file dragenter and keeps dragover accepting', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const t = fakeTransfer(['Files'])

    expect(gate.onEnter(t)).toBe(true)
    // Dragover fires many times before drop; each must stay true so the
    // app-lifetime HTML5Backend capture-time `dropEffect='none'` does not
    // suppress the eventual drop.
    expect(gate.onOver(t)).toBe(true)
    expect(gate.onOver(t)).toBe(true)
  })

  it('arms on a fully-empty dragenter (Windows sparse case) and keeps dragover open', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const empty = fakeTransfer()

    // First dragenter on the composer: empty transfer, but the only known
    // production signal of an external OS file drag. The gate must accept
    // (and the bubble-phase handler must preventDefault) so drop survives.
    expect(gate.onEnter(empty)).toBe(true)
    // Subsequent dragovers in the same drag are still empty — the gate
    // must stay armed.
    expect(gate.onOver(empty)).toBe(true)
    expect(gate.onOver(empty)).toBe(true)
  })

  it('REJECTS a sparse dragover that has no prior dragenter from this drag', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const empty = fakeTransfer()

    // No enter yet — a stray dragover with no preceding enter for this drag
    // must NOT arm the gate. This is the "don't blanket-classify every
    // empty DataTransfer" invariant: an in-app drag that happens to expose
    // an empty transfer cannot flash the file-drop overlay.
    expect(gate.onOver(empty)).toBe(false)
    expect(gate.onOver(empty)).toBe(false)
  })

  it('REJECTS a dragenter whose transfer carries a non-file, non-MIME type', () => {
    // A text drag (types=['text/plain']) is not a file drag and must not
    // arm the gate, even though it is not "fully empty".
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const text = fakeTransfer(['text/plain'])

    expect(gate.onEnter(text)).toBe(false)
    // And without an armed enter, dragover must also reject — no sticky
    // "file drag" overlay.
    expect(gate.onOver(text)).toBe(false)
  })

  it('disarms on onLeave(true) at root depth, so a new drag starts clean', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const empty = fakeTransfer()

    expect(gate.onEnter(empty)).toBe(true)
    expect(gate.onOver(empty)).toBe(true)
    gate.onLeave(true)
    // After leaving the root, a stray dragover must NOT be accepted —
    // and a brand-new drag must be re-evaluable from scratch.
    expect(gate.onOver(empty)).toBe(false)
    expect(gate.onEnter(empty)).toBe(true)
  })

  it('does NOT disarm on onLeave(false) (nested child leave preserves the drag)', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const empty = fakeTransfer()

    expect(gate.onEnter(empty)).toBe(true)
    gate.onLeave(false)
    // Nested-child dragleave must not close the gate — depth==0 is the
    // caller's job to track.
    expect(gate.onOver(empty)).toBe(true)
  })

  it('reset() always disarms (drop, unmount, abort)', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    const empty = fakeTransfer()

    expect(gate.onEnter(empty)).toBe(true)
    gate.reset()
    expect(gate.onOver(empty)).toBe(false)
  })

  it('rejects a null transfer on both enter and over', () => {
    const gate = createDragLifecycleGate(HERMES_PATHS_MIME)
    expect(gate.onEnter(null)).toBe(false)
    expect(gate.onOver(null)).toBe(false)
  })

  it('does not carry state between independent gate instances', () => {
    // Each drop zone must own its gate — an arm in one zone must not
    // leak into another. This is the React-ref-isolation invariant.
    const a = createDragLifecycleGate(HERMES_PATHS_MIME)
    const b = createDragLifecycleGate(HERMES_PATHS_MIME)
    const empty = fakeTransfer()

    a.onEnter(empty)
    expect(b.onOver(empty)).toBe(false)
  })
})
