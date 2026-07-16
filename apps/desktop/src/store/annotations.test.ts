import { beforeEach, describe, expect, it } from 'vitest'

import {
  $annotationContext,
  $annotationDiscardIntent,
  $annotationDraft,
  $annotationEditorCollapsed,
  $annotations,
  activateAnnotationContext,
  beginAnnotation,
  clearAnnotations,
  confirmDiscardAnnotationDraft,
  createReviewContext,
  editAnnotation,
  removeAnnotation,
  saveAnnotationDraft,
  updateAnnotationDraft
} from './annotations'

describe('annotation store', () => {
  beforeEach(() => {
    window.localStorage.clear()
    activateAnnotationContext(createReviewContext({ cwd: '/work/a', kind: 'document' }))
    clearAnnotations()
  })

  it('keeps complete edit data when replacing a dirty draft', () => {
    beginAnnotation({ kind: 'source', lineEnd: 2, lineStart: 2, path: 'a.ts' }, null, $annotationContext.get())
    updateAnnotationDraft({ comment: 'saved', labels: ['bug'], type: 'concern' })
    const saved = saveAnnotationDraft()

    beginAnnotation({ kind: 'source', lineEnd: 4, lineStart: 4, path: 'b.ts' }, null, $annotationContext.get())
    updateAnnotationDraft({ comment: 'unsaved' })
    editAnnotation(saved!.id)

    expect($annotationDiscardIntent.get()).toBe('replace')
    confirmDiscardAnnotationDraft()
    expect($annotationDraft.get()).toMatchObject({
      comment: 'saved',
      editingId: saved!.id,
      labels: ['bug'],
      type: 'concern'
    })
  })

  it('closes an edit draft when its annotation is removed', () => {
    beginAnnotation({ kind: 'file', path: 'a.ts' }, null, $annotationContext.get())
    updateAnnotationDraft({ comment: 'remove me' })
    const saved = saveAnnotationDraft()!
    editAnnotation(saved.id)

    removeAnnotation(saved.id)

    expect($annotations.get()).toEqual([])
    expect($annotationDraft.get()).toBeNull()
  })

  it('isolates annotations by exact review context', () => {
    const first = createReviewContext({ cwd: '/same', kind: 'git', reviewScope: 'uncommitted' })
    const second = createReviewContext({ cwd: '/same', headSha: 'next', kind: 'git', reviewScope: 'uncommitted' })

    activateAnnotationContext(first)
    beginAnnotation({ kind: 'file', path: 'a.ts' }, null, first)
    updateAnnotationDraft({ comment: 'first context' })
    saveAnnotationDraft()

    activateAnnotationContext(second)
    expect($annotations.get()).toEqual([])

    activateAnnotationContext(first)
    expect($annotations.get()).toHaveLength(1)
  })

  it('preserves a per-document draft minimized when switching contexts', () => {
    const first = createReviewContext({ artifactPath: '/work/a.md', contentHash: 'a1', kind: 'document' })
    const second = createReviewContext({ artifactPath: '/work/b.md', contentHash: 'b1', kind: 'document' })

    activateAnnotationContext(first)
    beginAnnotation({ contentHash: 'a1', kind: 'file', path: '/work/a.md' }, null, first)
    updateAnnotationDraft({ comment: 'unfinished' })
    activateAnnotationContext(second)

    expect($annotationDraft.get()).toBeNull()

    activateAnnotationContext(first)
    expect($annotationDraft.get()?.comment).toBe('unfinished')
    expect($annotationEditorCollapsed.get()).toBe(true)
  })

  it('carries annotations as stale only across a compatible revision', () => {
    const first = createReviewContext({ cwd: '/repo', headSha: 'one', kind: 'git', reviewScope: 'branch' })
    const second = createReviewContext({ cwd: '/repo', headSha: 'two', kind: 'git', reviewScope: 'branch' })

    activateAnnotationContext(first)
    beginAnnotation({ kind: 'file', path: 'a.ts' }, null, first)
    updateAnnotationDraft({ comment: 'check this' })
    saveAnnotationDraft()
    activateAnnotationContext(second, { carryStale: true })

    expect($annotations.get()).toMatchObject([{ status: 'stale' }])
  })

  it('persists editable normalized visual marks with the annotation', () => {
    const context = $annotationContext.get()
    beginAnnotation(
      {
        contentHash: 'image-v1',
        kind: 'visual',
        marks: [{ id: 'pin-1', point: { x: 0.25, y: 0.75 }, tool: 'pin' }],
        mediaKind: 'png',
        naturalHeight: 600,
        naturalWidth: 800,
        path: '/work/a/screenshot.png'
      },
      null,
      context
    )
    const draft = $annotationDraft.get()

    expect(draft?.anchor.kind).toBe('visual')

    if (!draft || draft.anchor.kind !== 'visual') {
      throw new Error('Expected a visual annotation draft')
    }

    updateAnnotationDraft({
      anchor: {
        ...draft.anchor,
        marks: [{ end: { x: 0.8, y: 0.9 }, id: 'box-1', start: { x: 0.1, y: 0.2 }, tool: 'rectangle' }]
      },
      comment: 'Move this control'
    })

    const saved = saveAnnotationDraft()

    expect(saved?.anchor).toMatchObject({
      kind: 'visual',
      marks: [{ id: 'box-1', tool: 'rectangle' }],
      naturalHeight: 600,
      naturalWidth: 800
    })
  })
})
