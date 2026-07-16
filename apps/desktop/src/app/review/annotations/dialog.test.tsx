import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import {
  $annotationDraft,
  $annotationEditorAnchor,
  $annotationEditorCollapsed,
  beginAnnotation,
  updateAnnotationDraft
} from '@/store/annotations'

import { AnnotationDialog } from './dialog'

describe('AnnotationDialog', () => {
  afterEach(() => {
    cleanup()
    $annotationDraft.set(null)
    $annotationEditorAnchor.set(null)
    $annotationEditorCollapsed.set(false)
  })

  it('opens an anchored non-modal editor without stealing focus', () => {
    const focusBefore = window.document.activeElement

    beginAnnotation(
      { excerpt: 'const selected = true', kind: 'source', lineEnd: 4, lineStart: 4, path: '/repo/a.ts' },
      { height: 20, width: 80, x: 120, y: 90 }
    )
    const rendered = render(<AnnotationDialog />)
    const editor = rendered.container.ownerDocument.querySelector('[data-annotation-editor]') as HTMLElement

    expect(editor).toBeTruthy()
    expect(editor.getAttribute('aria-modal')).toBe('false')
    expect(editor.style.left).toBe('120px')
    expect(editor.textContent).toContain('const selected = true')
    expect(window.document.activeElement).toBe(focusBefore)
  })

  it('stays open through outside interaction and minimizes only through an explicit action', async () => {
    beginAnnotation({ kind: 'file', path: '/repo/a.ts' })
    updateAnnotationDraft({ comment: 'Keep this draft' })
    render(<AnnotationDialog />)
    await waitFor(() => expect(window.document.querySelector('[data-annotation-editor]')).not.toBeNull())
    fireEvent.pointerDown(window.document.body)
    fireEvent.scroll(window.document.body)

    expect($annotationEditorCollapsed.get()).toBe(false)
    fireEvent.click(window.document.querySelector('[aria-label="Minimize annotation editor"]')!)

    expect($annotationEditorCollapsed.get()).toBe(true)
    expect(window.document.querySelector('[data-annotation-editor]')).toBeNull()
    expect($annotationDraft.get()?.comment).toBe('Keep this draft')
  })
})
