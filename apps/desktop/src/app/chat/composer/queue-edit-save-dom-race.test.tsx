import { afterEach, describe, expect, it, vi } from 'vitest'

import { saveQueuedEditFromEditor } from './queue-edit-save'

afterEach(() => {
  document.body.replaceChildren()
})

describe('queued-edit Save live DOM vs stale draft state (#57945)', () => {
  it('flushes the visible editor DOM before saving the queued prompt', () => {
    const editor = document.createElement('div')
    editor.contentEditable = 'true'
    editor.textContent = 'visible queued edit'
    document.body.appendChild(editor)

    let draftRef = 'stale queued edit'
    const calls: string[] = []
    const savedTexts: string[] = []

    const flushEditorToDraft = vi.fn((node: HTMLDivElement) => {
      calls.push('flush')
      draftRef = node.textContent ?? ''
    })

    const exitQueuedEdit = vi.fn((action: 'save') => {
      calls.push(`exit:${action}`)
      savedTexts.push(draftRef)

      return true
    })

    const saved = saveQueuedEditFromEditor({ editor, exitQueuedEdit, flushEditorToDraft })

    expect(saved).toBe(true)
    expect(flushEditorToDraft).toHaveBeenCalledWith(editor)
    expect(exitQueuedEdit).toHaveBeenCalledWith('save')
    expect(calls).toEqual(['flush', 'exit:save'])
    expect(savedTexts).toEqual(['visible queued edit'])
  })

  it('keeps the existing save fallback when the editor ref is unavailable', () => {
    let draftRef = 'existing queued edit'
    const savedTexts: string[] = []

    const flushEditorToDraft = vi.fn((node: HTMLDivElement) => {
      draftRef = node.textContent ?? ''
    })

    const exitQueuedEdit = vi.fn((action: 'save') => {
      savedTexts.push(draftRef)

      return action === 'save'
    })

    const saved = saveQueuedEditFromEditor({ editor: null, exitQueuedEdit, flushEditorToDraft })

    expect(saved).toBe(true)
    expect(flushEditorToDraft).not.toHaveBeenCalled()
    expect(exitQueuedEdit).toHaveBeenCalledWith('save')
    expect(savedTexts).toEqual(['existing queued edit'])
  })

  it('uses an empty visible editor to preserve the existing unsaved-empty behavior', () => {
    const editor = document.createElement('div')
    editor.contentEditable = 'true'
    editor.textContent = ''
    document.body.appendChild(editor)

    let draftRef = 'stale non-empty queued edit'

    const flushEditorToDraft = vi.fn((node: HTMLDivElement) => {
      draftRef = node.textContent ?? ''
    })

    const exitQueuedEdit = vi.fn(() => draftRef.trim().length > 0)

    const saved = saveQueuedEditFromEditor({ editor, exitQueuedEdit, flushEditorToDraft })

    expect(saved).toBe(false)
    expect(flushEditorToDraft).toHaveBeenCalledWith(editor)
    expect(exitQueuedEdit).toHaveBeenCalledWith('save')
  })
})
