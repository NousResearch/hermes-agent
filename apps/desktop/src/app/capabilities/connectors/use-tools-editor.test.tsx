import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { toolFixtures } from './fixtures'
import type { SaveResult } from './use-tools-editor'
import { useToolsEditor } from './use-tools-editor'

const tools = toolFixtures()

type Save = (disabled: string[], options: { overwrite: boolean }) => Promise<SaveResult>

function editor(onSave: Save, savedDisabled: string[] = []) {
  return renderHook(props => useToolsEditor(props), {
    initialProps: { onSave, savedDisabled, tools }
  })
}

describe('the editor machine', () => {
  it('moves the baseline only when the save lands', async () => {
    const onSave = vi.fn(async () => 'saved' as const)
    const { result } = editor(onSave)

    act(() => result.current.toggle('LINEAR_CREATE_ISSUE'))

    expect(result.current.dirty).toBe(true)

    await act(async () => {
      await result.current.save()
    })

    expect(onSave).toHaveBeenCalledWith(['LINEAR_CREATE_ISSUE'], { overwrite: false })
    expect(result.current.dirty).toBe(false)
    expect(result.current.phase).toBe('ready')
  })

  it('leaves the work on screen when the save fails', async () => {
    const { result } = editor(async () => 'failed')

    act(() => result.current.toggle('LINEAR_CREATE_ISSUE'))
    await act(async () => {
      await result.current.save()
    })

    expect(result.current.dirty).toBe(true)
    expect(result.current.local).toEqual(['LINEAR_CREATE_ISSUE'])
    expect(result.current.phase).toBe('ready')
  })

  it('stops on a conflict and keeps the edit until the reader chooses', async () => {
    const onSave = vi.fn<Save>(async () => 'conflict')
    const { result } = editor(onSave)

    act(() => result.current.toggle('LINEAR_CREATE_ISSUE'))
    await act(async () => {
      await result.current.save()
    })

    expect(result.current.phase).toBe('conflict')
    expect(result.current.local).toEqual(['LINEAR_CREATE_ISSUE'])

    // "Save over their version" is a save. It carries the overwrite in the call,
    // because a compare-and-set write cannot tell one from a second loser.
    onSave.mockResolvedValue('saved')

    await act(async () => {
      await result.current.keepMine()
    })

    expect(onSave).toHaveBeenLastCalledWith(['LINEAR_CREATE_ISSUE'], { overwrite: true })
    expect(result.current.phase).toBe('ready')
    expect(result.current.dirty).toBe(false)
  })

  it('carries an organisation-locked slug across a quick action instead of dropping it', () => {
    // The org already turned it off, so no quick action may put it back — and
    // silently deleting the person's own rule would report it as "back on".
    const { result } = editor(async () => 'saved', ['LINEAR_DELETE_PROJECT'])

    act(() => result.current.applyQuickAction('no-destructive'))

    expect(result.current.local).toContain('LINEAR_DELETE_PROJECT')
    expect(result.current.counts.backOn).toBe(0)
  })

  it('reloads for another connector even when the two saved rules read the same', () => {
    const { rerender, result } = renderHook(props => useToolsEditor(props), {
      initialProps: { editorKey: 'linear', onSave: async () => 'saved' as const, savedDisabled: [] as string[], tools }
    })

    act(() => result.current.toggle('LINEAR_CREATE_ISSUE'))

    // Equal by value, new by identity: the store feeding these components will
    // rebuild both arrays on every publish, and that must not wipe an edit.
    rerender({ editorKey: 'linear', onSave: async () => 'saved' as const, savedDisabled: [], tools: [...tools] })

    expect(result.current.local).toEqual(['LINEAR_CREATE_ISSUE'])

    rerender({ editorKey: 'github', onSave: async () => 'saved' as const, savedDisabled: [], tools: [...tools] })

    expect(result.current.local).toEqual([])
    expect(result.current.dirty).toBe(false)
  })

  it('refuses to write a rule the organisation already owns', () => {
    const { result } = editor(async () => 'saved')

    act(() => result.current.toggle('LINEAR_DELETE_PROJECT'))

    expect(result.current.local).toEqual([])
    expect(result.current.isOn('LINEAR_DELETE_PROJECT')).toBe(false)
  })

  it('lets a failed fetch outrank whatever the editor is doing', () => {
    const { result } = renderHook(props => useToolsEditor(props), {
      initialProps: { onSave: async () => 'saved' as const, savedDisabled: [], status: 'unavailable' as const, tools }
    })

    expect(result.current.phase).toBe('unavailable')
  })

  it('reloads itself when the saved rule changes underneath it', () => {
    const { rerender, result } = editor(async () => 'saved')

    act(() => result.current.toggle('LINEAR_CREATE_ISSUE'))
    rerender({ onSave: async () => 'saved' as const, savedDisabled: ['LINEAR_ARCHIVE_ISSUE'], tools })

    expect(result.current.local).toEqual(['LINEAR_ARCHIVE_ISSUE'])
    expect(result.current.dirty).toBe(false)
  })
})
