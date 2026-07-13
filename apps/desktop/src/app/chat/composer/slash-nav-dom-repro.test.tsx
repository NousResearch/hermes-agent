import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { act, fireEvent, render } from '@testing-library/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { useLiveCompletionAdapter } from './hooks/use-live-completion-adapter'
import { detectTrigger, type TriggerState } from './text-utils'

// Faithful mirror of index.tsx's trigger wiring, driven through REAL DOM
// keydown+keyup events on a contentEditable. Exercises the parts a direct
// reducer-call repro misses: the keyup -> refreshTrigger path, the
// keydown-set "consumed" ref that guards it, and per-press keydown+keyup
// ordering (critical for Escape, whose keydown nulls `trigger` before keyup).
function Harness({
  onState
}: {
  onState: (s: { active: number; items: readonly Unstable_TriggerItem[]; open: boolean }) => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const triggerKeyConsumedRef = useRef(false)
  const [trigger, setTrigger] = useState<TriggerState | null>(null)
  const [triggerActive, setTriggerActive] = useState(0)
  const [triggerItems, setTriggerItems] = useState<readonly Unstable_TriggerItem[]>([])

  const { adapter } = useLiveCompletionAdapter({
    enabled: true,
    debounceMs: 0,
    fetcher: async (query: string) => ({
      query,
      items: Array.from({ length: 5 }, (_, i) => ({ text: `/cmd${i}`, display: `/cmd${i}`, meta: '' }))
    }),
    toItem: (entry, index) => ({ id: `${entry.text}|${index}`, type: 'slash', label: entry.text.slice(1) })
  })

  const triggerAdapter: Unstable_TriggerAdapter | null = trigger?.kind === '/' ? adapter : null

  const refreshTrigger = useCallback(() => {
    const editor = editorRef.current

    if (!editor) {
      return
    }

    const raw = editor.textContent ?? ''

    if (!raw.includes('@') && !raw.includes('/')) {
      if (trigger) {
        setTrigger(null)
        setTriggerActive(0)
      }

      return
    }

    const detected = detectTrigger(raw)
    setTrigger(detected)

    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      setTriggerActive(0)
    }
  }, [trigger])

  useEffect(() => {
    if (!trigger || !triggerAdapter?.search) {
      setTriggerItems([])

      return
    }

    setTriggerItems(triggerAdapter.search(trigger.query))
  }, [trigger, triggerAdapter])

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  onState({ active: triggerActive, items: triggerItems, open: trigger !== null })

  const closeTrigger = () => {
    setTrigger(null)
    setTriggerItems([])
    setTriggerActive(0)
  }

  // Exact copies of index.tsx handlers, including the keydown-set "consumed"
  // ref that the keyup consults.
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (trigger && triggerItems.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx + 1) % triggerItems.length)

        return
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx - 1 + triggerItems.length) % triggerItems.length)

        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        closeTrigger()

        return
      }
    }
  }

  const handleKeyUp = () => {
    if (triggerKeyConsumedRef.current) {
      triggerKeyConsumedRef.current = false

      return
    }

    // index.tsx defers via setTimeout(refreshTrigger, 0); call synchronously
    // here so the test deterministically observes the keyup-driven refresh.
    refreshTrigger()
  }

  return (
    <div
      contentEditable
      data-testid="editor"
      onInput={() => refreshTrigger()}
      onKeyDown={handleKeyDown}
      onKeyUp={handleKeyUp}
      ref={editorRef}
      suppressContentEditableWarning
    />
  )
}

async function flush() {
  await act(async () => {
    await new Promise(r => setTimeout(r, 20))
  })
}

describe('slash menu navigation — real DOM keydown+keyup', () => {
  it('cycles through ALL items and Esc closes (and stays closed)', async () => {
    vi.useRealTimers()
    let latest = { active: 0, items: [] as readonly Unstable_TriggerItem[], open: false }
    const { getByTestId } = render(<Harness onState={s => (latest = s)} />)
    const editor = getByTestId('editor')

    // Simulate typing '/'.
    await act(async () => {
      editor.textContent = '/'
      fireEvent.input(editor)
    })
    await flush()

    expect(latest.open).toBe(true)
    expect(latest.items.length).toBe(5)

    // ArrowDown 6x with REAL keydown+keyup pairs. Bug = stuck [0,1,0,1,...].
    const seen: number[] = [latest.active]

    for (let i = 0; i < 6; i++) {
      await act(async () => {
        fireEvent.keyDown(editor, { key: 'ArrowDown' })
        fireEvent.keyUp(editor, { key: 'ArrowDown' })
        await Promise.resolve()
      })
      seen.push(latest.active)
    }

    expect(seen).toEqual([0, 1, 2, 3, 4, 0, 1])

    // Escape: keydown closes; keyup must NOT reopen (the '/' is still in text).
    await act(async () => {
      fireEvent.keyDown(editor, { key: 'Escape' })
      fireEvent.keyUp(editor, { key: 'Escape' })
      await Promise.resolve()
    })
    await flush()
    expect(latest.open).toBe(false)
  })
})
