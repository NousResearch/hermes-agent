import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { createDragDropManager } from 'dnd-core'
import { HTML5Backend } from 'react-dnd-html5-backend'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { startRosterDrag } from './roster-drag'
import { $draggingBot, BOT_DRAG_MIME } from './user-sections'
import { SectionDropZone, useEscapeCancelsBotDrag } from './user-sections-ui'

function transfer(): DataTransfer {
  const values = new Map<string, string>()
  const types: string[] = []

  return {
    types,
    effectAllowed: 'none',
    setData(type: string, value: string) {
      values.set(type, value)
      types.push(type)
    },
    getData(type: string) {
      return values.get(type) || ''
    }
  } as unknown as DataTransfer
}

function Fixture() {
  useEscapeCancelsBotDrag()

  return (
    <button draggable onDragStart={event => startRosterDrag(event, 'local:tech-admin')}>
      Drag
    </button>
  )
}

afterEach(() => {
  cleanup()
  $draggingBot.set(null)
})

describe('roster native drag with the file-tree backend', () => {
  it('reproduces cancellation of an unregistered source and protects roster drag', () => {
    const backend = createDragDropManager(HTML5Backend).getBackend()
    backend.setup()

    try {
      const plain = window.document.createElement('button')
      window.document.body.appendChild(plain)
      const event = new Event('dragstart', { bubbles: true, cancelable: true })
      Object.defineProperty(event, 'dataTransfer', { value: transfer() })
      expect(plain.dispatchEvent(event)).toBe(false)
      plain.remove()
      const view = render(<Fixture />)
      const dataTransfer = transfer()
      expect(fireEvent.dragStart(view.getByText('Drag'), { dataTransfer })).toBe(true)
      expect(dataTransfer.getData(BOT_DRAG_MIME)).toBe('local:tech-admin')
      expect($draggingBot.get()).toBe('local:tech-admin')
      fireEvent(window, new Event('dragend'))
      expect($draggingBot.get()).toBe(null)
    } finally {
      backend.teardown()
    }
  })

  it('delivers a section move and clears drag state', () => {
    const move = vi.fn()

    const view = render(
      <>
        <Fixture />
        <SectionDropZone isSource={false} onDropBot={move}>
          Target
        </SectionDropZone>
      </>
    )

    const dataTransfer = transfer()
    fireEvent.dragStart(view.getByText('Drag'), { dataTransfer })
    const backend = createDragDropManager(HTML5Backend).getBackend()
    backend.setup()

    try {
      fireEvent.dragOver(view.getByText('Target'), { dataTransfer })
      expect(dataTransfer.dropEffect).toBe('move')
      fireEvent.drop(view.getByText('Target'), { dataTransfer })
    } finally {
      backend.teardown()
    }

    expect(move).toHaveBeenCalledWith('local:tech-admin')
    expect($draggingBot.get()).toBe(null)
  })

  it.each(['blur', 'drop'])('clears cancelled state on %s', type => {
    render(<Fixture />)
    act(() => $draggingBot.set('local:tech-admin'))
    fireEvent(window, new Event(type))
    expect($draggingBot.get()).toBe(null)
  })
})
