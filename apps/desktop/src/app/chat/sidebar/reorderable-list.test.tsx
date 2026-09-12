import { DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable'
import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useSortableBindings } from './reorderable-list'

afterEach(cleanup)

// Regression test for the swallowed-Space bug: PR #82373 spread the full dnd-kit
// handle ({...dragHandleProps} — attributes + keyboard + pointer activators)
// onto the whole row SHELL. That made the shell itself a focusable drag
// activator: Space/Enter pressed while the shell (or a control inside it) had
// focus started keyboard drags, and while a drag session was active dnd-kit's
// KeyboardSensor arms a window keydown listener whose default `end` codes
// include Space — swallowing the keystroke in ANY text input (session rename
// dialog, first composer keystroke after launch).
//
// The split contract: the grabber keeps the FULL handle (keyboard reorder
// accessibility), the shell gets ONLY the pointer activator.
function Probe({ id }: { id: string }) {
  const bindings = useSortableBindings(id)

  return (
    <div>
      <pre data-keys={Object.keys(bindings.dragHandleProps ?? {}).join(',')} data-testid="grabber-keys" />
      <pre data-keys={Object.keys(bindings.shellDragProps ?? {}).join(',')} data-testid="shell-keys" />
    </div>
  )
}

function renderProbe() {
  return render(
    <DndContext>
      <SortableContext items={['a']} strategy={verticalListSortingStrategy}>
        <Probe id="a" />
      </SortableContext>
    </DndContext>
  )
}

describe('useSortableBindings', () => {
  it('keeps the full handle — keyboard activator + focusable attributes — on the grabber', () => {
    const { getByTestId } = renderProbe()

    const grabberKeys = (getByTestId('grabber-keys').dataset.keys ?? '').split(',')
    expect(grabberKeys).toContain('onKeyDown')
    expect(grabberKeys).toContain('onPointerDown')
  })

  it('gives the row shell only the pointer activator — no keyboard activator, no role/tabIndex', () => {
    const { getByTestId } = renderProbe()

    // The shell must never see the keyboard activator: Space/Enter pressed
    // while a shell descendant has focus would otherwise reach dnd-kit's
    // KeyboardSensor and be consumed by its default start/end codes. The
    // attributes (role=button / tabIndex) never enter shellDragProps either,
    // so the shell is not a focusable drag activator.
    expect((getByTestId('shell-keys').dataset.keys ?? '').split(',')).toEqual(['onPointerDown'])
  })
})

// Behavioral probe with the real sensors the sidebar uses (PointerSensor with
// the 6px activation constraint + KeyboardSensor with the default codes).
function ProbeHost({ id }: { id: string }) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor)
  )

  return (
    <DndContext onDragEnd={() => {}} sensors={sensors}>
      <SortableContext items={[id]} strategy={verticalListSortingStrategy}>
        <BehaviorProbe id={id} />
      </SortableContext>
    </DndContext>
  )
}

function BehaviorProbe({ id }: { id: string }) {
  const bindings = useSortableBindings(id)

  return (
    <div data-testid="shell" {...bindings.shellDragProps}>
      <button data-testid="control" type="button">
        Control
      </button>
      <div data-testid="grabber" {...bindings.dragHandleProps}>
        Grab
      </div>
      <output data-testid="state">{bindings.dragging ? 'DRAGGING' : 'IDLE'}</output>
    </div>
  )
}

const spaceKey = { key: ' ', code: 'Space', bubbles: true, cancelable: true } as const
// Enter is also a KeyboardSensor default start/end code — same swallow class.
const enterKey = { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true } as const

describe('useSortableBindings keyboard behavior', () => {
  it('does not swallow Space on the shell nor start a keyboard drag from it', () => {
    const { getByTestId } = render(<ProbeHost id="a" />)

    // Pre-fix, the shell carried the KeyboardSensor activator: Space here
    // started a drag (and was preventDefaulted). Post-fix, the shell has only
    // the pointer activator, so the keydown must pass through untouched.
    expect(fireEvent.keyDown(getByTestId('shell'), spaceKey)).toBe(true)
    expect(getByTestId('state').textContent).toBe('IDLE')

    // Enter is in the same default start/end codes — must pass through too.
    expect(fireEvent.keyDown(getByTestId('shell'), enterKey)).toBe(true)
    expect(getByTestId('state').textContent).toBe('IDLE')
  })

  it('does not swallow Space on a control inside the shell (⋯ menu, row body)', () => {
    const { getByTestId } = render(<ProbeHost id="a" />)

    expect(fireEvent.keyDown(getByTestId('control'), spaceKey)).toBe(true)
    expect(getByTestId('state').textContent).toBe('IDLE')
  })

  it('still starts a keyboard drag from the grabber (screen-reader reorder preserved)', () => {
    const { getByTestId } = render(<ProbeHost id="a" />)

    // dnd-kit's activator preventDefaults the start key, so fireEvent returns
    // false here — the drag STARTING is the assertion.
    fireEvent.keyDown(getByTestId('grabber'), spaceKey)
    expect(getByTestId('state').textContent).toBe('DRAGGING')

    // Space again on the grabber ends the drag (default `end` codes), so the
    // test tears down cleanly.
    fireEvent.keyDown(getByTestId('grabber'), spaceKey)
  })
})
