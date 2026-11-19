import { describe, expect, it } from 'vitest'

import { PEN_SELECTION_ATTACHMENT_ID, penSelectionAttachment, penSelectionLabel } from './pen-selection'

describe('pen selection chip', () => {
  it('labels by name, falling back to type then id, and counts the rest', () => {
    expect(penSelectionLabel([{ id: 'a1', name: 'Hero card' }])).toBe('Hero card')
    expect(penSelectionLabel([{ id: 'a1', type: 'frame' }, { id: 'b2' }, { id: 'c3' }])).toBe('frame +2')
    expect(penSelectionLabel([{ id: 'a1', name: '  ' }])).toBe('a1')
  })

  it('hands the agent each node with its type and id', () => {
    const chip = penSelectionAttachment('Login Screen', [
      { id: 'dE6qx', name: 'Hero card', type: 'frame' },
      { id: 'k9', name: 'Sign in' }
    ])

    expect(chip).toMatchObject({ detail: 'Login Screen', id: PEN_SELECTION_ATTACHMENT_ID, kind: 'canvas' })
    expect(chip.refText).toContain('"Login Screen"')
    expect(chip.refText).toContain('- Hero card (frame, id dE6qx)')
    expect(chip.refText).toContain('- Sign in (id k9)')
  })
})
