import { afterEach, describe, expect, it } from 'vitest'

import { _resetToolCardsForTests, cardToolName, isCardTool, isCardToolName, learnToolCard } from './tool-render-class'

afterEach(() => {
  _resetToolCardsForTests()
})

describe('learned tool cards', () => {
  it('maps a tool that declared the delegate card onto the delegate_task card', () => {
    expect(cardToolName('agents_run')).toBeUndefined()
    expect(isCardTool('agents_run')).toBe(false)

    learnToolCard('agents_run', 'delegate')

    expect(cardToolName('agents_run')).toBe('delegate_task')
    expect(isCardToolName('agents_run')).toBe(true)
    expect(isCardTool('agents_run')).toBe(true)
  })

  it('keeps built-in card names resolving to themselves', () => {
    expect(cardToolName('delegate_task')).toBe('delegate_task')
    expect(cardToolName('clarify')).toBe('clarify')
    expect(cardToolName('terminal')).toBeUndefined()
  })

  it('ignores empty names and unknown or non-string kinds without throwing', () => {
    learnToolCard('agents_run', 'nope')
    learnToolCard('agents_run', 'toString')
    learnToolCard('agents_run', undefined)
    learnToolCard('agents_run', null)
    learnToolCard('agents_run', 7)
    learnToolCard('', 'delegate')

    expect(cardToolName('agents_run')).toBeUndefined()
    expect(cardToolName('')).toBeUndefined()
  })

  it('forgets everything on reset', () => {
    learnToolCard('agents_run', 'delegate')
    _resetToolCardsForTests()

    expect(isCardTool('agents_run')).toBe(false)
  })
})
