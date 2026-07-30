import { describe, expect, it } from 'vitest'

import { group } from '../model'

import { rootChildSide } from './track-model'

describe('rootChildSide persistent panes', () => {
  it('does not side-collapse a persistent right-placement pane', () => {
    const child = group(['run-board'])

    const paneFor = (id: string) =>
      ({
        data: id === 'run-board' ? { persistent: true, placement: 'right' } : { placement: 'right' }
      }) as never

    expect(rootChildSide(child, paneFor)).toBeNull()
  })

  it('keeps ordinary right-placement panes owned by the right rail', () => {
    const child = group(['files'])
    const paneFor = () => ({ data: { placement: 'right' } }) as never

    expect(rootChildSide(child, paneFor)).toBe('right')
  })

  it('keeps the workspace as the only pane that needs main placement', () => {
    const child = group(['workspace'])
    const paneFor = () => ({ data: { placement: 'main', uncloseable: true } }) as never

    expect(rootChildSide(child, paneFor)).toBeNull()
  })
})
