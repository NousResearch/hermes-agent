import { describe, expect, it } from 'vitest'

import { installedDesktopHalfIds } from './desktop-half-enable'

const record = {
  id: 'secret-drop',
  name: 'Secret drop',
  packageName: 'secret-drop',
  status: 'disabled' as const
}

describe('installedDesktopHalfIds', () => {
  it('turns on the half the install just named', () => {
    expect(installedDesktopHalfIds([record], ['secret-drop'])).toEqual(['secret-drop'])
  })

  it('leaves a loaded half and an unrelated disabled plugin alone', () => {
    expect(
      installedDesktopHalfIds(
        [
          { ...record, status: 'loaded' },
          { id: 'other', name: 'Other', status: 'disabled' }
        ],
        ['secret-drop']
      )
    ).toEqual([])
  })

  it('does not match an empty name list', () => {
    expect(installedDesktopHalfIds([record], [null, undefined, ''])).toEqual([])
  })
})
