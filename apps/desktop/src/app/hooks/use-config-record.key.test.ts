import { describe, expect, it } from 'vitest'

import { HERMES_CONFIG_KEY, hermesConfigKey } from './use-config-record'

describe('hermesConfigKey', () => {
  it('always suffixes a profile so follow-active rows cannot share an in-flight fetch', () => {
    expect(hermesConfigKey()).toEqual([...HERMES_CONFIG_KEY, 'default'])
    expect(hermesConfigKey('coder')).toEqual([...HERMES_CONFIG_KEY, 'coder'])
    expect(hermesConfigKey('default')).toEqual([...HERMES_CONFIG_KEY, 'default'])
    expect(hermesConfigKey()).not.toEqual(HERMES_CONFIG_KEY)
  })
})
