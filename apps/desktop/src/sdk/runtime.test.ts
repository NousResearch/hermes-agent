import { describe, expect, it, vi } from 'vitest'

import { sdkImportMap, shimSource } from './runtime'

describe('plugin SDK shim source', () => {
  it('fails loudly at import time when the namespace is missing, naming the piece', () => {
    for (const namespace of [undefined, null]) {
      const source = shimSource('__HERMES_REACT_JSX_DEV__', namespace, 'react/jsx-dev-runtime')

      expect(source.startsWith('throw new Error(')).toBe(true)
      expect(source).toContain('react/jsx-dev-runtime')
      expect(source).toContain('__HERMES_REACT_JSX_DEV__')
    }
  })

  it('re-exports live members while skipping default and invalid identifiers', () => {
    const source = shimSource('__HERMES_PLUGIN_SDK__', { ping: 1, default: 2, 'not-an-identifier!': 3 }, '@hermes/plugin-sdk')

    expect(source).toContain('export const { ping } = m')
    expect(source).not.toContain('not-an-identifier!')
  })

  it('builds a usable shim URL for every supported specifier without throwing', () => {
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-shim')

    try {
      const map = sdkImportMap()

      for (const specifier of ['@hermes/plugin-sdk', 'react', 'react/jsx-runtime', 'react/jsx-dev-runtime']) {
        expect(map[specifier]).toBe('blob:fake-shim')
      }
    } finally {
      createObjectURL.mockRestore()
    }
  })
})
