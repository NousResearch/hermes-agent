import { describe, expect, it } from 'vitest'

import { en } from './en'
import { ru } from './ru'

const leafPaths = (value: unknown, prefix = ''): string[] => {
  if (typeof value === 'function' || typeof value !== 'object' || value === null || Array.isArray(value)) {
    return [prefix]
  }

  const paths: string[] = []

  for (const [key, child] of Object.entries(value)) {
    const childPrefix = prefix.length > 0 ? `${prefix}.${key}` : key
    paths.push(...leafPaths(child, childPrefix))
  }

  return paths.sort()
}

describe('desktop locale catalogs', () => {
  it('keeps Russian leaf paths identical to English', () => {
    expect(leafPaths(ru)).toEqual(leafPaths(en))
  })
})
