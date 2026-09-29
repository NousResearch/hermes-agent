import { describe, expect, it } from 'vitest'

import type { VaultNoteNode } from '@/api/vault'

import { folderHues } from './vault-graph'

const note = (id: string, folder: string): VaultNoteNode => ({
  excerpt: '',
  folder,
  id,
  label: id,
  links: 0,
  size: 0,
  tags: [],
  timestamp: 0
})

describe('folderHues', () => {
  it('gives every top-level folder its own hue, and sub-folders their root’s', () => {
    const hues = folderHues([note('a', 'Ludzie'), note('b', 'Ludzie/Klienci'), note('c', 'Projekty'), note('d', '')])

    expect([...hues.keys()].sort()).toEqual(['', 'Ludzie', 'Projekty'])
    expect(new Set(hues.values()).size).toBe(3)
  })
})
