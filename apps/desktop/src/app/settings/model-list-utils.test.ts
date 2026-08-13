import { describe, expect, it } from 'vitest'

import {
  CAPABILITY_ACTIVE_CLASS,
  CAPABILITY_CHIP_CLASS,
  CAPABILITY_ORDER,
  discoverButtonLabel,
  groupModelsByLetter,
  inferModelCapabilities,
  sortModelFamilies
} from './model-list-utils'
import type { ModelFamily } from '@/store/model-visibility'

const fam = (id: string): ModelFamily => ({ id, fastId: null })

describe('groupModelsByLetter', () => {
  it('returns empty for no families', () => {
    expect(groupModelsByLetter([])).toEqual([])
  })

  it('groups by first letter of display name', () => {
    const groups = groupModelsByLetter([fam('gpt-4o'), fam('claude-opus'), fam('gemini')])
    expect(groups.map(g => g.letter)).toEqual(['G', 'O'])
  })

  it('puts digits/symbols under #', () => {
    const groups = groupModelsByLetter([fam('1-mini'), fam('@special')])
    expect(groups.map(g => g.letter)).toEqual(['#'])
  })

  it('preserves family order within a letter', () => {
    const groups = groupModelsByLetter([fam('gpt-4o'), fam('gpt-4o-mini')])
    expect(groups[0].families.map(f => f.id)).toEqual(['gpt-4o', 'gpt-4o-mini'])
  })

  it('sorts letters alphabetically', () => {
    const groups = groupModelsByLetter([fam('zebra'), fam('alpha'), fam('mango')])
    expect(groups.map(g => g.letter)).toEqual(['A', 'M', 'Z'])
  })
})

describe('inferModelCapabilities', () => {
  it('infers vision from known id patterns', () => {
    expect(inferModelCapabilities('gpt-4o', 'GPT-4o').vision).toBe(true)
    expect(inferModelCapabilities('claude-3-5-sonnet', 'Claude 3.5 Sonnet').vision).toBe(true)
    expect(inferModelCapabilities('text-babbage', 'Babbage').vision).toBe(false)
  })

  it('infers multimodal from audio/whisper patterns', () => {
    expect(inferModelCapabilities('whisper-1', 'Whisper').multimodal).toBe(true)
    expect(inferModelCapabilities('gpt-4o', 'GPT-4o').multimodal).toBe(true)
  })

  it('infers reasoning from id patterns', () => {
    expect(inferModelCapabilities('o1-preview', 'o1').reasoning).toBe(true)
    expect(inferModelCapabilities('deepseek-reasoner', 'Reasoner').reasoning).toBe(true)
    expect(inferModelCapabilities('gpt-4o', 'GPT-4o').reasoning).toBe(false)
  })

  it('prefers backend flags when supplied', () => {
    expect(inferModelCapabilities('custom-model', 'Custom', { reasoning: true, fast: false }).reasoning).toBe(true)
    expect(inferModelCapabilities('custom-model-fast', 'Custom Fast', { fast: false }).fast).toBe(false)
  })

  it('infers fast from -fast suffix', () => {
    expect(inferModelCapabilities('gpt-4o-fast', 'GPT-4o Fast').fast).toBe(true)
  })
})

    describe('sortModelFamilies', () => {
    const list = [fam('alpha-model'), fam('beta-model'), fam('gamma-model')]
  
    it('activeFirst floats visible above hidden, each alphabetical', () => {
      const sorted = sortModelFamilies(list, 'activeFirst', id => id === 'beta-model')
      expect(sorted.map(f => f.id)).toEqual(['beta-model', 'alpha-model', 'gamma-model'])
    })
  
    it('az sorts alphabetically', () => {
      const sorted = sortModelFamilies(list, 'az', () => false)
      expect(sorted.map(f => f.id)).toEqual(['alpha-model', 'beta-model', 'gamma-model'])
    })
  
    it('za sorts reverse alphabetically', () => {
      const sorted = sortModelFamilies(list, 'za', () => false)
      expect(sorted.map(f => f.id)).toEqual(['gamma-model', 'beta-model', 'alpha-model'])
    })

  it('is stable for equal keys', () => {
    const stable = [fam('b'), fam('a'), fam('b')]
    const sorted = sortModelFamilies(stable, 'az', () => false)
    expect(sorted.map(f => f.id)).toEqual(['a', 'b', 'b'])
  })
})

describe('capability color meta', () => {
  it('exposes a stable order of all four capabilities', () => {
    expect(CAPABILITY_ORDER).toEqual(['vision', 'multimodal', 'reasoning', 'fast'])
  })

  it('gives every capability a distinct, readable active color class', () => {
    const classes = CAPABILITY_ORDER.map(key => CAPABILITY_ACTIVE_CLASS[key])
    // No two capabilities share the same color → filters read at a glance.
    expect(new Set(classes).size).toBe(CAPABILITY_ORDER.length)
    for (const cls of classes) {
      // /30 background (stronger than the old /20) + darker text for light theme.
      expect(cls).toMatch(/bg-.*\/30/)
      expect(cls).toMatch(/text-.*-[67]00/)
    }
  })

  it('gives every capability a non-empty resting chip class', () => {
    for (const key of CAPABILITY_ORDER) {
      expect(CAPABILITY_CHIP_CLASS[key].length).toBeGreaterThan(0)
    }
  })

  it('resting chip colors share the hue of their active badge color', () => {
    // The chip and its badge must read as the same capability: reasoning is
    // amber in both states, vision violet in both, etc. Extract the hue token
    // (the color name between "bg-" and "-500") and compare.
    const hue = (cls: string) => /bg-([a-z]+)-500/.exec(cls)?.[1]

    for (const key of CAPABILITY_ORDER) {
      expect(hue(CAPABILITY_CHIP_CLASS[key])).toBe(hue(CAPABILITY_ACTIVE_CLASS[key]))
    }
  })

  it('resting chip colors are pairwise distinct', () => {
    const classes = CAPABILITY_ORDER.map(key => CAPABILITY_CHIP_CLASS[key])
    expect(new Set(classes).size).toBe(CAPABILITY_ORDER.length)
  })
})

describe('discoverButtonLabel', () => {
  const copy = { discoverModels: 'Discover models', updateList: 'Update list' }

  it('returns discoverModels when the list is empty', () => {
    expect(discoverButtonLabel(0, copy)).toBe('Discover models')
  })

  it('returns updateList when models exist', () => {
    expect(discoverButtonLabel(5, copy)).toBe('Update list')
  })

  it('returns updateList for the boundary case of exactly 1 model', () => {
    expect(discoverButtonLabel(1, copy)).toBe('Update list')
  })
})
