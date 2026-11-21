import { existsSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { GALLERY, GALLERY_GROUPS, NOT_SHOWN } from './index'

const SRC = join(dirname(fileURLToPath(import.meta.url)), '../../../..')
const UI = join(SRC, 'components/ui')

/** `components/ui/` modules and stylesheets, by stem; a stylesheet belongs to its module. */
const uiStems = [
  ...new Set(
    readdirSync(UI, { withFileTypes: true })
      .filter(entry => entry.isFile() && !/\.test\.tsx?$/.test(entry.name))
      .map(entry => entry.name.replace(/\.(tsx?|css)$/, ''))
  )
].sort()

const stem = (source: string) => source.replace(/^components\/ui\//, '').replace(/\.(tsx?|css)$/, '')

const shown = new Set(
  GALLERY.flatMap(entry => [entry.source, ...(entry.covers ?? [])])
    .filter(source => source.startsWith('components/ui/'))
    .map(stem)
)

describe('the design system page covers components/ui', () => {
  it('shows every file there, or says why not', () => {
    const missing = uiStems.filter(name => !shown.has(name) && !(name in NOT_SHOWN))

    expect(missing, 'add each to a registry group (registry/*.tsx) or to NOT_SHOWN with a reason').toEqual([])
  })

  it('lists nothing as not shown that is shown or no longer exists', () => {
    expect(Object.keys(NOT_SHOWN).filter(name => shown.has(name) || !uiStems.includes(name))).toEqual([])
  })

  it('gives each not-shown file a reason', () => {
    expect(Object.entries(NOT_SHOWN).filter(([, reason]) => !reason.trim())).toEqual([])
  })
})

describe('the registry', () => {
  it('names files that exist', () => {
    const sources = GALLERY.flatMap(entry => [entry.source, ...(entry.covers ?? [])])

    expect(sources.filter(source => !existsSync(join(SRC, source)))).toEqual([])
  })

  it('has a use line and at least one example per entry, in a known group', () => {
    const incomplete = GALLERY.filter(
      entry => !entry.use.trim() || entry.examples.length === 0 || !GALLERY_GROUPS.includes(entry.group)
    )

    expect(incomplete.map(entry => entry.title)).toEqual([])
  })

  it('shows each primitive once', () => {
    const titles = GALLERY.map(entry => entry.title)

    expect(titles.filter((title, index) => titles.indexOf(title) !== index)).toEqual([])
  })
})
