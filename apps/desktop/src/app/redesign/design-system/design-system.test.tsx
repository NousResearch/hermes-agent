import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BUNDLED_LOCALES } from '@/i18n/catalog'
import { tokenFamilies } from '@/lib/design-tokens'
import { $shellChoice } from '@/store/shell-variant'

import { designSystemCopy } from './design-system-copy'
import { GALLERY } from './registry'
import type * as SourcesModule from './sources'
import { TOKEN_SOURCES } from './sources'

// The shipped token files' text. The page imports them with `?raw`, which vitest hands over empty, so
// the page under test reads the same files from disk instead.
const TOKENS = join(dirname(fileURLToPath(import.meta.url)), '../tokens')

const tokenText = vi.hoisted(() => ({ files: {} as Record<string, string> }))

vi.mock('./sources', async importOriginal => ({
  ...(await importOriginal<typeof SourcesModule>()),
  TOKEN_SOURCES: tokenText.files
}))

const stems = readdirSync(TOKENS)
  .filter(name => name.endsWith('.css'))
  .map(name => name.replace(/\.css$/, ''))

for (const stem of stems) {
  tokenText.files[stem] = readFileSync(join(TOKENS, `${stem}.css`), 'utf8')
}

const declared = tokenFamilies(tokenText.files).flatMap(family => family.tokens.map(token => token.name))

beforeEach(() => $shellChoice.set(true))

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

describe('the design system page', () => {
  it('reads every token file the folder holds', async () => {
    const { TOKEN_SOURCES: shipped } = await vi.importActual<typeof SourcesModule>('./sources')

    expect(Object.keys(shipped).sort()).toEqual([...stems].sort())
    expect(TOKEN_SOURCES).toBe(tokenText.files)
  })

  it('renders every token defined in the tokens folder, and every registry entry', async () => {
    const { DesignSystemView } = await import('./index')

    const { container } = render(
      <MemoryRouter>
        <DesignSystemView />
      </MemoryRouter>
    )

    const shown = new Set([...container.querySelectorAll('[data-token]')].map(row => row.getAttribute('data-token')))

    expect(declared.length).toBeGreaterThan(100)
    expect(declared.filter(name => !shown.has(name))).toEqual([])

    for (const entry of GALLERY) {
      expect(screen.getByRole('article', { name: entry.title })).toBeTruthy()
    }
  })

  it('names every token file and every group in every locale', () => {
    for (const locale of BUNDLED_LOCALES) {
      const copy = designSystemCopy(locale)

      expect(
        stems.filter(stem => stem !== 'index' && !copy.families[stem]),
        locale
      ).toEqual([])
    }
  })
})
