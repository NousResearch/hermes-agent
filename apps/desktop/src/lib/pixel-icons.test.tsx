import { readdirSync, readFileSync, statSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { Codicon } from '@/components/ui/codicon'
import { ToolIcon } from '@/components/ui/tool-icon'
import { ArrowBarToLeft, Plus, Settings } from '@/lib/icons'
import { $shellChoice } from '@/store/shell-variant'

import { CODICON_PIXEL, RTL_MIRRORED } from './pixel-icons'

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..')

/** Every codicon the font defines, read from the installed stylesheet. */
function codiconVocabulary(): Set<string> {
  const css = readFileSync(createRequire(import.meta.url).resolve('@vscode/codicons/dist/codicon.css'), 'utf8')

  return new Set([...css.matchAll(/^\.codicon-([a-z0-9-]+):before/gm)].map(match => match[1]))
}

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap(entry => {
    const path = join(dir, entry)

    if (statSync(path).isDirectory()) {
      return sourceFiles(path)
    }

    // The mapping modules name pixel glyphs, not codicons.
    return /\.(tsx?|css)$/.test(entry) && !/\.test\.|^pixel-icon/.test(entry) ? [path] : []
  })
}

/** Codicon names the app renders: `name=`/`icon:` literals, `codiconIcon()`, `codicon-*` classes
 *  and the language/file-type table. Derived from the source, so a new icon cannot ship unmapped. */
function codiconsInUse(): Map<string, string> {
  const vocabulary = codiconVocabulary()
  const used = new Map<string, string>()

  const patterns = [
    /(?:name|icon|codicon|iconName)\s*[:=]\s*\{?\s*['"]([a-z0-9-]+)['"]/g,
    /codiconIcon\(\s*['"]([a-z0-9-]+)['"]/g,
    /iconFor\(\s*['"]([a-z0-9-]+)['"]/g,
    /codicon-([a-z0-9-]+)/g,
    /name=\{[^}]*\?\s*['"]([a-z0-9-]+)['"]\s*:\s*['"]([a-z0-9-]+)['"]/g
  ]

  for (const file of sourceFiles(SRC)) {
    const text = readFileSync(file, 'utf8')
    const table = file.endsWith('markdown-code.ts') ? [/:\s*'([a-z0-9-]+)'/g] : []

    for (const pattern of [...patterns, ...table]) {
      for (const match of text.matchAll(pattern)) {
        for (const name of match.slice(1)) {
          if (name && vocabulary.has(name) && !used.has(name)) {
            used.set(name, file.slice(SRC.length + 1))
          }
        }
      }
    }
  }

  return used
}

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

describe('pixel icon mapping', () => {
  it('maps every codicon the app renders (or keeps its classic glyph on purpose)', () => {
    const unmapped = [...codiconsInUse()].filter(([name]) => !(name in CODICON_PIXEL))

    expect(unmapped.map(([name, file]) => `${name} (${file})`)).toEqual([])
  })
})

describe('shell-aware icons', () => {
  it('draws the codicon font in classic and the pixel glyph in portal, flipping live', () => {
    $shellChoice.set(false)
    const { container } = render(<Codicon name="settings-gear" />)

    expect(container.querySelector('i.codicon.codicon-settings-gear')).not.toBeNull()
    expect(container.querySelector('svg')).toBeNull()

    act(() => $shellChoice.set(true))

    const svg = container.querySelector('svg')

    expect(container.querySelector('i')).toBeNull()
    expect(svg?.dataset.pixelIcon).toBe('settings-2')
    // The legacy classes stay so codicon sizing/spin rules keep applying.
    expect(svg?.classList.contains('codicon-settings-gear')).toBe(true)
  })

  it('keeps the classic glyph for deliberate brand/language marks in portal', () => {
    $shellChoice.set(true)
    const { container } = render(<Codicon name="python" />)

    expect(container.querySelector('i.codicon-python')).not.toBeNull()
  })

  it('swaps Tabler aliases with the shell and carries size through', () => {
    $shellChoice.set(false)
    const { container } = render(<Plus size={18} />)

    expect(container.querySelector('svg')?.classList.contains('tabler-icon')).toBe(true)

    act(() => $shellChoice.set(true))

    const svg = container.querySelector('svg')

    expect(svg?.dataset.pixelIcon).toBe('plus')
    expect(svg?.getAttribute('width')).toBe('18')
  })

  it('draws tool rows with the pixel set in portal', () => {
    $shellChoice.set(true)
    const { container } = render(<ToolIcon name="terminal" />)

    expect(container.querySelector('svg')?.dataset.pixelIcon).toBe('terminal')
  })

  it('marks only reading-axis glyphs for RTL mirroring', () => {
    $shellChoice.set(true)

    const { container } = render(
      <>
        <ArrowBarToLeft />
        <Settings />
      </>
    )

    const [arrow, settings] = container.querySelectorAll('svg')

    expect(arrow.hasAttribute('data-rtl-mirror')).toBe(true)
    expect(settings.hasAttribute('data-rtl-mirror')).toBe(false)
    expect(RTL_MIRRORED.has('chevron-down')).toBe(false)
  })
})
