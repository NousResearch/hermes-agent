import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

const readSource = (relativePath: string) => readFileSync(new URL(relativePath, import.meta.url), 'utf8')

describe('Hermes CLI backdrop treatment', () => {
  it('hides the decorative backdrop image for the flat terminal-style theme', () => {
    const backdropSource = readSource('./Backdrop.tsx')
    const stylesSource = readSource('../styles.css')

    expect(backdropSource).toContain('data-slot="theme-backdrop-statue"')
    expect(stylesSource).toContain(":root[data-hermes-theme='hermes-cli'] [data-slot='theme-backdrop-statue']")
    expect(stylesSource).toMatch(
      /:root\[data-hermes-theme='hermes-cli'\] \[data-slot='theme-backdrop-statue'\]\s*\{\s*display: none;\s*\}/
    )
  })
})
