import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { findRawColors } from './check-design-tokens.mjs'

const roots = []

async function fixture(files) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-design-tokens-'))
  roots.push(root)
  await Promise.all(Object.entries(files).map(async ([relativePath, content]) => {
    const filePath = path.join(root, relativePath)
    await mkdir(path.dirname(filePath), { recursive: true })
    await writeFile(filePath, content)
  }))
  return root
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map(root => import('node:fs/promises').then(({ rm }) => rm(root, { recursive: true }))))
})

describe('check-design-tokens', () => {
  it('reports raw hex colors in component source while ignoring issue comments', async () => {
    const root = await fixture({
      'components/example.tsx': '// Follow-up: #103375\nexport const color = "#aabbcc"\n'
    })

    await expect(findRawColors(root)).resolves.toEqual(['components/example.tsx:2: #aabbcc'])
  })

  it('ignores issue references in prose strings but reports exact color strings', async () => {
    const root = await fixture({
      'components/example.tsx': [
        'const message = "Fixes #12345"',
        'const color = "#123456"',
      ].join('\n'),
    })

    await expect(findRawColors(root)).resolves.toEqual(['components/example.tsx:2: #123456'])
  })

  it('permits documented theme and brand sources', async () => {
    const root = await fixture({
      'themes/preset.ts': 'export const color = "#aabbcc"\n',
      'lib/mcp-brands.tsx': 'export const color = "#aabbcc"\n'
    })

    await expect(findRawColors(root)).resolves.toEqual([])
  })

  it('reports raw hex colors in authored CSS', async () => {
    const root = await fixture({
      'components/example.css': '.example { color: #123456; }\n'
    })

    await expect(findRawColors(root)).resolves.toEqual(['components/example.css:1: #123456'])
  })

  it('runs the token checker from the standard check command', async () => {
    const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))

    expect(packageJson.scripts.check).toContain('npm run check:design-tokens')
  })
})
