import type { ILink, Terminal as TerminalType } from '@xterm/xterm'
import { Terminal } from '@xterm/xterm'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $previewTabs, closeRightRail } from '@/store/preview'

import { clearTerminalFileLinkCache, pathCandidates, registerTerminalFileLinks } from './file-links'

const EXISTING = new Set(['/repo/src/app.ts', '/repo/README.md', '/home/u/notes.md'])

beforeEach(() => {
  clearTerminalFileLinkCache()
  closeRightRail()
  Object.defineProperty(navigator, 'platform', { configurable: true, value: 'MacIntel' })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      normalizePreviewTarget: vi.fn(async (raw: string, base?: string) => {
        const path = raw.startsWith('/') ? raw : `${base}/${raw.replace(/^\.\//, '')}`

        return EXISTING.has(path)
          ? { kind: 'file', label: path, path, previewKind: 'text', source: raw, url: `file://${path}` }
          : { kind: 'file', label: path, path, previewKind: 'missing', source: raw, url: `file://${path}` }
      })
    }
  })
})

async function linksIn(text: string, isLocal = true, cols = 120, row = 1): Promise<ILink[]> {
  const term = new Terminal({ allowProposedApi: true, cols, rows: 5 })
  const providers: Array<Parameters<TerminalType['registerLinkProvider']>[0]> = []
  const register = term.registerLinkProvider.bind(term)

  term.registerLinkProvider = provider => {
    providers.push(provider)

    return register(provider)
  }

  registerTerminalFileLinks(term, { isLocal: () => isLocal, resolveCwd: () => '/repo' })
  await new Promise<void>(resolve => term.write(`${text}\r\n`, resolve))

  return new Promise<ILink[]>(resolve => providers[0].provideLinks(row, found => resolve(found ?? [])))
}

describe('terminal file links', () => {
  it('links only paths that exist, keeps the location suffix in the range, and never takes URLs or versions', async () => {
    const line = 'error src/app.ts:12:3 see ./README.md, missing.ts v1.2.3 https://x.io/a.ts'
    const links = await linksIn(line)

    expect(links.map(link => link.text)).toEqual(['src/app.ts:12:3', './README.md'])
    expect(links[0].range.start.x).toBe(line.indexOf('src/') + 1)
    expect(pathCandidates('ran 1.2.3 tests in 0.5s').map(c => c.path)).toEqual([])
  })

  it('links a path the terminal soft-wrapped across rows, from either row', async () => {
    // 20 columns: "see ./README.md" wraps after "./READ".
    const line = 'aaaaaaaaaaaaaa ./README.md'

    for (const row of [1, 2]) {
      const [link] = await linksIn(line, true, 20, row)

      expect(link?.text).toBe('./README.md')
      expect(link?.range).toEqual({ end: { x: 6, y: 2 }, start: { x: 16, y: 1 } })
    }
  })

  it('opens a ⌘-clicked path in the preview pane; a bare click or an SSH shell opens nothing', async () => {
    const [link] = await linksIn('modified:   src/app.ts')

    link.activate(new MouseEvent('click'), link.text)
    await Promise.resolve()
    expect($previewTabs.get()).toHaveLength(0)

    link.activate(new MouseEvent('click', { metaKey: true }), link.text)
    await vi.waitFor(() => expect($previewTabs.get().at(-1)?.target.path).toBe('/repo/src/app.ts'))

    expect(await linksIn('modified:   src/app.ts', false)).toEqual([])
  })
})
