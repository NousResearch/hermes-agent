import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

vi.mock('./right-rail/preview', () => ({
  PreviewTilePane: () => null
}))

vi.mock('./right-rail/preview-console-store', () => ({
  forgetPreviewConsole: () => undefined
}))

import { DROPDOWN_KIT, type MenuKit } from '@/components/ui/actions-menu'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { registry } from '@/contrib/registry'
import { $previewTabs, closeRightRail, noteBrowserPage, openPreview } from '@/store/preview'
import { $connection } from '@/store/session'

import { fileTabPath, watchPreviewTiles } from './preview-tile'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installBridge(partial: Partial<Window['hermesDesktop']> = {}) {
  desktopWindow.hermesDesktop = {
    openExternal: vi.fn().mockResolvedValue(undefined),
    openPreviewInBrowser: vi.fn().mockResolvedValue(undefined),
    revealPath: vi.fn().mockResolvedValue(true),
    writeClipboard: vi.fn().mockResolvedValue(undefined),
    ...partial
  } as unknown as Window['hermesDesktop']

  return desktopWindow.hermesDesktop
}

/** The zone tab strip renders a tile's `tabMenuPrefix` at the top of its
 *  right-click menu. Read it back through the registry the way the strip does
 *  and mount it in a real open menu, so the rows are the rows the user sees. */
function tabMenuPrefixFor(tabId: string): ((kit: MenuKit) => React.ReactNode) | undefined {
  const entry = registry.getArea('panes').find(item => item.id === `preview-tile:${tabId}`)

  return (entry?.data as { tabMenuPrefix?: (kit: MenuKit) => React.ReactNode } | undefined)?.tabMenuPrefix
}

function mountPrefix(tabId: string) {
  const prefix = tabMenuPrefixFor(tabId)

  return render(
    <DropdownMenu open>
      <DropdownMenuTrigger>tab</DropdownMenuTrigger>
      <DropdownMenuContent>{prefix?.(DROPDOWN_KIT) ?? null}</DropdownMenuContent>
    </DropdownMenu>
  )
}

const fileTarget = (path: string) =>
  ({ kind: 'file', label: path.split('/').at(-1) ?? path, path, source: path, url: path }) as const

beforeAll(() => {
  watchPreviewTiles()
})

afterEach(() => {
  closeRightRail()
  $connection.set(null)
  cleanup()
  vi.restoreAllMocks()
  delete desktopWindow.hermesDesktop
})

describe('fileTabPath', () => {
  it('is the on-disk path for a file tab and null otherwise', () => {
    openPreview(fileTarget('/tmp/a.md'), 'file-browser')
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })

    const fileTab = $previewTabs.get().find(tab => tab.target.kind === 'file')!
    const urlTab = $previewTabs.get().find(tab => tab.target.kind === 'url')!

    expect(fileTabPath(fileTab.id)).toBe('/tmp/a.md')
    expect(fileTabPath(urlTab.id)).toBeNull()
  })
})

describe('preview tab menu — file tab', () => {
  it('offers open externally, reveal and copy path on a local gateway', async () => {
    installBridge()
    openPreview(fileTarget('/tmp/report.md'), 'file-browser')

    const tabId = $previewTabs.get()[0]!.id

    mountPrefix(tabId)

    expect(await screen.findByText('Open in external')).toBeTruthy()
    expect(screen.getByText(/Reveal in Finder|Reveal in File Explorer|Open containing folder/)).toBeTruthy()
    expect(screen.getByText('Copy path')).toBeTruthy()
  })

  it('copies the path and reveals the file through the desktop bridge', async () => {
    const bridge = installBridge()

    openPreview(fileTarget('/tmp/my report.md'), 'file-browser')

    const tabId = $previewTabs.get()[0]!.id

    mountPrefix(tabId)
    fireEvent.click(await screen.findByText('Copy path'))
    await waitFor(() => expect(bridge.writeClipboard).toHaveBeenCalledWith('/tmp/my report.md'))

    cleanup()
    mountPrefix(tabId)
    fireEvent.click(await screen.findByText(/Reveal in Finder|Reveal in File Explorer|Open containing folder/))
    await waitFor(() => expect(bridge.revealPath).toHaveBeenCalledWith('/tmp/my report.md'))
  })

  it('opens the file outside Hermes through the preview browser bridge', async () => {
    const bridge = installBridge()

    openPreview(fileTarget('/tmp/report.md'), 'file-browser')

    const tabId = $previewTabs.get()[0]!.id

    mountPrefix(tabId)
    fireEvent.click(await screen.findByText('Open in external'))

    await waitFor(() => expect(bridge.openPreviewInBrowser).toHaveBeenCalledWith('/tmp/report.md'))
  })

  it('keeps only copy path on a remote gateway — the file is not on this disk', async () => {
    $connection.set({ mode: 'remote' } as never)
    installBridge()
    openPreview(fileTarget('/srv/data/notes.txt'), 'file-browser')

    const tabId = $previewTabs.get()[0]!.id

    mountPrefix(tabId)

    expect(await screen.findByText('Copy path')).toBeTruthy()
    expect(screen.queryByText('Open in external')).toBeNull()
    expect(screen.queryByText(/Reveal in Finder|Reveal in File Explorer|Open containing folder/)).toBeNull()
  })
})

describe('preview tab menu — browser tab', () => {
  it('adds Copy URL beside Open in external, using the live page address', async () => {
    const bridge = installBridge()

    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })

    const tabId = $previewTabs.get()[0]!.id

    noteBrowserPage(tabId, { title: 'HN', url: 'https://news.ycombinator.com/' })
    mountPrefix(tabId)

    expect(await screen.findByText('Open in external')).toBeTruthy()
    fireEvent.click(screen.getByText('Copy URL'))

    await waitFor(() => expect(bridge.writeClipboard).toHaveBeenCalledWith('https://news.ycombinator.com/'))
  })
})
