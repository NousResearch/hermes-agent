import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

// Mock the desktop-fs module so readDesktopFileText is controllable.
const readFileText = vi.fn()
const readFileDataUrl = vi.fn()
const gitRoot = vi.fn()

vi.mock('@/lib/desktop-fs', () => ({
  desktopFileDiff: vi.fn(async () => ''),
  desktopGitRoot: (...args: unknown[]) => gitRoot(...args),
  readDesktopFileDataUrl: (...args: unknown[]) => readFileDataUrl(...args),
  readDesktopFileText: (...args: unknown[]) => readFileText(...args),
  writeDesktopFileText: vi.fn()
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key,
  useI18n: () => ({
    configLoadError: null,
    isLoadingConfig: false,
    isSavingLocale: false,
    locale: 'en' as const,
    saveError: null,
    setLocale: async () => {},
    t: {
      preview: {
        tab: '', closeTab: () => '', closeOthers: '', closeToRight: '', closeAll: '',
        closePane: '', loading: 'Loading…', unavailable: 'Unavailable', opening: '',
        hide: '', openPreview: '', openInBrowser: '', linkHint: '',
        sourceLineTitle: 'Source line', source: 'Source', renderedPreview: 'Rendered',
        diff: 'Diff', unknownSize: '', binaryTitle: 'Binary file',
        binaryBody: (l: string) => `${l} is binary`, largeTitle: 'File too large',
        largeBody: (_l: string, _s: string) => 'too large', previewAnyway: 'Preview anyway',
        truncated: 'Truncated', noInlineTitle: 'Cannot preview',
        noInlineBody: () => 'no inline', edit: 'Edit', editing: '', unsavedChanges: '',
        saveFailed: (e: string) => e, diskChangedTitle: 'Changed',
        diskChangedBody: 'Body', overwrite: 'Overwrite', discardReload: 'Discard',
      },
    },
  })
}))

function stubBridge() {
  vi.stubGlobal('window', {
    ...window,
    hermesDesktop: {
      gitRoot,
      readFileDataUrl,
      readFileText
    }
  })
}

describe('LocalFilePreview null-text handling', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
    $connection.set(null)
  })

  it('renders without crashing when readDesktopFileText returns null text for a binary file', async () => {
    stubBridge()
    // Simulate a binary file: IPC returns text=null, binary=true
    readFileText.mockResolvedValue({
      binary: true,
      byteSize: 1024,
      mimeType: 'application/pdf',
      path: '/test/file.pdf',
      text: null
    })

    render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          kind: 'file',
          label: 'file.pdf',
          path: '/test/file.pdf',
          previewKind: 'text',
          source: '/test/file.pdf',
          url: 'file:///test/file.pdf'
        }}
      />
    )

    // Should render the binary warning instead of crashing
    await waitFor(() => {
      expect(screen.queryByText('Loading…')).toBeNull()
    })
    // The component should render without throwing — the binary
    // guard blocks the null text from reaching chunkTextLines.
    expect(readFileText).toHaveBeenCalledWith('/test/file.pdf')
  })

  it('renders without crashing when readDesktopFileText returns null text and binary=false', async () => {
    stubBridge()
    // Edge case: text is null but binary flag is also false (IPC quirk)
    readFileText.mockResolvedValue({
      binary: false,
      byteSize: 512,
      mimeType: 'application/octet-stream',
      path: '/test/data.bin',
      text: null
    })

    render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          kind: 'file',
          label: 'data.bin',
          path: '/test/data.bin',
          previewKind: 'text',
          source: '/test/data.bin',
          url: 'file:///test/data.bin'
        }}
      />
    )

    // Should NOT crash — null text is normalized to undefined by the
    // ?? operator, so the rendering guard (state.text != null) blocks it.
    await waitFor(() => {
      expect(screen.queryByText('Loading…')).toBeNull()
    })
    expect(readFileText).toHaveBeenCalledWith('/test/data.bin')
  })
})