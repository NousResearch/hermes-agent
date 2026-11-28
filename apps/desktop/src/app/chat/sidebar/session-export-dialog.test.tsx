import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { exportSessionFile } from '@/lib/session-export'
import { notify } from '@/store/notifications'

import { SessionExportDialog } from './session-export-dialog'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel', close: 'Close', loading: 'Loading' },
      desktop: {
        sessionExportTitle: 'Export conversation',
        sessionExportFormat: 'File format',
        sessionExportRedact: 'Redact detected secrets',
        sessionExportRedactHint: 'Review before sharing',
        sessionExportAction: 'Export',
        sessionExported: 'Exported',
        sessionExportFailed: 'Export failed',
        sessionExportNeedsLive: 'Needs live',
        sessionExportContents: 'Transcript contents',
        sessionExportTools: 'Tool calls and results',
        sessionExportReasoning: 'Stored reasoning',
        sessionExportImages: 'Images',
        sessionExportVideos: 'Video references',
        sessionExportFiles: 'File references',
        sessionExportLinks: 'Clickable links',
        sessionExportAsides: '/btw asides',
        sessionExportCollapse: 'Collapse included details',
        sessionExportMediaHint: 'Images are embedded when available; videos and files remain references.',
        sessionExportLinkHint: 'Disabling clickable links preserves the original text.',
        sessionExportImagePrivacy: 'Image pixels are not scanned for secrets.',
        sessionExportFoldDocx: 'Folding depends on the Word reader.',
        sessionExportFoldOdt: 'Folding depends on the ODT editor.',
        sessionExportFoldHtml: 'Details can expand in the file.',
        sessionExportFoldMd: 'Folding and images depend on the Markdown viewer.',
        sessionExportFoldJson: 'JSON contains selected data without visual folding.',
        sessionExportUnsupported: 'Unsupported'
      }
    }
  })
}))
vi.mock('@/lib/session-export', () => ({ exportSessionFile: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

beforeEach(() => vi.resetAllMocks())
afterEach(cleanup)
const target = { sessionId: 'a', profile: 'personal', title: 'Conversation A' }

const defaultOptions = {
  include_tools: false,
  include_reasoning: false,
  include_images: true,
  include_videos: true,
  include_files: true,
  include_links: true,
  include_asides: true,
  collapse_details: true
}

function open() {
  const onClose = vi.fn()
  const view = render(<SessionExportDialog onClose={onClose} target={target} />)

  return { onClose, ...view }
}

it('offers all five requested formats with Markdown selected by default', () => {
  open()

  for (const name of ['DOCX', 'JSON', 'ODT', 'Markdown', 'HTML']) {
    expect(screen.getByRole('button', { name })).toBeTruthy()
  }

  expect(screen.getByRole('button', { name: 'Markdown' }).getAttribute('aria-pressed')).toBe('true')
})

it('starts with clean content and independently forwards each inclusion choice', async () => {
  vi.mocked(exportSessionFile).mockResolvedValue({ kind: 'canceled' })
  open()

  const labels = [
    'Tool calls and results',
    'Stored reasoning',
    'Images',
    'Video references',
    'File references',
    'Clickable links',
    '/btw asides',
    'Collapse included details'
  ]

  for (const [index, name] of labels.entries()) {
    expect(screen.getByRole('checkbox', { name }).getAttribute('aria-checked')).toBe(index < 2 ? 'false' : 'true')
    fireEvent.click(screen.getByRole('checkbox', { name }))
  }

  fireEvent.click(screen.getByRole('button', { name: 'Export' }))
  await waitFor(() => expect(exportSessionFile).toHaveBeenCalledTimes(1))
  expect(exportSessionFile).toHaveBeenCalledWith({
    ...target,
    format: 'md',
    redact: false,
    options: {
      include_tools: true,
      include_reasoning: true,
      include_images: false,
      include_videos: false,
      include_files: false,
      include_links: false,
      include_asides: false,
      collapse_details: false
    }
  })
})

it('keeps chosen options after picker cancellation and retries with them', async () => {
  vi.mocked(exportSessionFile)
    .mockResolvedValueOnce({ kind: 'canceled' })
    .mockResolvedValueOnce({ kind: 'saved', path: '/out.md' })
  const { onClose } = open()
  const dialog = screen.getByRole('dialog')
  expect(within(dialog).getByRole('button', { name: 'Markdown' }).getAttribute('aria-pressed')).toBe('true')
  fireEvent.click(within(dialog).getByRole('button', { name: 'Markdown' }))
  fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Redact detected secrets' }))
  fireEvent.click(within(dialog).getByRole('button', { name: 'Export' }))
  await waitFor(() => expect(exportSessionFile).toHaveBeenCalledTimes(1))
  expect(exportSessionFile).toHaveBeenCalledWith({ ...target, format: 'md', options: defaultOptions, redact: true })
  expect(onClose).not.toHaveBeenCalled()
  expect(notify).not.toHaveBeenCalled()
  await waitFor(() => expect(within(dialog).getByRole('button', { name: 'Export' })).toBeTruthy())
  fireEvent.click(within(dialog).getByRole('button', { name: 'Export' }))
  await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
  expect(notify).toHaveBeenCalledTimes(1)
})

it.each(['unsupported', 'needs-live-backend', 'failed'] as const)(
  'keeps the draft on %s and permits retry',
  async kind => {
    vi.mocked(exportSessionFile)
      .mockResolvedValueOnce(kind === 'failed' ? { kind, error: new Error('failure') } : { kind })
      .mockResolvedValueOnce({ kind: 'json-download' })
    const { onClose } = open()
    fireEvent.click(screen.getByRole('button', { name: 'HTML' }))
    fireEvent.click(screen.getByRole('button', { name: 'Export' }))
    expect(await screen.findByRole('alert')).toBeTruthy()
    expect(notify).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'HTML' }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'Export' }))
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
  }
)

it('does not advertise portable folding in JSON or ODT', () => {
  open()

  for (const format of ['JSON', 'ODT']) {
    fireEvent.click(screen.getByRole('button', { name: format }))
    expect((screen.getByRole('checkbox', { name: 'Collapse included details' }) as HTMLButtonElement).disabled).toBe(
      true
    )
  }

  fireEvent.click(screen.getByRole('button', { name: 'HTML' }))
  expect((screen.getByRole('checkbox', { name: 'Collapse included details' }) as HTMLButtonElement).disabled).toBe(
    false
  )
  expect(exportSessionFile).not.toHaveBeenCalled()
})

it('blocks duplicate submission and dismissal while a write is pending', async () => {
  let resolve!: (value: { kind: 'canceled' }) => void
  vi.mocked(exportSessionFile).mockReturnValue(
    new Promise(r => {
      resolve = r
    })
  )
  const { onClose } = open()
  const button = screen.getByRole('button', { name: 'Export' })
  act(() => {
    button.click()
    button.click()
  })
  expect(exportSessionFile).toHaveBeenCalledTimes(1)
  expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)

  for (const checkbox of screen.getAllByRole('checkbox')) {
    expect((checkbox as HTMLButtonElement).disabled).toBe(true)
  }

  expect((screen.getByRole('button', { name: 'HTML' }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.keyDown(window.document, { key: 'Escape' })
  expect(onClose).not.toHaveBeenCalled()
  expect(screen.getByRole('dialog')).toBeTruthy()
  await act(async () => resolve({ kind: 'canceled' }))
  fireEvent.keyDown(window.document, { key: 'Escape' })
  expect(onClose).toHaveBeenCalledTimes(1)
})
