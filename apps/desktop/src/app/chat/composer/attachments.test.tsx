import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { ComposerAttachment } from '@/store/composer'
import { $previewTabs } from '@/store/preview'

import { AttachmentList } from './attachments'

const DATA_URL = 'data:image/png;base64,iVBORw0KGgoAAAANS'
const THUMBNAIL_URL = 'data:image/png;base64,dGh1bWJuYWls'

function makeAttachment(id: string, label = 'test.pdf'): ComposerAttachment {
  return { id, kind: 'file', label }
}

async function renderWithI18n(ui: React.ReactNode) {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        {ui}
      </I18nProvider>
    )
  })

  return result!
}

describe('AttachmentList', () => {
  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.restoreAllMocks()
  })

  it('renders valid attachments', async () => {
    const attachments = [makeAttachment('a', 'doc.pdf'), makeAttachment('b', 'img.png')]
    await renderWithI18n(<AttachmentList attachments={attachments} />)
    expect(screen.getByText('doc.pdf')).toBeDefined()
    expect(screen.getByText('img.png')).toBeDefined()
  })

  it('renders empty list without error', async () => {
    const { container } = await renderWithI18n(<AttachmentList attachments={[]} />)

    const attachmentList = container.querySelector('[data-slot="composer-attachments"]')

    expect(attachmentList).toBeDefined()
  })

  it('does not crash when attachments array contains undefined entries', async () => {
    // Repro: session switch can leave stale/undefined entries in the
    // attachments array, causing a TypeError at attachment.refText.
    const attachments = [
      makeAttachment('a', 'good.pdf'),
      undefined as unknown as ComposerAttachment,
      makeAttachment('b', 'also-good.png')
    ]

    await expect(renderWithI18n(<AttachmentList attachments={attachments} />)).resolves.toBeTruthy()

    // Only valid attachments should render
    expect(screen.getByText('good.pdf')).toBeDefined()
    expect(screen.getByText('also-good.png')).toBeDefined()
  })

  it('does not crash when attachments array contains null entries', async () => {
    const attachments = [null as unknown as ComposerAttachment, makeAttachment('a', 'valid.txt')]

    await expect(renderWithI18n(<AttachmentList attachments={attachments} />)).resolves.toBeTruthy()

    expect(screen.getByText('valid.txt')).toBeDefined()
  })

  it('renders the thumbnail in the pill but opens the full-resolution image in the lightbox', async () => {
    $previewTabs.set([])

    const image: ComposerAttachment = {
      id: 'img',
      kind: 'image',
      label: 'shot.png',
      path: '/tmp/shot.png',
      previewUrl: DATA_URL,
      thumbnailUrl: THUMBNAIL_URL
    }

    await renderWithI18n(<AttachmentList attachments={[image]} />)

    expect(screen.getByAltText<HTMLImageElement>('shot.png').getAttribute('src')).toBe(THUMBNAIL_URL)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /shot\.png/ }))
    })

    // The lightbox renders the full-size image in a dialog; the rail stays empty.
    const lightbox = await screen.findByRole('dialog')

    expect(lightbox.querySelector<HTMLImageElement>('img')?.src).toBe(DATA_URL)
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('loads a path-backed full image only when opened and releases it when closed', async () => {
    const readFileDataUrl = vi.fn(async () => DATA_URL)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const image: ComposerAttachment = {
      id: 'img-on-demand',
      kind: 'image',
      label: 'shot.png',
      path: '/tmp/shot.png',
      thumbnailUrl: THUMBNAIL_URL
    }

    const { container } = await renderWithI18n(<AttachmentList attachments={[image]} />)

    expect(readFileDataUrl).not.toHaveBeenCalled()
    expect(screen.getByAltText<HTMLImageElement>('shot.png').getAttribute('src')).toBe(THUMBNAIL_URL)
    expect(container.querySelector(`img[src="${DATA_URL}"]`)).toBeNull()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /shot\.png/ }))
    })

    expect(readFileDataUrl).toHaveBeenCalledOnce()
    const lightboxImage = (await screen.findByRole('dialog')).querySelector<HTMLImageElement>('img')

    expect(lightboxImage?.getAttribute('src')).toBe(DATA_URL)

    await act(async () => {
      fireEvent.click(lightboxImage!)
    })

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(container.querySelector(`img[src="${DATA_URL}"]`)).toBeNull()
  })

  it('falls back to the original host path after an image was staged for a different filesystem', async () => {
    const stagedPath = '/root/.hermes/attachments/photo.png'
    const hostPath = 'C:\\Users\\alice\\Pictures\\photo.png'

    const readFileDataUrl = vi.fn(async (path: string) => {
      if (path === hostPath) {
        return DATA_URL
      }

      throw new Error(`not readable: ${path}`)
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const image: ComposerAttachment = {
      attachedSessionId: 'session-1',
      detail: hostPath,
      id: 'image:photo.png',
      kind: 'image',
      label: 'photo.png',
      path: stagedPath,
      thumbnailUrl: THUMBNAIL_URL
    }

    await renderWithI18n(<AttachmentList attachments={[image]} />)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /photo\.png/ }))
    })

    expect(readFileDataUrl).toHaveBeenCalledWith(stagedPath)
    expect(readFileDataUrl).toHaveBeenCalledWith(hostPath)
    expect((await screen.findByRole('dialog')).querySelector<HTMLImageElement>('img')?.src).toBe(DATA_URL)
  })

  it('does not let an old occurrence open a replacement lightbox after a deferred read', async () => {
    let resolveOldRead!: (value: string) => void

    const oldRead = new Promise<string>(resolve => {
      resolveOldRead = resolve
    })

    const readFileDataUrl = vi.fn((path: string) =>
      path === '/tmp/old.png' ? oldRead : Promise.resolve('data:image/png;base64,replacement')
    )

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl }
    })

    const oldOccurrence: ComposerAttachment = {
      id: 'image:same-path',
      kind: 'image',
      label: 'same.png',
      occurrenceId: 'occurrence-old',
      path: '/tmp/old.png',
      thumbnailUrl: THUMBNAIL_URL
    }

    const replacement: ComposerAttachment = {
      ...oldOccurrence,
      occurrenceId: 'occurrence-replacement',
      path: '/tmp/replacement.png'
    }

    const { rerender } = await renderWithI18n(<AttachmentList attachments={[oldOccurrence]} />)

    fireEvent.click(screen.getByRole('button', { name: /same\.png/ }))
    expect(readFileDataUrl).toHaveBeenCalledWith('/tmp/old.png')

    rerender(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <AttachmentList attachments={[replacement]} />
      </I18nProvider>
    )

    await act(async () => {
      resolveOldRead(DATA_URL)
      await oldRead
    })

    expect(screen.queryByRole('dialog')).toBeNull()

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /same\.png/ }))
    })

    expect(readFileDataUrl).toHaveBeenCalledWith('/tmp/replacement.png')
    expect((await screen.findByRole('dialog')).querySelector<HTMLImageElement>('img')?.src).toBe(
      'data:image/png;base64,replacement'
    )
  })

  it('removes an attachment from the composer chip', async () => {
    const onRemove = vi.fn()

    await renderWithI18n(<AttachmentList attachments={[makeAttachment('a', 'doc.pdf')]} onRemove={onRemove} />)

    fireEvent.click(screen.getByRole('button', { name: 'Remove doc.pdf' }))
    expect(onRemove).toHaveBeenCalledWith('a')
  })

  it('still routes a non-image attachment to the preview rail', async () => {
    $previewTabs.set([])

    const file: ComposerAttachment = { id: 'doc', kind: 'file', label: 'notes.md', path: '/tmp/notes.md' }

    await renderWithI18n(<AttachmentList attachments={[file]} />)

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /notes\.md/ }))
    })

    expect(screen.queryByRole('dialog')).toBeNull()
    expect($previewTabs.get().map(tab => tab.target.path)).toEqual(['/tmp/notes.md'])
  })
})
