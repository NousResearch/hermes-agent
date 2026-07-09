import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { ComposerAttachment } from '@/store/composer'

import { AttachmentList } from './attachments'

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
})
