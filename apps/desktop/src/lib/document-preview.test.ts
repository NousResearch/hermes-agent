import { afterEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $connection } from '@/store/session'

import { closePdfDocument, openPdfDocument, readPdfDocumentRange } from './document-preview'

afterEach(() => {
  $connection.set(null)
  vi.restoreAllMocks()
})

it('uses authenticated REST and decodes remote PDF ranges', async () => {
  $connection.set({ baseUrl: 'https://agent.example', mode: 'remote', profile: 'research' } as HermesConnection)

  const api = vi.fn(async (request: { path: string }) => {
    if (request.path.endsWith('/open')) {
      return {
        byteLength: 4,
        id: 'remote-document',
        initialData: btoa(String.fromCharCode(1, 2)),
        modifiedAt: 1,
        revision: '4:1'
      }
    }

    if (request.path.endsWith('/range')) {
      return { data: btoa(String.fromCharCode(3, 4)) }
    }

    return { closed: true }
  })

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  const descriptor = await openPdfDocument('/remote/paper.pdf')

  expect(descriptor.initialData).toEqual(new Uint8Array([1, 2]))
  expect(descriptor.transport).toBe('remote')
  expect(await readPdfDocumentRange(descriptor, 2, 4)).toEqual(new Uint8Array([3, 4]))
  expect(await closePdfDocument(descriptor)).toBe(true)
  expect(api).toHaveBeenCalledWith(
    expect.objectContaining({ method: 'POST', path: '/api/preview/pdf/open', profile: 'research' })
  )
})
