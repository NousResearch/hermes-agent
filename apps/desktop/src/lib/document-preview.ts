import type { HermesPdfDocument, HermesTexCompileResult } from '@/global'

import { desktopFsProfile, isDesktopFsRemoteMode } from './desktop-fs'

function bridge() {
  if (!window.hermesDesktop) {
    throw new Error('Hermes Desktop bridge is unavailable')
  }

  return window.hermesDesktop
}

function decodeBytes(value: string | Uint8Array): Uint8Array {
  if (value instanceof Uint8Array) {
    return value
  }

  const binary = window.atob(value)
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return bytes
}

function normalizeDocument(
  value: Omit<HermesPdfDocument, 'initialData'> & { initialData: string | Uint8Array },
  transport: HermesPdfDocument['transport']
): HermesPdfDocument {
  return { ...value, initialData: decodeBytes(value.initialData), transport }
}

async function remoteApi<T>(path: string, body: Record<string, unknown>): Promise<T> {
  return bridge().api<T>({ body, method: 'POST', path, profile: desktopFsProfile() })
}

export async function openPdfDocument(path: string): Promise<HermesPdfDocument> {
  if (!isDesktopFsRemoteMode()) {
    return normalizeDocument(await bridge().pdf.open(path), 'local')
  }

  return normalizeDocument(
    await remoteApi<Omit<HermesPdfDocument, 'initialData'> & { initialData: string }>('/api/preview/pdf/open', {
      path
    }),
    'remote'
  )
}

export async function readPdfDocumentRange(
  descriptor: HermesPdfDocument,
  begin: number,
  end: number
): Promise<Uint8Array> {
  if (descriptor.transport !== 'remote') {
    return bridge().pdf.readRange(descriptor.id, begin, end)
  }

  const result = await remoteApi<{ data: string }>('/api/preview/pdf/range', {
    begin,
    end,
    id: descriptor.id,
    revision: descriptor.revision
  })

  return decodeBytes(result.data)
}

export async function closePdfDocument(descriptor: HermesPdfDocument): Promise<boolean> {
  if (descriptor.transport !== 'remote') {
    return bridge().pdf.close(descriptor.id)
  }

  const result = await remoteApi<{ closed: boolean }>('/api/preview/pdf/close', { id: descriptor.id })

  return result.closed
}

function normalizeCompileResult(result: HermesTexCompileResult): HermesTexCompileResult {
  return {
    ...result,
    pdfDocument: result.pdfDocument
      ? normalizeDocument(result.pdfDocument, isDesktopFsRemoteMode() ? 'remote' : 'local')
      : undefined
  }
}

export async function compileTexDocument(
  path: string,
  requestId: string,
  workspaceRoot?: string
): Promise<HermesTexCompileResult> {
  if (!isDesktopFsRemoteMode()) {
    return normalizeCompileResult(await bridge().texPreview.compile({ path, requestId, workspaceRoot }))
  }

  return normalizeCompileResult(
    await remoteApi<HermesTexCompileResult>('/api/preview/tex/compile', { path, requestId, workspaceRoot })
  )
}

export async function cancelTexDocument(requestId: string): Promise<void> {
  if (!isDesktopFsRemoteMode()) {
    bridge().texPreview.cancel(requestId)

    return
  }

  await remoteApi('/api/preview/tex/cancel', { requestId })
}
