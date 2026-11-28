import type { ReactNode } from 'react'

import { pathExtension } from '@/lib/media'

/**
 * FILE PREVIEW RENDERERS — a plugin draws a file the preview rail opens.
 *
 * A claim names file extensions and MIME types. The first claim that matches
 * a file tab (lowest `order`, then contribution id) replaces the built-in
 * viewer, the binary/large refusal included; every other file keeps it.
 */

export const FILE_PREVIEW_RENDERER_AREA = 'preview.file'

export type FilePreviewReadResult =
  | { ok: true; bytes: Uint8Array; mimeType: string }
  | { ok: false; error: 'io-error' | 'missing' | 'scope-changed' | 'too-large' | 'unavailable'; message: string }

/** The file a preview tab shows, bound to the session and backend that own the tab. */
export interface FilePreviewFile {
  /** Path on the BACKEND's machine, as the tab names it. Never open it
   *  directly (the backend may be remote): use `read()`. */
  path: string
  name: string
  /** Lower-case, without the dot; `''` when the name has none. */
  extension: string
  /** The type the host reports for the tab, or null when unknown. */
  mimeType: null | string
  /** The chat that owns the tab, or null. */
  sessionId: null | string
  /** The whole file, read through the route the built-in viewer uses. Never
   *  rejects. `scope-changed` once the window's connection or profile moved,
   *  or the tab moved on to another file. */
  read: () => Promise<FilePreviewReadResult>
  /** Hand `data` to the OS save dialog under `name` (a basename, default the
   *  file's own). Never writes to `path`; the outcome is not reported. */
  saveCopy: (data: Blob | Uint8Array<ArrayBuffer>, name?: string) => void
}

export interface FilePreviewRendererProps {
  file: FilePreviewFile
  /** The built-in viewer for this file: return it for anything you don't draw. */
  fallback: ReactNode
}

/** Payload of a `preview.file` contribution's `data`. */
export interface FilePreviewRendererContribution {
  /** Case-insensitive, with or without the leading dot. */
  extensions?: readonly string[]
  /** Exact, case-insensitive. `application/octet-stream` never matches. */
  mimeTypes?: readonly string[]
  /** Mounted inside an error boundary whose fallback is the built-in viewer. */
  render: (props: FilePreviewRendererProps) => ReactNode
}

export interface FilePreviewClaim {
  id: string
  order?: number
  data?: unknown
}

const OCTET_STREAM = 'application/octet-stream'

/** A MIME type as claims compare it: no parameters, lower case, octet-stream as unknown. */
export function comparableMimeType(mimeType: null | string | undefined): null | string {
  const type = (mimeType ?? '').split(';', 1)[0]?.trim().toLowerCase() ?? ''

  return type && type !== OCTET_STREAM ? type : null
}

const optionalList = (value: unknown) => value === undefined || Array.isArray(value)

function isRendererContribution(data: unknown): data is FilePreviewRendererContribution {
  return (
    typeof data === 'object' &&
    data !== null &&
    'render' in data &&
    typeof data.render === 'function' &&
    optionalList('extensions' in data ? data.extensions : undefined) &&
    optionalList('mimeTypes' in data ? data.mimeTypes : undefined)
  )
}

function claimedExtensions(data: FilePreviewRendererContribution): string[] {
  return (data.extensions ?? [])
    .filter(value => typeof value === 'string')
    .map(value => value.replace(/^\./, '').toLowerCase())
}

function claimedMimeTypes(data: FilePreviewRendererContribution): (null | string)[] {
  return (data.mimeTypes ?? []).filter(value => typeof value === 'string').map(comparableMimeType)
}

/** The claim that draws `path`, and the extension it was matched by. */
export function resolveFilePreviewClaim<C extends FilePreviewClaim>(
  claims: readonly C[],
  path: string,
  mimeType: null | string
): { claim: C; data: FilePreviewRendererContribution; extension: string } | null {
  const valid = claims.flatMap(claim => (isRendererContribution(claim.data) ? [{ claim, data: claim.data }] : []))

  const known = new Set(valid.flatMap(({ data }) => claimedExtensions(data)))
  const extension = pathExtension(path, ext => known.has(ext)) ?? ''

  const match = valid
    .filter(
      ({ data }) =>
        (extension !== '' && claimedExtensions(data).includes(extension)) ||
        (mimeType !== null && claimedMimeTypes(data).includes(mimeType))
    )
    .sort(
      (a, b) =>
        (a.claim.order ?? 0) - (b.claim.order ?? 0) || Number(a.claim.id > b.claim.id) - Number(a.claim.id < b.claim.id)
    )[0]

  return match ? { ...match, extension } : null
}
