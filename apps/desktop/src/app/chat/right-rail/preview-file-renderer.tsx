import { useStore } from '@nanostores/react'
import { type ReactNode, useMemo, useRef } from 'react'

import { $apiRequestScope } from '@/api/client'
import { ErrorBoundary } from '@/components/error-boundary'
import { useContributions } from '@/contrib'
import { DesktopFileMissingError, readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { downloadBlob } from '@/lib/download'
import {
  comparableMimeType,
  FILE_PREVIEW_RENDERER_AREA,
  type FilePreviewFile,
  type FilePreviewReadResult,
  type FilePreviewRendererContribution,
  resolveFilePreviewClaim
} from '@/lib/file-preview-renderers'
import type { PreviewTarget } from '@/store/preview'
import { $connection } from '@/store/session'

import { fileEditScopeKey, filePathForTarget } from './preview-file'

interface FilePreviewSlotProps {
  /** The built-in viewer, mounted only when no plugin claims the file. */
  fallback: ReactNode
  sessionId?: string
  target: PreviewTarget
}

// One component on purpose: the built-in viewer must keep its tree position
// when renderers come and go, or every open file tab remounts and re-reads.
export function FilePreviewSlot({ fallback, sessionId, target }: FilePreviewSlotProps) {
  const claims = useContributions(FILE_PREVIEW_RENDERER_AREA)
  useStore($connection)
  useStore($apiRequestScope)
  const scope = fileEditScopeKey()
  const path = filePathForTarget(target)
  const mimeType = comparableMimeType(target.mimeType)
  const match = target.kind === 'file' ? resolveFilePreviewClaim(claims, path, mimeType) : null
  const extension = match?.extension ?? ''
  const current = useRef<FilePreviewFile | null>(null)

  const file = useMemo(
    () =>
      filePreviewHandle({
        extension,
        isCurrent: handle => current.current === handle,
        mimeType,
        path,
        scope,
        sessionId
      }),
    [extension, mimeType, path, scope, sessionId]
  )

  current.current = file

  if (!match) {
    return <>{fallback}</>
  }

  return (
    <ErrorBoundary fallback={() => fallback} key={`${match.claim.id}\u0000${path}`} label={`contrib:${match.claim.id}`}>
      <FilePreviewLeaf fallback={fallback} file={file} render={match.data.render} />
    </ErrorBoundary>
  )
}

function FilePreviewLeaf({
  fallback,
  file,
  render
}: {
  fallback: ReactNode
  file: FilePreviewFile
  render: FilePreviewRendererContribution['render']
}) {
  return <>{render({ fallback, file }) ?? fallback}</>
}

const baseName = (value: string) => value.split(/[\\/]/).filter(Boolean).pop() ?? ''

function filePreviewHandle({
  extension,
  isCurrent,
  mimeType,
  path,
  scope,
  sessionId
}: {
  extension: string
  isCurrent: (handle: FilePreviewFile) => boolean
  mimeType: null | string
  path: string
  scope: string
  sessionId?: string
}): FilePreviewFile {
  const name = baseName(path) || path

  const scopeChanged: FilePreviewReadResult = {
    ok: false,
    error: 'scope-changed',
    message: 'The preview moved to another file, connection or profile.'
  }

  const owned = () => isCurrent(handle) && fileEditScopeKey() === scope

  const handle: FilePreviewFile = {
    extension,
    mimeType,
    name,
    path,
    sessionId: sessionId ?? null,

    async read() {
      if (!owned()) {
        return scopeChanged
      }

      if (!window.hermesDesktop) {
        return { ok: false, error: 'unavailable', message: 'Hermes Desktop bridge is unavailable' }
      }

      try {
        const dataUrl = await readDesktopFileDataUrl(path, { sessionId })

        return owned() ? decodeDataUrl(dataUrl) : scopeChanged
      } catch (error) {
        return owned() ? readFailure(error) : scopeChanged
      }
    },

    saveCopy(data, saveName) {
      downloadBlob(data instanceof Blob ? data : new Blob([data]), baseName(saveName ?? '') || name)
    }
  }

  return handle
}

function decodeDataUrl(dataUrl: string): FilePreviewReadResult {
  const match = /^data:([^,;]*)[^,]*;base64,(.*)$/s.exec(dataUrl)

  if (!match) {
    return { ok: false, error: 'io-error', message: 'The file reader returned no data.' }
  }

  return {
    ok: true,
    bytes: Uint8Array.from(atob(match[2] ?? ''), char => char.charCodeAt(0)),
    mimeType: match[1] || 'application/octet-stream'
  }
}

function readFailure(error: unknown): FilePreviewReadResult {
  const message = error instanceof Error ? error.message : String(error)

  if (error instanceof DesktopFileMissingError || /(?:^|: )404: /.test(message)) {
    return { ok: false, error: 'missing', message }
  }

  const tooLarge = /(?:^|: )413: |failed: file is too large/.test(message)

  return { ok: false, error: tooLarge ? 'too-large' : 'io-error', message }
}
