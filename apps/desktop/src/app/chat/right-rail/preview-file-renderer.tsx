import { useStore } from '@nanostores/react'
import { type ReactNode, useLayoutEffect, useMemo, useRef } from 'react'

import { $apiRequestScope } from '@/api/client'
import { ErrorBoundary } from '@/components/error-boundary'
import { useContributions } from '@/contrib'
import { DesktopFileMissingError, fileEditScopeKey, readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { downloadBlob } from '@/lib/download'
import {
  comparableMimeType,
  FILE_PREVIEW_RENDERER_AREA,
  type FilePreviewFile,
  type FilePreviewReadResult,
  type FilePreviewRendererContribution,
  type FilePreviewRendererProps,
  resolveFilePreviewClaim
} from '@/lib/file-preview-renderers'
import { filePathForTarget } from '@/lib/local-preview'
import type { PreviewTarget } from '@/store/preview'
import { $connection } from '@/store/session'

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

  // Layout, not passive: StrictMode's replay re-runs this before any child's
  // passive effect, so a renderer that reads on mount never sees a retired handle.
  useLayoutEffect(() => {
    current.current = file

    return () => {
      if (current.current === file) {
        current.current = null
      }
    }
  }, [file])

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
  // One component per callback: a replaced renderer starts with fresh hooks,
  // and its own state updates re-run it.
  const Host = useMemo(
    () =>
      function FilePreviewRendererHost(props: FilePreviewRendererProps) {
        return <>{render(props) ?? props.fallback}</>
      },
    [render]
  )

  return <Host fallback={fallback} file={file} />
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

  const binary = atob(match[2] ?? '')
  const bytes = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }

  return {
    ok: true,
    bytes,
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
