import {
  copyTextToClipboard,
  desktopFilesystemKey,
  isDesktopFsRemoteMode,
  readDesktopFileText,
  revealDesktopPath
} from '@/lib/desktop-fs'
import { markdownArtifactTargetFromHref } from '@/lib/preview-targets'
import type { PreviewTarget } from '@/store/preview'

export type ArtifactAction = 'copy-contents' | 'copy-path' | 'open' | 'reveal'

function artifactPath(target: PreviewTarget): string {
  return target.path || target.source
}

function assertMarkdownArtifact(target: PreviewTarget): string {
  const path = artifactPath(target)

  if (target.kind !== 'file' || !target.artifact || !markdownArtifactTargetFromHref(path)) {
    throw new Error('This target is not a Markdown artifact')
  }

  return path
}

function isAbsoluteArtifactPath(path: string): boolean {
  return path.startsWith('/') || /^[a-z]:[\\/]/i.test(path)
}

async function validateLocalArtifactTarget(target: PreviewTarget): Promise<PreviewTarget> {
  const normalized = await window.hermesDesktop.normalizePreviewTarget(target.url || artifactPath(target))

  if (
    !normalized ||
    normalized.kind !== 'file' ||
    !markdownArtifactTargetFromHref(normalized.path || normalized.source)
  ) {
    throw new Error('Hermes could not validate this local Markdown artifact')
  }

  return normalized
}

export function artifactActionAvailable(action: ArtifactAction, target: PreviewTarget): boolean {
  if (!target.filesystemKey || target.filesystemKey !== desktopFilesystemKey()) {
    return false
  }

  return !isDesktopFsRemoteMode() || (action !== 'open' && action !== 'reveal')
}

function assertArtifactActionStillAvailable(
  action: ArtifactAction,
  target: PreviewTarget,
  filesystemKey: string | undefined
): void {
  if (!filesystemKey || target.filesystemKey !== filesystemKey || desktopFilesystemKey() !== filesystemKey) {
    throw new Error('This artifact belongs to a different filesystem or is not available as a local file')
  }

  if (!artifactActionAvailable(action, target)) {
    throw new Error('This artifact belongs to a different filesystem or is not available as a local file')
  }
}

export async function performArtifactAction(action: ArtifactAction, target: PreviewTarget): Promise<void> {
  const path = assertMarkdownArtifact(target)
  const filesystemKey = target.filesystemKey

  assertArtifactActionStillAvailable(action, target, filesystemKey)

  if (action === 'open') {
    const normalized = await validateLocalArtifactTarget(target)
    assertArtifactActionStillAvailable(action, target, filesystemKey)
    const url = new URL(normalized.url)

    if (url.protocol !== 'file:') {
      throw new Error('Only local file URLs can be opened')
    }

    await window.hermesDesktop.openExternal(url.toString())

    return
  }

  if (action === 'reveal') {
    const normalized = await validateLocalArtifactTarget(target)
    assertArtifactActionStillAvailable(action, target, filesystemKey)
    await revealDesktopPath(normalized.path || normalized.source)

    return
  }

  if (action === 'copy-path') {
    const resolvedPath = isAbsoluteArtifactPath(path) ? path : (await readDesktopFileText(path)).path

    assertArtifactActionStillAvailable(action, target, filesystemKey)

    if (!resolvedPath || !isAbsoluteArtifactPath(resolvedPath)) {
      throw new Error('Hermes could not resolve an absolute artifact path')
    }

    await copyTextToClipboard(resolvedPath)

    return
  }

  const result = await readDesktopFileText(path, { complete: true })

  assertArtifactActionStillAvailable(action, target, filesystemKey)

  if (result.binary) {
    throw new Error('Binary files cannot be copied as Markdown')
  }

  if (result.truncated) {
    throw new Error('Hermes could not read the complete file contents')
  }

  await copyTextToClipboard(result.text)
}
