import type { MessageDetails } from 'electron'

function formatRendererConsoleError(
  details: Pick<MessageDetails, 'level' | 'lineNumber' | 'message' | 'sourceUrl'> | null | undefined
): string | null {
  if (!details || details.level !== 3) {
    return null
  }

  return `[renderer console] ${details.message} (${details.sourceUrl}:${details.lineNumber})`
}

export { formatRendererConsoleError }
