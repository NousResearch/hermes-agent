import type { WebContentsConsoleMessageEventParams } from 'electron'

function formatRendererConsoleError(
  details: Pick<WebContentsConsoleMessageEventParams, 'level' | 'lineNumber' | 'message' | 'sourceId'> | null | undefined
): string | null {
  if (!details || details.level !== 'error') {
    return null
  }

  return `[renderer console] ${details.message} (${details.sourceId}:${details.lineNumber})`
}

export { formatRendererConsoleError }
