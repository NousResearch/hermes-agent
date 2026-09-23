import { annotateFlushPrompt, dataUrlToFile, packageAnnotateStack } from './pack'
import type { AnnotatePin } from './stack'

export interface AnnotateFlushPorts {
  attachImage: (blob: Blob) => void | Promise<void>
  insertText: (text: string) => void
  send?: () => void
}

export interface AnnotateFlushResult {
  count: number
  prompt: string
  sent: false
}

/**
 * Attach every pin to the composer as a numbered crop plus the packed prompt.
 * Saving / flushing never submits a turn — `send` is accepted so tests can
 * prove it stays untouched.
 */
export async function flushAnnotateStack(
  pins: readonly AnnotatePin[],
  ports: AnnotateFlushPorts,
  pageUrl?: string
): Promise<AnnotateFlushResult> {
  const items = packageAnnotateStack(pins)

  for (const item of items) {
    if (item.imageDataUrl) {
      await ports.attachImage(dataUrlToFile(item.imageDataUrl, `Comment_${item.number}.png`))
    }
  }

  const prompt = annotateFlushPrompt(items, pageUrl)

  if (prompt.trim()) {
    ports.insertText(prompt)
  }

  return { count: items.length, prompt, sent: false }
}
