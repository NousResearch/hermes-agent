import { downloadBlob } from '@/lib/download'

/** Save generated text content to a file via a blob download. */
export function downloadTextFile(name: string, content: string, mimeType = 'text/plain') {
  downloadBlob(new Blob([content], { type: `${mimeType};charset=utf-8` }), name)
}
