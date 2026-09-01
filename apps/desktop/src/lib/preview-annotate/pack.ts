import { formatIdentityLine, type CompactIdentity } from './identity'
import type { AnnotatePin } from './stack'

export interface ComposerReadyAnnotation {
  identity?: CompactIdentity
  imageDataUrl: string
  note: string
  number: number
  prompt: string
}

function identityBlock(pin: AnnotatePin): string {
  if (!pin.identity) {
    return `area on the page (${Math.round(pin.rect.width)}×${Math.round(pin.rect.height)}px)`
  }

  return formatIdentityLine(pin.identity)
}

export function packageAnnotatePin(pin: AnnotatePin): ComposerReadyAnnotation {
  const target = identityBlock(pin)
  const note = pin.note.trim()
  const prompt = [
    `Comment ${pin.number}`,
    `Target: ${target}`,
    note ? `Note: ${note}` : '',
    `Image ${pin.number} marks the target in blue.`
  ]
    .filter(Boolean)
    .join('\n')

  return {
    identity: pin.identity,
    imageDataUrl: pin.imageDataUrl,
    note,
    number: pin.number,
    prompt
  }
}

export function packageAnnotateStack(pins: readonly AnnotatePin[]): ComposerReadyAnnotation[] {
  return pins.map(packageAnnotatePin)
}

export function annotateFlushPrompt(items: readonly ComposerReadyAnnotation[], pageUrl?: string): string {
  const where = pageUrl ? ` on ${pageUrl}` : ''
  const count = items.length
  const header =
    count === 1
      ? `I left a comment${where} in the in-app browser. Address it and keep the scope narrow.`
      : `I left ${count} comments${where} in the in-app browser. Address them and keep the scope narrow.`

  return [header, '', ...items.map(item => item.prompt)].join('\n')
}

export function dataUrlToBlob(dataUrl: string): Blob {
  const comma = dataUrl.indexOf(',')
  const head = comma >= 0 ? dataUrl.slice(0, comma) : 'data:image/png;base64'
  const body = comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl
  const mime = /data:([^;]+)/.exec(head)?.[1] || 'image/png'
  const binary = atob(body)
  const bytes = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }

  return new Blob([bytes], { type: mime })
}

export function dataUrlToFile(dataUrl: string, name: string): File {
  const blob = dataUrlToBlob(dataUrl)

  return new File([blob], name, { type: blob.type })
}
