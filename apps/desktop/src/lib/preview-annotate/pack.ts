import { type CompactIdentity, formatIdentityLine } from './identity'
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

function cssBlock(identity: CompactIdentity): string {
  const entries = Object.entries(identity.css)

  if (!entries.length) {
    return ''
  }

  return `Styles: ${entries.map(([name, value]) => `${name}: ${value}`).join('; ')}`
}

/**
 * One comment, as much as the agent needs to find the element in source.
 *
 * The human-readable target line stays first and stays prose — it is what the
 * user actually pointed at. Selector, markup, and computed styles follow as
 * labelled lines, because the crop shows what is wrong and the DOM shows where
 * it lives; an agent given only the picture greps for the wrong div. Area pins
 * have no element, so they get the crop and the note and nothing invented.
 */
export function packageAnnotatePin(pin: AnnotatePin): ComposerReadyAnnotation {
  const target = identityBlock(pin)
  const note = pin.note.trim()
  const identity = pin.identity

  const prompt = [
    `Comment ${pin.number}`,
    `Target: ${target}`,
    identity?.selector ? `Selector: ${identity.selector}` : '',
    identity?.html ? `HTML: ${identity.html}` : '',
    identity ? cssBlock(identity) : '',
    note ? `Note: ${note}` : '',
    `Image ${pin.number} marks the target in blue.`
  ]
    .filter(Boolean)
    .join('\n')

  return {
    identity,
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
