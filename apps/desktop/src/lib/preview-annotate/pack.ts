import { type AnnotateGroup, groupAnnotations } from './group'
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

/** Below this a flat list is easier to read than a set of headed sections. */
const GROUP_THRESHOLD = 4

function groupHeading(group: AnnotateGroup, index: number): string {
  const what = group.label ? `\`${group.label}\`` : 'Unanchored (dragged areas)'

  return `Group ${index + 1} — ${what} (${group.items.length} comment${group.items.length === 1 ? '' : 's'})`
}

/**
 * How to work a batch this size.
 *
 * Two things the model gets wrong when handed a long flat list: it makes one
 * task per comment and grinds through them serially, and — told to parallelize
 * — it splits by theme, which puts several workers in the same component. So
 * say both. The groups below are structural (disjoint DOM subtrees, so usually
 * disjoint files), which is what makes handing them out concurrently safe;
 * "all the styling ones" is not.
 *
 * It stays advice, not instruction: the model can see whether these comments
 * are really one refactor, and a grouping computed from selectors cannot.
 */
function batchGuidance(groupCount: number, total: number): string {
  return [
    `These ${total} comments are pre-grouped by where they sit in the page — each group is a different part of the DOM, so the groups should touch mostly separate files.`,
    `Work them as ${groupCount} pieces of work, not ${total}. Fold comments in the same group into one change.`,
    'If you delegate, delegate whole groups — never split one group across workers, and never form new groups by theme (all the spacing ones, all the copy ones): those cut across the same files and the workers will collide.',
    'Regroup if the code disagrees with this split — it is derived from the page structure, not from your source layout.'
  ].join(' ')
}

export function annotateFlushPrompt(items: readonly ComposerReadyAnnotation[], pageUrl?: string): string {
  const where = pageUrl ? ` on ${pageUrl}` : ''
  const count = items.length

  if (count === 1) {
    return [
      `I left a comment${where} in the in-app browser. Address it and keep the scope narrow.`,
      '',
      ...items.map(item => item.prompt)
    ].join('\n')
  }

  const groups = groupAnnotations(items)

  if (count < GROUP_THRESHOLD || groups.length < 2) {
    return [
      `I left ${count} comments${where} in the in-app browser. Address them and keep the scope narrow.`,
      '',
      ...items.map(item => item.prompt)
    ].join('\n')
  }

  const sections = groups.flatMap((group, index) => [
    groupHeading(group, index),
    ...group.items.map(item => item.prompt),
    ''
  ])

  return [
    `I left ${count} comments${where} in the in-app browser. Address them and keep the scope narrow.`,
    batchGuidance(groups.length, count),
    '',
    ...sections
  ]
    .join('\n')
    .trimEnd()
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
