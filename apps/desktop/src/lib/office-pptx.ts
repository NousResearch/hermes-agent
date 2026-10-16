import { unzipSync } from 'fflate'

export interface SlidePreview {
  paragraphs: string[]
  images: string[]
}

const MAX_SLIDES = 40
const MAX_UNZIPPED_BYTES = 32 * 1024 * 1024

function textOf(bytes: Uint8Array | undefined): string {
  return bytes ? new TextDecoder().decode(bytes) : ''
}

function localName(element: Element): string {
  return element.localName
}

function descendants(root: ParentNode, name: string): Element[] {
  return Array.from(root.querySelectorAll('*')).filter(element => localName(element) === name)
}

function attribute(element: Element, name: string): string | null {
  for (const attr of element.attributes) {
    if (attr.localName === name) {
      return attr.value
    }
  }

  return null
}

function relationshipId(element: Element): string | null {
  for (const attr of element.attributes) {
    if (attr.localName === 'id' && /^rId/i.test(attr.value)) {
      return attr.value
    }
  }

  return null
}

function parseXml(xml: string): Document | null {
  if (!xml.trim()) {
    return null
  }

  const document = new DOMParser().parseFromString(xml, 'application/xml')

  return document.querySelector('parsererror') ? null : document
}

function entry(files: Record<string, Uint8Array>, path: string): Uint8Array | undefined {
  return files[path] ?? files[`/${path}`]
}

function zipPath(base: string, target: string): string {
  if (target.startsWith('/')) {
    return target.replace(/^\/+/, '')
  }

  const parts = base.split('/').slice(0, -1)

  for (const segment of target.split('/')) {
    if (segment === '..') {
      parts.pop()
    } else if (segment && segment !== '.') {
      parts.push(segment)
    }
  }

  return parts.join('/')
}

function relationships(files: Record<string, Uint8Array>, path: string): Map<string, string> {
  const document = parseXml(textOf(entry(files, path)))
  const targets = new Map<string, string>()

  if (!document) {
    return targets
  }

  for (const relationship of descendants(document, 'Relationship')) {
    const id = attribute(relationship, 'Id')
    const target = attribute(relationship, 'Target')

    if (id && target) {
      targets.set(id, zipPath(path.replace(/_rels\/[^/]+\.rels$/, ''), target))
    }
  }

  return targets
}

function dataUrl(bytes: Uint8Array, path: string): string | null {
  const ext = path.split('.').pop()?.toLowerCase()

  const mime =
    ext === 'png'
      ? 'image/png'
      : ext === 'jpg' || ext === 'jpeg'
        ? 'image/jpeg'
        : ext === 'gif'
          ? 'image/gif'
          : ext === 'webp'
            ? 'image/webp'
            : null

  if (!mime) {
    return null
  }

  let binary = ''

  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000))
  }

  return `data:${mime};base64,${btoa(binary)}`
}

function slideText(document: Document): string[] {
  return descendants(document, 'p')
    .map(paragraph =>
      descendants(paragraph, 't')
        .map(node => node.textContent ?? '')
        .join('')
        .trim()
    )
    .filter(Boolean)
}

function slideImages(document: Document, files: Record<string, Uint8Array>, slidePath: string): string[] {
  const rels = relationships(files, slidePath.replace(/([^/]+)$/, '_rels/$1.rels'))
  const images: string[] = []

  for (const blip of descendants(document, 'blip')) {
    const id = attribute(blip, 'embed')
    const path = id ? rels.get(id) : undefined
    const bytes = path ? entry(files, path) : undefined
    const url = bytes && path ? dataUrl(bytes, path) : null

    if (url) {
      images.push(url)
    }
  }

  return images
}

export function parsePptx(bytes: Uint8Array): SlidePreview[] {
  let files: Record<string, Uint8Array>

  try {
    files = unzipSync(bytes)
  } catch {
    throw new Error('Could not read this presentation')
  }

  let inflated = 0

  for (const part of Object.values(files)) {
    inflated += part.length

    if (inflated > MAX_UNZIPPED_BYTES) {
      throw new Error('Presentation is too large to preview')
    }
  }

  const presentation = parseXml(textOf(entry(files, 'ppt/presentation.xml')))
  const rels = relationships(files, 'ppt/_rels/presentation.xml.rels')
  const slideIds = presentation ? descendants(presentation, 'sldId') : []

  const paths = slideIds
    .map(slide => {
      const id = relationshipId(slide)

      return id ? rels.get(id) : undefined
    })
    .filter((path): path is string => Boolean(path))

  const fallback = Object.keys(files)
    .filter(path => /ppt\/slides\/slide\d+\.xml$/i.test(path))
    .sort()

  const slides = (paths.length ? paths : fallback).slice(0, MAX_SLIDES).map(path => {
    const document = parseXml(textOf(entry(files, path)))

    if (!document) {
      return { paragraphs: [], images: [] }
    }

    return { paragraphs: slideText(document), images: slideImages(document, files, path) }
  })

  if (!slides.length) {
    throw new Error('Could not read this presentation')
  }

  return slides
}
