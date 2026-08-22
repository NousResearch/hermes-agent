import type { ComposerAttachment } from '@/store/composer'

const AUTHORIZATION_TTL_MS = 30_000
const WINDOWS_RESERVED_NAME = /^(?:aux|con|nul|prn|com[1-9]|lpt[1-9])(?:\.|$)/i
const SAFE_MEDIA_TYPE = /^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/i

interface AttachmentGrant {
  authorizedAt: number
  changed: () => boolean
  expiresAt: number
  source: ComposerAttachment
  sourceShape: AttachmentShape
  shape: AttachmentShape
}

const authorized = new WeakMap<ComposerAttachment, AttachmentGrant>()

interface AttachmentShape {
  descriptors: Array<[PropertyKey, PropertyDescriptor]>
  prototype: object | null
}

export interface HostComposerAttachmentAuthority {}

interface HostComposerAttachmentAuthorityState {
  shapes: WeakMap<ComposerAttachment, AttachmentShape>
}

const hostAuthorityStates = new WeakMap<HostComposerAttachmentAuthority, HostComposerAttachmentAuthorityState>()

function captureAttachmentShape(attachment: ComposerAttachment): AttachmentShape | null {
  if (Object.getPrototypeOf(attachment) !== Object.prototype) {
    return null
  }

  return {
    descriptors: Reflect.ownKeys(attachment).map(key => [key, Object.getOwnPropertyDescriptor(attachment, key)!]),
    prototype: Object.getPrototypeOf(attachment)
  }
}

function attachmentMatchesShape(attachment: ComposerAttachment, shape: AttachmentShape): boolean {
  if (Object.getPrototypeOf(attachment) !== shape.prototype) {
    return false
  }

  const keys = Reflect.ownKeys(attachment)
  if (keys.length !== shape.descriptors.length || keys.some((key, index) => key !== shape.descriptors[index]?.[0])) {
    return false
  }

  return shape.descriptors.every(([key, expected]) => {
    const actual = Object.getOwnPropertyDescriptor(attachment, key)
    if (!actual) {
      return false
    }
    if (
      actual.configurable !== expected.configurable ||
      actual.enumerable !== expected.enumerable ||
      ('writable' in actual && actual.writable !== expected.writable)
    ) {
      return false
    }
    if ('value' in expected) {
      return 'value' in actual && Object.is(actual.value, expected.value)
    }

    return actual.get === expected.get && actual.set === expected.set
  })
}

export function captureHostComposerAttachmentAuthority(
  attachments: readonly ComposerAttachment[]
): HostComposerAttachmentAuthority {
  const shapes = new WeakMap<ComposerAttachment, AttachmentShape>()
  for (const attachment of attachments) {
    const shape = captureAttachmentShape(attachment)
    if (shape) {
      shapes.set(attachment, shape)
    }
  }

  const authority = Object.freeze(Object.create(null)) as HostComposerAttachmentAuthority
  hostAuthorityStates.set(authority, { shapes })

  return authority
}

function assertAuthorized(attachment: ComposerAttachment, now: number): AttachmentGrant {
  const grant = authorized.get(attachment)

  if (!grant || now > grant.expiresAt) {
    throw new Error('attachment is not an authorized composer attachment')
  }

  if (
    grant.changed() ||
    !attachmentMatchesShape(attachment, grant.shape) ||
    !attachmentMatchesShape(grant.source, grant.sourceShape)
  ) {
    throw new Error('composer attachment changed after authorization')
  }

  return grant
}

function safeAttachmentName(label: string): string {
  if (
    !label ||
    label.length > 255 ||
    [...label].some(character => {
      const code = character.charCodeAt(0)
      return code < 32 || code === 127
    }) ||
    /[\\/:]/.test(label) ||
    label === '.' ||
    label === '..' ||
    label.endsWith('.') ||
    label.endsWith(' ') ||
    WINDOWS_RESERVED_NAME.test(label)
  ) {
    throw new Error('invalid attachment name')
  }

  return label
}

function exactBytes(value: unknown): Uint8Array {
  if (!ArrayBuffer.isView(value) || (value as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT !== 1) {
    throw new Error('attachment reader did not return bytes')
  }

  const view = value as Uint8Array

  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer)

  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('')
}

export interface RelayAttachmentTarget {
  connectionId: string
  profile: string
  runtimeSessionId: string
  storedSessionId: string
  lineageRootId: string
}

export interface RelayStageInput {
  bytes: Uint8Array
  kind: 'file' | 'image'
  mediaType: string
  name: string
  order: number
  sha256: string
}

export interface RelayStageReceipt {
  attached: boolean
  bytes: number
  mediaType: string
  name: string
  order: number
  refText?: string
  runtimeSessionId: string
  sha256: string
  storedName: string
}

export interface RelayedAttachment {
  id: string
  kind: 'file' | 'image'
  mediaType: string
  name: string
  occurrenceId: null | string
  order: number
  provenance: {
    authorizedAt: number
    kind: 'composer'
    occurrenceId: null | string
    sourceId: string
  }
  refText?: string
  runtimeSessionId: string
  sha256: string
  size: number
  storedName: string
}

export interface RelayAttachmentDeps {
  maxBytes: number
  now: () => number
  read: (attachment: ComposerAttachment) => Promise<{ bytes: Uint8Array; mediaType: string }>
  revalidate: (target: RelayAttachmentTarget) => Promise<void>
  stage: (target: RelayAttachmentTarget, input: RelayStageInput) => Promise<RelayStageReceipt>
}

const ATTACHMENT_KEYS = [
  'id',
  'occurrenceId',
  'kind',
  'label',
  'detail',
  'refText',
  'previewUrl',
  'thumbnailUrl',
  'path',
  'attachedSessionId',
  'uploadState'
] as const

function copyAttachmentData(attachment: ComposerAttachment): ComposerAttachment {
  const copy: Record<string, unknown> = {}
  for (const key of ATTACHMENT_KEYS) {
    if (Object.prototype.hasOwnProperty.call(attachment, key)) {
      copy[key] = attachment[key]
    }
  }

  return copy as unknown as ComposerAttachment
}

function trackedAttachmentView(attachment: ComposerAttachment): {
  changed: () => boolean
  view: ComposerAttachment
} {
  let changed = false
  const target = copyAttachmentData(attachment)
  const mark = () => {
    changed = true
  }
  let view: ComposerAttachment
  view = new Proxy(target, {
    defineProperty(current, key, descriptor) {
      mark()
      return Reflect.defineProperty(current, key, descriptor)
    },
    deleteProperty(current, key) {
      mark()
      return Reflect.deleteProperty(current, key)
    },
    preventExtensions(current) {
      mark()
      return Reflect.preventExtensions(current)
    },
    set(current, key, value, receiver) {
      if (receiver === view) {
        mark()
      }
      return Reflect.set(current, key, value, receiver)
    },
    setPrototypeOf(current, prototype) {
      mark()
      return Reflect.setPrototypeOf(current, prototype)
    }
  })

  return { changed: () => changed, view }
}

export interface ComposerAttachmentAttempt {
  attachments: ComposerAttachment[]
  originForUnchangedView: (attachment: ComposerAttachment) => ComposerAttachment | null
  release: () => void
}

export function createComposerAttachmentAttempt(
  authority: HostComposerAttachmentAuthority,
  attachments: readonly ComposerAttachment[],
  origins: readonly (ComposerAttachment | null)[],
  options: { now?: () => number; ttlMs?: number } = {}
): ComposerAttachmentAttempt {
  const authorityState = hostAuthorityStates.get(authority)
  if (!authorityState || attachments.length !== origins.length) {
    throw new Error('invalid host composer attachment authority')
  }
  const now = options.now ?? Date.now
  const authorizedAt = now()
  const expiresAt = authorizedAt + (options.ttlMs ?? AUTHORIZATION_TTL_MS)
  const views: ComposerAttachment[] = []
  const attemptOrigins = new WeakMap<ComposerAttachment, ComposerAttachment>()

  for (const [index, attachment] of attachments.entries()) {
    const tracked = trackedAttachmentView(attachment)
    const origin = origins[index]
    const sourceShape = origin ? authorityState.shapes.get(origin) : undefined
    const viewShape = captureAttachmentShape(tracked.view)
    if (origin && sourceShape && viewShape && attachmentMatchesShape(origin, sourceShape)) {
      authorized.set(tracked.view, {
        authorizedAt,
        changed: tracked.changed,
        expiresAt,
        shape: viewShape,
        source: origin,
        sourceShape
      })
      attemptOrigins.set(tracked.view, origin)
    }
    views.push(tracked.view)
  }

  return {
    attachments: views,
    originForUnchangedView: attachment => {
      const origin = attemptOrigins.get(attachment)
      if (!origin) {
        return null
      }
      try {
        assertAuthorized(attachment, now())
        return origin
      } catch {
        return null
      }
    },
    release: () => {
      for (const attachment of views) {
        authorized.delete(attachment)
      }
    }
  }
}

export function composerAttachmentsAreAuthorized(
  attachments: readonly ComposerAttachment[],
  now = Date.now()
): boolean {
  try {
    for (const attachment of attachments) {
      assertAuthorized(attachment, now)
    }
    return true
  } catch {
    return false
  }
}

export async function relayComposerAttachments(
  target: RelayAttachmentTarget,
  attachments: readonly ComposerAttachment[],
  deps: RelayAttachmentDeps
): Promise<RelayedAttachment[]> {
  await deps.revalidate(target)
  const relayed: RelayedAttachment[] = []

  for (const [order, attachment] of attachments.entries()) {
    const grant = assertAuthorized(attachment, deps.now())
    const source = grant.source

    if (source.kind !== 'file' && source.kind !== 'image') {
      throw new Error('unsupported composer attachment kind')
    }

    const name = safeAttachmentName(source.label)
    await deps.revalidate(target)
    assertAuthorized(attachment, deps.now())
    const read = await deps.read(source)
    assertAuthorized(attachment, deps.now())
    await deps.revalidate(target)

    const bytes = exactBytes(read.bytes)

    if (!bytes.byteLength || bytes.byteLength > deps.maxBytes) {
      throw new Error('attachment exceeds the size limit')
    }
    if (!SAFE_MEDIA_TYPE.test(read.mediaType)) {
      throw new Error('invalid attachment media type')
    }

    const digest = await sha256(bytes)
    assertAuthorized(attachment, deps.now())
    await deps.revalidate(target)
    assertAuthorized(attachment, deps.now())
    const staged = await deps.stage(target, {
      bytes,
      kind: source.kind,
      mediaType: read.mediaType,
      name,
      order,
      sha256: digest
    })
    assertAuthorized(attachment, deps.now())
    await deps.revalidate(target)

    let storedName: string
    try {
      storedName = safeAttachmentName(staged.storedName)
    } catch {
      throw new Error('attachment relay integrity mismatch')
    }

    if (
      staged.attached !== true ||
      staged.bytes !== bytes.byteLength ||
      staged.mediaType !== read.mediaType ||
      staged.name !== name ||
      staged.order !== order ||
      staged.runtimeSessionId !== target.runtimeSessionId ||
      staged.sha256 !== digest ||
      storedName !== staged.storedName ||
      (staged.refText !== undefined && !staged.refText.includes(storedName))
    ) {
      throw new Error('attachment relay integrity mismatch')
    }

    relayed.push({
      id: source.id,
      kind: source.kind,
      mediaType: read.mediaType,
      name,
      occurrenceId: source.occurrenceId ?? null,
      order,
      provenance: {
        authorizedAt: grant.authorizedAt,
        kind: 'composer',
        occurrenceId: source.occurrenceId ?? null,
        sourceId: source.id
      },
      ...(staged.refText ? { refText: staged.refText } : {}),
      runtimeSessionId: target.runtimeSessionId,
      sha256: digest,
      size: bytes.byteLength,
      storedName
    })
  }

  return relayed
}
