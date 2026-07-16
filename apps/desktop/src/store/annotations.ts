import { atom, computed } from 'nanostores'

import { readKey, writeKey } from '@/lib/storage'

import { $connection, $currentCwd, $selectedStoredSessionId } from './session'

export type AnnotationStatus = 'active' | 'orphaned' | 'sent' | 'stale'
export type ReviewAnnotationType = 'comment' | 'concern' | 'suggestion'
export type ReviewAnnotationLabel = 'accessibility' | 'bug' | 'performance' | 'security' | 'style' | 'testing'

export const REVIEW_ANNOTATION_LABELS: ReviewAnnotationLabel[] = [
  'bug',
  'security',
  'performance',
  'accessibility',
  'testing',
  'style'
]

export interface ReviewContext {
  artifactPath?: string
  baseRef?: null | string
  connection: string
  contentHash?: string
  cwd: string
  headSha?: string
  id: string
  kind: 'document' | 'git' | 'plan'
  profile: string
  reviewScope?: 'branch' | 'lastTurn' | 'uncommitted'
  sessionId?: string
}

export interface TextPosition {
  endOffset: number
  endNodeOffset?: number
  endPath?: number[]
  prefix?: string
  quote: string
  startOffset: number
  startNodeOffset?: number
  startPath?: number[]
  suffix?: string
}

export interface SourceAnnotationAnchor {
  contentHash?: string
  excerpt?: string
  kind: 'source'
  lineEnd: number
  lineStart: number
  path: string
}

export interface DiffAnnotationAnchor {
  baseRef?: null | string
  contentHash?: string
  excerpt?: string
  headSha?: string
  kind: 'diff'
  lineEnd: number
  lineStart: number
  path: string
  side: 'new' | 'old'
}

export interface TextAnnotationAnchor extends TextPosition {
  contentHash?: string
  kind: 'html' | 'markdown' | 'svg'
  path: string
}

export interface VisualPoint {
  x: number
  y: number
}

export type VisualAnnotationMark =
  | { id: string; points: VisualPoint[]; tool: 'pen' }
  | { end: VisualPoint; id: string; start: VisualPoint; tool: 'arrow' }
  | { end: VisualPoint; id: string; start: VisualPoint; tool: 'rectangle' }
  | { id: string; point: VisualPoint; tool: 'pin' }

export interface VisualAnnotationAnchor {
  contentHash?: string
  kind: 'visual'
  marks: VisualAnnotationMark[]
  mediaKind: 'jpeg' | 'png' | 'svg'
  naturalHeight: number
  naturalWidth: number
  path: string
  quote?: string
}

export interface PdfAnnotationAnchor extends TextPosition {
  contentHash?: string
  documentKind?: 'pdf' | 'tex'
  kind: 'pdf'
  page: number
  path: string
}

export interface FileAnnotationAnchor {
  contentHash?: string
  kind: 'file'
  path: string
}

export type ReviewAnnotationAnchor =
  | DiffAnnotationAnchor
  | FileAnnotationAnchor
  | PdfAnnotationAnchor
  | SourceAnnotationAnchor
  | TextAnnotationAnchor
  | VisualAnnotationAnchor

export interface AnnotationEditorAnchor {
  boundary?: {
    height: number
    width: number
    x: number
    y: number
  }
  height: number
  width: number
  x: number
  y: number
}

export interface ReviewAnnotation {
  anchor: ReviewAnnotationAnchor
  comment: string
  contextId: string
  createdAt: number
  id: string
  labels: ReviewAnnotationLabel[]
  sentGeneration?: number
  status: AnnotationStatus
  suggestion?: string
  type: ReviewAnnotationType
  updatedAt?: number
}

export interface AnnotationDraft {
  anchor: ReviewAnnotationAnchor
  comment: string
  contextId: string
  editingId?: string
  labels: ReviewAnnotationLabel[]
  suggestion: string
  type: ReviewAnnotationType
}

interface ContextState {
  draft: AnnotationDraft | null
  generation: number
  items: ReviewAnnotation[]
  tombstoneGeneration: number
}

interface PersistedState {
  contexts: Record<string, ContextState>
  version: 3
}

export type AnnotationDiscardIntent = 'close' | 'replace'

const STORAGE_KEY = 'hermes.desktop.annotations.v3'
const VERSION = 3 as const
const MAX_ANNOTATION_CONTEXTS = 128
const MAX_ANNOTATIONS_PER_CONTEXT = 500
const MAX_PEN_POINTS = 2048

function canonicalContext(input: Omit<ReviewContext, 'id'>): string {
  return JSON.stringify({
    artifactPath: input.artifactPath ?? '',
    baseRef: input.baseRef ?? '',
    connection: input.connection,
    contentHash: input.contentHash ?? '',
    cwd: input.cwd,
    headSha: input.headSha ?? '',
    kind: input.kind,
    profile: input.profile,
    reviewScope: input.reviewScope ?? '',
    sessionId: input.sessionId ?? ''
  })
}

function connectionIdentity(): { connection: string; profile: string } {
  const connection = $connection.get()

  return {
    connection: `${connection?.mode ?? 'local'}:${connection?.baseUrl ?? ''}`,
    profile: connection?.profile ?? 'default'
  }
}

export function createReviewContext(
  input: Partial<Omit<ReviewContext, 'connection' | 'cwd' | 'id' | 'kind' | 'profile'>> & {
    cwd?: string
    kind: ReviewContext['kind']
  }
): ReviewContext {
  const identity = connectionIdentity()

  const bare: Omit<ReviewContext, 'id'> = {
    ...identity,
    artifactPath: input.artifactPath,
    baseRef: input.baseRef,
    contentHash: input.contentHash,
    cwd: input.cwd ?? $currentCwd.get() ?? '',
    headSha: input.headSha,
    kind: input.kind,
    reviewScope: input.reviewScope,
    sessionId: input.sessionId ?? $selectedStoredSessionId.get() ?? undefined
  }

  return { ...bare, id: canonicalContext(bare) }
}

function emptyContext(): ContextState {
  return { draft: null, generation: 0, items: [], tombstoneGeneration: 0 }
}

function validAnnotation(value: unknown): value is ReviewAnnotation {
  if (!value || typeof value !== 'object') {
    return false
  }

  const annotation = value as Partial<ReviewAnnotation>

  return (
    typeof annotation.id === 'string' &&
    typeof annotation.contextId === 'string' &&
    typeof annotation.comment === 'string' &&
    typeof annotation.createdAt === 'number' &&
    typeof annotation.anchor === 'object' &&
    annotation.anchor !== null &&
    Array.isArray(annotation.labels) &&
    ['comment', 'concern', 'suggestion'].includes(annotation.type ?? '') &&
    ['active', 'orphaned', 'sent', 'stale'].includes(annotation.status ?? '')
  )
}

function boundedAnchor(anchor: ReviewAnnotationAnchor): ReviewAnnotationAnchor {
  if (anchor.kind !== 'visual') {
    return anchor
  }

  return {
    ...anchor,
    marks: anchor.marks.map(mark =>
      mark.tool === 'pen' ? { ...mark, points: mark.points.slice(-MAX_PEN_POINTS) } : mark
    )
  }
}

function boundedContextState(value: unknown): ContextState | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const state = value as Partial<ContextState>

  if (
    !Array.isArray(state.items) ||
    !Number.isInteger(state.generation) ||
    !Number.isInteger(state.tombstoneGeneration)
  ) {
    return null
  }

  const draft =
    state.draft && typeof state.draft === 'object' && typeof state.draft.contextId === 'string'
      ? { ...state.draft, anchor: boundedAnchor(state.draft.anchor) }
      : null

  return {
    draft,
    generation: Math.max(0, state.generation ?? 0),
    items: state.items
      .filter(validAnnotation)
      .slice(-MAX_ANNOTATIONS_PER_CONTEXT)
      .map(item => ({ ...item, anchor: boundedAnchor(item.anchor) })),
    tombstoneGeneration: Math.max(0, state.tombstoneGeneration ?? 0)
  }
}

function loadState(): PersistedState {
  const raw = readKey(STORAGE_KEY)

  if (!raw) {
    return { contexts: {}, version: VERSION }
  }

  try {
    const parsed = JSON.parse(raw) as Partial<PersistedState>

    const contexts = Object.fromEntries(
      Object.entries(parsed.contexts ?? {})
        .slice(-MAX_ANNOTATION_CONTEXTS)
        .flatMap(([key, value]) => {
          const state = boundedContextState(value)

          return state ? [[key, state] as const] : []
        })
    )

    return { contexts, version: VERSION }
  } catch {
    return { contexts: {}, version: VERSION }
  }
}

const initialContext = createReviewContext({ kind: 'document' })
let persisted = loadState()
let pendingDraft: AnnotationDraft | null = null
let pendingEditorAnchor: AnnotationEditorAnchor | null = null

export const $annotationContext = atom<ReviewContext>(initialContext)
export const $annotations = atom<ReviewAnnotation[]>(persisted.contexts[initialContext.id]?.items ?? [])
export const $annotationDraft = atom<AnnotationDraft | null>(persisted.contexts[initialContext.id]?.draft ?? null)
export const $annotationGeneration = atom(persisted.contexts[initialContext.id]?.generation ?? 0)
export const $annotationStorageError = atom<string | null>(null)
export const $annotationDiscardIntent = atom<AnnotationDiscardIntent | null>(null)
export const $annotationEditorAnchor = atom<AnnotationEditorAnchor | null>(null)
export const $annotationEditorCollapsed = atom(false)
export const $annotationDraftAnchor = computed($annotationDraft, draft => draft?.anchor ?? null)

export const $annotationDraftDirty = computed($annotationDraft, draft =>
  Boolean(draft && (draft.comment.trim() || draft.suggestion.trim()))
)

function currentState(): ContextState {
  const contextId = $annotationContext.get().id
  const previous = persisted.contexts[contextId] ?? emptyContext()

  return {
    ...previous,
    draft: $annotationDraft.get(),
    generation: $annotationGeneration.get(),
    items: $annotations.get()
  }
}

function persistCurrent(): void {
  const contextId = $annotationContext.get().id

  const contexts = Object.fromEntries(
    [
      ...Object.entries(persisted.contexts).filter(([key]) => key !== contextId),
      [contextId, boundedContextState(currentState()) ?? emptyContext()]
    ].slice(-MAX_ANNOTATION_CONTEXTS)
  )

  persisted = {
    contexts,
    version: VERSION
  }
  const serialized = JSON.stringify(persisted)

  writeKey(STORAGE_KEY, serialized)
  $annotationStorageError.set(readKey(STORAGE_KEY) === serialized ? null : 'annotation-storage-unavailable')
}

function rehome(context: ReviewContext): void {
  if (context.id === $annotationContext.get().id) {
    return
  }

  persistCurrent()
  $annotationContext.set(context)
  const next = persisted.contexts[context.id] ?? emptyContext()
  $annotations.set(next.items)
  $annotationDraft.set(next.draft)
  $annotationGeneration.set(next.generation)
  $annotationDiscardIntent.set(null)
  $annotationEditorAnchor.set(null)
  // Drafts belong to their document, but changing documents must not make an
  // old editor jump open over the newly selected surface. The pane toolbar is
  // the explicit way to reopen a preserved draft.
  $annotationEditorCollapsed.set(Boolean(next.draft))
  pendingDraft = null
  pendingEditorAnchor = null
}

export function activateAnnotationContext(context: ReviewContext, options: { carryStale?: boolean } = {}): void {
  const previous = $annotationContext.get()
  const previousItems = $annotations.get()

  const compatibleRevision =
    previous.kind === context.kind &&
    previous.connection === context.connection &&
    previous.profile === context.profile &&
    previous.cwd === context.cwd &&
    previous.artifactPath === context.artifactPath &&
    previous.reviewScope === context.reviewScope

  rehome(context)

  if (options.carryStale && compatibleRevision && $annotations.get().length === 0 && previousItems.length > 0) {
    $annotations.set(
      previousItems.map(item => ({ ...item, contextId: context.id, status: 'stale' as const, updatedAt: Date.now() }))
    )
    $annotationGeneration.set($annotationGeneration.get() + 1)
    persistCurrent()
  }
}

export function documentReviewContext(path: string, contentHash?: string): ReviewContext {
  return createReviewContext({ artifactPath: path, contentHash, kind: 'document' })
}

export function planReviewContext(path: string, contentHash: string, sessionId?: string): ReviewContext {
  return createReviewContext({ artifactPath: path, contentHash, kind: 'plan', sessionId })
}

export function gitReviewContext(input: {
  baseRef?: null | string
  contentHash?: string
  cwd?: string
  headSha?: string
  reviewScope: 'branch' | 'lastTurn' | 'uncommitted'
}): ReviewContext {
  return createReviewContext({
    baseRef: input.baseRef,
    contentHash: input.contentHash,
    cwd: input.cwd,
    headSha: input.headSha,
    kind: 'git',
    reviewScope: input.reviewScope
  })
}

export function beginAnnotation(
  anchor: ReviewAnnotationAnchor,
  editorAnchor: AnnotationEditorAnchor | null = null,
  context = documentReviewContext(anchor.path, anchor.contentHash)
): void {
  rehome(context)

  const draft: AnnotationDraft = {
    anchor,
    comment: '',
    contextId: context.id,
    labels: [],
    suggestion: '',
    type: 'comment'
  }

  if ($annotationDraft.get() && $annotationDraftDirty.get()) {
    pendingDraft = draft
    pendingEditorAnchor = editorAnchor
    $annotationDiscardIntent.set('replace')

    return
  }

  $annotationDraft.set(draft)
  $annotationEditorAnchor.set(editorAnchor)
  $annotationEditorCollapsed.set(false)
  persistCurrent()
}

export function editAnnotation(id: string): void {
  const item = $annotations.get().find(annotation => annotation.id === id)

  if (!item) {
    return
  }

  const draft: AnnotationDraft = {
    anchor: item.anchor,
    comment: item.comment,
    contextId: item.contextId,
    editingId: item.id,
    labels: item.labels,
    suggestion: item.suggestion ?? '',
    type: item.type
  }

  if ($annotationDraft.get() && $annotationDraftDirty.get()) {
    pendingDraft = draft
    pendingEditorAnchor = null
    $annotationDiscardIntent.set('replace')

    return
  }

  $annotationDraft.set(draft)
  $annotationEditorAnchor.set(null)
  $annotationEditorCollapsed.set(false)
  persistCurrent()
}

export function updateAnnotationDraft(
  patch: Partial<Pick<AnnotationDraft, 'anchor' | 'comment' | 'labels' | 'suggestion' | 'type'>>
): void {
  const draft = $annotationDraft.get()

  if (draft) {
    $annotationDraft.set({ ...draft, ...patch })
    persistCurrent()
  }
}

export function toggleAnnotationDraftLabel(label: ReviewAnnotationLabel): void {
  const draft = $annotationDraft.get()

  if (!draft) {
    return
  }

  updateAnnotationDraft({
    labels: draft.labels.includes(label) ? draft.labels.filter(item => item !== label) : [...draft.labels, label]
  })
}

function createId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `annotation-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function saveAnnotationDraft(): ReviewAnnotation | null {
  const draft = $annotationDraft.get()
  const comment = draft?.comment.trim()

  if (!draft || !comment) {
    return null
  }

  const previous = draft.editingId ? $annotations.get().find(item => item.id === draft.editingId) : undefined
  const now = Date.now()

  const item: ReviewAnnotation = {
    anchor: draft.anchor,
    comment,
    contextId: draft.contextId,
    createdAt: previous?.createdAt ?? now,
    id: previous?.id ?? createId(),
    labels: draft.labels,
    status: 'active',
    suggestion: draft.type === 'suggestion' && draft.suggestion.trim() ? draft.suggestion.trim() : undefined,
    type: draft.type,
    updatedAt: previous ? now : undefined
  }

  $annotations.set(
    previous
      ? $annotations.get().map(value => (value.id === previous.id ? item : value))
      : [...$annotations.get(), item].slice(-MAX_ANNOTATIONS_PER_CONTEXT)
  )
  $annotationDraft.set(null)
  $annotationEditorAnchor.set(null)
  $annotationEditorCollapsed.set(false)
  $annotationGeneration.set($annotationGeneration.get() + 1)
  persistCurrent()

  return item
}

export function collapseAnnotationEditor(): void {
  if ($annotationDraft.get()) {
    $annotationEditorCollapsed.set(true)
  }
}

export function reopenAnnotationEditor(): void {
  if ($annotationDraft.get()) {
    $annotationEditorCollapsed.set(false)
  }
}

export function requestDiscardAnnotationDraft(): void {
  if ($annotationDraftDirty.get()) {
    $annotationDiscardIntent.set('close')
  } else {
    discardDraft()
  }
}

function discardDraft(): void {
  $annotationDraft.set(null)
  $annotationEditorAnchor.set(null)
  $annotationEditorCollapsed.set(false)
  persistCurrent()
}

export function cancelDiscardAnnotationDraft(): void {
  $annotationDiscardIntent.set(null)
  pendingDraft = null
  pendingEditorAnchor = null
}

export function confirmDiscardAnnotationDraft(): void {
  const replace = $annotationDiscardIntent.get() === 'replace' ? pendingDraft : null

  $annotationDiscardIntent.set(null)
  $annotationDraft.set(replace)
  $annotationEditorAnchor.set(replace ? pendingEditorAnchor : null)
  $annotationEditorCollapsed.set(false)
  pendingDraft = null
  pendingEditorAnchor = null
  persistCurrent()
}

export function removeAnnotation(id: string): void {
  const items = $annotations.get()
  const next = items.filter(item => item.id !== id)

  if (next.length === items.length) {
    return
  }

  $annotations.set(next)

  if ($annotationDraft.get()?.editingId === id) {
    $annotationDraft.set(null)
    $annotationEditorCollapsed.set(false)
  }

  const contextId = $annotationContext.get().id
  const current = persisted.contexts[contextId] ?? emptyContext()
  persisted.contexts[contextId] = { ...current, tombstoneGeneration: current.tombstoneGeneration + 1 }
  $annotationGeneration.set($annotationGeneration.get() + 1)
  persistCurrent()
}

export function clearAnnotations(): void {
  $annotations.set([])
  $annotationDraft.set(null)
  $annotationEditorCollapsed.set(false)
  const contextId = $annotationContext.get().id
  const current = persisted.contexts[contextId] ?? emptyContext()
  persisted.contexts[contextId] = { ...current, tombstoneGeneration: current.tombstoneGeneration + 1 }
  $annotationGeneration.set($annotationGeneration.get() + 1)
  persistCurrent()
}

export function markAnnotationsSent(ids: readonly string[]): void {
  const generation = $annotationGeneration.get() + 1
  const selected = new Set(ids)

  $annotations.set(
    $annotations
      .get()
      .map(item => (selected.has(item.id) ? { ...item, sentGeneration: generation, status: 'sent' as const } : item))
  )
  $annotationGeneration.set(generation)
  persistCurrent()
}

export function reconcileAnnotationAnchors(activeIds: ReadonlySet<string>, path: string): void {
  let changed = false

  const next = $annotations.get().map(item => {
    if (
      item.anchor.path !== path ||
      (item.anchor.kind !== 'html' && item.anchor.kind !== 'markdown' && item.anchor.kind !== 'pdf')
    ) {
      return item
    }

    const status: AnnotationStatus = activeIds.has(item.id) ? (item.status === 'sent' ? 'sent' : 'active') : 'orphaned'

    if (status === item.status) {
      return item
    }

    changed = true

    return { ...item, status }
  })

  if (changed) {
    $annotations.set(next)
    persistCurrent()
  }
}

export function annotationExcerpt(anchor: ReviewAnnotationAnchor): string {
  return 'quote' in anchor ? (anchor.quote ?? '') : 'excerpt' in anchor ? (anchor.excerpt ?? '') : ''
}

export function annotationStorageKey(): string {
  return STORAGE_KEY
}
