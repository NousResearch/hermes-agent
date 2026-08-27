/**
 * User-customisable prompt templates — the "Prompt templates" entry in the
 * composer's "+" menu.  The three built-in starters (code review,
 * implementation plan, explain this) seed the store the first time the user
 * opens the dialog; from then on the list is entirely user-owned.  Add, edit,
 * delete, re-order, and nest under folders all persist to localStorage through
 * the shared storage choke point so cross-window sync / telemetry hooks see the
 * writes.
 *
 * Storage shape: flat `PromptTemplate[]` ordered among siblings.  Folders are
 * nodes with `kind: 'folder'`; templates nest via `parentId`.  Legacy v1 rows
 * (no kind/parentId) normalize to root templates on read.
 *
 * i18n: the built-in defaults are read lazily via `translateNow` so they
 * reflect the user's active locale at first-launch time.  The store itself
 * starts empty and is seeded by `ensureSeeded()` when the dialog opens — this
 * avoids the circular import (runtime.ts ↔ store) AND the "English seed on a
 * Chinese UI" problem that a module-level constant would cause.
 */

import { translateNow } from '@/i18n/runtime'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { readKey } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.prompt-templates'

export type PromptNodeKind = 'folder' | 'template'

export interface PromptTemplate {
  /** folders only — UI open/closed; ignored on templates */
  collapsed?: boolean
  description: string
  id: string
  kind: PromptNodeKind
  label: string
  /** null = root. Must point at a folder id when set. */
  parentId: string | null
  text: string
}

/** One row in the tree list UI (depth-first, collapsed folders hide descendants). */
export interface PromptTreeRow {
  depth: number
  node: PromptTemplate
}

/** Stable ids for the three built-in starters.  The user may edit or delete
 *  these — the ids only matter for the initial seed and for reset. */
export const BUILTIN_TEMPLATE_IDS = ['codeReview', 'implementationPlan', 'explainThis'] as const

function newId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

/**
 * Build the three built-in starters using the active locale's translations.
 * Called lazily (not at module load) so the i18n runtime is initialised.
 * Returns fresh copies so callers can mutate without side effects.
 */
export function getBuiltInTemplates(): PromptTemplate[] {
  return BUILTIN_TEMPLATE_IDS.map(id => ({
    id,
    kind: 'template' as const,
    parentId: null,
    label: translateNow(`composer.templates.${id}.label`),
    description: translateNow(`composer.templates.${id}.description`),
    text: translateNow(`composer.templates.${id}.text`)
  }))
}

function normalizeNode(value: unknown): PromptTemplate | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const s = value as Record<string, unknown>

  if (typeof s.id !== 'string' || typeof s.label !== 'string') {
    return null
  }

  const kind: PromptNodeKind = s.kind === 'folder' ? 'folder' : 'template'
  const parentId = typeof s.parentId === 'string' && s.parentId.length > 0 ? s.parentId : null
  const description = typeof s.description === 'string' ? s.description : ''
  const text = typeof s.text === 'string' ? s.text : ''

  if (kind === 'folder') {
    return {
      id: s.id,
      kind,
      parentId,
      label: s.label,
      description,
      text: '',
      collapsed: s.collapsed === true
    }
  }

  // Templates still require string text/description (legacy always had them).
  if (typeof s.description !== 'string' || typeof s.text !== 'string') {
    return null
  }

  return {
    id: s.id,
    kind: 'template',
    parentId,
    label: s.label,
    description: s.description,
    text: s.text
  }
}

function sanitizeTemplates(raw: unknown): PromptTemplate[] {
  if (!Array.isArray(raw)) {
    return []
  }

  const nodes = raw.map(normalizeNode).filter((n): n is PromptTemplate => n !== null)
  return repairTree(nodes)
}

/** Drop broken parent links and folder→non-folder parents; keep stable order. */
export function repairTree(nodes: PromptTemplate[]): PromptTemplate[] {
  const byId = new Map(nodes.map(n => [n.id, n]))

  return nodes.map(node => {
    if (!node.parentId) {
      return node.parentId === null ? node : { ...node, parentId: null }
    }

    const parent = byId.get(node.parentId)

    if (!parent || parent.kind !== 'folder' || parent.id === node.id) {
      return { ...node, parentId: null }
    }

    // Prevent cycles: walk ancestors.
    let cursor: string | null = parent.parentId
    const seen = new Set<string>([node.id])

    while (cursor) {
      if (seen.has(cursor)) {
        return { ...node, parentId: null }
      }

      seen.add(cursor)
      cursor = byId.get(cursor)?.parentId ?? null
    }

    return node
  })
}

function isTemplateList(value: unknown): boolean {
  return Array.isArray(value) && value.every(item => normalizeNode(item) !== null)
}

// The empty array is a valid, user-owned state.  Track whether seeding is
// needed from the persisted payload rather than from the current list length.
// This read happens before persistentAtom's fallback subscription writes []
// for a missing key.
const persistedRaw = readKey(STORAGE_KEY)
let shouldSeed = persistedRaw === null

if (persistedRaw !== null) {
  try {
    shouldSeed = !isTemplateList(JSON.parse(persistedRaw) as unknown)
  } catch {
    shouldSeed = true
  }
}

export const $promptTemplates = persistentAtom<PromptTemplate[]>(STORAGE_KEY, [], Codecs.json(sanitizeTemplates))

function setList(next: PromptTemplate[]): void {
  $promptTemplates.set(repairTree(next))
}

/** Seed the store with locale-appropriate built-in templates the first time
 * the dialog is opened (or after a corrupted-payload reset).  A valid
 * persisted empty list is intentional and must remain empty. */
export function ensureSeeded(): void {
  if (!shouldSeed) {
    return
  }

  shouldSeed = false
  setList(getBuiltInTemplates())
}

export function siblingsOf(parentId: string | null, list = $promptTemplates.get()): PromptTemplate[] {
  return list.filter(n => n.parentId === parentId)
}

/** Depth-first visible rows; collapsed folders hide their descendants. */
export function visibleTreeRows(list = $promptTemplates.get()): PromptTreeRow[] {
  const rows: PromptTreeRow[] = []

  const walk = (parentId: string | null, depth: number) => {
    for (const node of siblingsOf(parentId, list)) {
      rows.push({ depth, node })

      if (node.kind === 'folder' && !node.collapsed) {
        walk(node.id, depth + 1)
      }
    }
  }

  walk(null, 0)
  return rows
}

export function collectDescendantIds(id: string, list = $promptTemplates.get()): Set<string> {
  const out = new Set<string>()
  const stack = [id]

  while (stack.length > 0) {
    const current = stack.pop()!

    for (const child of list) {
      if (child.parentId === current && !out.has(child.id)) {
        out.add(child.id)
        stack.push(child.id)
      }
    }
  }

  return out
}

/** Add a new template.  Returns the created node so the UI can enter edit mode. */
export function addTemplate(
  label = '',
  description = '',
  text = '',
  parentId: string | null = null
): PromptTemplate {
  const template: PromptTemplate = {
    id: newId('tpl'),
    kind: 'template',
    parentId,
    label,
    description,
    text
  }

  setList([...$promptTemplates.get(), template])
  return template
}

/** Add a folder (optionally nested).  Defaults expanded. */
export function addFolder(label = '', parentId: string | null = null): PromptTemplate {
  const folder: PromptTemplate = {
    id: newId('fld'),
    kind: 'folder',
    parentId,
    label,
    description: '',
    text: '',
    collapsed: false
  }

  setList([...$promptTemplates.get(), folder])
  return folder
}

/** Patch a node by id.  Unknown ids are ignored.  `id` / structural kind changes go through dedicated helpers. */
export function updateTemplate(
  id: string,
  patch: Partial<Omit<PromptTemplate, 'id' | 'kind'>>
): void {
  setList(
    $promptTemplates.get().map(s => {
      if (s.id !== id) {
        return s
      }

      const next = { ...s, ...patch, id: s.id, kind: s.kind }

      if (s.kind === 'folder') {
        next.text = ''
      }

      return next
    })
  )
}

/** Remove a node.  Folders remove all descendants.  Unknown ids are ignored. */
export function deleteTemplate(id: string): void {
  const list = $promptTemplates.get()

  if (!list.some(s => s.id === id)) {
    return
  }

  const drop = collectDescendantIds(id, list)
  drop.add(id)
  setList(list.filter(s => !drop.has(s.id)))
}

function moveAmongSiblings(id: string, direction: -1 | 1): void {
  const list = [...$promptTemplates.get()]
  const node = list.find(s => s.id === id)

  if (!node) {
    return
  }

  const siblingIndexes = list
    .map((s, index) => (s.parentId === node.parentId ? index : -1))
    .filter(index => index >= 0)
  const position = siblingIndexes.findIndex(index => list[index].id === id)

  if (position < 0) {
    return
  }

  const swapPos = position + direction

  if (swapPos < 0 || swapPos >= siblingIndexes.length) {
    return
  }

  const a = siblingIndexes[position]
  const b = siblingIndexes[swapPos]
  ;[list[a], list[b]] = [list[b], list[a]]
  setList(list)
}

/** Move one slot up among siblings (same parent).  No-op at the top. */
export function moveTemplateUp(id: string): void {
  moveAmongSiblings(id, -1)
}

/** Move one slot down among siblings.  No-op at the bottom. */
export function moveTemplateDown(id: string): void {
  moveAmongSiblings(id, 1)
}

export type DropPlacement = 'after' | 'before' | 'inside'

/**
 * Tree drop commit used by the dialog DnD layer.
 *
 * - `before` / `after`: become a sibling of `overId` (same parent), inserted
 *   before it or after it **and its remaining subtree**.
 * - `inside`: only valid when `overId` is a folder; become its last child and
 *   expand the folder so the move is visible.
 *
 * Moves the active node **and its descendants** as one block. Returns false
 * when the drop is illegal (unknown ids, drop into own subtree, inside a
 * non-folder) so the UI can leave the row alone and let dnd-kit snap back.
 */
export function placeNode(activeId: string, overId: string, placement: DropPlacement): boolean {
  if (activeId === overId) {
    return false
  }

  const list = $promptTemplates.get()
  const active = list.find(s => s.id === activeId)
  const over = list.find(s => s.id === overId)

  if (!active || !over) {
    return false
  }

  const blockIds = collectDescendantIds(activeId, list)

  blockIds.add(activeId)

  if (blockIds.has(overId)) {
    return false
  }

  let newParentId: string | null

  if (placement === 'inside') {
    if (over.kind !== 'folder') {
      return false
    }

    newParentId = over.id
  } else {
    newParentId = over.parentId
  }

  if (newParentId !== null) {
    const parent = list.find(s => s.id === newParentId)

    if (!parent || parent.kind !== 'folder' || blockIds.has(newParentId)) {
      return false
    }
  }

  const block = list
    .filter(s => blockIds.has(s.id))
    .map(s => (s.id === activeId ? { ...s, parentId: newParentId } : s))
  const rest = list
    .filter(s => !blockIds.has(s.id))
    .map(s => (placement === 'inside' && s.id === overId ? { ...s, collapsed: false } : s))

  const overIndex = rest.findIndex(s => s.id === overId)

  if (overIndex < 0) {
    return false
  }

  let insertAt: number

  if (placement === 'before') {
    insertAt = overIndex
  } else {
    // after | inside → after over and whatever descendants remain under it
    const overDesc = collectDescendantIds(overId, rest)

    insertAt = overIndex + 1

    while (insertAt < rest.length && overDesc.has(rest[insertAt].id)) {
      insertAt += 1
    }
  }

  const next = [...rest.slice(0, insertAt), ...block, ...rest.slice(insertAt)]
  const orderEqual =
    next.map(s => s.id).join(',') === list.map(s => s.id).join(',') &&
    next.find(s => s.id === activeId)?.parentId === active.parentId

  if (orderEqual) {
    return false
  }

  setList(next)

  return true
}

export function canMoveUp(id: string, list = $promptTemplates.get()): boolean {
  const node = list.find(s => s.id === id)

  if (!node) {
    return false
  }

  const siblings = siblingsOf(node.parentId, list)
  return siblings[0]?.id !== id
}

export function canMoveDown(id: string, list = $promptTemplates.get()): boolean {
  const node = list.find(s => s.id === id)

  if (!node) {
    return false
  }

  const siblings = siblingsOf(node.parentId, list)
  return siblings[siblings.length - 1]?.id !== id
}

/** Re-parent a node under a folder (or root).  Rejects cycles and non-folder parents. */
export function moveInto(id: string, parentId: string | null): void {
  const list = $promptTemplates.get()
  const node = list.find(s => s.id === id)

  if (!node) {
    return
  }

  if (parentId === node.parentId) {
    return
  }

  if (parentId !== null) {
    const parent = list.find(s => s.id === parentId)

    if (!parent || parent.kind !== 'folder') {
      return
    }

    if (parentId === id || collectDescendantIds(id, list).has(parentId)) {
      return
    }
  }

  setList(list.map(s => (s.id === id ? { ...s, parentId } : s)))
}

export function toggleFolderCollapsed(id: string): void {
  setList(
    $promptTemplates.get().map(s =>
      s.id === id && s.kind === 'folder' ? { ...s, collapsed: !s.collapsed } : s
    )
  )
}

/** Nest under the previous sibling when that sibling is a folder. */
export function indentNode(id: string): void {
  const list = $promptTemplates.get()
  const node = list.find(s => s.id === id)

  if (!node) {
    return
  }

  const siblings = siblingsOf(node.parentId, list)
  const index = siblings.findIndex(s => s.id === id)

  if (index <= 0) {
    return
  }

  const prev = siblings[index - 1]

  if (prev.kind !== 'folder') {
    return
  }

  // Expand target so the move is visible.
  setList(
    list.map(s => {
      if (s.id === prev.id) {
        return { ...s, collapsed: false }
      }

      if (s.id === id) {
        return { ...s, parentId: prev.id }
      }

      return s
    })
  )
}

/** Lift one level toward the root (parent's parent). */
export function outdentNode(id: string): void {
  const list = $promptTemplates.get()
  const node = list.find(s => s.id === id)

  if (!node?.parentId) {
    return
  }

  const parent = list.find(s => s.id === node.parentId)

  moveInto(id, parent?.parentId ?? null)
}

export function canIndent(id: string, list = $promptTemplates.get()): boolean {
  const node = list.find(s => s.id === id)

  if (!node) {
    return false
  }

  const siblings = siblingsOf(node.parentId, list)
  const index = siblings.findIndex(s => s.id === id)
  const prev = index > 0 ? siblings[index - 1] : null
  return prev?.kind === 'folder'
}

export function canOutdent(id: string, list = $promptTemplates.get()): boolean {
  const node = list.find(s => s.id === id)
  return Boolean(node?.parentId)
}

/** Restore the three built-in starters, discarding everything the user added. */
export function resetToBuiltins(): void {
  setList(getBuiltInTemplates())
}
