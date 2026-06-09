import { atom, map, type MapStore } from 'nanostores'

// --- Pane tree types ---

export interface PaneLeaf {
  type: 'leaf'
  id: string
}

export interface PaneSplit {
  type: 'split'
  direction: 'horizontal' | 'vertical'
  ratio: number
  first: PaneNode
  second: PaneNode
}

export type PaneNode = PaneLeaf | PaneSplit

// --- Store atoms ---

const STORAGE_KEY = 'hermes.desktop.terminalLayouts'

let nextId = 1
const newId = () => `term-${Date.now()}-${nextId++}`

function createDefaultTree(): PaneNode {
  return { type: 'leaf', id: newId() }
}

// --- Per-chat layout storage ---

function loadAllLayouts(): Record<string, PaneNode> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)

    if (raw) {
      const parsed = JSON.parse(raw) as Record<string, unknown>

      return Object.fromEntries(
        Object.entries(parsed)
          .filter(([, v]) => isPaneNode(v))
          .map(([k, v]) => [k, v as PaneNode])
      )
    }
  } catch {
    // Corrupted storage — fall through
  }

  return {}
}

function isPaneNode(value: unknown): value is PaneNode {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const obj = value as Record<string, unknown>

  if (obj.type === 'leaf') {
    return typeof obj.id === 'string'
  }

  if (obj.type === 'split') {
    return (
      (obj.direction === 'horizontal' || obj.direction === 'vertical') &&
      typeof obj.ratio === 'number' &&
      isPaneNode(obj.first) &&
      isPaneNode(obj.second)
    )
  }

  return false
}

// All layouts keyed by chat/session ID
export const $terminalLayouts: MapStore<Record<string, PaneNode>> = map(loadAllLayouts())

// Current chat ID — TerminalPanel sets this
export const $chatId = atom<string | null>(null)

// Current pane tree (derived from chatId + layouts)
export const $paneTree = atom<PaneNode>(createDefaultTree())
export const $focusedLeafId = atom<string | null>(null)

// When chatId changes, load that chat's layout (or create default)
$chatId.subscribe(chatId => {
  if (!chatId) {
    return
  }

  const layouts = $terminalLayouts.get()
  const tree = layouts[chatId] || createDefaultTree()

  $paneTree.set(tree)
  $focusedLeafId.set(findLeafIds(tree)[0] ?? null)
})

// Save layout when tree changes (under current chat ID)
$paneTree.subscribe(tree => {
  const chatId = $chatId.get()

  if (!chatId) {
    return
  }

  const layouts = $terminalLayouts.get()

  if (JSON.stringify(layouts[chatId]) === JSON.stringify(tree)) {
    return
  }

  $terminalLayouts.setKey(chatId, tree)

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify($terminalLayouts.get()))
  } catch {
    // Storage full — silent
  }
})

// Ensure focused leaf is valid when tree changes
$paneTree.subscribe(tree => {
  const ids = findLeafIds(tree)
  const current = $focusedLeafId.get()

  if (!current || !ids.includes(current)) {
    $focusedLeafId.set(ids[0] ?? null)
  }
})

// Set the active chat ID (call this from TerminalPanel)
export function setActiveChatId(chatId: string | null): void {
  if ($chatId.get() === chatId) {
    return
  }

  $chatId.set(chatId)
}

// --- Tree operations ---

export function findLeafIds(node: PaneNode): string[] {
  if (node.type === 'leaf') {
    return [node.id]
  }

  return [...findLeafIds(node.first), ...findLeafIds(node.second)]
}

export function findParentSplit(tree: PaneNode, leafId: string, parent: PaneSplit | null = null): PaneSplit | null {
  if (tree.type === 'leaf') {
    return tree.id === leafId ? parent : null
  }

  return findParentSplit(tree.first, leafId, tree) || findParentSplit(tree.second, leafId, tree)
}

function replaceLeaf(tree: PaneNode, leafId: string, replacement: PaneNode): PaneNode {
  if (tree.type === 'leaf') {
    return tree.id === leafId ? replacement : tree
  }

  return {
    ...tree,
    first: replaceLeaf(tree.first, leafId, replacement),
    second: replaceLeaf(tree.second, leafId, replacement)
  }
}

function replaceInTree(tree: PaneNode, target: PaneSplit, replacement: PaneNode): PaneNode {
  if (tree === target) {
    return replacement
  }

  if (tree.type === 'leaf') {
    return tree
  }

  return {
    ...tree,
    first: replaceInTree(tree.first, target, replacement),
    second: replaceInTree(tree.second, target, replacement)
  }
}

export function splitPane(direction: 'horizontal' | 'vertical'): void {
  const tree = $paneTree.get()
  const focusId = $focusedLeafId.get()

  if (!focusId) {
    return
  }

  const newLeaf: PaneLeaf = { type: 'leaf', id: newId() }

  const split: PaneSplit = {
    type: 'split',
    direction,
    ratio: 0.5,
    first: { type: 'leaf', id: focusId },
    second: newLeaf
  }

  const newTree = replaceLeaf(tree, focusId, split)

  $paneTree.set(newTree)
  $focusedLeafId.set(newLeaf.id)
}

export function addPane(): void {
  splitPane('horizontal')
}

export function closePane(leafId: string): void {
  const tree = $paneTree.get()

  if (findLeafIds(tree).length <= 1) {
    return
  }

  const parent = findParentSplit(tree, leafId)

  if (!parent) {
    return
  }

  const sibling = parent.first.type === 'leaf' && parent.first.id === leafId ? parent.second : parent.first
  const finalTree = replaceInTree(tree, parent, sibling)

  $paneTree.set(finalTree)
}

function findSplitBetween(tree: PaneNode, firstLeafId: string, secondLeafId: string): PaneSplit | null {
  if (tree.type === 'leaf') {
    return null
  }

  const firstIds = findLeafIds(tree.first)
  const secondIds = findLeafIds(tree.second)

  if (firstIds.includes(firstLeafId) && secondIds.includes(secondLeafId)) {
    return tree
  }

  if (firstIds.includes(firstLeafId) && firstIds.includes(secondLeafId)) {
    return findSplitBetween(tree.first, firstLeafId, secondLeafId)
  }

  if (secondIds.includes(firstLeafId) && secondIds.includes(secondLeafId)) {
    return findSplitBetween(tree.second, firstLeafId, secondLeafId)
  }

  return null
}

export function resizePane(firstLeafId: string, secondLeafId: string, newRatio: number): void {
  const tree = $paneTree.get()
  const split = findSplitBetween(tree, firstLeafId, secondLeafId)

  if (!split) {
    return
  }

  const clamped = Math.max(0.1, Math.min(0.9, newRatio))

  $paneTree.set(updateRatio(tree, split, clamped))
}

function updateRatio(tree: PaneNode, target: PaneSplit, ratio: number): PaneNode {
  if (tree === target) {
    return { ...tree, ratio }
  }

  if (tree.type === 'leaf') {
    return tree
  }

  return {
    ...tree,
    first: updateRatio(tree.first, target, ratio),
    second: updateRatio(tree.second, target, ratio)
  }
}

export function focusNext(): void {
  const tree = $paneTree.get()
  const ids = findLeafIds(tree)
  const current = $focusedLeafId.get()

  if (!ids.length) {
    return
  }

  const index = current ? ids.indexOf(current) : -1

  $focusedLeafId.set(ids[(index + 1) % ids.length])
}

export function focusPrev(): void {
  const tree = $paneTree.get()
  const ids = findLeafIds(tree)
  const current = $focusedLeafId.get()

  if (!ids.length) {
    return
  }

  const index = current ? ids.indexOf(current) : 0

  $focusedLeafId.set(ids[(index - 1 + ids.length) % ids.length])
}
