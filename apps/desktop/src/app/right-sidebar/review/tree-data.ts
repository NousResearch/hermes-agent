import type { HermesReviewFile } from '@/global'

// A node in the review changed-files tree. Directories aggregate their
// descendants' +/- so a collapsed folder still shows its total churn (Codex's
// folder hierarchy view).
export interface ReviewTreeNode {
  id: string
  name: string
  isDir: boolean
  added: number
  removed: number
  /** For a flat-list file row: the parent dir (relative), shown dimmed. */
  dir?: string
  file?: HermesReviewFile
  children?: ReviewTreeNode[]
  /** Supporting file (test/fixture/lockfile/generated) — rendered dimmed. */
  muted?: boolean
}

// ── Importance classification (smart order) ──────────────────────────────────
//
// Deterministic proxy for Amp's "intelligently ordered diffs": the files that
// best explain a change are the source files; tests, fixtures, lockfiles and
// generated artifacts SUPPORT it. No model call — classification must be free,
// instant, and identical on every refresh.

const SUPPORTING_DIR_SEGMENTS = new Set([
  '__fixtures__',
  '__mocks__',
  '__snapshots__',
  '__tests__',
  'fixtures',
  'snapshots',
  'spec',
  'test',
  'testdata',
  'tests'
])

const SUPPORTING_FILENAMES = new Set([
  'cargo.lock',
  'composer.lock',
  'gemfile.lock',
  'go.sum',
  'package-lock.json',
  'pnpm-lock.yaml',
  'poetry.lock',
  'uv.lock',
  'yarn.lock'
])

// Filename shapes: foo.test.ts / foo.spec.tsx / test_foo.py / foo_test.go /
// generated + minified + sourcemap artifacts / snapshot files.
const SUPPORTING_FILE_RE = /(\.(test|spec)\.[^.]+$)|(^test_.+\.py$)|(_test\.(py|go|rb|ts|js)$)|(\.snap$)|(\.min\.(js|css)$)|(\.map$)|(\.generated\.[^.]+$)|(_pb2(_grpc)?\.py$)/

/** True when a changed file is supporting material (test, fixture, lockfile,
 *  generated) rather than a change-explaining source file. */
export function isSupportingReviewPath(path: string): boolean {
  const segments = path.split('/').filter(Boolean)
  const name = (segments.pop() ?? path).toLowerCase()

  if (SUPPORTING_FILENAMES.has(name) || SUPPORTING_FILE_RE.test(name)) {
    return true
  }

  return segments.some(segment => SUPPORTING_DIR_SEGMENTS.has(segment.toLowerCase()))
}

// Smart flat list: change-explaining files first ordered by churn (the file
// with the most movement usually explains the change), supporting files after,
// dimmed. Ties break by path so the order is stable across refreshes.
export function buildReviewSmartList(files: HermesReviewFile[]): ReviewTreeNode[] {
  const churn = (f: HermesReviewFile) => f.added + f.removed

  return [...files]
    .sort((a, b) => {
      const aMuted = isSupportingReviewPath(a.path)
      const bMuted = isSupportingReviewPath(b.path)

      if (aMuted !== bMuted) {
        return aMuted ? 1 : -1
      }

      return churn(b) - churn(a) || a.path.localeCompare(b.path)
    })
    .map(file => {
      const segments = file.path.split('/').filter(Boolean)
      const name = segments.pop() ?? file.path

      return {
        id: file.path,
        name,
        dir: segments.join('/'),
        isDir: false,
        added: file.added,
        removed: file.removed,
        muted: isSupportingReviewPath(file.path),
        file
      }
    })
}

// Flat changed-file list (VS Code's default SCM "List" view): one row per file,
// filename + a dimmed parent-dir path, sorted by path. No folder nodes.
export function buildReviewFlatList(files: HermesReviewFile[]): ReviewTreeNode[] {
  return [...files]
    .sort((a, b) => a.path.localeCompare(b.path))
    .map(file => {
      const segments = file.path.split('/').filter(Boolean)
      const name = segments.pop() ?? file.path

      return {
        id: file.path,
        name,
        dir: segments.join('/'),
        isDir: false,
        added: file.added,
        removed: file.removed,
        file
      }
    })
}

interface MutableDir {
  id: string
  name: string
  added: number
  removed: number
  dirs: Map<string, MutableDir>
  files: ReviewTreeNode[]
}

const makeDir = (id: string, name: string): MutableDir => ({
  id,
  name,
  added: 0,
  removed: 0,
  dirs: new Map(),
  files: []
})

// Build a folder hierarchy from the flat changed-file list. With `compact`,
// single-child directory chains collapse into one row (`a/b/c`), the way VS Code
// and Codex render sparse trees.
export function buildReviewTree(files: HermesReviewFile[], compact = true): ReviewTreeNode[] {
  const root = makeDir('', '')

  for (const file of files) {
    const segments = file.path.split('/').filter(Boolean)
    const fileName = segments.pop() ?? file.path
    let dir = root

    dir.added += file.added
    dir.removed += file.removed

    let prefix = ''

    for (const segment of segments) {
      prefix = prefix ? `${prefix}/${segment}` : segment
      let child = dir.dirs.get(segment)

      if (!child) {
        child = makeDir(prefix, segment)
        dir.dirs.set(segment, child)
      }

      child.added += file.added
      child.removed += file.removed
      dir = child
    }

    dir.files.push({
      id: file.path,
      name: fileName,
      isDir: false,
      added: file.added,
      removed: file.removed,
      file
    })
  }

  const finalize = (dir: MutableDir): ReviewTreeNode[] => {
    const dirNodes: ReviewTreeNode[] = [...dir.dirs.values()]
      .sort((a, b) => a.name.localeCompare(b.name))
      .map(child => {
        let node: ReviewTreeNode = {
          id: child.id,
          name: child.name,
          isDir: true,
          added: child.added,
          removed: child.removed,
          children: finalize(child)
        }

        // Compact a chain: a folder whose only child is one folder merges into
        // `parent/child` so deep sparse paths read on one row.
        while (compact && node.children?.length === 1 && node.children[0].isDir) {
          const only = node.children[0]
          node = { ...only, name: `${node.name}/${only.name}` }
        }

        return node
      })

    const fileNodes = [...dir.files].sort((a, b) => a.name.localeCompare(b.name))

    return [...dirNodes, ...fileNodes]
  }

  return finalize(root)
}

// A row in the virtualized review list: the node plus the indentation depth it
// renders at (directory nesting level).
export interface ReviewFlatRow {
  node: ReviewTreeNode
  depth: number
}

// Total node count including every descendant — the cheap upper bound used to
// decide whether the tree needs virtualization. A single folder holding tens of
// thousands of untracked files is one top-level node but must still count as
// heavy.
export function countAllNodes(nodes: ReviewTreeNode[]): number {
  let total = 0

  for (const node of nodes) {
    total += 1

    if (node.children) {
      total += countAllNodes(node.children)
    }
  }

  return total
}

// Flatten the tree into the rows currently visible: a directory contributes
// its children only while open (per `isOpen`, which receives node ids), and
// every row carries its nesting depth. The virtualized scroller mounts only
// the rows in this list, so an open folder with tens of thousands of files
// never materializes every row in the DOM.
export function flattenReviewRows(
  nodes: ReviewTreeNode[],
  isOpen: (id: string) => boolean,
  depth = 0,
  rows: ReviewFlatRow[] = []
): ReviewFlatRow[] {
  for (const node of nodes) {
    rows.push({ depth, node })

    if (node.isDir && node.children && isOpen(node.id)) {
      flattenReviewRows(node.children, isOpen, depth + 1, rows)
    }
  }

  return rows
}
