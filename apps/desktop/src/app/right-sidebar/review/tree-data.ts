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
