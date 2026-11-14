// Per-change (hunk) stage / unstage / revert for the review pane. The renderer
// names a hunk by its exact text; the patch is rebuilt here from a fresh diff, so
// a hunk the file no longer contains (the agent kept editing) is refused instead
// of being applied somewhere it was never shown.

import fs from 'node:fs/promises'
import path from 'node:path'

import { singleHunkPatch } from '../../shared/src/diff-hunks'

import { gitFor } from './git-review-ops'
import { resolveRequestedPathForIpc } from './hardening'

export type ReviewHunkAction = 'revert' | 'stage' | 'unstage'

// `unstage` reverses the hunk out of the index (the staged diff); `stage` and
// `revert` act on the working-tree diff, into the index or out of the file.
const APPLY_FLAGS: Record<ReviewHunkAction, string[]> = {
  revert: ['-R'],
  stage: ['--cached'],
  unstage: ['--cached', '-R']
}

export const STALE_HUNK_ERROR = 'This change no longer matches the file. Refresh the review and try again.'

async function reviewHunk(repoPath, filePath, hunk, action: ReviewHunkAction, gitBin) {
  const flags = APPLY_FLAGS[action]

  if (!flags || !filePath || !hunk) {
    throw new Error('Invalid hunk request')
  }

  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: 'Review hunk' })
  const git = gitFor(cwd, gitBin)

  // Fixed a/ b/ prefixes: `git apply` strips one level, whatever diff.noprefix
  // or diff.mnemonicPrefix the user configured for the displayed diff.
  const diff = await git.diff([
    '--no-color',
    '--no-ext-diff',
    '--no-textconv',
    '--src-prefix=a/',
    '--dst-prefix=b/',
    ...(action === 'unstage' ? ['--cached'] : []),
    '--',
    filePath
  ])

  const patch = singleHunkPatch(diff, hunk)

  if (!patch) {
    throw new Error(STALE_HUNK_ERROR)
  }

  const gitDir = (await git.raw(['rev-parse', '--absolute-git-dir'])).trim()
  const patchPath = path.join(gitDir, `hermes-review-hunk-${process.pid}-${Date.now()}.patch`)

  await fs.writeFile(patchPath, patch, 'utf8')

  try {
    await git.raw(['apply', ...flags, '--', patchPath])
  } finally {
    await fs.rm(patchPath, { force: true })
  }

  return { ok: true }
}

export { reviewHunk }
