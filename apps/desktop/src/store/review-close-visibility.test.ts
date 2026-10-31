import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/oneshot', () => ({ requestOneShot: vi.fn() }))
vi.mock('./coding-status', () => ({ refreshRepoStatus: vi.fn(), repoStatusForCwd: () => ({ get: () => null }) }))

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

async function setup(primaryCwd: string) {
  const { computed } = await import('nanostores')
  const tree = await import('@/components/pane-shell/tree/store')
  const { group, split } = await import('@/components/pane-shell/tree/model')
  const { registry } = await import('@/contrib/registry')
  const review = await import('./review')
  const { $currentCwd } = await import('./session')

  $currentCwd.set(primaryCwd)
  review.$reviewOpen.set(false)
  tree.$dismissedPanes.set(new Set())
  tree.$hiddenTreePanes.set(new Set())

  for (const [id, placement] of [
    ['workspace', 'main'],
    ['tile:project', 'main'],
    ['review', 'right']
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data: { placement }, id, render: () => null, title: id }))
  }

  tree.$layoutTree.set(
    split('row', [
      group(['workspace'], { active: 'workspace', id: 'primary' }),
      group(['tile:project'], { active: 'tile:project', id: 'tile' }),
      group(['review'], { active: 'review', id: 'review-zone' })
    ])
  )

  // The controller's production binding: a detached primary draft can have
  // no cwd while a split session tile still owns a project and a transcript.
  const $hasWorkspace = computed($currentCwd, cwd => Boolean(cwd.trim()))
  tree.bindPaneVisibility(
    'review',
    computed([review.$reviewOpen, $hasWorkspace], (open, workspace) => open && workspace),
    review.closeReview,
    () => review.openReview(review.$reviewScopeCwd.get(), review.$reviewScopeTarget.get())
  )

  const file = { path: 'a.ts', status: 'M', staged: false, added: 1, removed: 0 }

  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
    git: {
      review: {
        list: vi.fn(async () => ({ files: [file] })),
        diff: vi.fn(async () => '@@ -1 +1 @@\n-old\n+new'),
        shipInfo: vi.fn(async () => ({ ghReady: false, pr: null }))
      }
    }
  }

  return { tree, review, file }
}

describe('closing a review revealed by a split session tile', () => {
  it.each([
    ['', 'review'],
    ['', 'file'],
    ['/primary-project', 'review'],
    ['/primary-project', 'file']
  ] as const)('closes the %s primary cwd / %s card action repeatedly', async (primaryCwd, action) => {
    const { tree, review, file } = await setup(primaryCwd)

    for (let attempt = 0; attempt < 2; attempt += 1) {
      // These are the Review button and file-row actions used by ChangedFilesCard.
      if (action === 'file') {
        await review.openReviewForPath('/tile-project/a.ts', '/tile-project', 'tile:project')
      } else {
        review.revealReview('/tile-project', 'tile:project')
        await review.selectReviewFile(file)
      }

      expect(review.$reviewScopeCwd.get()).toBe('/tile-project')
      expect(review.$reviewScopeTarget.get()).toBe('tile:project')
      expect(review.$reviewSelectedPath.get()).toBe('a.ts')
      expect(tree.isPaneVisible('review')).toBe(true)

      // The pane header's x invokes this same production action.
      review.closeReview()

      expect(review.$reviewOpen.get()).toBe(false)
      expect(review.$reviewScopeCwd.get()).toBeNull()
      expect(review.$reviewScopeTarget.get()).toBe('main')
      expect(review.$reviewSelectedPath.get()).toBeNull()
      expect(review.$reviewDiff.get()).toBeNull()
      expect(tree.isPaneVisible('review')).toBe(false)
      expect(tree.isPaneVisible('tile:project')).toBe(true)
      expect(tree.$dismissedPanes.get().has('review')).toBe(false)
      review.closeReview()
      expect(tree.isPaneVisible('review')).toBe(false)
    }
  })
})
