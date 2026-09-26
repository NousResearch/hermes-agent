import { atom, computed } from 'nanostores'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import type { HermesReviewCommit, HermesReviewCommitStack } from '@/global'
import { desktopGit } from '@/lib/desktop-git'

import { $reviewOpen, $reviewScopeTarget, reviewRepoCwd } from './review'
import { $busy, $currentCwd } from './session'
import { $workspaceChangeTick } from './workspace-events'

// The review pane's commit stack: the branch's commits since it split from
// trunk, and the one the user is reviewing. Pairs with "Restack", which hands
// the agent the job of rewriting that stack into reviewable commits — the tree
// stays byte-identical, only the history changes, so the pane re-reads the
// stack afterwards and lets you walk it commit by commit.

export const $reviewCommitStack = atom<HermesReviewCommitStack>({ base: null, commits: [] })
export const $reviewStackLoading = atom(false)
// Sha of the commit whose patch is showing; null = the working-tree view.
export const $reviewStackSelected = atom<null | string>(null)
export const $reviewStackDiff = atom<null | string>(null)
export const $reviewStackDiffLoading = atom(false)
// True from "Restack" press until the agent's turn settles.
export const $reviewRestackPending = atom(false)

export const $reviewStackCommits = computed($reviewCommitStack, stack => stack.commits)

export const $reviewStackSelectedCommit = computed(
  [$reviewCommitStack, $reviewStackSelected],
  (stack, sha): HermesReviewCommit | null => (sha ? stack.commits.find(c => c.sha === sha) ?? null : null)
)

let stackSeq = 0

type StackBridge = NonNullable<NonNullable<NonNullable<Window['hermesDesktop']>['git']>['review']>

function stackCtx(): { cwd: string; review: StackBridge } | null {
  const cwd = reviewRepoCwd()
  const review = desktopGit()?.review

  // Older preloads (or the remote-fs bridge) may lack the stack ops.
  return cwd && review?.commitStack ? { cwd, review } : null
}

export async function refreshCommitStack(): Promise<void> {
  const ctx = stackCtx()
  const seq = (stackSeq += 1)

  if (!$reviewOpen.get() || !ctx) {
    $reviewCommitStack.set({ base: null, commits: [] })
    $reviewStackLoading.set(false)

    return
  }

  $reviewStackLoading.set(true)

  try {
    const stack = await ctx.review.commitStack(ctx.cwd)

    if (seq !== stackSeq || reviewRepoCwd() !== ctx.cwd) {
      return
    }

    $reviewCommitStack.set(stack)

    // A restack rewrote the shas → the selected commit no longer exists.
    const selected = $reviewStackSelected.get()

    if (selected && !stack.commits.some(c => c.sha === selected)) {
      clearStackSelection()
    }
  } catch {
    if (seq === stackSeq) {
      $reviewCommitStack.set({ base: null, commits: [] })
    }
  } finally {
    if (seq === stackSeq) {
      $reviewStackLoading.set(false)
    }
  }
}

export async function selectStackCommit(sha: null | string, filePath: null | string = null): Promise<void> {
  if (!sha) {
    clearStackSelection()

    return
  }

  $reviewStackSelected.set(sha)

  const ctx = stackCtx()

  if (!ctx) {
    $reviewStackDiff.set(null)

    return
  }

  $reviewStackDiffLoading.set(true)

  try {
    const diff = await ctx.review.commitDiff(ctx.cwd, sha, filePath)

    if ($reviewStackSelected.get() === sha) {
      $reviewStackDiff.set(diff || '')
    }
  } catch {
    if ($reviewStackSelected.get() === sha) {
      $reviewStackDiff.set('')
    }
  } finally {
    if ($reviewStackSelected.get() === sha) {
      $reviewStackDiffLoading.set(false)
    }
  }
}

export function clearStackSelection(): void {
  $reviewStackSelected.set(null)
  $reviewStackDiff.set(null)
  $reviewStackDiffLoading.set(false)
}

// Build the restack prompt. The pane knows the exact base and the current
// stack, so the agent gets the range spelled out instead of guessing which
// commits are "the branch's". Returns null when there is nothing to restack:
// no trunk to measure from, or no commits AND a clean tree.
export function restackPrompt(
  stack: HermesReviewCommitStack,
  dirty: boolean,
  copy: { intro: string; rules: string; commitLine: (short: string, subject: string) => string; dirtyNote: string }
): null | string {
  if (!stack.base) {
    return null
  }

  if (stack.commits.length === 0 && !dirty) {
    return null
  }

  const lines = [copy.intro.replace('{base}', stack.base.slice(0, 12))]

  if (stack.commits.length > 0) {
    lines.push('', ...stack.commits.map(c => copy.commitLine(c.short, c.subject)))
  }

  if (dirty) {
    lines.push('', copy.dirtyNote)
  }

  lines.push('', copy.rules)

  return lines.join('\n')
}

// Hand the restack to the agent through the composer that owns this review
// scope. 'nothing' = no stack to rewrite; 'unavailable' = no visible composer
// to receive it (the caller toasts either way).
export function requestRestack(
  copy: Parameters<typeof restackPrompt>[2],
  dirty: boolean
): 'nothing' | 'sent' | 'unavailable' {
  const prompt = restackPrompt($reviewCommitStack.get(), dirty, copy)

  if (!prompt) {
    return 'nothing'
  }

  if (!requestComposerSubmit(prompt, { target: $reviewScopeTarget.get() })) {
    return 'unavailable'
  }

  $reviewRestackPending.set(true)
  clearStackSelection()

  return 'sent'
}

// ── Triggers ─────────────────────────────────────────────────────────────────

// The stack changes on commits, not file edits — but a rebase/restack shows up
// as a burst of tree changes, so share the pane's refresh edges.
$workspaceChangeTick.subscribe(() => {
  if ($reviewOpen.get()) {
    void refreshCommitStack()
  }
})

let prevBusy = $busy.get()

$busy.subscribe(busy => {
  if (prevBusy && !busy) {
    $reviewRestackPending.set(false)

    if ($reviewOpen.get()) {
      void refreshCommitStack()
    }
  }

  prevBusy = busy
})

$reviewOpen.subscribe(open => {
  if (open) {
    void refreshCommitStack()
  } else {
    clearStackSelection()
  }
})

$currentCwd.subscribe(() => {
  clearStackSelection()

  if ($reviewOpen.get()) {
    void refreshCommitStack()
  }
})

if (typeof window !== 'undefined') {
  window.addEventListener('focus', () => {
    if ($reviewOpen.get()) {
      void refreshCommitStack()
    }
  })
}
