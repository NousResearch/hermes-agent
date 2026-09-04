// update-root-policy.ts — pure classifier for the desktop self-update root.
//
// The desktop's git-based self-update arm runs `git fetch/merge` against a
// checkout it resolved (resolveUpdateRoot in main.ts). That root is only
// legitimate update territory when the install contract says so:
//
//   - Not a `.git` tree → there is nothing to pull; the desktop cannot
//     self-update here (bundled installs use the OS App Installer arm and
//     never reach this classifier).
//   - A `.git` tree whose install stamp says `updateMechanism: "self"` → the
//     install is a managed self-updating checkout (e.g. the desktop
//     bootstrap's clone, which writes exactly that stamp) — updatable.
//   - A `.git` tree with any OTHER mechanism (`external`: the store/steward
//     owns updates; `electron-updater`: the desktop package does) → the tree
//     is managed by someone else. Pulling into it from the desktop would
//     stash-and-move a user's checkout out from under its steward.
//   - No stamp at all (null mechanism; a dev source checkout) → provenance
//     unknown. The classifier reports `updatable` with `provenance:
//     'unknown'`: a developer's working tree keeps its existing update flow,
//     and callers log the ambiguity rather than silently widening the refusal.
//
// Pure and dependency-injected (same shape as update-gate.ts) so the policy
// is unit-testable without booting Electron, and every caller gets the same
// answer from one authority.

export type UpdateMechanism = 'self' | 'electron-updater' | 'external'

export type UpdateRootProvenance = 'managed-self' | 'steward-owned' | 'unknown' | 'not-a-checkout'

export interface UpdateRootClassification {
  /** Whether the desktop's git-based self-update may run against this root. */
  updatable: boolean
  /** Machine-readable verdict for update-check results and logs. */
  verdict: 'updatable' | 'not-a-checkout' | 'steward-owned-git-tree' | 'unmanaged-git-tree'
  /** Why the root is (or is not) updatable, for user-facing messages. */
  message: string | null
  /** The command that fixes it, when the user (not the app) must act. */
  advice: 'git pull' | null
  provenance: UpdateRootProvenance
}

export interface UpdateRootFacts {
  /** True when the resolved update root contains a `.git` entry. */
  isGitTree: boolean
  /** The install stamp's updateMechanism, or null when there is no stamp. */
  updateMechanism: UpdateMechanism | null
}

/**
 * Classify whether the desktop may self-update (git pull semantics) against
 * the resolved update root.
 */
export function classifyUpdateRoot(facts: UpdateRootFacts): UpdateRootClassification {
  if (!facts.isGitTree) {
    return {
      updatable: false,
      verdict: 'not-a-checkout',
      message: 'This install has no git checkout to update — the app or its steward owns the update loop.',
      advice: null,
      provenance: 'not-a-checkout'
    }
  }

  if (facts.updateMechanism === 'self') {
    return {
      updatable: true,
      verdict: 'updatable',
      message: null,
      advice: null,
      provenance: 'managed-self'
    }
  }

  if (facts.updateMechanism === 'external' || facts.updateMechanism === 'electron-updater') {
    return {
      updatable: false,
      verdict: 'steward-owned-git-tree',
      message:
        `This checkout is managed by its install method (${facts.updateMechanism}); ` +
        'the desktop will not run git updates against it.',
      advice: 'git pull',
      provenance: 'steward-owned'
    }
  }

  // No stamp: unknown provenance (typically a developer source checkout).
  return {
    updatable: true,
    verdict: 'updatable',
    message: null,
    advice: null,
    provenance: 'unknown'
  }
}
