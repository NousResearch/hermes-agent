/**
 * update-branch-ref.ts
 *
 * Validation for the self-update branch name.
 *
 * The branch arrives from a renderer IPC call (`hermes:updates:branch:set`),
 * is persisted, and then reaches two places that treat it as more than data:
 *
 *  - `git ls-remote --exit-code --heads <remote> <branch>`, where a value
 *    starting with `-` is read by git as an OPTION, not a pattern;
 *  - the Windows updater's argv join, where a trailing backslash used to eat
 *    its own closing quote (fixed in scripts/desktop-update/windows.ps1).
 *
 * Rejecting at the boundary is the fix; the two call sites additionally pass
 * a `refs/heads/` prefixed pattern and quote correctly, so neither depends on
 * this alone.
 *
 * The grammar is git-check-ref-format for a branch name, minus the rules that
 * cannot matter here.
 */

/** Path prefix that makes a branch unusable as a git option or pattern shorthand. */
export const GIT_BRANCH_REF_PREFIX = 'refs/heads/'

export function isValidUpdateBranchRef(branch: unknown): branch is string {
  if (typeof branch !== 'string') {return false}

  const value = branch.trim()

  if (!value || value !== branch) {return false}

  // ASCII control characters, DEL, and space.
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u0020\u007f]/.test(value)) {return false}

  // git-check-ref-format: no ~ ^ : ? * [ \ anywhere. The backslash ban also
  // removes the trailing-backslash argv hazard. A double quote is legal in a
  // git ref and useless in a branch name, so it is refused here too rather
  // than trusted to every quoting site downstream.
  if (/[~^:?*[\\"]/.test(value)) {return false}

  // A leading dash is an option to every git command that takes a pattern.
  if (value.startsWith('-')) {return false}

  if (value.includes('..') || value.includes('@{')) {return false}

  if (value === '@' || value === 'HEAD') {return false}

  if (value.startsWith('/') || value.endsWith('/') || value.includes('//')) {return false}

  if (value.endsWith('.') || value.endsWith('.lock')) {return false}

  // No path component may start with a dot or end with .lock.
  if (value.split('/').some(part => !part || part.startsWith('.') || part.endsWith('.lock'))) {return false}

  return value.length <= 255
}

/** The fully qualified ref for a validated branch, safe to hand to git as a pattern. */
export function updateBranchRefPattern(branch: string): string {
  return `${GIT_BRANCH_REF_PREFIX}${branch}`
}
