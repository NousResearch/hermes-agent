import fs from 'node:fs'
import path from 'node:path'

export type ValidatedSourceResolver<TArgs, TResult> = ((args: TArgs) => Promise<TResult>) & {
  current: () => string | undefined
}

export interface UpdateRootResolutionOptions {
  overrideRoot?: string
  isPackaged: boolean
  sourceRepoRoot: string
  activeHermesRoot: string
  validatedSourceInstallRoot?: string
  isHermesSourceRoot: (root: string) => boolean
  isGitCheckout: (root: string) => boolean
}

export function isGitCheckout(root: string): boolean {
  return fs.existsSync(path.join(root, '.git'))
}

export function createValidatedSourceResolver<TArgs, TResult>(
  resolveAttempt: (args: TArgs, select: (root: string) => void) => Promise<TResult>
): ValidatedSourceResolver<TArgs, TResult> {
  let generation = 0
  let selectedRoot: string | undefined

  const resolve = async (args: TArgs): Promise<TResult> => {
    const request = ++generation
    let candidate: string | undefined

    try {
      const result = await resolveAttempt(args, (root: string): void => {
        candidate = root
      })

      if (request === generation) {
        selectedRoot = candidate
      }

      return result
    } catch (error) {
      if (request === generation) {
        selectedRoot = undefined
      }

      throw error
    }
  }

  resolve.current = (): string | undefined => selectedRoot

  return resolve
}

/** Resolve the update checkout, preserving explicit, dev, and canonical precedence. */
export function resolveUpdateRoot(options: UpdateRootResolutionOptions): string {
  const candidates = [
    options.overrideRoot && path.resolve(options.overrideRoot),
    !options.isPackaged && options.isHermesSourceRoot(options.sourceRepoRoot) ? options.sourceRepoRoot : null,
    options.isHermesSourceRoot(options.activeHermesRoot) ? options.activeHermesRoot : null,
    options.validatedSourceInstallRoot && options.isHermesSourceRoot(options.validatedSourceInstallRoot)
      ? options.validatedSourceInstallRoot
      : null
  ].filter((candidate): candidate is string => Boolean(candidate))

  return candidates.find(options.isGitCheckout) || candidates[0] || options.activeHermesRoot
}
