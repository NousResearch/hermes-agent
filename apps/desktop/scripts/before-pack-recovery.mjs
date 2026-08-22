// Recovery wrapper around the canonical electron-builder beforePack hook.
// The base hook remains the owner of stale-app cleanup, rollback preservation,
// and native staging. This wrapper handles only one residual: a corrupt or
// unresolvable get-windows package root on a supported native build host.

import { Arch } from 'electron-builder'

import beforePack from './before-pack.mjs'
import {
  isMissingGetWindowsPackageError,
  stageGetWindowsWithRecovery
} from './stage-native-deps-recovery.mjs'

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

export default async function beforePackWithRecovery(context) {
  try {
    return await beforePack(context)
  } catch (error) {
    if (!isMissingGetWindowsPackageError(error)) {
      throw error
    }

    const platform = context && context.electronPlatformName
    const arch = context && typeof context.arch === 'number' ? Arch[context.arch] : undefined
    if (!platform || !arch) {
      throw error
    }

    console.warn(
      `[before-pack] canonical native staging found a corrupt get-windows package root; ` +
        `retrying ${platform}-${arch} from an isolated dependency realization`
    )
    try {
      stageGetWindowsWithRecovery({ platform, arch })
      console.log(`[before-pack] recovered and staged get-windows for target ${platform}-${arch}`)
    } catch (recoveryError) {
      throw new Error(
        `[before-pack] isolated get-windows recovery failed for ${platform}-${arch}: ${errorMessage(recoveryError)}`
      )
    }
  }
}
