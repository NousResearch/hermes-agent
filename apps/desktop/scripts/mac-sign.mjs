import path from 'node:path'

import { createMacSignCache, identityTeam, looseMachOFiles } from './mac-sign-cache.mjs'
import { rehashPayloadDigests } from './payload-digests.mjs'

/**
 * @param {Pick<import('app-builder-lib').ElectronSignOptions, 'entitlements' | 'entitlementsInherit' | 'hardenedRuntime'> & { ignore: (file: string) => boolean }} policy
 * @param {{ env?: NodeJS.ProcessEnv, makeCache?: typeof createMacSignCache, team?: typeof identityTeam, sign?: typeof import('@electron/osx-sign').sign }} [deps]
 * @returns {(opts: import('@electron/osx-sign').SignOptions, packager: { helper: Pick<import('app-builder-lib').MacPackager['helper'], 'getOptionsForFile'> }) => Promise<void>}
 */
export function createMacSigner(policy, { env = process.env, makeCache = createMacSignCache, team = identityTeam, sign: signApp } = {}) {
  return async (opts, packager) => {
    const target = opts.platform === 'mas' ? (opts.type === 'development' ? 'mas-dev' : 'mas') : 'mac'
    const optionsForFile = await packager.helper.getOptionsForFile(opts.app, target, policy)
    const inheritedIgnore = opts.ignore
    /** @param {string} file */
    const ignored = file => (typeof inheritedIgnore === 'function' && inheritedIgnore(file)) || policy.ignore(file)
    // Release builds (HERMES_MAC_SIGNATURE_CACHE) sign loose payload binaries through
    // the signature cache with the arguments osx-sign would use (inherited
    // entitlements, hardened runtime, secure timestamp); osx-sign then skips them
    // and still seals every bundle and the app.
    const identity = opts.identity
    const enabled = target === 'mac' && env.HERMES_MAC_SIGNATURE_CACHE && identity &&
      policy.entitlementsInherit && policy.hardenedRuntime
    const cache = enabled && policy.entitlementsInherit ? makeCache({
      root: env.HERMES_MAC_SIGNATURE_CACHE, teamId: await team(identity, opts.keychain), identity,
      keychain: opts.keychain, entitlements: policy.entitlementsInherit
    }) : null
    /** @type {Set<string>} */
    const presigned = new Set()
    if (cache) {
      const files = looseMachOFiles(opts.app, ignored)
      const plan = await cache.prepare(files)
      await cache.sign(plan)
      for (const file of files) presigned.add(file)
      console.log(`mac signature cache: ${files.length} loose binaries, ${plan.restored} restored, ` +
        `${plan.pending.length} signed`)
    }
    const sign = signApp ?? (await import('@electron/osx-sign')).sign
    await sign({
      ...opts,
      ignore: file => presigned.has(file) || ignored(file),
      // Batching defers child signatures until after all option callbacks.
      batchCodesignCalls: false,
      optionsForFile: file => {
        if (file === opts.app) {
          rehashPayloadDigests(path.join(opts.app, 'Contents', 'Resources', 'agent-payload'))
        }
        return optionsForFile(file)
      }
    })
  }
}
