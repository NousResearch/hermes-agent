import path from 'node:path'

import { rehashPayloadDigests } from './payload-digests.mjs'

/**
 * @param {Pick<import('app-builder-lib').ElectronSignOptions, 'entitlements' | 'entitlementsInherit' | 'hardenedRuntime'> & { ignore: (file: string) => boolean }} policy
 * @returns {(opts: import('@electron/osx-sign').SignOptions, packager: import('app-builder-lib').MacPackager) => Promise<void>}
 */
export function createMacSigner(policy) {
  return async (opts, packager) => {
    const target = opts.platform === 'mas' ? (opts.type === 'development' ? 'mas-dev' : 'mas') : 'mac'
    const optionsForFile = await packager.helper.getOptionsForFile(opts.app, target, policy)
    const inheritedIgnore = opts.ignore
    const { sign } = await import('@electron/osx-sign')
    await sign({
      ...opts,
      ignore: file => (typeof inheritedIgnore === 'function' && inheritedIgnore(file)) || policy.ignore(file),
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
