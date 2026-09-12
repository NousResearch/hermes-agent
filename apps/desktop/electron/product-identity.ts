// product-identity.ts — the typed build-time product identity.
//
// product-identity.cjs is the single derivation of every name-shaped
// value a variant owns (product name, appId, channel). Dev bundles and
// test runs import the .cjs itself so the derivation exists in exactly
// one place.

import { mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

import type devIdentity from '../product-identity.cjs'

/** Mirrors product-identity.cjs (see product-identity.d.cts). */
export type ProductIdentity = typeof devIdentity

declare const __HERMES_PRODUCT_IDENTITY__: ProductIdentity

/** The baked identity of this artifact (dev bundles derive it live). */
export const PRODUCT_IDENTITY: Readonly<ProductIdentity> =
  typeof __HERMES_PRODUCT_IDENTITY__ === 'undefined'
    ? Object.freeze(createRequire(import.meta.url)('../product-identity.cjs') as ProductIdentity)
    : Object.freeze(__HERMES_PRODUCT_IDENTITY__)

/** Pin before the first userData lookup and single-instance lock. Electron's
 * later display-name changes must not redirect an installed build's state.
 * Stable keeps its historical userData and runtime naming behavior.
 */
export function applyDesktopIdentity(
  app: { getPath(name: 'appData'): string; setPath(name: 'userData', value: string): void; setName(name: string): void },
  identity: Readonly<ProductIdentity> = PRODUCT_IDENTITY
): string | null {
  if (identity.appNamePascal === identity.artifactNamePascal) {return null}
  const userData: string = path.join(app.getPath('appData'), identity.appNamePascal)
  mkdirSync(userData, { recursive: true })
  app.setPath('userData', userData)
  app.setName(identity.displayName)

  return identity.displayName
}
