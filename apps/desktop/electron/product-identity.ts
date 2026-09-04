// product-identity.ts — the typed build-time product identity.
//
// product-identity.cjs is the single derivation of every name-shaped
// value a variant owns (product name, appId, channel). Dev bundles and
// test runs import the .cjs itself so the derivation exists in exactly
// one place.

import devIdentity from '../product-identity.cjs'

/** Mirrors product-identity.cjs (see product-identity.d.cts). */
export type ProductIdentity = typeof devIdentity

declare const __HERMES_PRODUCT_IDENTITY__: ProductIdentity

/** The baked identity of this artifact (dev bundles derive it live). */
export const PRODUCT_IDENTITY: Readonly<ProductIdentity> =
  typeof __HERMES_PRODUCT_IDENTITY__ === 'undefined'
    ? Object.freeze(devIdentity)
    : Object.freeze(__HERMES_PRODUCT_IDENTITY__)
