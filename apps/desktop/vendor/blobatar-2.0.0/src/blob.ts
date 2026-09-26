import { makeBlobatar, type BlobatarOptions } from "./render";
import { style } from "./styles/blob";
import type { Layout } from "./styles/compose";
import type { Traits } from "./traits";

export type { BlobatarOptions };
export type { TraitOverrides } from "./traits";

/**
 * Every silhouette name Blobatar 2 can produce.
 *
 * Hand-written because which names are possible is a property of the private
 * band table, not of the generic composer. Kept narrow because callers use the
 * layout export to filter bulk seeds for a named silhouette, where `string`
 * would let a typo through silently.
 */
export type Shape =
  | "round" | "boxy" | "organic" | "cloud" | "sun" | "nub"
  | "capsule" | "triangle" | "hexagon" | "droplet";

/**
 * The renderer alone, without the colour and trait utilities the barrel also
 * carries. Import this when all you do is render.
 */
export const blobatar = makeBlobatar(style);

/**
 * The numeric layout for a set of traits, without resolving a palette or
 * rendering. Exposed for callers that need a seed's `shape` in bulk — filtering
 * thousands of seeds down to the rare silhouettes costs a hash and some
 * arithmetic this way, where going through `_layout` would also resolve an
 * OKLCh palette per candidate.
 *
 * The cast is the one place the narrower `shape` type is asserted rather than
 * inferred, and it is sound by the same thing that makes it necessary: the band
 * table in `styles/blob.ts` names exactly these silhouettes, and the golden
 * fixture freezes that it always will within this major.
 */
export const layout = style.layout as (
  t: Traits,
) => Omit<Layout, "shape"> & { shape: Shape };
