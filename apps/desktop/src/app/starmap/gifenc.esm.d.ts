// Type declarations for the vendored gifenc ESM build (gifenc.esm.js).
export class GIFEncoder {
  writeFrame(index: Uint8Array, width: number, height: number, options?: { delay?: number; palette?: Uint8Array; transparent?: boolean }): void
  finish(): void
  bytes(): Uint8Array
  reset(): void
}
export function quantize(rgba: Uint8Array, maxColors: number): Uint8Array
export function applyPalette(rgba: Uint8Array, palette: Uint8Array, format?: string): Uint8Array
export function nearestColorIndex(palette: Uint8Array, pixel: Uint8Array): number
export function nearestColorIndexWithDistance(palette: Uint8Array, pixel: Uint8Array, maxDistance?: number): [number, number]
export function nearestColor(palette: Uint8Array, pixel: Uint8Array): Uint8Array
export function prequantize(rgba: Uint8Array, opts?: { format?: string }): Uint8Array
export function snapColorsToPalette(palette: Uint8Array, pixels: Uint8Array, out?: Uint8Array): Uint8Array
