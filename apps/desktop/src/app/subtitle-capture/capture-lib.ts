// Re-export the shared sampling helpers. The live loop now snapshots in
// Electron main (periodic window thumbnails, not a persistent display
// stream); these stay importable from the renderer test file.
export {
  BRIGHT_LUMA,
  brightMaskHash,
  cropRectFor,
  hammingDistance,
  HASH_HEIGHT,
  HASH_WIDTH,
  SAME_TEXT_MAX_DISTANCE,
  SHIP_MAX_WIDTH,
  shipSize
} from '../../../electron/subtitle-capture'
