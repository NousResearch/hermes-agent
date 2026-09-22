const FILE_EXTENSION =
  '(?:aac|avi|csv|docx?|flac|gif|html?|jpe?g|json|m4a|md|mp3|mp4|ogg|pdf|png|py|rs|tsx?|txt|wav|webm|webp|ya?ml)'

const FILE_PATH = new RegExp(`(?:[A-Za-z]:)?(?:[\\\\/][^\\s\\\\/]+)+\\.${FILE_EXTENSION}\\b`, 'gi')
const FILE_NAME = new RegExp(`\\b[A-Za-z0-9][A-Za-z0-9._-]*\\.${FILE_EXTENSION}\\b`, 'gi')
const UUID = /\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b/gi
const DENSE_IDENTIFIER = /\b(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b/g
const CODE_IDENTIFIER = /\b[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+\(\)/g

/** Replaces tokens that sound like corrupted audio when passed verbatim to TTS. */
export function normalizeTextForSpeech(text: string): string {
  return text
    .replace(FILE_PATH, 'a file')
    .replace(FILE_NAME, 'a file')
    .replace(UUID, 'an identifier')
    .replace(DENSE_IDENTIFIER, 'an identifier')
    .replace(CODE_IDENTIFIER, 'an identifier')
}
