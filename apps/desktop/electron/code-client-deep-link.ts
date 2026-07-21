export type CodeClient = 'claude-code' | 'codex'

export interface CodeClientLaunch {
  client: CodeClient
  cwd: string
  prompt: string
}

const PARENT_SEGMENT = /(?:^|[\\/])\.\.(?:[\\/]|$)/
const MAX_PROMPT_LENGTH = 5000

function hasUnsafePathCharacters(value: string): boolean {
  return [...value].some(character => {
    const codePoint = character.codePointAt(0) ?? 0

    return (
      codePoint <= 0x1f ||
      (codePoint >= 0x7f && codePoint <= 0x9f) ||
      (codePoint >= 0x200b && codePoint <= 0x200f) ||
      (codePoint >= 0x202a && codePoint <= 0x202e) ||
      (codePoint >= 0x2060 && codePoint <= 0x206f)
    )
  })
}

function validLocalDirectory(value: string, platform: NodeJS.Platform): boolean {
  const fullyQualified = platform === 'win32' ? /^[A-Za-z]:[\\/]/.test(value) : value.startsWith('/')

  return (
    value.length > 0 &&
    fullyQualified &&
    !/^[\\/]{2}/.test(value) &&
    !hasUnsafePathCharacters(value) &&
    !PARENT_SEGMENT.test(value)
  )
}

/** Build an official, review-before-send code-client deep link. */
export function codeClientDeepLink(value: unknown, platform: NodeJS.Platform = process.platform): string {
  if (!value || typeof value !== 'object') {
    throw new Error('Invalid code client request')
  }

  const input = value as Partial<CodeClientLaunch>

  if (typeof input.client !== 'string' || typeof input.cwd !== 'string' || typeof input.prompt !== 'string') {
    throw new Error('Invalid code client request')
  }

  if (input.client !== 'codex' && input.client !== 'claude-code') {
    throw new Error('Unsupported code client')
  }

  if (!validLocalDirectory(input.cwd, platform)) {
    throw new Error('Invalid local working directory')
  }

  if (input.prompt.length > MAX_PROMPT_LENGTH) {
    throw new Error(`Prompt exceeds ${MAX_PROMPT_LENGTH} characters`)
  }

  if (input.client === 'codex') {
    const prompt = input.prompt ? `&prompt=${encodeURIComponent(input.prompt)}` : ''

    return `codex://new?path=${encodeURIComponent(input.cwd)}${prompt}`
  }

  const prompt = input.prompt ? `&q=${encodeURIComponent(input.prompt)}` : ''

  return `claude-cli://open?cwd=${encodeURIComponent(input.cwd)}${prompt}`
}
