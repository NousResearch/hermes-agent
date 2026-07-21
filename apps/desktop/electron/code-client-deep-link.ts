import path from 'node:path'

export type CodeClient = 'claude-code' | 'codex'

export interface CodeClientLaunch {
  client: CodeClient
  cwd: string
  prompt: string
}

const UNSAFE_PATH_CHARACTERS = /[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u206f]/
const PARENT_SEGMENT = /(?:^|[\\/])\.\.(?:[\\/]|$)/
const MAX_PROMPT_LENGTH = 5000

function validLocalDirectory(value: string): boolean {
  return (
    value.length > 0 &&
    (path.posix.isAbsolute(value) || path.win32.isAbsolute(value)) &&
    !value.startsWith('//') &&
    !value.startsWith('\\\\') &&
    !UNSAFE_PATH_CHARACTERS.test(value) &&
    !PARENT_SEGMENT.test(value)
  )
}

/** Build an official, review-before-send code-client deep link. */
export function codeClientDeepLink(value: unknown): string {
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

  if (!validLocalDirectory(input.cwd)) {
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
