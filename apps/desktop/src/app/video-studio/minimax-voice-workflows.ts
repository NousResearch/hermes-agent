export interface MiniMaxCloneValidationInput {
  cloneFile: boolean
  promptFile: boolean
  promptText: string
  voiceId: string
}

export function createMiniMaxCloneVoiceId(prefix = 'Hermes', now = new Date()): string {
  const safePrefix = prefix.replace(/[^A-Za-z0-9]/g, '') || 'Hermes'
  const stamp = now.toISOString().replace(/\D/g, '').slice(4, 14)

  return `${safePrefix}Clone${stamp}`
}

export function miniMaxVoiceErrorMessage(message = ''): string {
  return /voice (?:clone )?voice id duplicate/i.test(message)
    ? '该 ID 已存在；请在已有音色中使用它，或为克隆生成新的 ID。'
    : message
}

export function miniMaxVoiceName(voiceId: string): string {
  return `minimax:${voiceId.trim()}`
}

export function validateMiniMaxCloneInput(input: MiniMaxCloneValidationInput): string {
  const voiceId = input.voiceId.trim()

  if (!voiceId) {
    return '请填写新的克隆 Voice ID。'
  }

  if (!/^[A-Za-z][A-Za-z0-9_-]{6,254}[A-Za-z0-9]$/.test(voiceId)) {
    return 'Voice ID 需为 8-256 位，首字符为英文字母，且只能包含字母、数字、- 和 _。'
  }

  if (!input.cloneFile) {
    return '请选择用于复刻的音频文件。'
  }

  if (input.promptFile !== Boolean(input.promptText.trim())) {
    return '参考音频和参考音频文本必须同时提供。'
  }

  return ''
}
