import type { FieldCopyTree } from '@/app/settings/field-copy'

// Settings > Memory & Context > Context & compression copy for this locale.
export const zhCompressionFieldLabels: FieldCopyTree = {
  context: {
    engine: '上下文引擎'
  },
  compression: {
    enabled: '自动压缩',
    threshold: '压缩阈值',
    codexGpt55Autoraise: 'Codex 压缩自动提高',
    targetRatio: '压缩目标',
    protectLastN: '保护最近消息',
    warmHandoff: '热缓存交接'
  },
  auxiliary: {
    compression: {
      timeout: '压缩模型超时（秒）'
    }
  }
}

export const zhCompressionFieldDescriptions: FieldCopyTree = {
  context: {
    engine: '在接近上下文上限时管理长对话的策略。'
  },
  compression: {
    enabled: '当对话变大时对较早的上下文进行摘要。',
    codexGpt55Autoraise: '为受支持的 ChatGPT Codex OAuth 模型将压缩阈值提高到 85%。',
    warmHandoff:
      '由主模型在其已缓存的提示上撰写压缩摘要，支持提示缓存的服务器只需读取新消息。压缩使用同一模型时更快。自动：仅当压缩使用主模型且服务器报告缓存令牌时启用。开启：始终尝试。关闭：始终使用压缩模型。任何失败都会回退到常规摘要。'
  },
  auxiliary: {
    compression: {
      timeout: '每次调用辅助压缩模型的等待秒数（默认 120）。本地模型较慢时请调高。'
    }
  }
}
