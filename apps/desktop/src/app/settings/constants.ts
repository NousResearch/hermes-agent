import {
  Brain,
  type IconComponent,
  Lock,
  MessageCircle,
  Mic,
  Monitor,
  Moon,
  Palette,
  Sparkles,
  Sun,
  Wrench
} from '@/lib/icons'
import type { ThemeMode } from '@/themes/context'

import type { DesktopConfigSection } from './types'

interface ProviderPrefix {
  prefix: string
  name: string
  priority: number
}

export const EMPTY_SELECT_VALUE = '__hermes_empty__'
export const CONTROL_TEXT = 'text-[0.8125rem]'

export const PROVIDER_GROUPS: ProviderPrefix[] = [
  { prefix: 'NOUS_', name: 'Nous Portal', priority: 0 },
  { prefix: 'ANTHROPIC_', name: 'Anthropic', priority: 1 },
  { prefix: 'DASHSCOPE_', name: 'DashScope (Qwen)', priority: 2 },
  { prefix: 'HERMES_QWEN_', name: 'DashScope (Qwen)', priority: 2 },
  { prefix: 'DEEPSEEK_', name: 'DeepSeek', priority: 3 },
  { prefix: 'GOOGLE_', name: 'Gemini', priority: 4 },
  { prefix: 'GEMINI_', name: 'Gemini', priority: 4 },
  { prefix: 'GLM_', name: 'GLM / Z.AI', priority: 5 },
  { prefix: 'ZAI_', name: 'GLM / Z.AI', priority: 5 },
  { prefix: 'Z_AI_', name: 'GLM / Z.AI', priority: 5 },
  { prefix: 'HF_', name: 'Hugging Face', priority: 6 },
  { prefix: 'KIMI_', name: 'Kimi / Moonshot', priority: 7 },
  { prefix: 'MINIMAX_', name: 'MiniMax', priority: 8 },
  { prefix: 'MINIMAX_CN_', name: 'MiniMax (China)', priority: 9 },
  { prefix: 'OPENCODE_GO_', name: 'OpenCode Go', priority: 10 },
  { prefix: 'OPENCODE_ZEN_', name: 'OpenCode Zen', priority: 11 },
  { prefix: 'OPENROUTER_', name: 'OpenRouter', priority: 12 },
  { prefix: 'XIAOMI_', name: 'Xiaomi MiMo', priority: 13 }
]

export const BUILTIN_PERSONALITIES = [
  'helpful',
  'concise',
  'technical',
  'creative',
  'teacher',
  'kawaii',
  'catgirl',
  'pirate',
  'shakespeare',
  'surfer',
  'noir',
  'uwu',
  'philosopher',
  'hype'
]

// Schema-side select overrides for desktop-relevant enum fields whose
// backend schema only declares a string type.
export const ENUM_OPTIONS: Record<string, string[]> = {
  'agent.image_input_mode': ['auto', 'native', 'text'],
  'approvals.mode': ['manual', 'smart', 'off'],
  'code_execution.mode': ['project', 'strict'],
  'context.engine': ['compressor', 'default', 'custom'],
  'delegation.reasoning_effort': ['', 'minimal', 'low', 'medium', 'high', 'xhigh'],
  'memory.provider': ['', 'builtin', 'honcho'],
  'stt.elevenlabs.model_id': ['scribe_v2', 'scribe_v1'],
  'stt.local.model': ['tiny', 'base', 'small', 'medium', 'large-v3'],
  'tts.openai.voice': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
}

export const FIELD_LABELS: Record<string, string> = {
  model: '默认模型',
  model_context_length: '上下文窗口',
  fallback_providers: '备用模型',
  toolsets: '已启用的工具集',
  timezone: '时区',
  'display.personality': '个性设置',
  'display.show_reasoning': '推理显示',
  'agent.max_turns': '最大代理步骤',
  'agent.image_input_mode': '图片附件',
  'terminal.cwd': '工作目录',
  'terminal.backend': '执行后端',
  'terminal.timeout': '命令超时',
  'terminal.persistent_shell': '持久 Shell',
  'terminal.env_passthrough': '环境变量透传',
  file_read_max_chars: '文件读取限制',
  'tool_output.max_bytes': '终端输出限制',
  'tool_output.max_lines': '文件页数限制',
  'tool_output.max_line_length': '行长度限制',
  'code_execution.mode': '代码执行模式',
  'approvals.mode': '审批模式',
  'approvals.timeout': '审批超时',
  'approvals.mcp_reload_confirm': '确认 MCP 重载',
  command_allowlist: '命令白名单',
  'security.redact_secrets': '隐藏密钥',
  'security.allow_private_urls': '允许私有 URL',
  'browser.allow_private_urls': '浏览器私有 URL',
  'browser.auto_local_for_private_urls': '私有 URL 本地浏览器',
  'checkpoints.enabled': '文件检查点',
  'checkpoints.max_snapshots': '检查点限制',
  'voice.record_key': '语音快捷键',
  'voice.max_recording_seconds': '最长录音时长',
  'voice.auto_tts': '朗读回复',
  'stt.enabled': '语音转文字',
  'stt.provider': '语音转文字提供商',
  'stt.local.model': '本地转录模型',
  'stt.local.language': '转录语言',
  'stt.elevenlabs.model_id': 'ElevenLabs STT 模型',
  'stt.elevenlabs.language_code': 'ElevenLabs 语言',
  'stt.elevenlabs.tag_audio_events': '标记音频事件',
  'stt.elevenlabs.diarize': '说话人分离',
  'tts.provider': '文字转语音提供商',
  'tts.edge.voice': 'Edge 语音',
  'tts.openai.model': 'OpenAI TTS 模型',
  'tts.openai.voice': 'OpenAI 语音',
  'tts.elevenlabs.voice_id': 'ElevenLabs 语音',
  'tts.elevenlabs.model_id': 'ElevenLabs 模型',
  'memory.memory_enabled': '持久记忆',
  'memory.user_profile_enabled': '用户画像',
  'memory.memory_char_limit': '记忆预算',
  'memory.user_char_limit': '画像预算',
  'memory.provider': '记忆提供商',
  'context.engine': '上下文引擎',
  'compression.enabled': '自动压缩',
  'compression.threshold': '压缩阈值',
  'compression.target_ratio': '压缩目标',
  'compression.protect_last_n': '受保护的最新消息',
  'agent.api_max_retries': 'API 重试次数',
  'agent.service_tier': '服务层级',
  'agent.tool_use_enforcement': '工具使用强制',
  'delegation.model': '子代理模型',
  'delegation.provider': '子代理提供商',
  'delegation.max_iterations': '子代理轮次限制',
  'delegation.max_concurrent_children': '并行子代理',
  'delegation.child_timeout_seconds': '子代理超时',
  'delegation.reasoning_effort': '子代理推理力度'
}

export const FIELD_DESCRIPTIONS: Record<string, string> = {
  model: 'Used for new chats unless you pick a different model in the composer.',
  model_context_length: "Leave at 0 to use the selected model's detected context window.",
  fallback_providers: 'Backup provider:model entries to try if the default model fails.',
  'display.personality': 'Default assistant style for new sessions.',
  timezone: 'Used when Hermes needs local time context. Blank uses the system timezone.',
  'display.show_reasoning': 'Show reasoning sections when the backend provides them.',
  'agent.image_input_mode': 'Controls how image attachments are sent to the model.',
  'terminal.cwd': 'Default project folder for tool and terminal work.',
  'code_execution.mode': 'How strictly code execution is scoped to the current project.',
  'terminal.persistent_shell': 'Keep shell state between commands when the backend supports it.',
  'terminal.env_passthrough': 'Environment variables to pass into tool execution.',
  file_read_max_chars: 'Maximum characters Hermes can read from one file request.',
  'approvals.mode': 'How Hermes handles commands that need explicit approval.',
  'approvals.timeout': 'How long approval prompts wait before timing out.',
  'security.redact_secrets': 'Hide detected secrets from model-visible content when possible.',
  'checkpoints.enabled': 'Create rollback snapshots before file edits.',
  'memory.memory_enabled': 'Save durable memories that can help future sessions.',
  'memory.user_profile_enabled': 'Maintain a compact profile of user preferences.',
  'context.engine': 'Strategy for managing long conversations near the context limit.',
  'compression.enabled': 'Summarize older context when conversations get large.',
  'voice.auto_tts': 'Automatically speak assistant responses.',
  'stt.enabled': 'Enable local or provider-backed speech transcription.',
  'stt.elevenlabs.language_code': 'Optional ISO-639-3 language code. Blank lets ElevenLabs auto-detect.',
  'agent.max_turns': 'Upper bound for tool-calling turns before Hermes stops a run.'
}

// Curated desktop config surface: only fields a user might tune from the app.
export const SECTIONS: DesktopConfigSection[] = [
  {
    id: 'model',
    label: '模型',
    icon: Sparkles,
    keys: ['model_context_length', 'fallback_providers']
  },
  {
    id: 'chat',
    label: '对话',
    icon: MessageCircle,
    keys: ['display.personality', 'timezone', 'display.show_reasoning', 'agent.image_input_mode']
  },
  {
    id: 'appearance',
    label: '外观',
    icon: Palette,
    keys: []
  },
  {
    id: 'workspace',
    label: '工作区',
    icon: Monitor,
    keys: [
      'terminal.cwd',
      'code_execution.mode',
      'terminal.persistent_shell',
      'terminal.env_passthrough',
      'file_read_max_chars'
    ]
  },
  {
    id: 'safety',
    label: '安全',
    icon: Lock,
    keys: [
      'approvals.mode',
      'approvals.timeout',
      'approvals.mcp_reload_confirm',
      'command_allowlist',
      'security.redact_secrets',
      'security.allow_private_urls',
      'browser.allow_private_urls',
      'browser.auto_local_for_private_urls',
      'checkpoints.enabled'
    ]
  },
  {
    id: 'memory',
    label: '记忆与上下文',
    icon: Brain,
    keys: [
      'memory.memory_enabled',
      'memory.user_profile_enabled',
      'memory.memory_char_limit',
      'memory.user_char_limit',
      'memory.provider',
      'context.engine',
      'compression.enabled',
      'compression.threshold',
      'compression.target_ratio',
      'compression.protect_last_n'
    ]
  },
  {
    id: 'voice',
    label: '语音',
    icon: Mic,
    keys: [
      'tts.provider',
      'stt.enabled',
      'stt.provider',
      'voice.auto_tts',
      'tts.edge.voice',
      'tts.openai.model',
      'tts.openai.voice',
      'tts.elevenlabs.voice_id',
      'tts.elevenlabs.model_id',
      'stt.local.model',
      'stt.local.language',
      'stt.elevenlabs.model_id',
      'stt.elevenlabs.language_code',
      'stt.elevenlabs.tag_audio_events',
      'stt.elevenlabs.diarize',
      'voice.record_key',
      'voice.max_recording_seconds'
    ]
  },
  {
    id: 'advanced',
    label: '高级',
    icon: Wrench,
    keys: [
      'toolsets',
      'terminal.backend',
      'terminal.timeout',
      'tool_output.max_bytes',
      'tool_output.max_lines',
      'tool_output.max_line_length',
      'checkpoints.max_snapshots',
      'agent.max_turns',
      'agent.api_max_retries',
      'agent.service_tier',
      'agent.tool_use_enforcement',
      'delegation.model',
      'delegation.provider',
      'delegation.max_iterations',
      'delegation.max_concurrent_children',
      'delegation.child_timeout_seconds',
      'delegation.reasoning_effort'
    ]
  }
]

export interface ModeOption {
  id: ThemeMode
  label: string
  description: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  { id: 'light', label: '浅色', description: '明亮的桌面界面', icon: Sun },
  { id: 'dark', label: '深色', description: '低眩光工作区', icon: Moon },
  { id: 'system', label: '跟随系统', description: '跟随操作系统外观', icon: Monitor }
]

export const SEARCH_PLACEHOLDER: Record<'about' | 'config' | 'gateway' | 'keys' | 'mcp' | 'sessions', string> = {
  about: '关于 Hermes 桌面版',
  config: '搜索设置...',
  gateway: '网关连接...',
  keys: '搜索 API 密钥...',
  mcp: '搜索 MCP 服务...',
  sessions: '搜索已存档的会话...'
}
