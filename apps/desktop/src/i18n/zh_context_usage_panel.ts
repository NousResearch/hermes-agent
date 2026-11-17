import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const zhContextUsagePanel = {
  categories: {
    conversation: '对话',
    mcp: 'MCP',
    memory: '记忆',
    rules: '规则',
    skills: '技能',
    subagent_definitions: '子代理定义',
    system_prompt: '系统提示词',
    tool_definitions: '工具定义'
  },
  compact: {
    action: '压缩对话',
    busyHint: '请先完成或停止当前回复',
    confirm: '压缩',
    hint: '总结较早的上下文以腾出空间，继续当前对话。',
    question: '将此对话压缩为摘要？',
    questionNamed: title => `将“${title}”压缩为摘要？`,
    working: '正在压缩…'
  },
  empty: '暂无上下文数据',
  loading: '正在加载明细…',
  percentFull: percent => `已用 ${percent}%`,
  title: '上下文用量',
  tokenSummary: (used, max) => `${used} / ${max} Tokens`,
  tabs: { inspector: '检查器', usage: '用量' },
  inspector: {
    changes: {
      model_changed: (before, after) => `模型已更改: ${before} → ${after}`,
      prompt_changed: '提示词已更改',
      provider_changed: (before, after) => `提供商已更改: ${before} → ${after}`,
      tool_added: name => `新增工具: ${name}`,
      tool_removed: name => `移除工具: ${name}`
    },
    error: '无法读取代理看到的内容',
    lineage: count => `压缩谱系 · ${count} 个分段`,
    loading: '正在读取已保存的快照…',
    mcpServers: 'MCP 服务器',
    model: '模型',
    noChanges: '无变化',
    noPrompt: '尚无已保存的提示词：首轮之后才会保存。',
    none: '无',
    promptChars: count => `${count} 字符`,
    provider: '提供商',
    segment: index => `分段 ${index}`,
    sourceLive: '运行中的代理',
    sourceStored: '已保存的快照',
    systemPrompt: '系统提示词',
    tools: count => `工具 (${count})`
  }
} satisfies ContextUsagePanelCopy
