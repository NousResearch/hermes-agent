import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const jaContextUsagePanel = {
  categories: {
    conversation: '会話',
    mcp: 'MCP',
    memory: 'メモリ',
    rules: 'ルール',
    skills: 'スキル',
    subagent_definitions: 'サブエージェント定義',
    system_prompt: 'システムプロンプト',
    tool_definitions: 'ツール定義'
  },
  compact: {
    action: '会話を圧縮',
    busyHint: '先に実行中の応答を終了するか停止してください',
    confirm: '圧縮',
    hint: '古いコンテキストを要約して空きを作ります。同じ会話を続けられます。',
    question: 'この会話を要約に圧縮しますか？',
    questionNamed: title => `「${title}」を要約に圧縮しますか？`,
    working: '圧縮中…'
  },
  empty: 'コンテキストデータはまだありません',
  loading: '内訳を読み込み中…',
  percentFull: percent => `${percent}% 使用中`,
  title: 'コンテキスト使用状況',
  tokenSummary: (used, max) => `${used} / ${max} Tokens`,
  tabs: { inspector: 'インスペクター', usage: '使用量' },
  inspector: {
    changes: {
      model_changed: (before, after) => `モデル変更: ${before} → ${after}`,
      prompt_changed: 'プロンプト変更',
      provider_changed: (before, after) => `プロバイダー変更: ${before} → ${after}`,
      tool_added: name => `ツール追加: ${name}`,
      tool_removed: name => `ツール削除: ${name}`
    },
    error: 'エージェントが見た内容を読み込めませんでした',
    lineage: count => `圧縮の系譜 · ${count} セグメント`,
    loading: '保存済みスナップショットを読み込み中…',
    mcpServers: 'MCP サーバー',
    model: 'モデル',
    noChanges: '変更なし',
    noPrompt: '保存済みのプロンプトはまだありません。最初のターン後に保存されます。',
    none: 'なし',
    promptChars: count => `${count} 文字`,
    provider: 'プロバイダー',
    segment: index => `セグメント ${index}`,
    sourceLive: '稼働中のエージェント',
    sourceStored: '保存済みスナップショット',
    systemPrompt: 'システムプロンプト',
    tools: count => `ツール (${count})`
  }
} satisfies ContextUsagePanelCopy
