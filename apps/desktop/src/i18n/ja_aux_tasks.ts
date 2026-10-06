import type { AuxTaskCopyMap } from './types_aux_tasks'

export const jaAuxTasks: AuxTaskCopyMap = {
  vision: { label: 'ビジョン', hint: '画像分析' },
  compression: { label: '圧縮', hint: 'コンテキストの圧縮' },
  skills_hub: { label: 'スキルハブ', hint: 'スキル検索' },
  approval: { label: '承認', hint: 'スマート自動承認' },
  mcp: { label: 'MCP', hint: 'MCP ツールルーティング' },
  title_generation: { label: 'タイトル生成', hint: 'セッションタイトル' },
  review: { label: 'レビュー', hint: '/review レビューサブエージェント' },
  triage_specifier: { label: 'トリアージ指定', hint: 'カンバン仕様の具体化' },
  kanban_decomposer: { label: 'カンバン分解', hint: 'タスク分解' },
  profile_describer: { label: 'プロファイル記述', hint: 'プロファイル概要の自動生成' },
  curator: { label: 'キュレーター', hint: 'スキル使用レビュー' }
}
