import type { AuxTaskCopyMap } from './types_aux_tasks'

export const jaAuxTasks: AuxTaskCopyMap = {
  vision: { label: 'ビジョン', hint: '画像分析' },
  compression: { label: '圧縮', hint: 'コンテキストの圧縮' },
  approval: { label: '承認', hint: 'スマート自動承認' },
  mcp: { label: 'MCP', hint: 'MCP ツールルーティング' },
  title_generation: { label: 'タイトル生成', hint: 'セッションタイトル' },
  review: { label: 'レビュー', hint: '/review レビューサブエージェント' },
  voice_chat: { label: '音声チャット', hint: '音声モードの応答' },
  triage_specifier: { label: 'トリアージ指定', hint: 'カンバン仕様の具体化' },
  kanban_decomposer: { label: 'カンバン分解', hint: 'タスク分解' },
  profile_describer: { label: 'プロファイル記述', hint: 'プロファイル概要の自動生成' },
  curator: { label: 'キュレーター', hint: 'スキル使用レビュー' },
  memory_query_rewrite: { label: 'メモリクエリ書き換え', hint: 'メモリ検索クエリ' },
  tts_audio_tags: { label: 'TTS オーディオタグ', hint: 'Gemini TTS のタグ挿入' },
  group_router: { label: 'グループルーター', hint: '自動ルーティングのグループで応答者を決定' },
  goal_judge: { label: 'ゴール判定', hint: '/goal 契約の作成と達成判定' },
  background_review: { label: 'バックグラウンドレビュー', hint: 'ターン後のメモリとスキルの更新' },
  monitor: { label: 'モニター', hint: '重要メールのスコアリング' },
  moa_reference: { label: 'MoA 参照', hint: 'エージェント混合の参照モデル' },
  moa_aggregator: { label: 'MoA 集約', hint: 'エージェント混合の集約ステップ' }
}
