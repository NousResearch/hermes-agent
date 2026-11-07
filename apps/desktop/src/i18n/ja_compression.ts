import type { FieldCopyTree } from '@/app/settings/field-copy'

// Settings > Memory & Context > Context & compression copy for this locale.
export const jaCompressionFieldLabels: FieldCopyTree = {
  context: {
    engine: 'コンテキストエンジン'
  },
  compression: {
    enabled: '自動圧縮',
    threshold: '圧縮しきい値',
    codexGpt55Autoraise: 'Codex 圧縮の自動引き上げ',
    targetRatio: '圧縮目標',
    protectLastN: '保護する直近メッセージ',
    warmHandoff: 'ウォームハンドオフ'
  },
  auxiliary: {
    compression: {
      timeout: '圧縮モデルのタイムアウト（秒）'
    }
  }
}

export const jaCompressionFieldDescriptions: FieldCopyTree = {
  context: {
    engine: '長い会話がコンテキスト上限に近づいたときの管理戦略です。'
  },
  compression: {
    enabled: '会話が大きくなったとき、古いコンテキストを要約します。',
    codexGpt55Autoraise: '対応する ChatGPT Codex OAuth モデルの圧縮しきい値を 85% に引き上げます。',
    warmHandoff:
      'メインモデルがキャッシュ済みのプロンプト上で圧縮サマリーを書くため、プロンプトキャッシュ対応サーバーは新しいメッセージだけを読み込みます。圧縮に同じモデルを使う場合に高速です。自動: 圧縮がメインモデルを使い、サーバーがキャッシュ済みトークンを報告した場合のみ。オン: 常に試行。オフ: 常に圧縮モデルを使用。失敗時は通常のサマリーに戻ります。'
  },
  auxiliary: {
    compression: {
      timeout: '補助圧縮モデルの呼び出しごとに待機する秒数（既定 120）。遅いローカルモデルでは値を上げてください。'
    }
  }
}
