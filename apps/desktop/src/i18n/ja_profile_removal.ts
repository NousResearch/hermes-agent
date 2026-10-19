// The profile delete / reset confirmation copy, composed by ja.ts as `profiles.removal`.
import type { TranslationOverrides } from './define-locale'

export const jaProfileRemoval: NonNullable<TranslationOverrides['profiles']>['removal'] = {
  deleteTitle: 'プロファイルを削除しますか？',
  deleteDescPrefix: 'これにより ',
  deleteDescMid: ' が削除され、その ',
  deleteDescSuffix: ' ディレクトリが削除されます。この操作は元に戻せません。',
  deleting: '削除中...',
  resetTitle: 'デフォルトプロファイルをリセットしますか？',
  resetDesc:
    'デフォルトプロファイルは削除できません。そのディレクトリには Hermes 本体と他のプロファイルが含まれています。' +
    'リセットすると、メモリ、スキル、設定、SOUL.md、チャットが消去され、cron ジョブも削除されます。' +
    '他のプロファイル、Hermes 本体、API キーとログイン情報は保持されます。この操作は元に戻せません。',
  resetAction: 'リセット',
  resetting: 'リセット中...',
  resetDone: 'プロファイルをリセットしました'
}
