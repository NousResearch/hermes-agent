/** settings.config shard — keepAwake + managed-scope copy, split out of
 *  ja.ts (FILE_LINES cap). */
export const jaSettingsConfig = {
  keepAwakeTitle: 'コンピューターをスリープさせない',
  keepAwakeDesc:
    '本体のスリープを防ぎます。「実行中のみ」はターンの実行中だけ有効になるため、夜通しの実行を継続しつつ、ノートPCを一週間つけたままにはしません。画面は暗転できます。',
  keepAwakeOff: 'オフ',
  keepAwakeWhileWorking: '実行中のみ',
  keepAwakeAlways: '常に',
  managedFieldHint: '管理者により管理されています{source} — 読み取り専用',
  managedRejectedTitle: '一部の設定は保存されませんでした',
  managedRejectedNotice: '保存されませんでした（管理者により管理）: {keys}',
}
