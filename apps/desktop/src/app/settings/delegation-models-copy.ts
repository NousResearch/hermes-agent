/** Feature-local copy keeps this bounded integration out of the large locale owners. */
const en = {
  title: 'Delegation models',
  loading: 'Loading delegation settings…',
  unsupported: 'Update this gateway’s Hermes runtime to configure delegation fallbacks here. Existing settings remain unchanged.',
  description: 'Default model and ordered backups for newly delegated work. Your main conversation is unchanged.',
  provider: 'Delegation provider',
  model: 'Delegation model',
  mainProvider: 'Use main provider',
  mainModel: 'Use main model',
  direct: 'Existing direct endpoint',
  directHint: 'A direct endpoint override is active. Choosing a provider replaces its endpoint and authentication overrides.',
  customModel: 'Custom model ID…',
  policy: 'Delegation fallback behavior',
  auto: 'Default behavior',
  autoHint: 'Inherit the main backups only when no delegation model or provider is pinned.',
  inherit: 'Use main fallbacks',
  inheritHint: 'Explicitly share the main conversation’s fallback chain, even with a delegation model selected.',
  none: 'No fallbacks',
  custom: 'Choose delegation fallbacks',
  customHint: 'Try these provider/model pairs in order. Each backup keeps its own route and credentials.',
  add: 'Add delegation fallback',
  remove: 'Remove fallback',
  up: 'Move fallback earlier',
  down: 'Move fallback later',
  apply: 'Apply delegation settings',
  reset: 'Discard edits',
  refresh: 'Reload settings',
  saving: 'Saving…',
  failed: 'Could not save and confirm these settings. Your edits are retained; retry after checking the connection.',
  loadFailed: 'Could not load delegation settings.',
  offline: 'The model catalog is unavailable. Enter provider and model IDs manually, or retry.',
  conflict: 'Delegation settings changed elsewhere. Reload before applying to avoid overwriting newer settings.',
  invalid: 'Complete every provider/model pair and choose a fallback behavior before applying.',
  noProviders: 'No model catalog is available for this profile. You can enter IDs manually.'
}

export type DelegationModelsCopy = typeof en

const zh: DelegationModelsCopy = {
  loading: '正在加载委派设置…',
  unsupported: '请更新此网关的 Hermes 运行时以在此配置委派备用模型。现有设置保持不变。',
  title: '委派模型', description: '为新委派任务设置默认模型和有序备用模型。主对话保持不变。',
  provider: '委派提供商', model: '委派模型', mainProvider: '使用主提供商', mainModel: '使用主模型',
  direct: '现有直接端点', directHint: '直接端点覆盖已启用。选择提供商将替换其端点和身份验证覆盖。',
  customModel: '自定义模型 ID…', policy: '委派备用行为', auto: '默认行为',
  autoHint: '仅在未指定委派模型或提供商时继承主备用链。', inherit: '使用主备用链',
  inheritHint: '明确共享主对话的备用链，即使已选择委派模型。', none: '不使用备用模型', custom: '选择委派备用模型',
  customHint: '按顺序尝试这些提供商和模型。每个备用模型保留自己的路由和凭据。',
  add: '添加委派备用模型', remove: '删除备用模型', up: '提前尝试', down: '推后尝试',
  apply: '应用委派设置', reset: '放弃编辑', refresh: '重新加载设置', saving: '正在保存…',
  failed: '无法保存并确认设置。编辑已保留；请检查连接后重试。', loadFailed: '无法加载委派设置。',
  offline: '模型目录不可用。请手动输入提供商和模型 ID，或重试。', conflict: '委派设置已在其他位置更改。请重新加载后再应用。',
  invalid: '应用前请填写完整的提供商和模型，并选择备用行为。', noProviders: '此配置没有可用的模型目录。可以手动输入 ID。'
}

const zhHant: DelegationModelsCopy = {
  loading: '正在載入委派設定…',
  unsupported: '請更新此閘道的 Hermes 執行環境，以在此設定委派備用模型。現有設定保持不變。',
  title: '委派模型', description: '為新委派工作設定預設模型及有序備用模型。主對話保持不變。',
  provider: '委派供應商', model: '委派模型', mainProvider: '使用主供應商', mainModel: '使用主模型',
  direct: '現有直接端點', directHint: '直接端點覆寫已啟用。選擇供應商將取代其端點和驗證覆寫。',
  customModel: '自訂模型 ID…', policy: '委派備用行為', auto: '預設行為',
  autoHint: '僅在未指定委派模型或供應商時繼承主備用鏈。', inherit: '使用主備用鏈',
  inheritHint: '明確共用主對話的備用鏈，即使已選擇委派模型。', none: '不使用備用模型', custom: '選擇委派備用模型',
  customHint: '依序嘗試這些供應商和模型。每個備用模型保留自己的路由和憑證。',
  add: '新增委派備用模型', remove: '刪除備用模型', up: '提前嘗試', down: '延後嘗試',
  apply: '套用委派設定', reset: '捨棄編輯', refresh: '重新載入設定', saving: '儲存中…',
  failed: '無法儲存並確認設定。編輯已保留；請檢查連線後重試。', loadFailed: '無法載入委派設定。',
  offline: '模型目錄無法使用。請手動輸入供應商和模型 ID，或重試。', conflict: '委派設定已在其他位置變更。請重新載入後再套用。',
  invalid: '套用前請填妥供應商和模型，並選擇備用行為。', noProviders: '此設定檔沒有可用的模型目錄。可手動輸入 ID。'
}

const ja: DelegationModelsCopy = {
  loading: '委任設定を読み込み中…',
  unsupported: '委任の代替モデルを設定するには、このゲートウェイの Hermes を更新してください。既存の設定は変更されません。',
  title: '委任モデル', description: '新しい委任タスクの既定モデルと順序付き代替モデルを設定します。メイン会話は変更されません。',
  provider: '委任プロバイダー', model: '委任モデル', mainProvider: 'メインプロバイダーを使用', mainModel: 'メインモデルを使用',
  direct: '既存の直接エンドポイント', directHint: '直接エンドポイントが設定されています。プロバイダーの選択でエンドポイントと認証の上書き設定が置き換わります。',
  customModel: 'カスタムモデル ID…', policy: '委任のフォールバック動作', auto: '既定の動作',
  autoHint: '委任モデルもプロバイダーも指定されていない場合のみ、メインの代替モデルを継承します。', inherit: 'メインの代替モデルを使用',
  inheritHint: '委任モデルが選択されていても、メイン会話の代替モデルを明示的に共有します。', none: '代替モデルなし', custom: '委任の代替モデルを選択',
  customHint: 'これらのプロバイダーとモデルを順番に試します。各代替モデルのルートと認証情報は維持されます。',
  add: '委任の代替モデルを追加', remove: '代替モデルを削除', up: '順序を上げる', down: '順序を下げる',
  apply: '委任設定を適用', reset: '編集を破棄', refresh: '設定を再読み込み', saving: '保存中…',
  failed: '設定を保存して確認できませんでした。編集は保持されています。接続を確認して再試行してください。', loadFailed: '委任設定を読み込めませんでした。',
  offline: 'モデル一覧を取得できません。プロバイダーとモデル ID を手入力するか、再試行してください。', conflict: '委任設定が別の場所で変更されました。適用前に再読み込みしてください。',
  invalid: 'プロバイダーとモデルをすべて入力し、代替動作を選択してください。', noProviders: 'このプロファイルにはモデル一覧がありません。ID を手入力できます。'
}

const ar: DelegationModelsCopy = {
  loading: 'جارٍ تحميل إعدادات التفويض…',
  unsupported: 'حدّث بيئة Hermes لهذه البوابة لإعداد بدائل التفويض هنا. تبقى الإعدادات الحالية دون تغيير.',
  title: 'نماذج التفويض', description: 'النموذج الافتراضي والبدائل المرتبة للمهام المفوضة الجديدة. لا تتغير المحادثة الرئيسية.',
  provider: 'مزود التفويض', model: 'نموذج التفويض', mainProvider: 'استخدام المزود الرئيسي', mainModel: 'استخدام النموذج الرئيسي',
  direct: 'نقطة الاتصال المباشرة الحالية', directHint: 'يوجد تجاوز لنقطة الاتصال المباشرة. اختيار مزود يستبدل تجاوزات الاتصال والمصادقة.',
  customModel: 'معرّف نموذج مخصص…', policy: 'سلوك بدائل التفويض', auto: 'السلوك الافتراضي',
  autoHint: 'توريث البدائل الرئيسية فقط عندما لا يكون نموذج التفويض أو مزوده محددًا.', inherit: 'استخدام البدائل الرئيسية',
  inheritHint: 'مشاركة سلسلة البدائل الرئيسية صراحةً، حتى مع اختيار نموذج للتفويض.', none: 'بدون بدائل', custom: 'اختيار بدائل التفويض',
  customHint: 'تجربة أزواج المزود والنموذج بالترتيب. يحتفظ كل بديل بمساره وبيانات اعتماده.',
  add: 'إضافة بديل للتفويض', remove: 'إزالة البديل', up: 'تقديم البديل', down: 'تأخير البديل',
  apply: 'تطبيق إعدادات التفويض', reset: 'تجاهل التعديلات', refresh: 'إعادة تحميل الإعدادات', saving: 'جارٍ الحفظ…',
  failed: 'تعذر حفظ الإعدادات وتأكيدها. تم الاحتفاظ بالتعديلات؛ تحقق من الاتصال وأعد المحاولة.', loadFailed: 'تعذر تحميل إعدادات التفويض.',
  offline: 'دليل النماذج غير متاح. أدخل معرّفات المزود والنموذج يدويًا، أو أعد المحاولة.', conflict: 'تغيرت إعدادات التفويض في مكان آخر. أعد التحميل قبل التطبيق.',
  invalid: 'أكمل جميع أزواج المزود والنموذج واختر سلوك البدائل قبل التطبيق.', noProviders: 'لا يوجد دليل نماذج لهذا الملف. يمكنك إدخال المعرّفات يدويًا.'
}

const copies: Record<string, DelegationModelsCopy> = { en, zh, 'zh-Hant': zhHant, 'zh-hant': zhHant, ja, ar }
export const delegationModelsCopy = (locale: string): DelegationModelsCopy => copies[locale] ?? en
