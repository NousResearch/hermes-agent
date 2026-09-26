import type { Locale } from '@/i18n'

export interface PromptCopy {
  title: string
  subtitle: string
  search: string
  add: string
  name: string
  content: string
  save: string
  remove: string
  copy: string
  use: string
  copied: string
  saved: string
  removed: string
  undo: string
  empty: string
  untitled: string
  error: string
}
type Labels = [
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string
]

const rows: Record<Locale, Labels> = {
  pl: [
    'Moje prompty',
    'Twoja biblioteka poleceń. Zapisywana lokalnie, osobno dla każdego profilu.',
    'Szukaj w promptach',
    'Nowy prompt',
    'Nazwa promptu',
    'Treść promptu',
    'Zapisz prompt',
    'Usuń',
    'Kopiuj',
    'Użyj w rozmowie',
    'Prompt skopiowany',
    'Prompt zapisany',
    'Prompt usunięty',
    'Cofnij',
    'Zapisz pierwszy prompt, aby mieć go zawsze pod ręką.',
    'Nowy prompt',
    'Nie udało się zapisać zmian. Spróbuj ponownie.'
  ],
  en: [
    'My prompts',
    'Your command library. Saved locally, separately for each profile.',
    'Search prompts',
    'New prompt',
    'Prompt name',
    'Prompt content',
    'Save prompt',
    'Delete',
    'Copy',
    'Use in chat',
    'Prompt copied',
    'Prompt saved',
    'Prompt deleted',
    'Undo',
    'Save your first prompt to keep it close at hand.',
    'New prompt',
    'Changes could not be saved. Try again.'
  ],
  ja: [
    'マイプロンプト',
    'プロファイルごとにローカル保存する指示集。',
    'プロンプトを検索',
    '新規プロンプト',
    '名前',
    '内容',
    '保存',
    '削除',
    'コピー',
    '会話で使用',
    'コピーしました',
    '保存しました',
    '削除しました',
    '元に戻す',
    '最初のプロンプトを保存しましょう。',
    '新規プロンプト',
    '保存できませんでした。再試行してください。'
  ],
  zh: [
    '我的提示词',
    '按配置分别保存在本机的指令库。',
    '搜索提示词',
    '新建提示词',
    '名称',
    '内容',
    '保存提示词',
    '删除',
    '复制',
    '用于对话',
    '已复制',
    '已保存',
    '已删除',
    '撤销',
    '保存第一个提示词，方便随时使用。',
    '新建提示词',
    '无法保存，请重试。'
  ],
  'zh-hant': [
    '我的提示詞',
    '按設定檔分別儲存在本機的指令庫。',
    '搜尋提示詞',
    '新增提示詞',
    '名稱',
    '內容',
    '儲存提示詞',
    '刪除',
    '複製',
    '用於對話',
    '已複製',
    '已儲存',
    '已刪除',
    '復原',
    '儲存第一個提示詞，方便隨時使用。',
    '新增提示詞',
    '無法儲存，請重試。'
  ],
  ru: [
    'Мои промпты',
    'Библиотека команд. Сохраняется локально для каждого профиля.',
    'Поиск промптов',
    'Новый промпт',
    'Название',
    'Текст',
    'Сохранить',
    'Удалить',
    'Копировать',
    'Использовать в чате',
    'Скопировано',
    'Сохранено',
    'Удалено',
    'Отменить',
    'Сохраните первый промпт, чтобы он всегда был под рукой.',
    'Новый промпт',
    'Не удалось сохранить. Попробуйте снова.'
  ],
  ar: [
    'توجيهاتي',
    'مكتبة أوامر محفوظة محلياً لكل ملف على حدة.',
    'البحث في التوجيهات',
    'توجيه جديد',
    'الاسم',
    'المحتوى',
    'حفظ التوجيه',
    'حذف',
    'نسخ',
    'استخدام في المحادثة',
    'تم النسخ',
    'تم الحفظ',
    'تم الحذف',
    'تراجع',
    'احفظ أول توجيه لتستخدمه بسهولة.',
    'توجيه جديد',
    'تعذر الحفظ. حاول مرة أخرى.'
  ]
}

export function promptCopy(locale: Locale): PromptCopy {
  const [
    title,
    subtitle,
    search,
    add,
    name,
    content,
    save,
    remove,
    copy,
    use,
    copied,
    saved,
    removed,
    undo,
    empty,
    untitled,
    error
  ] = rows[locale]

  return {
    title,
    subtitle,
    search,
    add,
    name,
    content,
    save,
    remove,
    copy,
    use,
    copied,
    saved,
    removed,
    undo,
    empty,
    untitled,
    error
  }
}
