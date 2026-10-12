import type { TranslationOverrides } from './define-locale'

// Simple-mode surfaces (interface-mode picker, chat starters), spread into ru.ts.
export const ruSimpleMode = {
  interfaceMode: {
    title: 'Режим интерфейса',
    hint: 'Меняет то, что показано, а не то, что умеет Hermes.',
    sessionNote:
      'Задано простым режимом. Изменение здесь действует до конца сеанса; переключитесь в расширенный, чтобы сделать его своим.',
    switchToAdvanced: 'Перейти в расширенный',
    simple: {
      label: 'Простой',
      description: 'Для общения с Hermes. Боковая панель и чат; без терминала, файлов и панелей diff.'
    },
    advanced: {
      label: 'Расширенный',
      description: 'Для разработчиков. Терминал, файлы, diff, строка состояния и раскладки — как вы их настроили.'
    }
  },
  chatStarters: {
    label: 'Идеи для начала',
    intro: 'Спросите меня о чём угодно — или начните с одной из этих идей.',
    email: {
      title: 'Помоги написать письмо',
      prompt: 'Помоги мне написать письмо. Вот кому оно и что я хочу сказать: '
    },
    dinners: {
      title: 'Спланируй ужины на неделю',
      prompt: 'Придумай пять простых ужинов на эту неделю и составь один общий список покупок для всех.'
    },
    summarize: {
      title: 'Кратко перескажи документ',
      prompt: 'Я прикреплю файл — выдели главное.'
    },
    lookup: {
      title: 'Найди кое-что для меня',
      prompt: 'Найди это и расскажи, что удалось узнать: '
    }
  }
} satisfies Pick<TranslationOverrides, 'interfaceMode' | 'chatStarters'>
