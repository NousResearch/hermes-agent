import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
import { setRuntimeI18nLocale, translateNow } from './runtime'
import { zh } from './zh'

interface RussianStaticCopyCase {
  expected: string
  path: string
}

interface RussianFunctionCopyCase {
  args: readonly unknown[]
  expected: string
  path: string
}

interface RussianCounterCase {
  args: (count: number) => readonly unknown[]
  expected: { [count: number]: string }
  path: string
}

const RUSSIAN_STATIC_COPY_CASES: RussianStaticCopyCase[] = [
  {
    path: 'boot.steps.connectingGateway',
    expected: 'Подключение к шлюзу Hermes Desktop'
  },
  {
    path: 'settings.plugins.blurb',
    expected:
      'Расширения интерфейса, загружаемые в это приложение: встроенные в сборку или добавленные в папку desktop-plugins (включая плагины, созданные Hermes). Отключение немедленно выгружает плагин, а настройка сохраняется после перезапуска.'
  },
  {
    path: 'settings.plugins.kinds.runtime',
    expected: 'динамический'
  },
  {
    path: 'settings.sessions.emptyArchivedDesc',
    expected: 'Архивируйте чат, чтобы он появился здесь.'
  },
  {
    path: 'profiles.soulPlaceholderCloned',
    expected: 'клонированный вариант по умолчанию'
  },
  {
    path: 'profiles.soulPlaceholderEmpty',
    expected: 'пустой вариант по умолчанию'
  },
  {
    path: 'sidebar.projects.convertBranchDesc',
    expected: 'Откройте ветви, уже открытые в рабочих деревьях, или создайте рабочее дерево для свободной ветви.'
  },
  {
    path: 'sidebar.projects.branchSwitchHome',
    expected: 'переключить основное рабочее дерево'
  },
  {
    path: 'preview.web.stillWorking',
    expected:
      'Hermes всё ещё работает, но результат перезапуска ещё не получен. Возможно, команда сервера выполняется на переднем плане.'
  },
  {
    path: 'zones.closeRunningBody',
    expected:
      'Этот чат всё ещё работает или ждёт вашего ответа. Закрытие вкладки только скроет её — сессия продолжит работу, и её можно будет снова открыть из боковой панели.'
  },
  {
    path: 'assistant.clarify.lateAnswerHint',
    expected: 'Этот запрос больше не ждёт. Выберите вариант, чтобы добавить ответ в черновик следующего сообщения.'
  },
  {
    path: 'prompts.secretDesc',
    expected: 'Hermes требуются учётные данные для продолжения.'
  }
]

const RUSSIAN_FUNCTION_COPY_CASES: RussianFunctionCopyCase[] = [
  {
    path: 'boot.failure.remoteSignInHint',
    args: ['Войти в Hermes Cloud'],
    expected:
      'Будет выполнен выход из сохранённой удалённой сессии браузера, затем откроется Войти в Hermes Cloud. Чтобы переключиться на встроенный бэкенд, используйте локальный шлюз.'
  },
  {
    path: 'profiles.soulPlaceholder',
    args: ['клонированный вариант по умолчанию'],
    expected:
      'Системный промпт и описание персоны этого профиля.\nОставьте поле пустым, чтобы сохранить клонированный вариант по умолчанию.'
  },
  {
    path: 'cron.everyDayOfWeekAt',
    args: ['Понедельник', '09:00'],
    expected: 'Каждую неделю: Понедельник, 09:00'
  },
  {
    path: 'artifacts.goToPage',
    args: ['Изображения', 3],
    expected: 'Изображения: перейти на страницу 3'
  }
]

const RUSSIAN_COUNTER_VALUES = [1, 2, 5, 11, 14, 21] as const

const RUSSIAN_COUNTER_CASES: RussianCounterCase[] = [
  {
    path: 'profiles.count',
    args: count => [count],
    expected: {
      1: '1 профиль',
      2: '2 профиля',
      5: '5 профилей',
      11: '11 профилей',
      14: '14 профилей',
      21: '21 профиль'
    }
  },
  {
    path: 'settings.plugins.count',
    args: count => [count],
    expected: {
      1: 'Установлен 1 плагин',
      2: 'Установлено 2 плагина',
      5: 'Установлено 5 плагинов',
      11: 'Установлено 11 плагинов',
      14: 'Установлено 14 плагинов',
      21: 'Установлен 21 плагин'
    }
  },
  {
    path: 'settings.sessions.messages',
    args: count => [count],
    expected: {
      1: '1 сообщение',
      2: '2 сообщения',
      5: '5 сообщений',
      11: '11 сообщений',
      14: '14 сообщений',
      21: '21 сообщение'
    }
  },
  {
    path: 'cron.count',
    args: count => [count],
    expected: {
      1: '1 задание',
      2: '2 задания',
      5: '5 заданий',
      11: '11 заданий',
      14: '14 заданий',
      21: '21 задание'
    }
  },
  {
    path: 'shell.statusbar.subagents',
    args: count => [count],
    expected: {
      1: '1 субагент',
      2: '2 субагента',
      5: '5 субагентов',
      11: '11 субагентов',
      14: '14 субагентов',
      21: '21 субагент'
    }
  },
  {
    path: 'preview.web.filesChanged',
    args: count => [count, 'http://localhost:3000'],
    expected: {
      1: '1 изменение файла, перезагрузка предпросмотра: http://localhost:3000',
      2: '2 изменения файлов, перезагрузка предпросмотра: http://localhost:3000',
      5: '5 изменений файлов, перезагрузка предпросмотра: http://localhost:3000',
      11: '11 изменений файлов, перезагрузка предпросмотра: http://localhost:3000',
      14: '14 изменений файлов, перезагрузка предпросмотра: http://localhost:3000',
      21: '21 изменение файла, перезагрузка предпросмотра: http://localhost:3000'
    }
  }
]

describe('desktop i18n runtime translator', () => {
  beforeEach(() => {
    setRuntimeI18nLocale('en')
  })

  afterEach(() => {
    setRuntimeI18nLocale('en')
  })

  it('translates string paths for the active runtime locale', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('boot.ready')).toBe('Hermes 桌面版已就绪')
    expect(translateNow('notifications.voice.noSpeechDetected')).toBe('没有检测到语音')
    expect(translateNow('composer.lookupNoMatches')).toBe('没有匹配项。')
    expect(translateNow('assistant.tool.statusRecovered')).toBe('已恢复')
  })

  it('passes arguments to function translations', () => {
    expect(translateNow('notifications.updateReadyMessage', 2)).toBe('2 new changes available.')
  })

  it('translates migrated overlap keys for newly supported locales', () => {
    setRuntimeI18nLocale('ja')
    expect(translateNow('common.save')).toBe('保存')

    setRuntimeI18nLocale('zh-hant')
    expect(translateNow('cron.promptPlaceholder')).toBe('代理每次執行時應做什麼？')
  })

  it('translates settings copy for newly supported locales', () => {
    setRuntimeI18nLocale('ja')
    expect(translateNow('settings.appearance.title')).toBe('外観')
    expect(translateNow('settings.nav.providers')).toBe('プロバイダー')

    setRuntimeI18nLocale('zh-hant')
    expect(translateNow('settings.appearance.title')).toBe('外觀')
    expect(translateNow('settings.nav.providerApiKeys')).toBe('API 金鑰')

    setRuntimeI18nLocale('ar')
    expect(translateNow('settings.appearance.reasoningCollapsedTitle')).toBe('طي التفكير افتراضيًا')
    expect(translateNow('settings.appearance.reasoningCollapsedDesc')).toBe(
      'أبقِ التفكير المتدفق متاحًا دون توسيعه حتى تفتحه.'
    )
  })

  it('translates Russian strings', () => {
    setRuntimeI18nLocale('ru')

    expect(translateNow('common.save')).toBe('Сохранить')
    expect(translateNow('settings.appearance.title')).toBe('Внешний вид')
  })

  for (const copyCase of RUSSIAN_STATIC_COPY_CASES) {
    it(`keeps Russian runtime semantics for ${copyCase.path}`, () => {
      setRuntimeI18nLocale('ru')

      expect(translateNow(copyCase.path)).toBe(copyCase.expected)
    })
  }

  for (const copyCase of RUSSIAN_FUNCTION_COPY_CASES) {
    it(`composes Russian runtime copy for ${copyCase.path}`, () => {
      setRuntimeI18nLocale('ru')

      expect(translateNow(copyCase.path, ...copyCase.args)).toBe(copyCase.expected)
    })
  }

  it('composes the Russian empty-cloud-agents message around its link', () => {
    setRuntimeI18nLocale('ru')

    const message = [
      translateNow('settings.gateway.cloudNoAgents.before'),
      translateNow('settings.gateway.cloudNoAgents.linkText'),
      translateNow('settings.gateway.cloudNoAgents.after')
    ].join('')

    expect(message).toBe('Агенты на этом аккаунте не найдены. Создайте агента на портале Nous, затем обновите.')
  })

  for (const counterCase of RUSSIAN_COUNTER_CASES) {
    it(`inflects Russian entity counts for ${counterCase.path}`, () => {
      setRuntimeI18nLocale('ru')

      for (const count of RUSSIAN_COUNTER_VALUES) {
        expect(translateNow(counterCase.path, ...counterCase.args(count))).toBe(counterCase.expected[count])
      }
    })
  }

  it('formats the Russian context token summary without inflecting a formatted maximum', () => {
    setRuntimeI18nLocale('ru')

    expect(translateNow('shell.statusbar.contextUsagePanel.tokenSummary', '~1,2 тыс.', '8 тыс.')).toBe(
      'Токены: ~1,2 тыс. / 8 тыс.'
    )
  })

  it('keeps translated settings field copy addressable from schema keys', () => {
    const field = ['display', 'show_reasoning'].join('.')

    expect(fieldCopyForSchemaKey(zh.settings.fieldLabels, field)).toBe('推理过程块')
    expect(fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)).toBe('当后端提供推理内容时予以显示。')
  })

  it('falls back to English when the active locale cannot resolve a key', () => {
    const boot = TRANSLATIONS.ja.boot as { ready?: string }
    const originalReady = boot.ready

    try {
      boot.ready = undefined
      setRuntimeI18nLocale('ja')

      expect(translateNow('boot.ready')).toBe('Hermes Desktop is ready')
    } finally {
      boot.ready = originalReady
    }
  })

  it('returns the key when no locale can resolve a path', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('missing.path')).toBe('missing.path')
  })
})
