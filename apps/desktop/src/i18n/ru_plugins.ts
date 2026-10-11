import type { TranslationOverrides } from './define-locale'

type SettingsOverrides = NonNullable<TranslationOverrides['settings']>

/** Russian copy for the plugin settings surfaces: Settings ▸ Plugins pages and
 *  the Desktop plugin install flow. */
export const ruPluginSettings = {
  pluginPages: {
    blurb:
      'Параметры, которые добавляют установленные плагины. У каждого плагина своя страница, у некоторых есть подстраницы.',
    empty: 'Пока ни у одного плагина нет настроек.',
    manage: 'Управление плагинами',
    agentSettings: 'Настройки агента',
    pageCount: (n: number) => `Страниц: ${n}`,
    missing: 'У этого плагина нет страницы настроек. Возможно, он отключён или удалён.'
  },
  plugins: {
    title: 'Плагины приложения',
    openFolder: 'Открыть папку плагинов приложения',
    rescan: 'Пересканировать',
    reveal: 'Показать в файловом менеджере',
    failed: 'ошибка',
    kinds: { bundled: 'встроенный', disk: 'на диске', runtime: 'runtime' },
    installModal: {
      title: 'Установка плагина',
      description: 'Перед установкой посмотрите, что содержит этот репозиторий.',
      repoLabel: 'Репозиторий',
      includesHeading: 'Состав пакета',
      agentLabel: 'Плагин агента',
      desktopLabel: 'UI приложения',
      profileLabel: 'Установить для профиля',
      agentTargetLocal: (profile, dir) => `Устанавливается в локальный бэкенд ${profile} (${dir})`,
      agentTargetRemote: profile => `Устанавливается в подключённый бэкенд ${profile}`,
      desktopTarget: 'Устанавливается в локальную папку desktop-plugins этого приложения',
      desktopOnlyNote: 'Пакеты только для приложения не устанавливают плагин агента.',
      insecureWarning:
        'Этот URL использует небезопасную или локальную схему. Для боевой установки предпочитайте https:// или git@.',
      securityHeading: 'Перед установкой',
      securityIntro:
        'Устанавливайте только из проверенных источников — при желании просмотрите репозиторий ниже, чтобы увидеть, что будет добавлено.',
      sourceHeading: 'Исходный код',
      viewRepository: 'Посмотреть репозиторий',
      viewPluginFiles: 'Посмотреть файлы плагина',
      gitCloneLabel: 'URL для git clone',
      enableAgent: 'Включить плагин агента после установки',
      forceReinstall: 'Принудительная переустановка (заменить, если уже установлен)',
      pinToCommit: 'Закрепить на коммите (необязательно)',
      pinToCommitPlaceholder: 'Полный SHA коммита (40 символов)',
      pinToCommitHint:
        'Все, кто установит этот SHA, получат одинаковый код; плагин перестанет обновляться до смены пина. Оставьте пустым для последнего коммита.',
      pinToCommitInvalid: 'Нужен полный SHA коммита из 40 символов (ветки и теги не принимаются).',
      install: 'Установить',
      installing: 'Установка…',
      probing: 'Осмотр репозитория…',
      probeUnavailable: 'Осмотр плагинов недоступен в этом окружении.',
      desktopUnavailable: 'Установка плагинов приложения недоступна в этом окружении.',
      selectComponent: 'Выберите хотя бы один компонент для установки.',
      agentSuccess: name => `Плагин агента ${name} установлен`,
      desktopSuccess: name => `Плагин приложения ${name} установлен`,
      agentFailed: 'Не удалось установить плагин агента',
      installUncertain:
        'Hermes перестал ждать результат установки, но плагин может всё ещё устанавливаться. Закройте это окно и обновите список плагинов перед повторной установкой.',
      desktopFailed: 'Не удалось установить плагин приложения',
      missingEnv: (_name, vars) => `Не хватает переменных окружения: ${vars}. Добавьте их в Настройки → Ключи.`,
      toolsConnected: n => `Подключено инструментов: ${n}`,
      skillsReady: names => (names.length === 1 ? `навык ${names[0]} готов` : `готово навыков: ${names.length}`),
      nextChat: 'остальные инструменты появятся в следующем чате',
      serverNotConnected: (server, reason) => `MCP-сервер ${server} не подключён${reason ? `: ${reason}` : '.'}`
    }
  }
} satisfies Pick<SettingsOverrides, 'pluginPages' | 'plugins'>
