import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Machines copy where the connection cannot change, spread into ru.ts.
export const ruGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Настройки компьютера недоступны',
  unavailableDesc: 'IPC-мост приложения не предоставляет настройки компьютера.',
  webappHostTitle: 'Хост Hermes',
  webappHostDesc:
    'Webapp всегда работает на хосте Hermes, который её обслуживает. Сохранённые подключения ниже — список этого компьютера, общий с приложением Hermes Desktop; другие компьютеры доступны через этот хост.',
  webappOauthPaste: {
    title: 'Завершить вход',
    description:
      'Если вкладка входа остановилась на адресе, который не открывается (этот браузер не на компьютере с Hermes), вставьте этот адрес сюда.',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: 'Завершить',
    failed: 'Не удалось завершить вход'
  },
  webAccess: {
    title: 'Веб-доступ',
    intro:
      'Откройте Hermes этого компьютера с телефона или другого компьютера, войдя через аккаунт Nous. Вошедший может запускать команды на этом компьютере.',
    failed: 'Не удалось изменить веб-доступ',
    running: 'Работает',
    stopped: 'Остановлен',
    starting: 'Запускается',
    stopping: 'Останавливается',
    restartNeeded: 'Нужен перезапуск',
    unavailable: 'Недоступен',
    notRunningDetail: 'Запустите его, чтобы открыть Hermes с другого устройства.',
    needsNous: 'Сначала войдите в аккаунт Nous в разделе «Провайдеры».',
    runningLan: 'Откройте этот адрес на устройстве в той же сети.',
    runningPublic: 'Доступен через ваш URL.',
    runningLocal:
      'Открыть его можно только на этом компьютере. Остановите и снова запустите его, чтобы открыть с других устройств.',
    pendingOn: 'Сохранено. Hermes откроет доступ к веб-приложению после перезапуска.',
    pendingOff: 'Выключено, но Hermes продолжает предоставлять доступ к веб-приложению до перезапуска.',
    unavailableDetail: reason => `Hermes не удалось открыть доступ к веб-приложению: ${reason}`,
    restartBusy:
      'Сохранено. Идёт чат, поэтому Hermes не перезапущен. Когда чат закончится, нажмите «Перезапустить Hermes».',
    restartFailed:
      'Сохранено, но Hermes не удалось перезапустить. Нажмите «Перезапустить Hermes», чтобы попробовать снова.',
    restartFailedBecause: reason => `Не удалось перезапустить Hermes: ${reason}`,
    startTimedOut: 'Hermes не открыл доступ к веб-приложению за 90 секунд.',
    stopTimedOut: 'Hermes не прекратил предоставлять доступ к веб-приложению за 90 секунд.',
    startStages: {
      preparing: 'Готовим веб-приложение. Сборка может занять несколько минут.',
      building: 'Собираем веб-приложение. Это может занять несколько минут.',
      registering: 'Настраиваем вход через Nous для этого адреса.',
      restarting: 'Перезапускаем Hermes, чтобы открыть веб-приложение.'
    },
    startElapsed: time => `Прошло: ${time}`,
    start: 'Запустить',
    stop: 'Остановить',
    restartGateway: 'Перезапустить Hermes',
    modeLan: 'Эта сеть',
    modeLanDetail: url => `Устройства в вашей сети Wi-Fi открывают ${url}. Используйте только в доверенной сети.`,
    lanUnencrypted: 'Соединение не зашифровано: другие в этой сети могут прочитать то, что вы отправляете и получаете.',
    noLan: 'У этого компьютера нет сетевого адреса, доступного другим устройствам.',
    modePublic: 'Мой собственный URL',
    modePublicDetail: port =>
      `Уже работающий HTTPS-адрес, например Tailscale Serve или туннель, который перенаправляет на http://127.0.0.1:${port}.`
  }
}
