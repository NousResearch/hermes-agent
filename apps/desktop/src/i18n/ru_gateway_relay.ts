import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in ru.ts.
export const ruGatewayRelay: GatewayRelayTranslations = {
  title: 'Другие шлюзы',
  intro:
    'Открывайте другой шлюз Hermes из этого Webapp. Его профили и чаты появятся в переключателе шлюзов; запросы идут через этот шлюз, подписываются его собственным ключом и выполняются от имени учётной записи, которую вы подтвердите там.',
  empty: 'Других шлюзов пока нет.',
  addTitle: 'Подключить шлюз',
  urlLabel: 'Адрес Webapp',
  labelLabel: 'Название',
  connect: 'Подключить',
  connecting: 'Открываем другой шлюз…',
  connectFailed: 'Не удалось начать подключение к этому шлюзу',
  connected: label => `${label} подключён`,
  denied: 'Другой шлюз не подключён: запрос там отклонили.',
  failed: detail => `Другой шлюз не подключён: ${detail}`,
  remove: 'Удалить',
  removeConfirm: label =>
    `Удалить ${label}? Этот шлюз больше не сможет до него достучаться, а его разрешение там будет отозвано.`,
  removeFailed: 'Не удалось удалить этот шлюз',
  test: 'Проверить',
  reachable: 'Доступен',
  unreachable: error => `Недоступен: ${error}`,
  scopes: scopes => `Также разрешено: ${scopes}`,
  coreOnly: 'Только чаты, профили, сеансы и настройки',
  grantsTitle: 'Шлюзы, которые обращаются к этому',
  grantsIntro:
    'Шлюзы, которым вы разрешили передавать сюда ваш Webapp. Любой, кто вошёл в один из них под вашей учётной записью, может через него пользоваться этим компьютером.',
  grantsEmpty: 'Ни один другой шлюз не обращается к этому.',
  grantFrom: (origin, when) => `${origin} · разрешён ${when}`,
  revoke: 'Отозвать',
  revokeConfirm: name =>
    `Отозвать ${name}? Он больше не сможет обращаться к этому шлюзу, а его открытые соединения закроются.`,
  revokeFailed: 'Не удалось отозвать этот шлюз',
  loadFailed: 'Не удалось загрузить ретрансляцию шлюзов'
}
