import { expect, it } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CROSS_GATEWAY_GROUP_LOCALES } from './cross-gateway-locales'

it.each([
  ['ja', 'ボット'],
  ['zh', '机器人'],
  ['zh-hant', '機器人']
] as const)('keeps translated bot terms joined to surrounding text in %s', (locale, term) => {
  for (const [key, text] of Object.entries(CANONICAL_GROUP_LOCALES[locale])) {
    expect(text, `${locale}.${key}`).not.toMatch(new RegExp(`\\s${term}|${term}\\s`))
  }
})


const advice = [
  ['en', 'computer running the group chat', 'asleep or offline', 'once it', 'this device', 'this device'],
  ['ja', 'グループチャットを実行しているコンピューター', 'スリープ中かオフライン', '復帰', 'このデバイス', 'このデバイス'],
  ['zh', '运行群聊的电脑', '休眠或离线', '恢复', '此设备', '此设备'],
  ['zh-hant', '執行群組聊天的電腦', '休眠或離線', '恢復', '此裝置', '此裝置'],
  ['ar', 'الجهاز الذي يشغّل الدردشة الجماعية', 'السكون أو غير متصل', 'عودته', 'هذا الجهاز', 'هذا الجهاز'],
  ['ru', 'компьютере, на котором работает групповой чат', 'спящем режиме или не в сети', 'снова', 'этом устройстве', 'это устройство'],
  ['fr', 'l’ordinateur qui exécute la discussion de groupe', 'veille ou hors ligne', 'de nouveau', 'cet appareil', 'cet appareil'],
  ['de', 'Computer, auf dem der Gruppenchat läuft', 'Ruhezustand oder offline', 'wieder', 'diesem Gerät', 'Dieses Gerät'],
  ['es', 'equipo que ejecuta el chat de grupo', 'suspendido o sin conexión', 'vuelva', 'este dispositivo', 'este dispositivo']
] as const

it.each(advice)('directs missing-profile recovery to the group owner (%s)', (locale, owner) => {
  const text = CANONICAL_GROUP_LOCALES[locale].disbandMemberHomeUnknown
  expect(text).toContain(owner)
  expect(text).toContain('hosted_rooms.profiles')
})

it.each(advice)('explains when to retry an unconfirmed Disband (%s)', (locale, _owner, offline, back) => {
  const text = CANONICAL_GROUP_LOCALES[locale].disbandUnconfirmed
  expect(text).toContain(offline)
  expect(text).toContain(back)
})

it.each(advice)('uses natural local-device wording and preserves remote names (%s)', (locale, _owner, _offline, _back, onDevice, device) => {
  const words = CROSS_GATEWAY_GROUP_LOCALES[locale]
  expect(words.gatewayPeer('This device', true)).toContain(onDevice)
  expect(words.peerCheckFailed('This device', true)).toContain(device)
  expect(words.gatewayPeer('Homelab')).toContain('Homelab')
  expect(words.peerCheckFailed('Homelab')).toContain('Homelab')
})

it.each(['peerJoinUnconfirmed', 'disbandMemberHomeUnknown', 'peerStartWithoutRetry', 'peerStartWithoutKept'] as const)(
  'uses the formal German register in %s', key => {
    const text = CANONICAL_GROUP_LOCALES.de[key]
    expect(text).not.toMatch(/\b(?:Prüfe|Füge|Versuche|Verwende|Löse|du)\b/)
    expect(text).toContain('Sie')
  }
)
