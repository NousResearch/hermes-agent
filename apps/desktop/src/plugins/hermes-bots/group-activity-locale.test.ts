import { afterEach, expect, it, vi } from 'vitest'

const { notify } = vi.hoisted(() => ({ notify: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async () => (await import('./group-test-utils')).pluginSdkMock({ notify }))

import { readCanonicalGroupSend } from './canonical-group-send'
import { groupActivityLabel } from './group-activity'
import { filesToGroupAttachments } from './group-attachments'
import { botsText } from './i18n'
import { translateBotsIn } from './i18n-test-helper'
import { setPluginCtx } from './shared'

afterEach(() => { setPluginCtx(null); notify.mockClear() })

it('reads the current language for activity, send errors and attachments with one translator', async () => {
  let locale = 'en'
  const t = (key: string, ...args: unknown[]) => translateBotsIn(locale)(key, ...args)
  setPluginCtx({ i18n: { t } } as never)
  const held = botsText().group
  const actual = []
  const expected = []
  const file = new File([], '')
  Object.defineProperty(file, 'size', { value: 15_000_001 })

  for (locale of ['en', 'ar', 'ru']) {
    const translate = translateBotsIn(locale)
    const error = await readCanonicalGroupSend({ connectionId: '', profile: '', roomId: '' }).catch((e: Error) => e.message)
    await filesToGroupAttachments([file])
    actual.push([groupActivityLabel({ at: 1, epoch: 0, kind: 'working', member: 'Helper' }), held.activityWorking,
      error, notify.mock.lastCall?.[0].message])
    expected.push([`Helper ${translate('group.activityWorking')}`, translate('group.activityWorking'),
      translate('group.sendRouteRequired'), translate('group.attachmentTooLarge', translate('group.attachmentFallback'))])
  }

  expect(actual).toEqual(expected)
})

it.each([
  ['en', 'You sent a message', 'You stopped the group chat — remaining turns are held until resumed'],
  ['ja', 'あなたがメッセージを送信しました', 'あなたがグループチャットを停止しました — 残りのターンは再開まで一時停止中です'],
  ['zh', '你发送了消息', '你停止了群聊 — 剩余轮次暂停，等待恢复'],
  ['zh-hant', '您傳送了訊息', '您停止了群組聊天 — 剩餘回合暫停，等待恢復'],
  ['fr', 'Vous avez envoyé un message', 'Vous avez arrêté la discussion de groupe — les tours restants sont en pause'],
  ['de', 'Du hast eine Nachricht gesendet', 'Du hast den Gruppenchat gestoppt — verbleibende Durchgänge sind pausiert'],
  ['es', 'Tú enviaste un mensaje', 'Tú detuviste el chat de grupo — los turnos restantes están en pausa'],
  ['ru', 'Вы отправили сообщение', 'Вы остановили групповой чат — оставшиеся ходы приостановлены'],
  ['ar', 'أنت أرسلت رسالة', 'أنت أوقفت محادثة المجموعة — الأدوار المتبقية متوقفة مؤقتًا']
])('agrees with the user and keeps bot verbs in %s', (locale, queued, stopped) => {
  const t = translateBotsIn(locale)
  setPluginCtx({ i18n: { t } } as never)

  for (const [kind, expected, key] of [['queued', queued, 'activityQueued'], ['stopped', stopped, 'activityStopped']] as const) {
    const event = { at: 1, epoch: 0, kind, member: 'You' }
    expect(groupActivityLabel(event)).toBe(expected)
    expect(groupActivityLabel({ ...event, member: 'Helper' })).toBe(`Helper ${t(`group.${key}`)}`)
  }
})
