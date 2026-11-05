import { expect, it } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { BOTS_LOCALES } from './i18n'

it.each([
  ['ja', 'ボット'],
  ['zh', '机器人'],
  ['zh-hant', '機器人']
] as const)('keeps translated bot terms joined to surrounding text in %s', (locale, term) => {
  for (const [key, text] of Object.entries(CANONICAL_GROUP_LOCALES[locale])) {
    expect(text, `${locale}.${key}`).not.toMatch(new RegExp(`\\s${term}|${term}\\s`))
  }
})

const bannedTerms = {
  en: /\brooms?\b|gateway[- ]groups?|group driver|canonical|hosted room|RoomLink/iu,
  ja: /ルーム|部屋|ゲートウェイ(?:の)?グループ|グループの実行機能|正規の(?:グループ|チャット)|ホスト型/iu,
  zh: /房间|网关群组|群组运行程序|规范群|托管房间/iu,
  'zh-hant': /群聊|房間|閘道群組|群組執行程式|規範群|託管房間/iu,
  ar: /غرف|مجموع(?:ة|ات) (?:ال)?بوابة|مشغ[ّ]?ل المجموعة|الدردشة الجماعية|(?:ال)?محادثة (?:ال)?جماعية/iu,
  ru: /комнат|групп[а-яё]* шлюза|исполнител[а-яё]* групп|драйвер[а-яё]*\s+групп[а-яё]*|каноническ/iu,
  fr: /chats? de groupe|\bsalons?\b|groupes? de (?:la )?passerelle|pilote de groupe|canonique/iu,
  de: /\b(?:gruppen|chat)?(?:raum|räume)|Gateway-Gruppe|Gruppentreiber|kanonisch/iu,
  es: /chats? grupal(?:es)?|(?<![\p{L}\p{M}])salas?(?![\p{L}\p{M}])|grupos? de (?:la )?pasarela|controlador de grupo|canónic/iu
} satisfies Record<keyof typeof BOTS_LOCALES, RegExp>

function visibleStrings(value: unknown): string[] {
  if (typeof value === 'string') {
    return [value]
  }

  if (typeof value === 'function') {
    return [0, 1, 2].map(count => value(count, count, count))
  }

  return Object.values(value as Record<string, unknown>).flatMap(visibleStrings)
}

it.each(Object.entries(BOTS_LOCALES))('uses consistent group chat wording in %s', (locale, bundle) => {
  for (const text of visibleStrings(bundle)) {
    expect(text).not.toMatch(bannedTerms.en)
    expect(text).not.toMatch(bannedTerms[locale as keyof typeof bannedTerms])
  }
})

// Examples exercise the regex table itself, including the false friends seen in other UI copy.
it.each([
  ['en', 'Open gateway-group'],
  ['en', 'Open gateway groups'],
  ['en', 'Hosted room settings'],
  ['ja', 'ゲートウェイグループを開く'],
  ['ja', 'ゲートウェイのグループを開く'],
  ['ja', '正規のグループ'],
  ['zh', '房间设置'],
  ['zh-hant', '開啟群聊'],
  ['ar', 'فتح المحادثة الجماعية'],
  ['ar', 'محادثة جماعية'],
  ['ru', 'Нет драйвера группы'],
  ['fr', 'Ouvrir le chat de groupe'],
  ['de', 'Gruppenraum öffnen'],
  ['de', 'Chaträume öffnen'],
  ['es', 'Abrir chat grupal'],
  ['es', 'Ajustes de la sala'],
  ['es', 'Ajustes de las salas']
] as const)('rejects competing feature wording in %s: %s', (locale, text) => {
  expect(text).toMatch(bannedTerms[locale])
})

it.each([
  ['en', 'A group chat hosted on this gateway'],
  ['ja', '正規のURLを入力してください。'],
  ['zh', '群聊'],
  ['zh-hant', '群組聊天'],
  ['ar', 'محادثة المجموعة المستضافة على هذه البوابة'],
  ['ru', 'Обновите драйвер видеокарты.'],
  ['fr', 'Ouvrir la discussion de groupe'],
  ['de', 'Zeitraum auswählen'],
  ['es', 'Úsalas para adjuntar archivos.']
] as const)('allows ordinary wording in %s: %s', (locale, text) => {
  expect(text).not.toMatch(bannedTerms[locale])
})

it.each([
  ['en', /hosted on this gateway/],
  ['ja', /このゲートウェイで運用される/],
  ['zh', /在此网关上托管的/],
  ['zh-hant', /在此閘道上託管的/],
  ['ar', /مستضافة على هذه البوابة/],
  ['ru', /размещённ[а-яё]+ на этом шлюзе/],
  ['fr', /hébergée sur cette passerelle/],
  ['de', /auf diesem Gateway gehostete[ rn]/],
  ['es', /alojado en esta pasarela/]
] as const)('limits gateway hosting restrictions to that gateway in %s', (locale, qualifier) => {
  for (const key of ['createGroupSize', 'createOtherConnection', 'createDuplicateMember', 'legacyRoomElsewhere', 'legacyRoomBlocked'] as const) {
    expect(CANONICAL_GROUP_LOCALES[locale][key], key).toMatch(qualifier)
  }
})
