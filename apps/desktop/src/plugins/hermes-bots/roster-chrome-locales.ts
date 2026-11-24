import type { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

export interface RosterChromeMessages {
  title: string
  toastsOn: string
  toastsOff: string
  newMenu: string
  filter: string
  filtersActive: (count: number) => string
  allGateways: string
  currentGateway: string
  hidden: string
}

export const ROSTER_CHROME = {
  'en': {
    title: "Bots",
    toastsOn: "Activity toasts on — click to silence",
    toastsOff: "Activity toasts off — click to enable",
    newMenu: "New…",
    filter: "Filter roster",
    filtersActive: count => `Filter roster, ${count} active`,
    allGateways: "All gateways",
    currentGateway: "Current gateway",
    hidden: "Hidden"
  },
  'ja': {
    title: "ボット",
    toastsOn: "活動通知オン — クリックでオフ",
    toastsOff: "活動通知オフ — クリックでオン",
    newMenu: "新規…",
    filter: "一覧を絞り込む",
    filtersActive: count => `一覧を絞り込む（${count}件適用中）`,
    allGateways: "すべてのゲートウェイ",
    currentGateway: "現在のゲートウェイ",
    hidden: "非表示"
  },
  'zh': {
    title: "机器人",
    toastsOn: "活动通知已开启 — 点击关闭",
    toastsOff: "活动通知已关闭 — 点击开启",
    newMenu: "新建…",
    filter: "筛选列表",
    filtersActive: count => `筛选列表，已启用 ${count} 项`,
    allGateways: "所有网关",
    currentGateway: "当前网关",
    hidden: "已隐藏"
  },
  'zh-hant': {
    title: "機器人",
    toastsOn: "活動通知已開啟 — 點擊關閉",
    toastsOff: "活動通知已關閉 — 點擊開啟",
    newMenu: "新增…",
    filter: "篩選清單",
    filtersActive: count => `篩選清單，已啟用 ${count} 項`,
    allGateways: "所有閘道",
    currentGateway: "目前的閘道",
    hidden: "已隱藏"
  },
  'fr': {
    title: "Bots",
    toastsOn: "Notifications d’activité activées — cliquer pour désactiver",
    toastsOff: "Notifications d’activité désactivées — cliquer pour activer",
    newMenu: "Nouveau…",
    filter: "Filtrer la liste",
    filtersActive: count => `Filtrer la liste, ${count} filtre${count === 1 ? '' : 's'} actif${count === 1 ? '' : 's'}`,
    allGateways: "Toutes les passerelles",
    currentGateway: "Passerelle actuelle",
    hidden: "Masqués"
  },
  'de': {
    title: "Bots",
    toastsOn: "Aktivitätshinweise an — zum Ausschalten klicken",
    toastsOff: "Aktivitätshinweise aus — zum Einschalten klicken",
    newMenu: "Neu…",
    filter: "Liste filtern",
    filtersActive: count => `Liste filtern, ${count} aktiv`,
    allGateways: "Alle Gateways",
    currentGateway: "Aktuelles Gateway",
    hidden: "Ausgeblendet"
  },
  'es': {
    title: "Bots",
    toastsOn: "Avisos de actividad activados — haz clic para silenciarlos",
    toastsOff: "Avisos de actividad desactivados — haz clic para activarlos",
    newMenu: "Nuevo…",
    filter: "Filtrar la lista",
    filtersActive: count => `Filtrar la lista, ${count} ${count === 1 ? 'activo' : 'activos'}`,
    allGateways: "Todas las pasarelas",
    currentGateway: "Pasarela actual",
    hidden: "Ocultos"
  },
  'ru': {
    title: "Боты",
    toastsOn: "Уведомления об активности включены — нажмите, чтобы отключить",
    toastsOff: "Уведомления об активности отключены — нажмите, чтобы включить",
    newMenu: "Создать…",
    filter: "Фильтровать список",
    filtersActive: count => `Фильтровать список, активно: ${count}`,
    allGateways: "Все шлюзы",
    currentGateway: "Текущий шлюз",
    hidden: "Скрытые"
  },
  'ar': {
    title: "الروبوتات",
    toastsOn: "إشعارات النشاط مفعّلة — انقر لكتمها",
    toastsOff: "إشعارات النشاط معطّلة — انقر لتفعيلها",
    newMenu: "جديد…",
    filter: "تصفية القائمة",
    filtersActive: count => `تصفية القائمة، عدد المرشحات النشطة: ${count}`,
    allGateways: "كل البوابات",
    currentGateway: "البوابة الحالية",
    hidden: "مخفي"
  },
} satisfies Record<keyof typeof CANONICAL_GROUP_LOCALES, RosterChromeMessages>
