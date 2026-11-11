import type { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

export interface RosterChromeMessages {
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
  en: {
    toastsOn: 'Activity toasts on — click to silence',
    toastsOff: 'Activity toasts off — click to enable',
    newMenu: 'New…',
    filter: 'Filter roster',
    filtersActive: count => `Filter roster, ${count} active`,
    allGateways: 'All machines',
    currentGateway: 'Current machine',
    hidden: 'Hidden'
  },
  ja: {
    toastsOn: '活動通知オン — クリックでオフ',
    toastsOff: '活動通知オフ — クリックでオン',
    newMenu: '新規…',
    filter: '一覧を絞り込む',
    filtersActive: count => `一覧を絞り込む（${count}件適用中）`,
    allGateways: 'すべてのマシン',
    currentGateway: '現在のマシン',
    hidden: '非表示'
  },
  zh: {
    toastsOn: '活动通知已开启 — 点击关闭',
    toastsOff: '活动通知已关闭 — 点击开启',
    newMenu: '新建…',
    filter: '筛选列表',
    filtersActive: count => `筛选列表，已启用 ${count} 项`,
    allGateways: '所有机器',
    currentGateway: '当前机器',
    hidden: '已隐藏'
  },
  'zh-hant': {
    toastsOn: '活動通知已開啟 — 點擊關閉',
    toastsOff: '活動通知已關閉 — 點擊開啟',
    newMenu: '新增…',
    filter: '篩選清單',
    filtersActive: count => `篩選清單，已啟用 ${count} 項`,
    allGateways: '所有機器',
    currentGateway: '目前的機器',
    hidden: '已隱藏'
  },
  fr: {
    toastsOn: 'Notifications d’activité activées — cliquer pour désactiver',
    toastsOff: 'Notifications d’activité désactivées — cliquer pour activer',
    newMenu: 'Nouveau…',
    filter: 'Filtrer la liste',
    filtersActive: count => `Filtrer la liste, ${count} filtre${count === 1 ? '' : 's'} actif${count === 1 ? '' : 's'}`,
    allGateways: 'Toutes les machines',
    currentGateway: 'Machine actuelle',
    hidden: 'Masqués'
  },
  de: {
    toastsOn: 'Aktivitätshinweise an — zum Ausschalten klicken',
    toastsOff: 'Aktivitätshinweise aus — zum Einschalten klicken',
    newMenu: 'Neu…',
    filter: 'Liste filtern',
    filtersActive: count => `Liste filtern, ${count} aktiv`,
    allGateways: 'Alle Rechner',
    currentGateway: 'Aktueller Rechner',
    hidden: 'Ausgeblendet'
  },
  es: {
    toastsOn: 'Avisos de actividad activados — haz clic para silenciarlos',
    toastsOff: 'Avisos de actividad desactivados — haz clic para activarlos',
    newMenu: 'Nuevo…',
    filter: 'Filtrar la lista',
    filtersActive: count => `Filtrar la lista, ${count} ${count === 1 ? 'activo' : 'activos'}`,
    allGateways: 'Todas las máquinas',
    currentGateway: 'Máquina actual',
    hidden: 'Ocultos'
  },
  ru: {
    toastsOn: 'Уведомления об активности включены — нажмите, чтобы отключить',
    toastsOff: 'Уведомления об активности отключены — нажмите, чтобы включить',
    newMenu: 'Создать…',
    filter: 'Фильтровать список',
    filtersActive: count => `Фильтровать список, активно: ${count}`,
    allGateways: 'Все компьютеры',
    currentGateway: 'Текущий компьютер',
    hidden: 'Скрытые'
  },
  ar: {
    toastsOn: 'إشعارات النشاط مفعّلة — انقر لكتمها',
    toastsOff: 'إشعارات النشاط معطّلة — انقر لتفعيلها',
    newMenu: 'جديد…',
    filter: 'تصفية القائمة',
    filtersActive: count => `تصفية القائمة، عدد المرشحات النشطة: ${count}`,
    allGateways: 'كل الأجهزة',
    currentGateway: 'الجهاز الحالي',
    hidden: 'مخفي'
  }
} satisfies Record<keyof typeof CANONICAL_GROUP_LOCALES, RosterChromeMessages>
