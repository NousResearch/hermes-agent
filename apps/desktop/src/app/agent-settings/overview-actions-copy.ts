import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

const copy = {
  en: {
    disconnectedTitle: '{name} is disconnected',
    disconnectedHelp: 'Messages sent there won’t reach this agent until you reconnect.',
    duplicate: 'Duplicate agent',
    setDefault: 'Set as default agent',
    copyPath: 'Copy agent folder path',
    defaultUnavailable: 'Setting a default agent is unavailable for this machine in this app.'
  },
  fr: {
    disconnectedTitle: '{name} est déconnecté',
    disconnectedHelp:
      'Les messages envoyés sur ce canal ne parviendront pas à cet agent tant que vous ne l’aurez pas reconnecté.',
    duplicate: 'Dupliquer l’agent',
    setDefault: 'Définir comme agent par défaut',
    copyPath: 'Copier le chemin du dossier de l’agent',
    defaultUnavailable: 'Définir un agent par défaut n’est pas disponible pour cette machine dans cette application.'
  },
  de: {
    disconnectedTitle: '{name} ist nicht verbunden',
    disconnectedHelp:
      'Dort gesendete Nachrichten erreichen diesen Agenten erst, wenn du die Verbindung wiederherstellst.',
    duplicate: 'Agent duplizieren',
    setDefault: 'Als Standardagent festlegen',
    copyPath: 'Pfad des Agentenordners kopieren',
    defaultUnavailable: 'Für diesen Rechner kann in dieser App kein Standardagent festgelegt werden.'
  },
  es: {
    disconnectedTitle: '{name} está desconectado',
    disconnectedHelp: 'Los mensajes enviados allí no llegarán a este agente hasta que vuelvas a conectarlo.',
    duplicate: 'Duplicar agente',
    setDefault: 'Establecer como agente predeterminado',
    copyPath: 'Copiar ruta de la carpeta del agente',
    defaultUnavailable: 'No se puede establecer un agente predeterminado para esta máquina en esta aplicación.'
  },
  ja: {
    disconnectedTitle: '{name} は切断されています',
    disconnectedHelp: '再接続するまで、そこで送信されたメッセージはこのエージェントに届きません。',
    duplicate: 'エージェントを複製',
    setDefault: '既定のエージェントに設定',
    copyPath: 'エージェントフォルダーのパスをコピー',
    defaultUnavailable: 'このアプリでは、このマシンの既定のエージェントを設定できません。'
  },
  zh: {
    disconnectedTitle: '{name} 已断开连接',
    disconnectedHelp: '重新连接前，在那里发送的消息不会送达此智能体。',
    duplicate: '复制智能体',
    setDefault: '设为默认智能体',
    copyPath: '复制智能体文件夹路径',
    defaultUnavailable: '此应用无法为此机器设置默认智能体。'
  },
  'zh-hant': {
    disconnectedTitle: '{name} 已中斷連線',
    disconnectedHelp: '重新連線前，在那裡傳送的訊息不會送達此智慧體。',
    duplicate: '複製智慧體',
    setDefault: '設為預設智慧體',
    copyPath: '複製智慧體資料夾路徑',
    defaultUnavailable: '此應用程式無法為此機器設定預設智慧體。'
  },
  ru: {
    disconnectedTitle: '{name}: нет подключения',
    disconnectedHelp: 'Отправленные туда сообщения не дойдут до этого агента, пока вы не восстановите подключение.',
    duplicate: 'Дублировать агента',
    setDefault: 'Сделать агентом по умолчанию',
    copyPath: 'Копировать путь к папке агента',
    defaultUnavailable: 'В этом приложении нельзя выбрать агента по умолчанию для этого компьютера.'
  },
  ar: {
    disconnectedTitle: '{name} غير متصل',
    disconnectedHelp: 'لن تصل الرسائل المُرسلة هناك إلى هذا الوكيل حتى تعيد الاتصال.',
    duplicate: 'تكرار الوكيل',
    setDefault: 'تعيين كوكيل افتراضي',
    copyPath: 'نسخ مسار مجلد الوكيل',
    defaultUnavailable: 'تعيين وكيل افتراضي غير متاح لهذا الجهاز في هذا التطبيق.'
  }
} satisfies Record<BundledLocale, Record<string, string>>

export function useOverviewActionsCopy() {
  const { locale } = useI18n()

  return isBundledLocale(locale) ? copy[locale] : copy.en
}
