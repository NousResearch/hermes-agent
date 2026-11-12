import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** The redesign's page descriptions for routed pages whose heading had none (or only a raw count),
 *  and titles that used to carry a count. Counts move to the heading row's quiet meta slot. */
export interface PageFrameCopy {
  agentsDescription: string
  schedulesDescription: string
  webhooksTitle: string
}

const COPY: Record<BundledLocale, PageFrameCopy> = {
  en: {
    agentsDescription:
      'Every agent on this device and the gateways you connect. Open one to see and change its settings.',
    schedulesDescription: 'Prompts that run on their own at the times you set, and post the results where you choose.',
    webhooksTitle: 'Webhooks'
  },
  fr: {
    agentsDescription:
      'Tous les agents de cet appareil et des passerelles connectées. Ouvrez-en un pour voir et modifier ses réglages.',
    schedulesDescription:
      'Des prompts qui s’exécutent seuls aux heures choisies et publient leurs résultats où vous le souhaitez.',
    webhooksTitle: 'Webhooks'
  },
  de: {
    agentsDescription:
      'Alle Agenten auf diesem Gerät und den verbundenen Gateways. Öffne einen, um seine Einstellungen zu sehen und zu ändern.',
    schedulesDescription:
      'Prompts, die zu festgelegten Zeiten von selbst laufen und ihre Ergebnisse dorthin senden, wo du willst.',
    webhooksTitle: 'Webhooks'
  },
  es: {
    agentsDescription:
      'Todos los agentes de este dispositivo y de las pasarelas que conectes. Abre uno para ver y cambiar sus ajustes.',
    schedulesDescription:
      'Prompts que se ejecutan solos a las horas que elijas y publican los resultados donde prefieras.',
    webhooksTitle: 'Webhooks'
  },
  ja: {
    agentsDescription: 'このデバイスと接続したゲートウェイ上のすべてのエージェント。開くと設定を確認・変更できます。',
    schedulesDescription: '指定した時刻に自動で実行され、結果を選んだ場所に投稿するプロンプトです。',
    webhooksTitle: 'Webhook'
  },
  zh: {
    agentsDescription: '此设备和已连接网关上的所有智能体。打开一个即可查看和修改其设置。',
    schedulesDescription: '在你设定的时间自动运行，并把结果发到你选择的地方的提示词。',
    webhooksTitle: 'Webhook'
  },
  'zh-hant': {
    agentsDescription: '此裝置與已連線閘道上的所有代理。開啟一個即可查看並修改其設定。',
    schedulesDescription: '在你設定的時間自動執行，並將結果發布到你選擇之處的提示詞。',
    webhooksTitle: 'Webhook'
  },
  ru: {
    agentsDescription:
      'Все агенты на этом устройстве и на подключённых шлюзах. Откройте агента, чтобы посмотреть и изменить его настройки.',
    schedulesDescription:
      'Промпты, которые запускаются сами в заданное время и публикуют результаты туда, куда вы выберете.',
    webhooksTitle: 'Вебхуки'
  },
  ar: {
    agentsDescription: 'كل الوكلاء على هذا الجهاز وعلى البوابات التي تتصل بها. افتح أحدها لعرض إعداداته وتغييرها.',
    schedulesDescription: 'أوامر تعمل تلقائيًا في الأوقات التي تحددها وتنشر النتائج حيث تختار.',
    webhooksTitle: 'خطافات الويب'
  }
}

export function usePageFrameCopy(): PageFrameCopy {
  const { locale } = useI18n()

  return COPY[isBundledLocale(locale) ? locale : 'en']
}
