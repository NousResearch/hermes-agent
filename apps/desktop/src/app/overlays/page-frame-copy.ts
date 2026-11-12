import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** The redesign's page descriptions for routed pages whose heading had none (or only a raw count),
 *  and titles that used to carry a count. Counts move to the heading row's quiet meta slot. */
export interface PageFrameCopy {
  agentsDescription: string
  libraryDescription: string
  messagingDescription: string
  schedulesDescription: string
  webhooksTitle: string
}

const COPY: Record<BundledLocale, PageFrameCopy> = {
  en: {
    agentsDescription:
      'Every agent on this device and the gateways you connect. Open one to see and change its settings.',
    libraryDescription: 'Images, files and links your agents made, gathered from every chat.',
    messagingDescription: 'Reach your agents from the chat apps you already use, and choose who may talk to them.',
    schedulesDescription: 'Prompts that run on their own at the times you set, and post the results where you choose.',
    webhooksTitle: 'Webhooks'
  },
  fr: {
    agentsDescription:
      'Tous les agents de cet appareil et des passerelles connectées. Ouvrez-en un pour voir et modifier ses réglages.',
    libraryDescription: 'Les images, fichiers et liens créés par vos agents, réunis depuis toutes vos discussions.',
    messagingDescription: 'Parlez à vos agents depuis vos messageries habituelles et choisissez qui peut leur écrire.',
    schedulesDescription:
      'Des prompts qui s’exécutent seuls aux heures choisies et publient leurs résultats où vous le souhaitez.',
    webhooksTitle: 'Webhooks'
  },
  de: {
    agentsDescription:
      'Alle Agenten auf diesem Gerät und den verbundenen Gateways. Öffne einen, um seine Einstellungen zu sehen und zu ändern.',
    libraryDescription: 'Bilder, Dateien und Links, die deine Agenten erstellt haben, aus allen Chats gesammelt.',
    messagingDescription:
      'Erreiche deine Agenten über die Chat-Apps, die du schon nutzt, und lege fest, wer mit ihnen schreiben darf.',
    schedulesDescription:
      'Prompts, die zu festgelegten Zeiten von selbst laufen und ihre Ergebnisse dorthin senden, wo du willst.',
    webhooksTitle: 'Webhooks'
  },
  es: {
    agentsDescription:
      'Todos los agentes de este dispositivo y de las pasarelas que conectes. Abre uno para ver y cambiar sus ajustes.',
    libraryDescription: 'Imágenes, archivos y enlaces que crearon tus agentes, reunidos de todos tus chats.',
    messagingDescription: 'Habla con tus agentes desde las apps de chat que ya usas y elige quién puede escribirles.',
    schedulesDescription:
      'Prompts que se ejecutan solos a las horas que elijas y publican los resultados donde prefieras.',
    webhooksTitle: 'Webhooks'
  },
  ja: {
    agentsDescription: 'このデバイスと接続したゲートウェイ上のすべてのエージェント。開くと設定を確認・変更できます。',
    libraryDescription: 'エージェントが作成した画像・ファイル・リンクを、すべてのチャットから集めて表示します。',
    messagingDescription: '使い慣れたチャットアプリからエージェントに話しかけ、誰がやり取りできるかを選べます。',
    schedulesDescription: '指定した時刻に自動で実行され、結果を選んだ場所に投稿するプロンプトです。',
    webhooksTitle: 'Webhook'
  },
  zh: {
    agentsDescription: '此设备和已连接网关上的所有智能体。打开一个即可查看和修改其设置。',
    libraryDescription: '智能体在所有对话中生成的图片、文件和链接，集中在这里。',
    messagingDescription: '通过你常用的聊天应用联系智能体，并选择谁可以与它们对话。',
    schedulesDescription: '在你设定的时间自动运行，并把结果发到你选择的地方的提示词。',
    webhooksTitle: 'Webhook'
  },
  'zh-hant': {
    agentsDescription: '此裝置與已連線閘道上的所有代理。開啟一個即可查看並修改其設定。',
    libraryDescription: '代理在所有對話中產生的圖片、檔案與連結，集中在這裡。',
    messagingDescription: '透過你常用的聊天應用程式聯絡代理，並選擇誰可以與它們對話。',
    schedulesDescription: '在你設定的時間自動執行，並將結果發布到你選擇之處的提示詞。',
    webhooksTitle: 'Webhook'
  },
  ru: {
    agentsDescription:
      'Все агенты на этом устройстве и на подключённых шлюзах. Откройте агента, чтобы посмотреть и изменить его настройки.',
    libraryDescription: 'Изображения, файлы и ссылки, созданные вашими агентами, собранные из всех чатов.',
    messagingDescription: 'Пишите агентам из привычных мессенджеров и выбирайте, кто может с ними общаться.',
    schedulesDescription:
      'Промпты, которые запускаются сами в заданное время и публикуют результаты туда, куда вы выберете.',
    webhooksTitle: 'Вебхуки'
  },
  ar: {
    agentsDescription: 'كل الوكلاء على هذا الجهاز وعلى البوابات التي تتصل بها. افتح أحدها لعرض إعداداته وتغييرها.',
    libraryDescription: 'الصور والملفات والروابط التي أنشأها وكلاؤك، مجمّعة من كل المحادثات.',
    messagingDescription: 'تواصل مع وكلائك من تطبيقات الدردشة التي تستخدمها، واختر من يمكنه مراسلتهم.',
    schedulesDescription: 'أوامر تعمل تلقائيًا في الأوقات التي تحددها وتنشر النتائج حيث تختار.',
    webhooksTitle: 'خطافات الويب'
  }
}

export function usePageFrameCopy(): PageFrameCopy {
  const { locale } = useI18n()

  return COPY[isBundledLocale(locale) ? locale : 'en']
}
