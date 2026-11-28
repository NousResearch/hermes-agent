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
      'Every agent on this device and the machines you connect. Open one to see and change its settings.',
    libraryDescription: 'Images, files and links your agents made, gathered from every chat.',
    messagingDescription: 'Reach your agents from the chat apps you already use, and choose who may talk to them.',
    schedulesDescription: 'Tasks your agents run on a schedule. Change what a routine does in its agent’s settings.',
    webhooksTitle: 'Webhooks'
  },
  fr: {
    agentsDescription:
      'Tous les agents de cet appareil et des machines connectées. Ouvrez-en un pour voir et modifier ses réglages.',
    libraryDescription: 'Les images, fichiers et liens créés par vos agents, réunis depuis toutes vos discussions.',
    messagingDescription: 'Parlez à vos agents depuis vos messageries habituelles et choisissez qui peut leur écrire.',
    schedulesDescription: 'Les tâches que vos agents exécutent selon un horaire. Modifiez une routine dans les réglages de son agent.',
    webhooksTitle: 'Webhooks'
  },
  de: {
    agentsDescription:
      'Alle Agenten auf diesem Gerät und den verbundenen Rechnern. Öffne einen, um seine Einstellungen zu sehen und zu ändern.',
    libraryDescription: 'Bilder, Dateien und Links, die deine Agenten erstellt haben, aus allen Chats gesammelt.',
    messagingDescription:
      'Erreiche deine Agenten über die Chat-Apps, die du schon nutzt, und lege fest, wer mit ihnen schreiben darf.',
    schedulesDescription: 'Aufgaben, die deine Agenten nach Zeitplan ausführen. Was eine Routine tut, änderst du in den Einstellungen ihres Agenten.',
    webhooksTitle: 'Webhooks'
  },
  es: {
    agentsDescription:
      'Todos los agentes de este dispositivo y de las máquinas que conectes. Abre uno para ver y cambiar sus ajustes.',
    libraryDescription: 'Imágenes, archivos y enlaces que crearon tus agentes, reunidos de todos tus chats.',
    messagingDescription: 'Habla con tus agentes desde las apps de chat que ya usas y elige quién puede escribirles.',
    schedulesDescription: 'Tareas que tus agentes ejecutan según un horario. Cambia lo que hace una rutina en los ajustes de su agente.',
    webhooksTitle: 'Webhooks'
  },
  ja: {
    agentsDescription: 'このデバイスと接続したマシン上のすべてのエージェント。開くと設定を確認・変更できます。',
    libraryDescription: 'エージェントが作成した画像・ファイル・リンクを、すべてのチャットから集めて表示します。',
    messagingDescription: '使い慣れたチャットアプリからエージェントに話しかけ、誰がやり取りできるかを選べます。',
    schedulesDescription: 'エージェントがスケジュールどおりに実行するタスクです。内容はエージェントの設定で変更します。',
    webhooksTitle: 'Webhook'
  },
  zh: {
    agentsDescription: '此设备和已连接机器上的所有智能体。打开一个即可查看和修改其设置。',
    libraryDescription: '智能体在所有对话中生成的图片、文件和链接，集中在这里。',
    messagingDescription: '通过你常用的聊天应用联系智能体，并选择谁可以与它们对话。',
    schedulesDescription: '智能体按计划自动执行的任务。在智能体的设置中更改例行任务的内容。',
    webhooksTitle: 'Webhook'
  },
  'zh-hant': {
    agentsDescription: '此裝置與你連線的機器上的所有智慧體。開啟一個即可查看並修改其設定。',
    libraryDescription: '你的智慧體在所有對話中產生的圖片、檔案與連結，集中在這裡。',
    messagingDescription: '透過你常用的聊天應用程式聯絡智慧體，並選擇誰可以與它們對話。',
    schedulesDescription: '智慧體依排程自動執行的任務。在智慧體的設定中變更例行任務的內容。',
    webhooksTitle: 'Webhooks'
  },
  ru: {
    agentsDescription:
      'Все агенты на этом устройстве и на подключённых компьютерах. Откройте агента, чтобы посмотреть и изменить его настройки.',
    libraryDescription: 'Изображения, файлы и ссылки, созданные вашими агентами, собранные из всех чатов.',
    messagingDescription: 'Пишите агентам из привычных мессенджеров и выбирайте, кто может с ними общаться.',
    schedulesDescription: 'Задачи, которые ваши агенты выполняют по расписанию. Что делает задача, меняется в настройках её агента.',
    webhooksTitle: 'Вебхуки'
  },
  ar: {
    agentsDescription: 'كل الوكلاء على هذا الجهاز وعلى الأجهزة التي تتصل بها. افتح أحدها لعرض إعداداته وتغييرها.',
    libraryDescription: 'الصور والملفات والروابط التي أنشأها وكلاؤك، مجمّعة من كل المحادثات.',
    messagingDescription: 'تواصل مع وكلائك من تطبيقات الدردشة التي تستخدمها، واختر من يمكنه مراسلتهم.',
    schedulesDescription: 'مهام ينفّذها وكلاؤك وفق جدول زمني. غيّر ما تفعله المهمة من إعدادات وكيلها.',
    webhooksTitle: 'خطافات الويب'
  }
}

export function usePageFrameCopy(): PageFrameCopy {
  const { locale } = useI18n()

  return COPY[isBundledLocale(locale) ? locale : 'en']
}
