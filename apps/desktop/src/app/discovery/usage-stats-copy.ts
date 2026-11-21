import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** The redesign's usage-stats offer (Figma 1335:32952, 1365:34379): the sidebar card, its short
 *  dialog and its tray entry. Classic keeps the composer strip and the long dialog
 *  (`t.sharedMetrics`). The promise is "a random ID", never "anonymous": that is what the
 *  product can truthfully say. */
export interface UsageStatsCopy {
  card: {
    title: string
    /** Followed by the Details link, inside the same sentence. */
    body: string
    details: string
    share: string
    decline: string
    /** The ×: it puts the card away into the notifications tray without answering. */
    later: string
  }
  dialog: {
    title: string
    intro: string
    sending: string
    seeShared: string
    share: string
    notNow: string
  }
  tray: {
    title: string
    description: string
    action: string
    /** The ×: the person asked not to be offered it here again. */
    clear: string
  }
}

const en: UsageStatsCopy = {
  card: {
    title: 'Share usage stats?',
    body: 'Help improve your experience by sharing Hermes usage.',
    details: 'Details',
    share: 'Share with Nous',
    decline: 'No thanks',
    later: 'Move to notifications'
  },
  dialog: {
    title: 'Help improve Hermes',
    intro: 'Share usage stats so we can find bugs and see what works. Your chats and files are never shared.',
    sending: 'Sent once a day with a random ID. Turn it off anytime in Settings.',
    seeShared: 'See what’s shared',
    share: 'Share stats',
    notNow: 'Not now'
  },
  tray: {
    title: 'Share usage stats?',
    description: 'Help improve Hermes. Your chats and files are never shared.',
    action: 'Review',
    clear: 'Clear'
  }
}

const fr: UsageStatsCopy = {
  card: {
    title: 'Partager les statistiques d’utilisation ?',
    body: 'Aidez à améliorer votre expérience en partageant votre utilisation de Hermes.',
    details: 'Détails',
    share: 'Partager avec Nous',
    decline: 'Non merci',
    later: 'Déplacer vers les notifications'
  },
  dialog: {
    title: 'Aidez à améliorer Hermes',
    intro:
      'Partagez les statistiques d’utilisation pour nous aider à trouver les bugs et voir ce qui fonctionne. Vos discussions et vos fichiers ne sont jamais partagés.',
    sending:
      'Envoyées une fois par jour avec un identifiant aléatoire. Désactivez-les à tout moment dans les Réglages.',
    seeShared: 'Voir ce qui est partagé',
    share: 'Partager les statistiques',
    notNow: 'Pas maintenant'
  },
  tray: {
    title: 'Partager les statistiques d’utilisation ?',
    description: 'Aidez à améliorer Hermes. Vos discussions et vos fichiers ne sont jamais partagés.',
    action: 'Examiner',
    clear: 'Effacer'
  }
}

const de: UsageStatsCopy = {
  card: {
    title: 'Nutzungsstatistiken teilen?',
    body: 'Helfen Sie, Ihr Erlebnis zu verbessern, indem Sie Ihre Hermes-Nutzung teilen.',
    details: 'Details',
    share: 'Mit Nous teilen',
    decline: 'Nein, danke',
    later: 'In die Benachrichtigungen verschieben'
  },
  dialog: {
    title: 'Hermes verbessern',
    intro:
      'Teilen Sie Nutzungsstatistiken, damit wir Fehler finden und sehen, was funktioniert. Ihre Chats und Dateien werden nie geteilt.',
    sending: 'Einmal täglich mit einer zufälligen ID gesendet. Jederzeit in den Einstellungen abschaltbar.',
    seeShared: 'Sehen, was geteilt wird',
    share: 'Statistiken teilen',
    notNow: 'Nicht jetzt'
  },
  tray: {
    title: 'Nutzungsstatistiken teilen?',
    description: 'Helfen Sie, Hermes zu verbessern. Ihre Chats und Dateien werden nie geteilt.',
    action: 'Ansehen',
    clear: 'Entfernen'
  }
}

const es: UsageStatsCopy = {
  card: {
    title: '¿Compartir estadísticas de uso?',
    body: 'Ayuda a mejorar tu experiencia compartiendo el uso de Hermes.',
    details: 'Detalles',
    share: 'Compartir con Nous',
    decline: 'No, gracias',
    later: 'Mover a notificaciones'
  },
  dialog: {
    title: 'Ayuda a mejorar Hermes',
    intro:
      'Comparte estadísticas de uso para que podamos encontrar errores y ver qué funciona. Tus chats y archivos nunca se comparten.',
    sending: 'Se envían una vez al día con un ID aleatorio. Desactívalo cuando quieras en Ajustes.',
    seeShared: 'Ver qué se comparte',
    share: 'Compartir estadísticas',
    notNow: 'Ahora no'
  },
  tray: {
    title: '¿Compartir estadísticas de uso?',
    description: 'Ayuda a mejorar Hermes. Tus chats y archivos nunca se comparten.',
    action: 'Revisar',
    clear: 'Quitar'
  }
}

const ja: UsageStatsCopy = {
  card: {
    title: '利用統計を共有しますか？',
    body: 'Hermes の利用状況を共有して、使い心地の改善にご協力ください。',
    details: '詳細',
    share: 'Nous と共有する',
    decline: '共有しない',
    later: '通知に移動'
  },
  dialog: {
    title: 'Hermes の改善にご協力ください',
    intro:
      '利用統計を共有いただくと、不具合の発見や効果の確認に役立ちます。チャットやファイルが共有されることはありません。',
    sending: '1 日 1 回、ランダムな ID で送信されます。設定からいつでもオフにできます。',
    seeShared: '共有される内容を見る',
    share: '統計を共有する',
    notNow: '今はしない'
  },
  tray: {
    title: '利用統計を共有しますか？',
    description: 'Hermes の改善にご協力ください。チャットやファイルが共有されることはありません。',
    action: '確認する',
    clear: '消去'
  }
}

const zh: UsageStatsCopy = {
  card: {
    title: '分享使用统计？',
    body: '分享 Hermes 的使用情况，帮助改进你的体验。',
    details: '详情',
    share: '分享给 Nous',
    decline: '不用了',
    later: '移到通知'
  },
  dialog: {
    title: '帮助改进 Hermes',
    intro: '分享使用统计，帮助我们发现问题、了解哪些功能有效。你的对话和文件绝不会被分享。',
    sending: '每天发送一次，使用随机 ID。可随时在设置中关闭。',
    seeShared: '查看分享的内容',
    share: '分享统计',
    notNow: '暂不'
  },
  tray: {
    title: '分享使用统计？',
    description: '帮助改进 Hermes。你的对话和文件绝不会被分享。',
    action: '查看',
    clear: '清除'
  }
}

const zhHant: UsageStatsCopy = {
  card: {
    title: '分享使用統計？',
    body: '分享 Hermes 的使用情況，協助改善你的體驗。',
    details: '詳情',
    share: '分享給 Nous',
    decline: '不用了',
    later: '移到通知'
  },
  dialog: {
    title: '協助改善 Hermes',
    intro: '分享使用統計，協助我們找出問題、了解哪些功能有效。你的對話和檔案絕不會被分享。',
    sending: '每天傳送一次，使用隨機 ID。可隨時在設定中關閉。',
    seeShared: '查看分享的內容',
    share: '分享統計',
    notNow: '暫不'
  },
  tray: {
    title: '分享使用統計？',
    description: '協助改善 Hermes。你的對話和檔案絕不會被分享。',
    action: '查看',
    clear: '清除'
  }
}

const ru: UsageStatsCopy = {
  card: {
    title: 'Делиться статистикой использования?',
    body: 'Помогите сделать работу удобнее, поделившись статистикой использования Hermes.',
    details: 'Подробнее',
    share: 'Делиться с Nous',
    decline: 'Нет, спасибо',
    later: 'Перенести в уведомления'
  },
  dialog: {
    title: 'Помогите улучшить Hermes',
    intro:
      'Делитесь статистикой использования, чтобы мы находили ошибки и видели, что работает. Ваши чаты и файлы никогда не передаются.',
    sending: 'Отправляется раз в день со случайным ID. Отключить можно в любой момент в настройках.',
    seeShared: 'Что именно передаётся',
    share: 'Делиться статистикой',
    notNow: 'Не сейчас'
  },
  tray: {
    title: 'Делиться статистикой использования?',
    description: 'Помогите улучшить Hermes. Ваши чаты и файлы никогда не передаются.',
    action: 'Посмотреть',
    clear: 'Убрать'
  }
}

const ar: UsageStatsCopy = {
  card: {
    title: 'مشاركة إحصاءات الاستخدام؟',
    body: 'ساعد في تحسين تجربتك بمشاركة استخدامك لـ Hermes.',
    details: 'التفاصيل',
    share: 'المشاركة مع Nous',
    decline: 'لا، شكرًا',
    later: 'نقل إلى الإشعارات'
  },
  dialog: {
    title: 'ساعد في تحسين Hermes',
    intro: 'شارك إحصاءات الاستخدام لنكتشف الأخطاء ونعرف ما الذي ينجح. لا تتم مشاركة محادثاتك وملفاتك أبدًا.',
    sending: 'تُرسل مرة يوميًا مع معرّف عشوائي. يمكنك إيقافها في أي وقت من الإعدادات.',
    seeShared: 'اطّلع على ما تتم مشاركته',
    share: 'مشاركة الإحصاءات',
    notNow: 'ليس الآن'
  },
  tray: {
    title: 'مشاركة إحصاءات الاستخدام؟',
    description: 'ساعد في تحسين Hermes. لا تتم مشاركة محادثاتك وملفاتك أبدًا.',
    action: 'مراجعة',
    clear: 'مسح'
  }
}

const COPY: Record<BundledLocale, UsageStatsCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function usageStatsCopy(locale: string): UsageStatsCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useUsageStatsCopy(): UsageStatsCopy {
  return usageStatsCopy(useI18n().locale)
}
