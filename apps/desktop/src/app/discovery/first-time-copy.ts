import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'
import type { PortalAccent } from '@/store/portal-accent'
import type { StarterIdea } from '@/store/starter-ideas'

/** The redesign's first-run surfaces for someone new to Hermes: the first-time splash, its tray
 *  entries, the empty chat's greeting and starter ideas, and the reply to a message held until sign
 *  up. Only the redesign renders them. The destinations' names (Chats, Agents, ...) are the rail's
 *  own labels (renamed-names.ts). */
export interface FirstTimeCopy {
  splash: {
    eyebrow: string
    title: string
    subtitle: string
    /** The rows' list, for assistive tech. */
    listLabel: string
    close: string
    lines: Record<'agents' | 'chats' | 'library' | 'routines', { line: string; example: string }>
    theme: {
      label: string
      mode: string
      light: string
      dark: string
      color: string
      colors: Record<PortalAccent, string>
    }
    ready: string
    footnote: string
    signUp: string
    start: string
  }
  tray: {
    signUp: { title: string; description: string; action: string }
    meet: { title: string; action: string }
  }
  landing: {
    greeting: string
    /** The ideas' group, for assistive tech. */
    ideasLabel: string
    ideas: Record<StarterIdea, string>
    hide: string
    /** The Hide button's accessible name. */
    hideLabel: string
    showIdeas: string
    /** The composer's resting text while the ideas are shown, and once they are gone. */
    placeholderIdeas: string
    placeholderRetired: string
  }
  held: {
    /** Who said what, for assistive tech. */
    you: string
    hermes: string
    title: string
    body: string
    signUp: string
    failedTitle: string
    failedBody: string
    tryAgain: string
  }
}

const en: FirstTimeCopy = {
  splash: {
    eyebrow: '// Welcome',
    title: 'Welcome to Hermes Agent',
    subtitle: 'Hover a row to learn more.',
    listLabel: 'What Hermes does',
    close: 'Close',
    lines: {
      chats: {
        line: 'Don’t just ask, assign.',
        example: 'e.g. “Research my 5 biggest competitors, compare their pricing, cite sources”'
      },
      routines: {
        line: 'Set it once, it keeps checking.',
        example: 'e.g. a table for 2 opening up, a sold-out item back in stock'
      },
      agents: {
        line: 'A team of experts. Each one gets smarter over time.',
        example: 'e.g. a travel agent that knows your preferences, a research agent that knows your industry'
      },
      library: {
        line: 'Everything your agents make, saved in one place.',
        example: 'e.g. every brief, report and doc it made, from every chat'
      }
    },
    theme: {
      label: 'Make it yours',
      mode: 'Light or dark',
      light: 'Light',
      dark: 'Dark',
      color: 'Color',
      colors: { blue: 'Blue', violet: 'Violet', green: 'Green', orange: 'Orange', pink: 'Pink' }
    },
    ready: 'Ready to get started?',
    footnote: 'Click anywhere in the nav to close. Find this again in notifications.',
    signUp: 'Sign up to start',
    start: 'Let’s get to work'
  },
  tray: {
    signUp: {
      title: 'Ready to get started?',
      description: 'Your own Hermes, set up in about a minute.',
      action: 'Sign up to start'
    },
    meet: { title: 'Meet Hermes', action: 'Open' }
  },
  landing: {
    greeting: 'Hi, I’m Hermes.',
    ideasLabel: 'Ideas to start with',
    ideas: {
      morning_briefing: 'Send me a morning briefing',
      research_options: 'Research and compare options',
      summarize_document: 'Summarize a long document',
      clean_inbox: 'Clean up my inbox'
    },
    hide: 'Hide',
    hideLabel: 'Hide ideas',
    showIdeas: 'Show ideas',
    placeholderIdeas: 'Assign a task, or pick one below',
    placeholderRetired: 'Give your agent a task'
  },
  held: {
    you: 'You',
    hermes: 'Hermes',
    title: 'I can do that.',
    body: 'Sign up and I’ll start on it right away.',
    signUp: 'Sign up to start',
    failedTitle: 'Couldn’t sign you in.',
    failedBody: 'Your request is still saved.',
    tryAgain: 'Try again'
  }
}

const fr: FirstTimeCopy = {
  splash: {
    eyebrow: '// Bienvenue',
    title: 'Bienvenue dans Hermes Agent',
    subtitle: 'Survolez une ligne pour en savoir plus.',
    listLabel: 'Ce que fait Hermes',
    close: 'Fermer',
    lines: {
      chats: {
        line: 'Ne demandez pas, confiez.',
        example: 'ex. « Étudie mes 5 plus gros concurrents, compare leurs prix, cite tes sources »'
      },
      routines: {
        line: 'Réglez-la une fois, elle continue de vérifier.',
        example: 'ex. une table pour 2 qui se libère, un article épuisé de nouveau en stock'
      },
      agents: {
        line: 'Une équipe d’experts. Chacun s’améliore avec le temps.',
        example: 'ex. un agent voyage qui connaît vos préférences, un agent recherche qui connaît votre secteur'
      },
      library: {
        line: 'Tout ce que font vos agents, rangé au même endroit.',
        example: 'ex. chaque note, rapport et document créé, de chaque discussion'
      }
    },
    theme: {
      label: 'À votre image',
      mode: 'Clair ou sombre',
      light: 'Clair',
      dark: 'Sombre',
      color: 'Couleur',
      colors: { blue: 'Bleu', violet: 'Violet', green: 'Vert', orange: 'Orange', pink: 'Rose' }
    },
    ready: 'Prêt à commencer ?',
    footnote:
      'Cliquez n’importe où dans la navigation pour fermer. Retrouvez cette page dans les notifications.',
    signUp: 'S’inscrire pour commencer',
    start: 'Au travail'
  },
  tray: {
    signUp: {
      title: 'Prêt à commencer ?',
      description: 'Votre propre Hermes, prêt en une minute environ.',
      action: 'S’inscrire pour commencer'
    },
    meet: { title: 'Découvrir Hermes', action: 'Ouvrir' }
  },
  landing: {
    greeting: 'Bonjour, je suis Hermes.',
    ideasLabel: 'Des idées pour commencer',
    ideas: {
      morning_briefing: 'Envoie-moi un point du matin',
      research_options: 'Compare plusieurs options',
      summarize_document: 'Résume un long document',
      clean_inbox: 'Fais le tri dans ma boîte mail'
    },
    hide: 'Masquer',
    hideLabel: 'Masquer les idées',
    showIdeas: 'Afficher des idées',
    placeholderIdeas: 'Confiez une tâche, ou choisissez-en une ci-dessous',
    placeholderRetired: 'Confiez une tâche à votre agent'
  },
  held: {
    you: 'Vous',
    hermes: 'Hermes',
    title: 'Je peux m’en charger.',
    body: 'Inscrivez-vous et je m’y mets tout de suite.',
    signUp: 'S’inscrire pour commencer',
    failedTitle: 'Impossible de vous connecter.',
    failedBody: 'Votre demande est toujours enregistrée.',
    tryAgain: 'Réessayer'
  }
}

const de: FirstTimeCopy = {
  splash: {
    eyebrow: '// Willkommen',
    title: 'Willkommen bei Hermes Agent',
    subtitle: 'Zeige auf eine Zeile, um mehr zu erfahren.',
    listLabel: 'Was Hermes kann',
    close: 'Schließen',
    lines: {
      chats: {
        line: 'Nicht nur fragen, beauftragen.',
        example: 'z. B. „Recherchiere meine 5 größten Wettbewerber, vergleiche ihre Preise, nenne Quellen“'
      },
      routines: {
        line: 'Einmal einrichten, sie prüft weiter.',
        example: 'z. B. ein Tisch für 2, der frei wird, ein ausverkaufter Artikel wieder auf Lager'
      },
      agents: {
        line: 'Ein Team aus Profis. Jeder wird mit der Zeit klüger.',
        example: 'z. B. ein Reiseagent, der deine Vorlieben kennt, ein Rechercheagent, der deine Branche kennt'
      },
      library: {
        line: 'Alles, was deine Agenten erstellen, an einem Ort.',
        example: 'z. B. jedes Briefing, jeder Bericht und jedes Dokument, aus jedem Chat'
      }
    },
    theme: {
      label: 'Mach es zu deinem',
      mode: 'Hell oder dunkel',
      light: 'Hell',
      dark: 'Dunkel',
      color: 'Farbe',
      colors: { blue: 'Blau', violet: 'Violett', green: 'Grün', orange: 'Orange', pink: 'Pink' }
    },
    ready: 'Bereit loszulegen?',
    footnote: 'Klicke irgendwo in die Navigation, um zu schließen. Du findest das wieder unter Benachrichtigungen.',
    signUp: 'Registrieren und loslegen',
    start: 'An die Arbeit'
  },
  tray: {
    signUp: {
      title: 'Bereit loszulegen?',
      description: 'Dein eigenes Hermes, in etwa einer Minute eingerichtet.',
      action: 'Registrieren und loslegen'
    },
    meet: { title: 'Hermes kennenlernen', action: 'Öffnen' }
  },
  landing: {
    greeting: 'Hallo, ich bin Hermes.',
    ideasLabel: 'Ideen für den Anfang',
    ideas: {
      morning_briefing: 'Schick mir ein Morgen-Briefing',
      research_options: 'Optionen recherchieren und vergleichen',
      summarize_document: 'Ein langes Dokument zusammenfassen',
      clean_inbox: 'Mein Postfach aufräumen'
    },
    hide: 'Ausblenden',
    hideLabel: 'Ideen ausblenden',
    showIdeas: 'Ideen zeigen',
    placeholderIdeas: 'Gib eine Aufgabe oder wähle unten eine aus',
    placeholderRetired: 'Gib deinem Agenten eine Aufgabe'
  },
  held: {
    you: 'Du',
    hermes: 'Hermes',
    title: 'Das kann ich erledigen.',
    body: 'Registriere dich und ich lege sofort los.',
    signUp: 'Registrieren und loslegen',
    failedTitle: 'Anmeldung fehlgeschlagen.',
    failedBody: 'Deine Anfrage ist weiterhin gespeichert.',
    tryAgain: 'Erneut versuchen'
  }
}

const es: FirstTimeCopy = {
  splash: {
    eyebrow: '// Bienvenida',
    title: 'Te damos la bienvenida a Hermes Agent',
    subtitle: 'Pasa el cursor por una fila para saber más.',
    listLabel: 'Qué hace Hermes',
    close: 'Cerrar',
    lines: {
      chats: {
        line: 'No solo preguntes, encarga.',
        example: 'p. ej. «Investiga a mis 5 mayores competidores, compara sus precios y cita las fuentes»'
      },
      routines: {
        line: 'Configúrala una vez y sigue comprobando.',
        example: 'p. ej. una mesa para 2 que se libera, un artículo agotado que vuelve a estar disponible'
      },
      agents: {
        line: 'Un equipo de expertos. Cada uno mejora con el tiempo.',
        example: 'p. ej. un agente de viajes que conoce tus gustos, un agente de investigación que conoce tu sector'
      },
      library: {
        line: 'Todo lo que crean tus agentes, en un solo lugar.',
        example: 'p. ej. cada resumen, informe y documento que creó, de cada chat'
      }
    },
    theme: {
      label: 'Hazlo tuyo',
      mode: 'Claro u oscuro',
      light: 'Claro',
      dark: 'Oscuro',
      color: 'Color',
      colors: { blue: 'Azul', violet: 'Violeta', green: 'Verde', orange: 'Naranja', pink: 'Rosa' }
    },
    ready: '¿Listo para empezar?',
    footnote: 'Haz clic en cualquier parte de la navegación para cerrar. Encuéntralo de nuevo en las notificaciones.',
    signUp: 'Regístrate para empezar',
    start: 'Manos a la obra'
  },
  tray: {
    signUp: {
      title: '¿Listo para empezar?',
      description: 'Tu propio Hermes, listo en un minuto aproximadamente.',
      action: 'Regístrate para empezar'
    },
    meet: { title: 'Conoce Hermes', action: 'Abrir' }
  },
  landing: {
    greeting: 'Hola, soy Hermes.',
    ideasLabel: 'Ideas para empezar',
    ideas: {
      morning_briefing: 'Envíame un resumen matutino',
      research_options: 'Investiga y compara opciones',
      summarize_document: 'Resume un documento largo',
      clean_inbox: 'Ordena mi bandeja de entrada'
    },
    hide: 'Ocultar',
    hideLabel: 'Ocultar ideas',
    showIdeas: 'Mostrar ideas',
    placeholderIdeas: 'Encarga una tarea o elige una de abajo',
    placeholderRetired: 'Dale una tarea a tu agente'
  },
  held: {
    you: 'Tú',
    hermes: 'Hermes',
    title: 'Puedo hacerlo.',
    body: 'Regístrate y empiezo ahora mismo.',
    signUp: 'Regístrate para empezar',
    failedTitle: 'No pudimos iniciar tu sesión.',
    failedBody: 'Tu petición sigue guardada.',
    tryAgain: 'Reintentar'
  }
}

const ja: FirstTimeCopy = {
  splash: {
    eyebrow: '// ようこそ',
    title: 'Hermes Agent へようこそ',
    subtitle: '各行にカーソルを合わせると詳しく表示されます。',
    listLabel: 'Hermes にできること',
    close: '閉じる',
    lines: {
      chats: {
        line: '質問するだけでなく、任せる。',
        example: '例：「大手競合 5 社を調べて、価格を比較し、出典を示して」'
      },
      routines: {
        line: '一度設定すれば、確認し続けます。',
        example: '例：2 名の席が空いたとき、売り切れ商品が再入荷したとき'
      },
      agents: {
        line: '専門家のチーム。それぞれが使うほど賢くなります。',
        example: '例：好みを知っている旅行エージェント、業界を知っているリサーチエージェント'
      },
      library: {
        line: 'エージェントが作ったものを、すべて一か所に。',
        example: '例：すべてのチャットで作った要約、レポート、ドキュメント'
      }
    },
    theme: {
      label: '自分好みに',
      mode: 'ライトまたはダーク',
      light: 'ライト',
      dark: 'ダーク',
      color: 'カラー',
      colors: { blue: 'ブルー', violet: 'バイオレット', green: 'グリーン', orange: 'オレンジ', pink: 'ピンク' }
    },
    ready: 'さっそく始めますか？',
    footnote: 'ナビゲーションのどこかをクリックすると閉じます。通知からもう一度開けます。',
    signUp: '登録して始める',
    start: 'さっそく始める'
  },
  tray: {
    signUp: {
      title: 'さっそく始めますか？',
      description: 'あなた専用の Hermes を、1 分ほどで準備します。',
      action: '登録して始める'
    },
    meet: { title: 'Hermes について', action: '開く' }
  },
  landing: {
    greeting: 'こんにちは、Hermes です。',
    ideasLabel: '最初のアイデア',
    ideas: {
      morning_briefing: '朝のブリーフィングを送って',
      research_options: '選択肢を調べて比較して',
      summarize_document: '長いドキュメントを要約して',
      clean_inbox: '受信トレイを整理して'
    },
    hide: '非表示',
    hideLabel: 'アイデアを非表示',
    showIdeas: 'アイデアを表示',
    placeholderIdeas: 'タスクを任せるか、下から選んでください',
    placeholderRetired: 'エージェントにタスクを任せる'
  },
  held: {
    you: 'あなた',
    hermes: 'Hermes',
    title: 'お任せください。',
    body: '登録すれば、すぐに取りかかります。',
    signUp: '登録して始める',
    failedTitle: 'サインインできませんでした。',
    failedBody: 'リクエストは保存されたままです。',
    tryAgain: 'もう一度試す'
  }
}

const zh: FirstTimeCopy = {
  splash: {
    eyebrow: '// 欢迎',
    title: '欢迎使用 Hermes Agent',
    subtitle: '将指针悬停在某一行上了解详情。',
    listLabel: 'Hermes 能做什么',
    close: '关闭',
    lines: {
      chats: {
        line: '不只是提问，而是委派。',
        example: '例如：“调研我的 5 个最大竞争对手，比较他们的价格，并注明来源”'
      },
      routines: {
        line: '设置一次，持续检查。',
        example: '例如：有 2 人桌空出来时、售罄商品重新到货时'
      },
      agents: {
        line: '一支专家团队，每位都会越用越聪明。',
        example: '例如：了解你偏好的旅行智能体、了解你所在行业的研究智能体'
      },
      library: {
        line: '智能体做出的一切，集中保存在一处。',
        example: '例如：来自每个聊天的每份简报、报告和文档'
      }
    },
    theme: {
      label: '打造你的风格',
      mode: '浅色或深色',
      light: '浅色',
      dark: '深色',
      color: '颜色',
      colors: { blue: '蓝色', violet: '紫色', green: '绿色', orange: '橙色', pink: '粉色' }
    },
    ready: '准备好开始了吗？',
    footnote: '点击导航中的任意位置即可关闭。可在通知中再次找到此页。',
    signUp: '注册并开始',
    start: '开始工作'
  },
  tray: {
    signUp: {
      title: '准备好开始了吗？',
      description: '你自己的 Hermes，大约一分钟即可设置好。',
      action: '注册并开始'
    },
    meet: { title: '认识 Hermes', action: '打开' }
  },
  landing: {
    greeting: '你好，我是 Hermes。',
    ideasLabel: '入门灵感',
    ideas: {
      morning_briefing: '给我发一份晨间简报',
      research_options: '调研并比较几个选项',
      summarize_document: '总结一份长文档',
      clean_inbox: '整理我的收件箱'
    },
    hide: '隐藏',
    hideLabel: '隐藏灵感',
    showIdeas: '显示灵感',
    placeholderIdeas: '布置一个任务，或从下方选一个',
    placeholderRetired: '给你的智能体布置任务'
  },
  held: {
    you: '你',
    hermes: 'Hermes',
    title: '这个我能做。',
    body: '注册后我会立刻开始。',
    signUp: '注册并开始',
    failedTitle: '无法为你登录。',
    failedBody: '你的请求仍已保存。',
    tryAgain: '重试'
  }
}

const zhHant: FirstTimeCopy = {
  splash: {
    eyebrow: '// 歡迎',
    title: '歡迎使用 Hermes Agent',
    subtitle: '將指標停在任一列上以了解更多。',
    listLabel: 'Hermes 能做什麼',
    close: '關閉',
    lines: {
      chats: {
        line: '不只是提問，而是交辦。',
        example: '例如：「研究我的 5 個最大競爭對手，比較他們的價格，並註明來源」'
      },
      routines: {
        line: '設定一次，持續檢查。',
        example: '例如：有 2 人桌空出來時、售完的商品重新到貨時'
      },
      agents: {
        line: '一支專家團隊，每位都會越用越聰明。',
        example: '例如：了解你偏好的旅遊代理、了解你所在產業的研究代理'
      },
      library: {
        line: '代理做出的一切，集中保存在一處。',
        example: '例如：來自每個聊天的每份簡報、報告和文件'
      }
    },
    theme: {
      label: '打造你的風格',
      mode: '淺色或深色',
      light: '淺色',
      dark: '深色',
      color: '顏色',
      colors: { blue: '藍色', violet: '紫色', green: '綠色', orange: '橘色', pink: '粉紅色' }
    },
    ready: '準備好開始了嗎？',
    footnote: '點按導覽中的任何位置即可關閉。可在通知中再次找到此頁。',
    signUp: '註冊並開始',
    start: '開始工作'
  },
  tray: {
    signUp: {
      title: '準備好開始了嗎？',
      description: '你自己的 Hermes，大約一分鐘即可設定好。',
      action: '註冊並開始'
    },
    meet: { title: '認識 Hermes', action: '開啟' }
  },
  landing: {
    greeting: '你好，我是 Hermes。',
    ideasLabel: '入門靈感',
    ideas: {
      morning_briefing: '給我一份晨間簡報',
      research_options: '研究並比較幾個選項',
      summarize_document: '摘要一份長文件',
      clean_inbox: '整理我的收件匣'
    },
    hide: '隱藏',
    hideLabel: '隱藏靈感',
    showIdeas: '顯示靈感',
    placeholderIdeas: '交辦一項任務，或從下方選一個',
    placeholderRetired: '交辦任務給你的代理'
  },
  held: {
    you: '你',
    hermes: 'Hermes',
    title: '這個我能做。',
    body: '註冊後我會馬上開始。',
    signUp: '註冊並開始',
    failedTitle: '無法為你登入。',
    failedBody: '你的請求仍然保存著。',
    tryAgain: '再試一次'
  }
}

const ru: FirstTimeCopy = {
  splash: {
    eyebrow: '// Добро пожаловать',
    title: 'Добро пожаловать в Hermes Agent',
    subtitle: 'Наведите на строку, чтобы узнать больше.',
    listLabel: 'Что умеет Hermes',
    close: 'Закрыть',
    lines: {
      chats: {
        line: 'Не просто спрашивайте, поручайте.',
        example: 'напр. «Изучи 5 моих главных конкурентов, сравни их цены, укажи источники»'
      },
      routines: {
        line: 'Настройте один раз, и она продолжит проверять.',
        example: 'напр. освободился столик на двоих, распроданный товар снова в наличии'
      },
      agents: {
        line: 'Команда экспертов. Каждый со временем становится умнее.',
        example: 'напр. агент по путешествиям, знающий ваши предпочтения, исследовательский агент, знающий вашу отрасль'
      },
      library: {
        line: 'Всё, что создают ваши агенты, в одном месте.',
        example: 'напр. каждая сводка, отчёт и документ из каждого чата'
      }
    },
    theme: {
      label: 'Под себя',
      mode: 'Светлая или тёмная',
      light: 'Светлая',
      dark: 'Тёмная',
      color: 'Цвет',
      colors: { blue: 'Синий', violet: 'Фиолетовый', green: 'Зелёный', orange: 'Оранжевый', pink: 'Розовый' }
    },
    ready: 'Готовы начать?',
    footnote: 'Нажмите в любом месте навигации, чтобы закрыть. Эту страницу можно снова найти в уведомлениях.',
    signUp: 'Зарегистрироваться',
    start: 'За работу'
  },
  tray: {
    signUp: {
      title: 'Готовы начать?',
      description: 'Ваш собственный Hermes, готов примерно за минуту.',
      action: 'Зарегистрироваться'
    },
    meet: { title: 'Знакомство с Hermes', action: 'Открыть' }
  },
  landing: {
    greeting: 'Привет, я Hermes.',
    ideasLabel: 'Идеи для начала',
    ideas: {
      morning_briefing: 'Присылай мне утреннюю сводку',
      research_options: 'Изучи и сравни варианты',
      summarize_document: 'Кратко перескажи длинный документ',
      clean_inbox: 'Разбери мою почту'
    },
    hide: 'Скрыть',
    hideLabel: 'Скрыть идеи',
    showIdeas: 'Показать идеи',
    placeholderIdeas: 'Поручите задачу или выберите одну ниже',
    placeholderRetired: 'Дайте агенту задачу'
  },
  held: {
    you: 'Вы',
    hermes: 'Hermes',
    title: 'Я могу это сделать.',
    body: 'Зарегистрируйтесь, и я сразу начну.',
    signUp: 'Зарегистрироваться',
    failedTitle: 'Не удалось войти.',
    failedBody: 'Ваш запрос по-прежнему сохранён.',
    tryAgain: 'Попробовать снова'
  }
}

const ar: FirstTimeCopy = {
  splash: {
    eyebrow: '// مرحبًا',
    title: 'مرحبًا بك في Hermes Agent',
    subtitle: 'مرّر المؤشر فوق صف لمعرفة المزيد.',
    listLabel: 'ما يفعله Hermes',
    close: 'إغلاق',
    lines: {
      chats: {
        line: 'لا تكتفِ بالسؤال، بل كلّف.',
        example: 'مثلًا: «ابحث عن أكبر 5 منافسين لي، وقارن أسعارهم، واذكر المصادر»'
      },
      routines: {
        line: 'اضبطه مرة واحدة، ويواصل التحقق.',
        example: 'مثلًا: طاولة لشخصين أصبحت متاحة، منتج نفد عاد إلى المخزون'
      },
      agents: {
        line: 'فريق من الخبراء. يزداد كل منهم ذكاءً مع الوقت.',
        example: 'مثلًا: وكيل سفر يعرف تفضيلاتك، ووكيل بحث يعرف مجال عملك'
      },
      library: {
        line: 'كل ما يصنعه وكلاؤك، محفوظ في مكان واحد.',
        example: 'مثلًا: كل ملخص وتقرير ومستند صنعه، من كل محادثة'
      }
    },
    theme: {
      label: 'اجعله على ذوقك',
      mode: 'فاتح أو داكن',
      light: 'فاتح',
      dark: 'داكن',
      color: 'اللون',
      colors: { blue: 'أزرق', violet: 'بنفسجي', green: 'أخضر', orange: 'برتقالي', pink: 'وردي' }
    },
    ready: 'هل أنت مستعد للبدء؟',
    footnote: 'انقر في أي مكان في شريط التنقل للإغلاق. ستجد هذه الصفحة مجددًا في الإشعارات.',
    signUp: 'سجّل للبدء',
    start: 'لنبدأ العمل'
  },
  tray: {
    signUp: {
      title: 'هل أنت مستعد للبدء؟',
      description: 'نسختك الخاصة من Hermes، جاهزة في نحو دقيقة.',
      action: 'سجّل للبدء'
    },
    meet: { title: 'تعرّف على Hermes', action: 'فتح' }
  },
  landing: {
    greeting: 'مرحبًا، أنا Hermes.',
    ideasLabel: 'أفكار للبدء',
    ideas: {
      morning_briefing: 'أرسل لي موجزًا صباحيًا',
      research_options: 'ابحث في الخيارات وقارن بينها',
      summarize_document: 'لخّص مستندًا طويلًا',
      clean_inbox: 'نظّم صندوق بريدي'
    },
    hide: 'إخفاء',
    hideLabel: 'إخفاء الأفكار',
    showIdeas: 'عرض الأفكار',
    placeholderIdeas: 'كلّف بمهمة، أو اختر واحدة أدناه',
    placeholderRetired: 'أعطِ وكيلك مهمة'
  },
  held: {
    you: 'أنت',
    hermes: 'Hermes',
    title: 'يمكنني فعل ذلك.',
    body: 'سجّل وسأبدأ عليه فورًا.',
    signUp: 'سجّل للبدء',
    failedTitle: 'تعذّر تسجيل دخولك.',
    failedBody: 'طلبك لا يزال محفوظًا.',
    tryAgain: 'حاول مرة أخرى'
  }
}

const COPY: Record<BundledLocale, FirstTimeCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function firstTimeCopy(locale: string): FirstTimeCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useFirstTimeCopy(): FirstTimeCopy {
  return firstTimeCopy(useI18n().locale)
}
