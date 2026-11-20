import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'
import type { FirstTimeSplashPlace } from '@/store/first-time-splash'
import type { PortalAccent } from '@/store/portal-accent'
import type { StarterIdea } from '@/store/starter-ideas'

import type { Artwork } from '../redesign/artwork'

/** The redesign's first-run surfaces for someone new to Hermes (Figma "Prod - Desktop" › Onboarding,
 *  01 Discover & sign in): the first-time splash and its "Make it yours" step, its tray entries, the
 *  empty chat's heading and starter ideas, the sidebar's sign-up card, and the reply to a message held
 *  until sign up. Only the redesign renders them. The destinations' names (Chats, Agents, ...) are the
 *  rail's own labels (renamed-names.ts). */
export interface FirstTimeCopy {
  splash: {
    eyebrow: string
    title: string
    subtitle: string
    /** The rows' list, for assistive tech. */
    listLabel: string
    lines: Record<FirstTimeSplashPlace, { line: string; example: string }>
    /** A row's seen check, for assistive tech. */
    seen: string
    /** Under the rows: the way to the second step. */
    style: { title: string; line: string; action: string }
    /** The second step. */
    theme: {
      eyebrow: string
      title: string
      subtitle: string
      back: string
      mode: string
      light: string
      dark: string
      color: string
      colors: Record<PortalAccent, string>
      artwork: { title: string; selected: string; names: Record<Artwork, string> }
    }
    ready: string
    footnote: string
    signUp: string
    lookAround: string
    start: string
  }
  tray: {
    signUp: { title: string; description: string; action: string }
    meet: { title: string; description: string; action: string }
  }
  landing: {
    /** Over a newcomer's empty chat: Hermes' name where sign up is required (the hosted Portal), and
     *  the splash's own eyebrow elsewhere. */
    eyebrowPortal: string
    title: string
    subtitle: string
    /** In the composer while sign up is required. */
    signUp: string
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
  /** The card at the foot of the chat sidebar while sign up is required. */
  account: { title: string; body: string }
  held: {
    /** Who said what, for assistive tech. */
    you: string
    hermes: string
    /** When the held message was sent: it is always the moment before. */
    justNow: string
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
    eyebrow: 'Hermes Agent',
    title: 'Welcome to Hermes',
    subtitle: 'Give it a task. It learns how you work and keeps going while you’re away.',
    listLabel: 'What Hermes does',
    lines: {
      chats: {
        line: 'Don’t just ask, assign.',
        example: 'e.g. “research my 5 biggest competitors, compare their pricing, cite sources”'
      },
      agents: {
        line: 'A team of experts. Each one gets smarter over time.',
        example: 'e.g. a research agent that knows your industry'
      },
      routines: {
        line: 'Set it once. Hermes keeps checking.',
        example: 'e.g. a table for 2 opening up, a sold-out item back in stock'
      },
      library: {
        line: 'Everything your agents make, in one place.',
        example: 'e.g. every brief, report and doc it made, from every chat'
      }
    },
    seen: 'Seen',
    style: {
      title: 'Select your style',
      line: 'Theme and choose your Hermes artwork.',
      action: 'Make it yours'
    },
    theme: {
      eyebrow: 'Your Hermes',
      title: 'Make it yours',
      subtitle: 'Give it a task. It learns how you work and keeps going while you’re away.',
      back: 'Back',
      mode: 'Light or dark',
      light: 'Light',
      dark: 'Dark',
      color: 'Color',
      colors: {
        blue: 'Blue',
        violet: 'Violet',
        pink: 'Pink',
        red: 'Red',
        orange: 'Orange',
        yellow: 'Yellow',
        green: 'Green',
        cyan: 'Cyan'
      },
      artwork: {
        title: 'Make it beautiful',
        selected: 'Selected',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: 'Ready to get started?',
    footnote: 'Click anywhere in the nav to close. Find this again in notifications.',
    signUp: 'Sign up',
    lookAround: 'Look around',
    start: 'Let’s get to work'
  },
  tray: {
    signUp: {
      title: 'Ready to get started?',
      description: 'Your own Hermes, set up in about a minute.',
      action: 'Sign up to start'
    },
    meet: {
      title: 'Meet Hermes',
      description: 'See what your agent can do.',
      action: 'Open'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Welcome to Hermes',
    subtitle: 'Assign a task, or pick an idea below.',
    signUp: 'Sign up to start',
    ideasLabel: 'Ideas to start with',
    ideas: {
      morning_briefing: 'Send me a morning briefing',
      research_options: 'Research and compare options',
      summarize_document: 'Summarise a long document',
      clean_inbox: 'Clean up my inbox'
    },
    hide: 'Hide',
    hideLabel: 'Hide ideas',
    showIdeas: 'Show ideas',
    placeholderIdeas: 'Give Hermes a task',
    placeholderRetired: 'Give your agent a task'
  },
  account: {
    title: 'Create an Account',
    body: 'Access 200+ cutting-edge models and built-in tools.'
  },
  held: {
    you: 'You',
    hermes: 'Hermes',
    justNow: 'Just now',
    title: 'I can do that',
    body: 'Sign up and I’ll start on it right away. Your own agent will be ready in about a minute.',
    signUp: 'Sign up to start',
    failedTitle: 'Couldn’t sign you in.',
    failedBody: 'Your request is still saved.',
    tryAgain: 'Try again'
  }
}

const fr: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Bienvenue dans Hermes',
    subtitle: 'Confiez-lui une tâche. Il apprend votre façon de travailler et continue pendant votre absence.',
    listLabel: 'Ce que fait Hermes',
    lines: {
      chats: {
        line: 'Ne demandez pas, confiez.',
        example: 'ex. « étudie mes 5 plus gros concurrents, compare leurs prix, cite tes sources »'
      },
      agents: {
        line: 'Une équipe d’experts. Chacun s’améliore avec le temps.',
        example: 'ex. un agent recherche qui connaît votre secteur'
      },
      routines: {
        line: 'Réglez-la une fois. Hermes continue de vérifier.',
        example: 'ex. une table pour 2 qui se libère, un article épuisé de nouveau en stock'
      },
      library: {
        line: 'Tout ce que font vos agents, au même endroit.',
        example: 'ex. chaque note, rapport et document créé, de chaque discussion'
      }
    },
    seen: 'Vu',
    style: {
      title: 'Choisissez votre style',
      line: 'Thème et illustration de votre Hermes.',
      action: 'À votre image'
    },
    theme: {
      eyebrow: 'Votre Hermes',
      title: 'À votre image',
      subtitle: 'Confiez-lui une tâche. Il apprend votre façon de travailler et continue pendant votre absence.',
      back: 'Retour',
      mode: 'Clair ou sombre',
      light: 'Clair',
      dark: 'Sombre',
      color: 'Couleur',
      colors: {
        blue: 'Bleu',
        violet: 'Violet',
        pink: 'Rose',
        red: 'Rouge',
        orange: 'Orange',
        yellow: 'Jaune',
        green: 'Vert',
        cyan: 'Cyan'
      },
      artwork: {
        title: 'Rendez-le beau',
        selected: 'Choisi',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: 'Prêt à commencer ?',
    footnote: 'Cliquez n’importe où dans la navigation pour fermer. Retrouvez cette page dans les notifications.',
    signUp: 'S’inscrire',
    lookAround: 'Jeter un œil',
    start: 'Au travail'
  },
  tray: {
    signUp: {
      title: 'Prêt à commencer ?',
      description: 'Votre propre Hermes, prêt en une minute environ.',
      action: 'S’inscrire pour commencer'
    },
    meet: {
      title: 'Découvrir Hermes',
      description: 'Voyez ce que votre agent sait faire.',
      action: 'Ouvrir'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Bienvenue dans Hermes',
    subtitle: 'Confiez une tâche, ou choisissez une idée ci-dessous.',
    signUp: 'S’inscrire pour commencer',
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
    placeholderIdeas: 'Confiez une tâche à Hermes',
    placeholderRetired: 'Confiez une tâche à votre agent'
  },
  account: {
    title: 'Créer un compte',
    body: 'Plus de 200 modèles de pointe et des outils intégrés.'
  },
  held: {
    you: 'Vous',
    hermes: 'Hermes',
    justNow: 'À l’instant',
    title: 'Je peux m’en charger',
    body: 'Inscrivez-vous et je m’y mets tout de suite. Votre propre agent sera prêt en une minute environ.',
    signUp: 'S’inscrire pour commencer',
    failedTitle: 'Impossible de vous connecter.',
    failedBody: 'Votre demande est toujours enregistrée.',
    tryAgain: 'Réessayer'
  }
}

const de: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Willkommen bei Hermes',
    subtitle: 'Gib ihm eine Aufgabe. Er lernt, wie du arbeitest, und macht weiter, während du weg bist.',
    listLabel: 'Was Hermes kann',
    lines: {
      chats: {
        line: 'Nicht nur fragen, beauftragen.',
        example: 'z. B. „recherchiere meine 5 größten Wettbewerber, vergleiche ihre Preise, nenne Quellen“'
      },
      agents: {
        line: 'Ein Team aus Profis. Jeder wird mit der Zeit klüger.',
        example: 'z. B. ein Rechercheagent, der deine Branche kennt'
      },
      routines: {
        line: 'Einmal einrichten. Hermes prüft weiter.',
        example: 'z. B. ein Tisch für 2, der frei wird, ein ausverkaufter Artikel wieder auf Lager'
      },
      library: {
        line: 'Alles, was deine Agenten erstellen, an einem Ort.',
        example: 'z. B. jedes Briefing, jeder Bericht und jedes Dokument, aus jedem Chat'
      }
    },
    seen: 'Gesehen',
    style: {
      title: 'Wähle deinen Stil',
      line: 'Design und Hermes-Artwork auswählen.',
      action: 'Mach es zu deinem'
    },
    theme: {
      eyebrow: 'Dein Hermes',
      title: 'Mach es zu deinem',
      subtitle: 'Gib ihm eine Aufgabe. Er lernt, wie du arbeitest, und macht weiter, während du weg bist.',
      back: 'Zurück',
      mode: 'Hell oder dunkel',
      light: 'Hell',
      dark: 'Dunkel',
      color: 'Farbe',
      colors: {
        blue: 'Blau',
        violet: 'Violett',
        pink: 'Pink',
        red: 'Rot',
        orange: 'Orange',
        yellow: 'Gelb',
        green: 'Grün',
        cyan: 'Cyan'
      },
      artwork: {
        title: 'Mach es schön',
        selected: 'Ausgewählt',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: 'Bereit loszulegen?',
    footnote: 'Klicke irgendwo in die Navigation, um zu schließen. Du findest das wieder unter Benachrichtigungen.',
    signUp: 'Registrieren',
    lookAround: 'Umschauen',
    start: 'An die Arbeit'
  },
  tray: {
    signUp: {
      title: 'Bereit loszulegen?',
      description: 'Dein eigenes Hermes, in etwa einer Minute eingerichtet.',
      action: 'Registrieren und loslegen'
    },
    meet: {
      title: 'Hermes kennenlernen',
      description: 'Sieh dir an, was dein Agent kann.',
      action: 'Öffnen'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Willkommen bei Hermes',
    subtitle: 'Gib eine Aufgabe oder wähle unten eine Idee.',
    signUp: 'Registrieren und loslegen',
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
    placeholderIdeas: 'Gib Hermes eine Aufgabe',
    placeholderRetired: 'Gib deinem Agenten eine Aufgabe'
  },
  account: {
    title: 'Konto erstellen',
    body: 'Über 200 aktuelle Modelle und integrierte Werkzeuge.'
  },
  held: {
    you: 'Du',
    hermes: 'Hermes',
    justNow: 'Gerade eben',
    title: 'Das kann ich erledigen',
    body: 'Registriere dich und ich lege sofort los. Dein eigener Agent ist in etwa einer Minute bereit.',
    signUp: 'Registrieren und loslegen',
    failedTitle: 'Anmeldung fehlgeschlagen.',
    failedBody: 'Deine Anfrage ist weiterhin gespeichert.',
    tryAgain: 'Erneut versuchen'
  }
}

const es: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Te damos la bienvenida a Hermes',
    subtitle: 'Dale una tarea. Aprende cómo trabajas y sigue mientras no estás.',
    listLabel: 'Qué hace Hermes',
    lines: {
      chats: {
        line: 'No solo preguntes, encarga.',
        example: 'p. ej. «investiga a mis 5 mayores competidores, compara sus precios y cita las fuentes»'
      },
      agents: {
        line: 'Un equipo de expertos. Cada uno mejora con el tiempo.',
        example: 'p. ej. un agente de investigación que conoce tu sector'
      },
      routines: {
        line: 'Configúrala una vez. Hermes sigue comprobando.',
        example: 'p. ej. una mesa para 2 que se libera, un artículo agotado que vuelve a estar disponible'
      },
      library: {
        line: 'Todo lo que crean tus agentes, en un solo lugar.',
        example: 'p. ej. cada resumen, informe y documento que creó, de cada chat'
      }
    },
    seen: 'Visto',
    style: {
      title: 'Elige tu estilo',
      line: 'Tema e ilustración de tu Hermes.',
      action: 'Hazlo tuyo'
    },
    theme: {
      eyebrow: 'Tu Hermes',
      title: 'Hazlo tuyo',
      subtitle: 'Dale una tarea. Aprende cómo trabajas y sigue mientras no estás.',
      back: 'Atrás',
      mode: 'Claro u oscuro',
      light: 'Claro',
      dark: 'Oscuro',
      color: 'Color',
      colors: {
        blue: 'Azul',
        violet: 'Violeta',
        pink: 'Rosa',
        red: 'Rojo',
        orange: 'Naranja',
        yellow: 'Amarillo',
        green: 'Verde',
        cyan: 'Cian'
      },
      artwork: {
        title: 'Hazlo bonito',
        selected: 'Elegida',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: '¿Listo para empezar?',
    footnote: 'Haz clic en cualquier parte de la navegación para cerrar. Encuéntralo de nuevo en las notificaciones.',
    signUp: 'Regístrate',
    lookAround: 'Echar un vistazo',
    start: 'Manos a la obra'
  },
  tray: {
    signUp: {
      title: '¿Listo para empezar?',
      description: 'Tu propio Hermes, listo en un minuto aproximadamente.',
      action: 'Regístrate para empezar'
    },
    meet: {
      title: 'Conoce Hermes',
      description: 'Mira lo que puede hacer tu agente.',
      action: 'Abrir'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Te damos la bienvenida a Hermes',
    subtitle: 'Encarga una tarea o elige una idea de abajo.',
    signUp: 'Regístrate para empezar',
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
    placeholderIdeas: 'Dale una tarea a Hermes',
    placeholderRetired: 'Dale una tarea a tu agente'
  },
  account: {
    title: 'Crear una cuenta',
    body: 'Más de 200 modelos punteros y herramientas integradas.'
  },
  held: {
    you: 'Tú',
    hermes: 'Hermes',
    justNow: 'Justo ahora',
    title: 'Puedo hacerlo',
    body: 'Regístrate y empiezo ahora mismo. Tu propio agente estará listo en un minuto aproximadamente.',
    signUp: 'Regístrate para empezar',
    failedTitle: 'No pudimos iniciar tu sesión.',
    failedBody: 'Tu petición sigue guardada.',
    tryAgain: 'Reintentar'
  }
}

const ja: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Hermes へようこそ',
    subtitle: 'タスクを任せてください。あなたの働き方を学び、留守の間も進め続けます。',
    listLabel: 'Hermes にできること',
    lines: {
      chats: {
        line: '質問するだけでなく、任せる。',
        example: '例：「大手競合 5 社を調べて、価格を比較し、出典を示して」'
      },
      agents: {
        line: '専門家のチーム。それぞれが使うほど賢くなります。',
        example: '例：業界を知っているリサーチエージェント'
      },
      routines: {
        line: '一度設定すれば、Hermes が確認し続けます。',
        example: '例：2 名の席が空いたとき、売り切れ商品が再入荷したとき'
      },
      library: {
        line: 'エージェントが作ったものを、すべて一か所に。',
        example: '例：すべてのチャットで作った要約、レポート、ドキュメント'
      }
    },
    seen: '確認済み',
    style: {
      title: 'スタイルを選ぶ',
      line: 'テーマと Hermes のアートワークを選びます。',
      action: '自分好みに'
    },
    theme: {
      eyebrow: 'あなたの Hermes',
      title: '自分好みに',
      subtitle: 'タスクを任せてください。あなたの働き方を学び、留守の間も進め続けます。',
      back: '戻る',
      mode: 'ライトまたはダーク',
      light: 'ライト',
      dark: 'ダーク',
      color: 'カラー',
      colors: {
        blue: 'ブルー',
        violet: 'バイオレット',
        pink: 'ピンク',
        red: 'レッド',
        orange: 'オレンジ',
        yellow: 'イエロー',
        green: 'グリーン',
        cyan: 'シアン'
      },
      artwork: {
        title: '美しく彩る',
        selected: '選択中',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: 'さっそく始めますか？',
    footnote: 'ナビゲーションのどこかをクリックすると閉じます。通知からもう一度開けます。',
    signUp: '登録',
    lookAround: '見てまわる',
    start: 'さっそく始める'
  },
  tray: {
    signUp: {
      title: 'さっそく始めますか？',
      description: 'あなた専用の Hermes を、1 分ほどで準備します。',
      action: '登録して始める'
    },
    meet: {
      title: 'Hermes について',
      description: 'エージェントにできることを見てみましょう。',
      action: '開く'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Hermes へようこそ',
    subtitle: 'タスクを任せるか、下のアイデアを選んでください。',
    signUp: '登録して始める',
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
    placeholderIdeas: 'Hermes にタスクを任せる',
    placeholderRetired: 'エージェントにタスクを任せる'
  },
  account: {
    title: 'アカウントを作成',
    body: '200 以上の最先端モデルと組み込みツールを利用できます。'
  },
  held: {
    you: 'あなた',
    hermes: 'Hermes',
    justNow: 'たった今',
    title: 'お任せください',
    body: '登録すれば、すぐに取りかかります。あなた専用のエージェントは 1 分ほどで準備できます。',
    signUp: '登録して始める',
    failedTitle: 'サインインできませんでした。',
    failedBody: 'リクエストは保存されたままです。',
    tryAgain: 'もう一度試す'
  }
}

const zh: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: '欢迎使用 Hermes',
    subtitle: '交给它一个任务。它会了解你的工作方式，并在你离开时继续推进。',
    listLabel: 'Hermes 能做什么',
    lines: {
      chats: {
        line: '不只是提问，而是委派。',
        example: '例如：“调研我的 5 个最大竞争对手，比较他们的价格，并注明来源”'
      },
      agents: {
        line: '一支专家团队，每位都会越用越聪明。',
        example: '例如：了解你所在行业的研究智能体'
      },
      routines: {
        line: '设置一次，Hermes 会持续检查。',
        example: '例如：有 2 人桌空出来时、售罄商品重新到货时'
      },
      library: {
        line: '智能体做出的一切，集中在一处。',
        example: '例如：来自每个聊天的每份简报、报告和文档'
      }
    },
    seen: '已查看',
    style: {
      title: '选择你的风格',
      line: '选择主题和你的 Hermes 插画。',
      action: '打造你的风格'
    },
    theme: {
      eyebrow: '你的 Hermes',
      title: '打造你的风格',
      subtitle: '交给它一个任务。它会了解你的工作方式，并在你离开时继续推进。',
      back: '返回',
      mode: '浅色或深色',
      light: '浅色',
      dark: '深色',
      color: '颜色',
      colors: {
        blue: '蓝色',
        violet: '紫色',
        pink: '粉色',
        red: '红色',
        orange: '橙色',
        yellow: '黄色',
        green: '绿色',
        cyan: '青色'
      },
      artwork: {
        title: '让它更美',
        selected: '已选',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: '准备好开始了吗？',
    footnote: '点击导航中的任意位置即可关闭。可在通知中再次找到此页。',
    signUp: '注册',
    lookAround: '先看看',
    start: '开始工作'
  },
  tray: {
    signUp: {
      title: '准备好开始了吗？',
      description: '你自己的 Hermes，大约一分钟即可设置好。',
      action: '注册并开始'
    },
    meet: {
      title: '认识 Hermes',
      description: '看看你的智能体能做什么。',
      action: '打开'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: '欢迎使用 Hermes',
    subtitle: '布置一个任务，或从下方选一个灵感。',
    signUp: '注册并开始',
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
    placeholderIdeas: '给 Hermes 布置任务',
    placeholderRetired: '给你的智能体布置任务'
  },
  account: {
    title: '创建账户',
    body: '使用 200 多个前沿模型和内置工具。'
  },
  held: {
    you: '你',
    hermes: 'Hermes',
    justNow: '刚刚',
    title: '这个我能做',
    body: '注册后我会立刻开始。你自己的智能体大约一分钟就能准备好。',
    signUp: '注册并开始',
    failedTitle: '无法为你登录。',
    failedBody: '你的请求仍已保存。',
    tryAgain: '重试'
  }
}

const zhHant: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: '歡迎使用 Hermes',
    subtitle: '交給它一項任務。它會了解你的工作方式，並在你離開時繼續進行。',
    listLabel: 'Hermes 能做什麼',
    lines: {
      chats: {
        line: '不只是提問，而是交辦。',
        example: '例如：「研究我的 5 個最大競爭對手，比較他們的價格，並註明來源」'
      },
      agents: {
        line: '一支專家團隊，每位都會越用越聰明。',
        example: '例如：了解你所在產業的研究代理'
      },
      routines: {
        line: '設定一次，Hermes 會持續檢查。',
        example: '例如：有 2 人桌空出來時、售完的商品重新到貨時'
      },
      library: {
        line: '代理做出的一切，集中在一處。',
        example: '例如：來自每個聊天的每份簡報、報告和文件'
      }
    },
    seen: '已查看',
    style: {
      title: '選擇你的風格',
      line: '選擇主題和你的 Hermes 插畫。',
      action: '打造你的風格'
    },
    theme: {
      eyebrow: '你的 Hermes',
      title: '打造你的風格',
      subtitle: '交給它一項任務。它會了解你的工作方式，並在你離開時繼續進行。',
      back: '返回',
      mode: '淺色或深色',
      light: '淺色',
      dark: '深色',
      color: '顏色',
      colors: {
        blue: '藍色',
        violet: '紫色',
        pink: '粉紅色',
        red: '紅色',
        orange: '橘色',
        yellow: '黃色',
        green: '綠色',
        cyan: '青色'
      },
      artwork: {
        title: '讓它更美',
        selected: '已選',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: '準備好開始了嗎？',
    footnote: '點按導覽中的任何位置即可關閉。可在通知中再次找到此頁。',
    signUp: '註冊',
    lookAround: '先看看',
    start: '開始工作'
  },
  tray: {
    signUp: {
      title: '準備好開始了嗎？',
      description: '你自己的 Hermes，大約一分鐘即可設定好。',
      action: '註冊並開始'
    },
    meet: {
      title: '認識 Hermes',
      description: '看看你的代理能做什麼。',
      action: '開啟'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: '歡迎使用 Hermes',
    subtitle: '交辦一項任務，或從下方選一個靈感。',
    signUp: '註冊並開始',
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
    placeholderIdeas: '交辦任務給 Hermes',
    placeholderRetired: '交辦任務給你的代理'
  },
  account: {
    title: '建立帳戶',
    body: '使用 200 多個前沿模型和內建工具。'
  },
  held: {
    you: '你',
    hermes: 'Hermes',
    justNow: '剛剛',
    title: '這個我能做',
    body: '註冊後我會馬上開始。你自己的代理大約一分鐘就能準備好。',
    signUp: '註冊並開始',
    failedTitle: '無法為你登入。',
    failedBody: '你的請求仍然保存著。',
    tryAgain: '再試一次'
  }
}

const ru: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Добро пожаловать в Hermes',
    subtitle: 'Дайте задачу. Hermes изучит, как вы работаете, и продолжит, пока вас нет.',
    listLabel: 'Что умеет Hermes',
    lines: {
      chats: {
        line: 'Не просто спрашивайте, поручайте.',
        example: 'напр. «изучи 5 моих главных конкурентов, сравни их цены, укажи источники»'
      },
      agents: {
        line: 'Команда экспертов. Каждый со временем становится умнее.',
        example: 'напр. исследовательский агент, знающий вашу отрасль'
      },
      routines: {
        line: 'Настройте один раз. Hermes продолжит проверять.',
        example: 'напр. освободился столик на двоих, распроданный товар снова в наличии'
      },
      library: {
        line: 'Всё, что создают ваши агенты, в одном месте.',
        example: 'напр. каждая сводка, отчёт и документ из каждого чата'
      }
    },
    seen: 'Просмотрено',
    style: {
      title: 'Выберите стиль',
      line: 'Тема и иллюстрация вашего Hermes.',
      action: 'Под себя'
    },
    theme: {
      eyebrow: 'Ваш Hermes',
      title: 'Под себя',
      subtitle: 'Дайте задачу. Hermes изучит, как вы работаете, и продолжит, пока вас нет.',
      back: 'Назад',
      mode: 'Светлая или тёмная',
      light: 'Светлая',
      dark: 'Тёмная',
      color: 'Цвет',
      colors: {
        blue: 'Синий',
        violet: 'Фиолетовый',
        pink: 'Розовый',
        red: 'Красный',
        orange: 'Оранжевый',
        yellow: 'Жёлтый',
        green: 'Зелёный',
        cyan: 'Голубой'
      },
      artwork: {
        title: 'Сделайте красиво',
        selected: 'Выбрано',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: 'Готовы начать?',
    footnote: 'Нажмите в любом месте навигации, чтобы закрыть. Эту страницу можно снова найти в уведомлениях.',
    signUp: 'Зарегистрироваться',
    lookAround: 'Осмотреться',
    start: 'За работу'
  },
  tray: {
    signUp: {
      title: 'Готовы начать?',
      description: 'Ваш собственный Hermes, готов примерно за минуту.',
      action: 'Зарегистрироваться'
    },
    meet: {
      title: 'Знакомство с Hermes',
      description: 'Посмотрите, что умеет ваш агент.',
      action: 'Открыть'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Добро пожаловать в Hermes',
    subtitle: 'Поручите задачу или выберите идею ниже.',
    signUp: 'Зарегистрироваться',
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
    placeholderIdeas: 'Дайте Hermes задачу',
    placeholderRetired: 'Дайте агенту задачу'
  },
  account: {
    title: 'Создать аккаунт',
    body: 'Более 200 передовых моделей и встроенные инструменты.'
  },
  held: {
    you: 'Вы',
    hermes: 'Hermes',
    justNow: 'Только что',
    title: 'Я могу это сделать',
    body: 'Зарегистрируйтесь, и я сразу начну. Ваш собственный агент будет готов примерно за минуту.',
    signUp: 'Зарегистрироваться',
    failedTitle: 'Не удалось войти.',
    failedBody: 'Ваш запрос по-прежнему сохранён.',
    tryAgain: 'Попробовать снова'
  }
}

const ar: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'مرحبًا بك في Hermes',
    subtitle: 'أعطه مهمة. يتعلّم طريقة عملك ويواصل أثناء غيابك.',
    listLabel: 'ما يفعله Hermes',
    lines: {
      chats: {
        line: 'لا تكتفِ بالسؤال، بل كلّف.',
        example: 'مثلًا: «ابحث عن أكبر 5 منافسين لي، وقارن أسعارهم، واذكر المصادر»'
      },
      agents: {
        line: 'فريق من الخبراء. يزداد كل منهم ذكاءً مع الوقت.',
        example: 'مثلًا: وكيل بحث يعرف مجال عملك'
      },
      routines: {
        line: 'اضبطه مرة واحدة. يواصل Hermes التحقق.',
        example: 'مثلًا: طاولة لشخصين أصبحت متاحة، منتج نفد عاد إلى المخزون'
      },
      library: {
        line: 'كل ما يصنعه وكلاؤك، في مكان واحد.',
        example: 'مثلًا: كل ملخص وتقرير ومستند صنعه، من كل محادثة'
      }
    },
    seen: 'تمت المشاهدة',
    style: {
      title: 'اختر أسلوبك',
      line: 'السمة والعمل الفني لـ Hermes الخاص بك.',
      action: 'اجعله على ذوقك'
    },
    theme: {
      eyebrow: 'Hermes الخاص بك',
      title: 'اجعله على ذوقك',
      subtitle: 'أعطه مهمة. يتعلّم طريقة عملك ويواصل أثناء غيابك.',
      back: 'رجوع',
      mode: 'فاتح أو داكن',
      light: 'فاتح',
      dark: 'داكن',
      color: 'اللون',
      colors: {
        blue: 'أزرق',
        violet: 'بنفسجي',
        pink: 'وردي',
        red: 'أحمر',
        orange: 'برتقالي',
        yellow: 'أصفر',
        green: 'أخضر',
        cyan: 'سماوي'
      },
      artwork: {
        title: 'اجعله جميلًا',
        selected: 'محدد',
        names: {
          wing: 'Wing',
          angel: 'Angel',
          ascend: 'Ascend',
          karborn: 'Karborn',
          vanilla: 'Vanilla'
        }
      }
    },
    ready: 'هل أنت مستعد للبدء؟',
    footnote: 'انقر في أي مكان في شريط التنقل للإغلاق. ستجد هذه الصفحة مجددًا في الإشعارات.',
    signUp: 'سجّل',
    lookAround: 'ألقِ نظرة',
    start: 'لنبدأ العمل'
  },
  tray: {
    signUp: {
      title: 'هل أنت مستعد للبدء؟',
      description: 'نسختك الخاصة من Hermes، جاهزة في نحو دقيقة.',
      action: 'سجّل للبدء'
    },
    meet: {
      title: 'تعرّف على Hermes',
      description: 'اكتشف ما يستطيع وكيلك فعله.',
      action: 'فتح'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'مرحبًا بك في Hermes',
    subtitle: 'كلّف بمهمة، أو اختر فكرة أدناه.',
    signUp: 'سجّل للبدء',
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
    placeholderIdeas: 'أعطِ Hermes مهمة',
    placeholderRetired: 'أعطِ وكيلك مهمة'
  },
  account: {
    title: 'أنشئ حسابًا',
    body: 'أكثر من 200 نموذج متطور وأدوات مدمجة.'
  },
  held: {
    you: 'أنت',
    hermes: 'Hermes',
    justNow: 'الآن',
    title: 'يمكنني فعل ذلك',
    body: 'سجّل وسأبدأ عليه فورًا. سيكون وكيلك الخاص جاهزًا في نحو دقيقة.',
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
