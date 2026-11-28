import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'
import type { FirstTimeSplashPlace } from '@/store/first-time-splash'
import type { PortalAccent } from '@/store/portal-accent'

import type { Artwork } from '../redesign/artwork'

/** The redesign's first-run surfaces for someone new to Hermes (Figma "Prod - Desktop" › Onboarding,
 *  01 Discover & sign in): the first-time splash and its "Make it yours" step, its tray entries, the
 *  empty chat's heading, the sidebar's sign-up card, and the reply to a message held
 *  until sign up. Only the redesign renders them. The destinations' names (Chats, Agents, ...) are the
 *  rail's own labels (renamed-names.ts). */
export interface FirstTimeCopy {
  splash: {
    eyebrow: string
    title: string
    subtitle: string
    /** The rows' list, for assistive tech. */
    listLabel: string
    /** Connections is the redesign's word for the rail's Capabilities destination; the other three
     *  places are named by the rail itself (renamed-names.ts). */
    placeNames: { connections: string }
    lines: Record<FirstTimeSplashPlace, { line: string; example: string }>
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
    checkIn: { title: string; description: string; action: string }
  }
  landing: {
    /** Over a newcomer's empty chat: Hermes' name where sign up is required (the hosted Portal), and
     *  the splash's own eyebrow elsewhere. */
    eyebrowPortal: string
    title: string
    /** In the composer while sign up is required. */
    signUp: string
    /** The composer's resting text before a first task, and after. */
    placeholderIdeas: string
    placeholderRetired: string
    /** After a first finished task (C4, C5): the heading; the line every newcomer's heading carries. */
    back: string
    subtitleRetired: string
    /** Back after a long stay away (C8). */
    awayTitle: string
    awayLine: string
  }
  /** "How's Hermes going so far?" (Onboarding C7 18677:28705) on the new chat's landing. */
  checkIn: {
    eyebrow: string
    title: string
    subtitle: string
    /** The suggestions' list, for assistive tech. */
    listLabel: string
    video: string
    videoLine: string
    weekly: string
    weeklyLine: string
    tour: string
    tourLine: string
    try: string
    open: string
    goingWell: string
    couldBeBetter: string
    dismiss: string
  }
  /** "Want the developer view?" (Onboarding C6 18677:28595), offered in a chat about code. */
  developerView: {
    eyebrow: string
    title: string
    body: string
    preview: string
    previewLine: string
    turnOn: string
    notNow: string
    settingsNote: string
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
  }
  /** "Let's try signing in again" (Onboarding 18732:33066, 18695:19224): sign up came back without
   *  signing the visitor in. */
  recovery: {
    eyebrow: string
    title: string
    /** With a message held, and without one. */
    subtitleHeld: string
    subtitle: string
    /** The rows' list, for assistive tech. */
    listLabel: string
    account: string
    accountLine: string
    unverified: string
    request: string
    saved: string
    opening: string
    openingLine: string
    waiting: string
    tryAgain: string
    otherEmail: string
    expired: string
    requestNew: string
  }
}

const en: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Welcome to Hermes',
    subtitle: 'Give it a task. It learns how you work and keeps going while you’re away.',
    listLabel: 'What Hermes does',
    placeNames: { connections: 'Connections' },
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
      connections: {
        line: 'Connect your everyday apps.',
        example: 'e.g. “check my gmail, add the meeting to my calendar, post the notes in slack and create tasks in Linear”'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'How’s Hermes going so far?',
      description: 'Keep going with what works, or try something new.',
      action: 'Open'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Welcome to Hermes',
    signUp: 'Sign up to start',
    placeholderIdeas: 'Give Hermes a task',
    placeholderRetired: 'Give your agent a task',
    back: 'Welcome back.',
    subtitleRetired: 'What would you like your agent to work on?',
    awayTitle: 'Good to see you again.',
    awayLine: 'Your chats and Library are here whenever you need them.',
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
    signUp: 'Sign up to start'
  },
  recovery: {
    eyebrow: 'Sign in / Recovery',
    title: 'Let’s try signing in again',
    subtitleHeld: 'The sign-in window closed before we could finish. Your request is still saved.',
    subtitle: 'The sign-in window closed before we could finish.',
    listLabel: 'Where signing in stands',
    account: 'Your account',
    accountLine: 'One account for Hermes and Nous',
    unverified: 'Unverified',
    request: 'Your request',
    saved: 'Saved',
    opening: 'Opening Hermes',
    openingLine: 'This should only take a moment',
    waiting: 'Waiting',
    tryAgain: 'Try again',
    otherEmail: 'Use another email',
    expired: 'Email link expired?',
    requestNew: 'Request a new one.'
  },
  checkIn: {
    eyebrow: 'Return visit / Check in',
    title: 'How’s Hermes going so far?',
    subtitle: 'Keep going with what works, or try something new.',
    listLabel: 'Things to try',
    video: 'Turn a video into a page',
    videoLine: 'A useful way to save what you learn',
    weekly: 'Keep me updated weekly',
    weeklyLine: 'Let a routine follow a topic for you',
    tour: 'Show me around again',
    tourLine: 'Reopen the product introduction',
    try: 'Try',
    open: 'Open',
    goingWell: 'Going well',
    couldBeBetter: 'Could be better',
    dismiss: 'Dismiss'
  },
  developerView: {
    eyebrow: 'Contextual offer / Code detected',
    title: 'Want the developer view?',
    body: 'Watch your agent work: the terminal, files and diffs it uses, beside the conversation.',
    preview: 'See the work as it happens',
    previewLine: 'Terminal, files and diffs alongside the conversation',
    turnOn: 'Turn on developer view',
    notNow: 'Not now',
    settingsNote: 'Change this anytime in Settings.'
  }
}

const fr: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Bienvenue dans Hermes',
    subtitle: 'Confiez-lui une tâche. Il apprend votre façon de travailler et continue pendant votre absence.',
    listLabel: 'Ce que fait Hermes',
    placeNames: { connections: 'Connexions' },
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
      connections: {
        line: 'Connectez vos applications du quotidien.',
        example: 'ex. « consulte mes mails, ajoute la réunion à mon agenda, publie les notes sur Slack et crée les tâches dans Linear »'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'Comment se passe Hermes jusqu’ici ?',
      description: 'Continuez ce qui marche, ou essayez autre chose.',
      action: 'Ouvrir'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Bienvenue dans Hermes',
    signUp: 'S’inscrire pour commencer',
    placeholderIdeas: 'Confiez une tâche à Hermes',
    placeholderRetired: 'Confiez une tâche à votre agent',
    back: 'Bon retour.',
    subtitleRetired: 'Sur quoi votre agent doit-il travailler ?',
    awayTitle: 'Ravi de vous revoir.',
    awayLine: 'Vos discussions et votre Bibliothèque sont là quand vous en avez besoin.',
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
    signUp: 'S’inscrire pour commencer'
  },
  recovery: {
    eyebrow: 'Connexion / Reprise',
    title: 'Réessayons de vous connecter',
    subtitleHeld: 'La fenêtre de connexion s’est fermée avant la fin. Votre demande est toujours enregistrée.',
    subtitle: 'La fenêtre de connexion s’est fermée avant la fin.',
    listLabel: 'Où en est la connexion',
    account: 'Votre compte',
    accountLine: 'Un seul compte pour Hermes et Nous',
    unverified: 'Non vérifié',
    request: 'Votre demande',
    saved: 'Enregistrée',
    opening: 'Ouverture de Hermes',
    openingLine: 'Cela ne devrait prendre qu’un instant',
    waiting: 'En attente',
    tryAgain: 'Réessayer',
    otherEmail: 'Utiliser une autre adresse',
    expired: 'Lien e-mail expiré ?',
    requestNew: 'En demander un nouveau.'
  },
  checkIn: {
    eyebrow: 'Retour / Point d’étape',
    title: 'Comment se passe Hermes jusqu’ici ?',
    subtitle: 'Continuez ce qui marche, ou essayez autre chose.',
    listLabel: 'Idées à essayer',
    video: 'Transformer une vidéo en page',
    videoLine: 'Une façon utile de garder ce que vous apprenez',
    weekly: 'Me tenir au courant chaque semaine',
    weeklyLine: 'Une routine suit un sujet pour vous',
    tour: 'Refaire la visite',
    tourLine: 'Rouvrir la présentation du produit',
    try: 'Essayer',
    open: 'Ouvrir',
    goingWell: 'Ça se passe bien',
    couldBeBetter: 'Ça pourrait être mieux',
    dismiss: 'Fermer'
  },
  developerView: {
    eyebrow: 'Suggestion / Code détecté',
    title: 'Voir la vue développeur ?',
    body: 'Regardez votre agent travailler : le terminal, les fichiers et les diffs qu’il utilise, à côté de la discussion.',
    preview: 'Voir le travail en direct',
    previewLine: 'Terminal, fichiers et diffs à côté de la discussion',
    turnOn: 'Activer la vue développeur',
    notNow: 'Pas maintenant',
    settingsNote: 'Modifiable à tout moment dans les Réglages.'
  }
}

const de: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Willkommen bei Hermes',
    subtitle: 'Gib ihm eine Aufgabe. Er lernt, wie du arbeitest, und macht weiter, während du weg bist.',
    listLabel: 'Was Hermes kann',
    placeNames: { connections: 'Verbindungen' },
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
      connections: {
        line: 'Verbinde deine Alltags-Apps.',
        example: 'z. B. „prüf meine Gmail, trag den Termin in meinen Kalender ein, poste die Notizen in Slack und leg Aufgaben in Linear an“'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'Wie läuft es bisher mit Hermes?',
      description: 'Mach weiter mit dem, was funktioniert, oder probier etwas Neues.',
      action: 'Öffnen'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Willkommen bei Hermes',
    signUp: 'Registrieren und loslegen',
    placeholderIdeas: 'Gib Hermes eine Aufgabe',
    placeholderRetired: 'Gib deinem Agenten eine Aufgabe',
    back: 'Willkommen zurück.',
    subtitleRetired: 'Woran soll dein Agent arbeiten?',
    awayTitle: 'Schön, dich wiederzusehen.',
    awayLine: 'Deine Chats und deine Bibliothek sind da, wann immer du sie brauchst.',
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
    signUp: 'Registrieren und loslegen'
  },
  recovery: {
    eyebrow: 'Anmelden / Wiederherstellen',
    title: 'Versuchen wir die Anmeldung noch einmal',
    subtitleHeld:
      'Das Anmeldefenster wurde geschlossen, bevor wir fertig waren. Deine Anfrage ist weiterhin gespeichert.',
    subtitle: 'Das Anmeldefenster wurde geschlossen, bevor wir fertig waren.',
    listLabel: 'Stand der Anmeldung',
    account: 'Dein Konto',
    accountLine: 'Ein Konto für Hermes und Nous',
    unverified: 'Nicht bestätigt',
    request: 'Deine Anfrage',
    saved: 'Gespeichert',
    opening: 'Hermes wird geöffnet',
    openingLine: 'Das dauert nur einen Moment',
    waiting: 'Wartet',
    tryAgain: 'Erneut versuchen',
    otherEmail: 'Andere E-Mail verwenden',
    expired: 'E-Mail-Link abgelaufen?',
    requestNew: 'Neuen anfordern.'
  },
  checkIn: {
    eyebrow: 'Wiederbesuch / Rückfrage',
    title: 'Wie läuft es bisher mit Hermes?',
    subtitle: 'Mach weiter mit dem, was funktioniert, oder probier etwas Neues.',
    listLabel: 'Zum Ausprobieren',
    video: 'Ein Video in eine Seite verwandeln',
    videoLine: 'Eine nützliche Art, Gelerntes festzuhalten',
    weekly: 'Halte mich wöchentlich auf dem Laufenden',
    weeklyLine: 'Eine Routine verfolgt ein Thema für dich',
    tour: 'Zeig mir alles noch einmal',
    tourLine: 'Die Produkteinführung erneut öffnen',
    try: 'Ausprobieren',
    open: 'Öffnen',
    goingWell: 'Läuft gut',
    couldBeBetter: 'Könnte besser sein',
    dismiss: 'Schließen'
  },
  developerView: {
    eyebrow: 'Vorschlag / Code erkannt',
    title: 'Entwickleransicht gefällig?',
    body: 'Sieh deinem Agenten bei der Arbeit zu: Terminal, Dateien und Diffs neben der Unterhaltung.',
    preview: 'Die Arbeit live sehen',
    previewLine: 'Terminal, Dateien und Diffs neben der Unterhaltung',
    turnOn: 'Entwickleransicht einschalten',
    notNow: 'Nicht jetzt',
    settingsNote: 'Jederzeit in den Einstellungen änderbar.'
  }
}

const es: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Te damos la bienvenida a Hermes',
    subtitle: 'Dale una tarea. Aprende cómo trabajas y sigue mientras no estás.',
    listLabel: 'Qué hace Hermes',
    placeNames: { connections: 'Conexiones' },
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
      connections: {
        line: 'Conecta tus aplicaciones del día a día.',
        example: 'p. ej. «revisa mi Gmail, añade la reunión a mi calendario, publica las notas en Slack y crea tareas en Linear»'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: '¿Qué tal te va con Hermes?',
      description: 'Sigue con lo que funciona o prueba algo nuevo.',
      action: 'Abrir'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Te damos la bienvenida a Hermes',
    signUp: 'Regístrate para empezar',
    placeholderIdeas: 'Dale una tarea a Hermes',
    placeholderRetired: 'Dale una tarea a tu agente',
    back: 'Hola de nuevo.',
    subtitleRetired: '¿En qué quieres que trabaje tu agente?',
    awayTitle: 'Qué bueno verte de nuevo.',
    awayLine: 'Tus chats y tu Biblioteca están aquí cuando los necesites.',
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
    signUp: 'Regístrate para empezar'
  },
  recovery: {
    eyebrow: 'Iniciar sesión / Recuperación',
    title: 'Intentemos iniciar sesión de nuevo',
    subtitleHeld: 'La ventana de inicio de sesión se cerró antes de terminar. Tu petición sigue guardada.',
    subtitle: 'La ventana de inicio de sesión se cerró antes de terminar.',
    listLabel: 'Estado del inicio de sesión',
    account: 'Tu cuenta',
    accountLine: 'Una cuenta para Hermes y Nous',
    unverified: 'Sin verificar',
    request: 'Tu petición',
    saved: 'Guardada',
    opening: 'Abriendo Hermes',
    openingLine: 'Solo tardará un momento',
    waiting: 'En espera',
    tryAgain: 'Reintentar',
    otherEmail: 'Usar otro correo',
    expired: '¿Caducó el enlace del correo?',
    requestNew: 'Pide uno nuevo.'
  },
  checkIn: {
    eyebrow: 'Regreso / Consulta',
    title: '¿Qué tal te va con Hermes?',
    subtitle: 'Sigue con lo que funciona o prueba algo nuevo.',
    listLabel: 'Cosas para probar',
    video: 'Convertir un vídeo en una página',
    videoLine: 'Una forma útil de guardar lo que aprendes',
    weekly: 'Mantenme al día cada semana',
    weeklyLine: 'Una rutina sigue un tema por ti',
    tour: 'Enséñame todo otra vez',
    tourLine: 'Volver a abrir la presentación',
    try: 'Probar',
    open: 'Abrir',
    goingWell: 'Va bien',
    couldBeBetter: 'Podría ir mejor',
    dismiss: 'Descartar'
  },
  developerView: {
    eyebrow: 'Sugerencia / Código detectado',
    title: '¿Quieres la vista de desarrollador?',
    body: 'Mira trabajar a tu agente: la terminal, los archivos y los diffs que usa, junto a la conversación.',
    preview: 'Ver el trabajo en directo',
    previewLine: 'Terminal, archivos y diffs junto a la conversación',
    turnOn: 'Activar la vista de desarrollador',
    notNow: 'Ahora no',
    settingsNote: 'Puedes cambiarlo cuando quieras en Ajustes.'
  }
}

const ja: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Hermes へようこそ',
    subtitle: 'タスクを任せてください。あなたの働き方を学び、留守の間も進め続けます。',
    listLabel: 'Hermes にできること',
    placeNames: { connections: '接続' },
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
      connections: {
        line: 'いつものアプリをつなぎます。',
        example: '例：「Gmail を確認して、予定をカレンダーに入れて、メモを Slack に投稿して、Linear にタスクを作って」'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'Hermes の使い心地はいかがですか？',
      description: 'うまくいっていることを続けるか、新しいことを試しましょう。',
      action: '開く'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Hermes へようこそ',
    signUp: '登録して始める',
    placeholderIdeas: 'Hermes にタスクを任せる',
    placeholderRetired: 'エージェントにタスクを任せる',
    back: 'おかえりなさい。',
    subtitleRetired: 'エージェントに何を任せますか？',
    awayTitle: 'またお会いできてうれしいです。',
    awayLine: 'チャットとライブラリはいつでもここにあります。',
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
    signUp: '登録して始める'
  },
  recovery: {
    eyebrow: 'サインイン / 復旧',
    title: 'もう一度サインインしましょう',
    subtitleHeld: '完了する前にサインイン画面が閉じました。リクエストは保存されたままです。',
    subtitle: '完了する前にサインイン画面が閉じました。',
    listLabel: 'サインインの状況',
    account: 'アカウント',
    accountLine: 'Hermes と Nous で共通のアカウント',
    unverified: '未確認',
    request: 'リクエスト',
    saved: '保存済み',
    opening: 'Hermes を開いています',
    openingLine: 'すぐに終わります',
    waiting: '待機中',
    tryAgain: 'もう一度試す',
    otherEmail: '別のメールアドレスを使う',
    expired: 'メールのリンクの期限が切れましたか？',
    requestNew: '新しいリンクを請求する。'
  },
  checkIn: {
    eyebrow: '再訪 / チェックイン',
    title: 'Hermes の使い心地はいかがですか？',
    subtitle: 'うまくいっていることを続けるか、新しいことを試しましょう。',
    listLabel: '試してみること',
    video: '動画をページにまとめる',
    videoLine: '学んだことを残す便利な方法',
    weekly: '毎週最新情報を届けて',
    weeklyLine: 'ルーティンがテーマを追いかけます',
    tour: 'もう一度案内して',
    tourLine: '製品紹介をもう一度開く',
    try: '試す',
    open: '開く',
    goingWell: '順調です',
    couldBeBetter: '改善の余地あり',
    dismiss: '閉じる'
  },
  developerView: {
    eyebrow: '提案 / コードを検出',
    title: '開発者ビューを使いますか？',
    body: 'エージェントの作業を見られます：使っているターミナル、ファイル、差分を会話の横に。',
    preview: '作業の様子をそのまま見る',
    previewLine: '会話の横にターミナル、ファイル、差分',
    turnOn: '開発者ビューをオンにする',
    notNow: '今はしない',
    settingsNote: '設定からいつでも変更できます。'
  }
}

const zh: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: '欢迎使用 Hermes',
    subtitle: '交给它一个任务。它会了解你的工作方式，并在你离开时继续推进。',
    listLabel: 'Hermes 能做什么',
    placeNames: { connections: '连接' },
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
      connections: {
        line: '连接你常用的应用。',
        example: '例如：“查看我的 Gmail，把会议加到我的日历，把笔记发到 Slack，再在 Linear 里创建任务”'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'Hermes 用得怎么样？',
      description: '继续用顺手的，或者试试新东西。',
      action: '打开'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: '欢迎使用 Hermes',
    signUp: '注册并开始',
    placeholderIdeas: '给 Hermes 布置任务',
    placeholderRetired: '给你的智能体布置任务',
    back: '欢迎回来。',
    subtitleRetired: '你想让智能体做些什么？',
    awayTitle: '很高兴再次见到你。',
    awayLine: '你的聊天和资料库随时都在这里。',
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
    signUp: '注册并开始'
  },
  recovery: {
    eyebrow: '登录 / 恢复',
    title: '我们再试一次登录',
    subtitleHeld: '登录窗口在完成前关闭了。你的请求仍已保存。',
    subtitle: '登录窗口在完成前关闭了。',
    listLabel: '登录进度',
    account: '你的账户',
    accountLine: 'Hermes 与 Nous 共用一个账户',
    unverified: '未验证',
    request: '你的请求',
    saved: '已保存',
    opening: '正在打开 Hermes',
    openingLine: '马上就好',
    waiting: '等待中',
    tryAgain: '重试',
    otherEmail: '使用其他邮箱',
    expired: '邮件链接已过期？',
    requestNew: '重新获取一个。'
  },
  checkIn: {
    eyebrow: '回访 / 问候',
    title: 'Hermes 用得怎么样？',
    subtitle: '继续用顺手的，或者试试新东西。',
    listLabel: '可以试试',
    video: '把视频变成一页',
    videoLine: '保存所学的好方法',
    weekly: '每周给我更新',
    weeklyLine: '让例程替你追踪一个话题',
    tour: '再带我看看',
    tourLine: '重新打开产品介绍',
    try: '试试',
    open: '打开',
    goingWell: '挺好的',
    couldBeBetter: '还可以更好',
    dismiss: '关闭'
  },
  developerView: {
    eyebrow: '推荐 / 检测到代码',
    title: '想要开发者视图吗？',
    body: '看着智能体工作：它使用的终端、文件和差异就在对话旁边。',
    preview: '实时查看工作过程',
    previewLine: '对话旁边的终端、文件和差异',
    turnOn: '打开开发者视图',
    notNow: '暂不',
    settingsNote: '随时可在设置中更改。'
  }
}

const zhHant: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: '歡迎使用 Hermes',
    subtitle: '交給它一項任務。它會了解你的工作方式，並在你離開時繼續進行。',
    listLabel: 'Hermes 能做什麼',
    placeNames: { connections: '連線' },
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
      connections: {
        line: '連接你常用的應用程式。',
        example: '例如：「查看我的 Gmail，把會議加到我的行事曆，把筆記發到 Slack，再到 Linear 建立任務」'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'Hermes 用得怎麼樣？',
      description: '繼續用順手的，或者試試新東西。',
      action: '開啟'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: '歡迎使用 Hermes',
    signUp: '註冊並開始',
    placeholderIdeas: '交辦任務給 Hermes',
    placeholderRetired: '交辦任務給你的代理',
    back: '歡迎回來。',
    subtitleRetired: '你想讓代理做些什麼？',
    awayTitle: '很高興再次見到你。',
    awayLine: '你的聊天和資料庫隨時都在這裡。',
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
    signUp: '註冊並開始'
  },
  recovery: {
    eyebrow: '登入 / 復原',
    title: '我們再試一次登入',
    subtitleHeld: '登入視窗在完成前關閉了。你的請求仍然保存著。',
    subtitle: '登入視窗在完成前關閉了。',
    listLabel: '登入進度',
    account: '你的帳戶',
    accountLine: 'Hermes 與 Nous 共用一個帳戶',
    unverified: '未驗證',
    request: '你的請求',
    saved: '已儲存',
    opening: '正在開啟 Hermes',
    openingLine: '馬上就好',
    waiting: '等待中',
    tryAgain: '再試一次',
    otherEmail: '使用其他電子郵件',
    expired: '郵件連結已過期？',
    requestNew: '重新索取一個。'
  },
  checkIn: {
    eyebrow: '回訪 / 問候',
    title: 'Hermes 用得怎麼樣？',
    subtitle: '繼續用順手的，或者試試新東西。',
    listLabel: '可以試試',
    video: '把影片變成一頁',
    videoLine: '保存所學的好方法',
    weekly: '每週給我更新',
    weeklyLine: '讓例行程序替你追蹤一個主題',
    tour: '再帶我看看',
    tourLine: '重新開啟產品介紹',
    try: '試試',
    open: '開啟',
    goingWell: '挺好的',
    couldBeBetter: '還可以更好',
    dismiss: '關閉'
  },
  developerView: {
    eyebrow: '推薦 / 偵測到程式碼',
    title: '想要開發者檢視嗎？',
    body: '看著代理工作：它使用的終端機、檔案和差異就在對話旁邊。',
    preview: '即時查看工作過程',
    previewLine: '對話旁邊的終端機、檔案和差異',
    turnOn: '開啟開發者檢視',
    notNow: '暫不',
    settingsNote: '隨時可在設定中變更。'
  }
}

const ru: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'Добро пожаловать в Hermes',
    subtitle: 'Дайте задачу. Hermes изучит, как вы работаете, и продолжит, пока вас нет.',
    listLabel: 'Что умеет Hermes',
    placeNames: { connections: 'Подключения' },
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
      connections: {
        line: 'Подключите свои повседневные приложения.',
        example: 'напр. «проверь мою почту Gmail, добавь встречу в календарь, опубликуй заметки в Slack и создай задачи в Linear»'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'Как вам Hermes?',
      description: 'Продолжайте то, что работает, или попробуйте новое.',
      action: 'Открыть'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'Добро пожаловать в Hermes',
    signUp: 'Зарегистрироваться',
    placeholderIdeas: 'Дайте Hermes задачу',
    placeholderRetired: 'Дайте агенту задачу',
    back: 'С возвращением.',
    subtitleRetired: 'Над чем поработать вашему агенту?',
    awayTitle: 'Рады видеть вас снова.',
    awayLine: 'Ваши чаты и Библиотека здесь, когда понадобятся.',
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
    signUp: 'Зарегистрироваться'
  },
  recovery: {
    eyebrow: 'Вход / Восстановление',
    title: 'Попробуем войти ещё раз',
    subtitleHeld: 'Окно входа закрылось раньше, чем мы закончили. Ваш запрос по-прежнему сохранён.',
    subtitle: 'Окно входа закрылось раньше, чем мы закончили.',
    listLabel: 'Как идёт вход',
    account: 'Ваш аккаунт',
    accountLine: 'Один аккаунт для Hermes и Nous',
    unverified: 'Не подтверждён',
    request: 'Ваш запрос',
    saved: 'Сохранён',
    opening: 'Открываем Hermes',
    openingLine: 'Это займёт всего момент',
    waiting: 'Ожидание',
    tryAgain: 'Попробовать снова',
    otherEmail: 'Другой адрес почты',
    expired: 'Ссылка из письма устарела?',
    requestNew: 'Запросите новую.'
  },
  checkIn: {
    eyebrow: 'Возвращение / Как дела',
    title: 'Как вам Hermes?',
    subtitle: 'Продолжайте то, что работает, или попробуйте новое.',
    listLabel: 'Что попробовать',
    video: 'Превратить видео в страницу',
    videoLine: 'Удобный способ сохранить узнанное',
    weekly: 'Присылай новости раз в неделю',
    weeklyLine: 'Регулярная задача следит за темой за вас',
    tour: 'Покажи всё ещё раз',
    tourLine: 'Снова открыть знакомство с продуктом',
    try: 'Попробовать',
    open: 'Открыть',
    goingWell: 'Всё хорошо',
    couldBeBetter: 'Могло быть лучше',
    dismiss: 'Закрыть'
  },
  developerView: {
    eyebrow: 'Предложение / Найден код',
    title: 'Включить вид разработчика?',
    body: 'Смотрите, как работает агент: терминал, файлы и изменения рядом с разговором.',
    preview: 'Видеть работу вживую',
    previewLine: 'Терминал, файлы и изменения рядом с разговором',
    turnOn: 'Включить вид разработчика',
    notNow: 'Не сейчас',
    settingsNote: 'Это можно изменить в Настройках.'
  }
}

const ar: FirstTimeCopy = {
  splash: {
    eyebrow: 'Hermes Agent',
    title: 'مرحبًا بك في Hermes',
    subtitle: 'أعطه مهمة. يتعلّم طريقة عملك ويواصل أثناء غيابك.',
    listLabel: 'ما يفعله Hermes',
    placeNames: { connections: 'الاتصالات' },
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
      connections: {
        line: 'اربط تطبيقاتك اليومية.',
        example: 'مثلًا: «تحقق من بريدي في Gmail، وأضف الاجتماع إلى تقويمي، وانشر الملاحظات في Slack، وأنشئ مهامًا في Linear»'
      }
    },
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
          ascend: 'Ascend',
          cathedral: 'Cathedral',
          forest: 'Forest',
          angel: 'Angel',
          sword: 'Sword',
          throne: 'Throne',
          lattice: 'Lattice',
          wing: 'Wing',
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
    },
    checkIn: {
      title: 'كيف تجد Hermes حتى الآن؟',
      description: 'واصل ما ينجح، أو جرّب شيئًا جديدًا.',
      action: 'فتح'
    }
  },
  landing: {
    eyebrowPortal: 'Hermes Portal',
    title: 'مرحبًا بك في Hermes',
    signUp: 'سجّل للبدء',
    placeholderIdeas: 'أعطِ Hermes مهمة',
    placeholderRetired: 'أعطِ وكيلك مهمة',
    back: 'مرحبًا بعودتك.',
    subtitleRetired: 'على ماذا تريد أن يعمل وكيلك؟',
    awayTitle: 'سعداء برؤيتك مجددًا.',
    awayLine: 'محادثاتك ومكتبتك هنا متى احتجت إليها.',
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
    signUp: 'سجّل للبدء'
  },
  recovery: {
    eyebrow: 'تسجيل الدخول / الاسترداد',
    title: 'لنحاول تسجيل الدخول مرة أخرى',
    subtitleHeld: 'أُغلقت نافذة تسجيل الدخول قبل أن ننتهي. طلبك لا يزال محفوظًا.',
    subtitle: 'أُغلقت نافذة تسجيل الدخول قبل أن ننتهي.',
    listLabel: 'حالة تسجيل الدخول',
    account: 'حسابك',
    accountLine: 'حساب واحد لـ Hermes و Nous',
    unverified: 'غير مُتحقَّق',
    request: 'طلبك',
    saved: 'محفوظ',
    opening: 'جارٍ فتح Hermes',
    openingLine: 'لن يستغرق ذلك سوى لحظة',
    waiting: 'في الانتظار',
    tryAgain: 'حاول مرة أخرى',
    otherEmail: 'استخدم بريدًا آخر',
    expired: 'انتهت صلاحية رابط البريد؟',
    requestNew: 'اطلب رابطًا جديدًا.'
  },
  checkIn: {
    eyebrow: 'زيارة مجددة / اطمئنان',
    title: 'كيف تجد Hermes حتى الآن؟',
    subtitle: 'واصل ما ينجح، أو جرّب شيئًا جديدًا.',
    listLabel: 'أشياء لتجربتها',
    video: 'حوّل مقطع فيديو إلى صفحة',
    videoLine: 'طريقة مفيدة لحفظ ما تتعلمه',
    weekly: 'أبقني على اطلاع أسبوعيًا',
    weeklyLine: 'دع روتينًا يتابع موضوعًا لك',
    tour: 'أرني المكان مرة أخرى',
    tourLine: 'أعد فتح التعريف بالمنتج',
    try: 'جرّب',
    open: 'فتح',
    goingWell: 'يسير جيدًا',
    couldBeBetter: 'يمكن أن يكون أفضل',
    dismiss: 'إغلاق'
  },
  developerView: {
    eyebrow: 'اقتراح / تم اكتشاف شيفرة',
    title: 'هل تريد عرض المطوّر؟',
    body: 'شاهد وكيلك يعمل: الطرفية والملفات والفروق التي يستخدمها، بجانب المحادثة.',
    preview: 'شاهد العمل أثناء حدوثه',
    previewLine: 'الطرفية والملفات والفروق بجانب المحادثة',
    turnOn: 'شغّل عرض المطوّر',
    notNow: 'ليس الآن',
    settingsNote: 'يمكنك تغيير هذا في أي وقت من الإعدادات.'
  }
}

const COPY: Record<BundledLocale, FirstTimeCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function firstTimeCopy(locale: string): FirstTimeCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useFirstTimeCopy(): FirstTimeCopy {
  return firstTimeCopy(useI18n().locale)
}
