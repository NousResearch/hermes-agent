import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

import type { AgentSettingsSection } from './settings-routes'

interface EditorCopy {
  general: string
  access: string
  runtime: string
  limits: string
  browser: string
  providers: string
  safety: string
  invalid: string
  soulDescription: string
  characters: string
  voiceModeDescription: string
  voiceGroups: Record<'conversation' | 'listening' | 'speaking', string>
  voiceAdvancedDescription: string
  descriptions: Record<Exclude<AgentSettingsSection, 'tools-advanced'>, string>
}

const copy = {
  en: {
    general: 'General',
    access: 'Access',
    runtime: 'Runtime',
    limits: 'Runtime & Limits',
    browser: 'Browser & Logins',
    providers: 'Providers & Keys',
    safety: 'Safety & Approvals',
    invalid: 'This setting is not part of this agent. Choose a section below.',
    soulDescription: 'Plain text or Markdown. Changes apply to new chats.',
    characters: 'Characters',
    voiceGroups: { conversation: 'Conversation', listening: 'Listening', speaking: 'Speaking' },
    voiceAdvancedDescription: 'Recording, transcription, and provider options',
    voiceModeDescription:
      'Chained: speech-to-text, the agent, then text-to-speech. GPT-Live: one live voice model that hands real tasks to the agent.',
    descriptions: {
      general: 'Name, color, and where this agent runs.',
      instructions: 'The system prompt and persona for this agent. Every new chat starts from it.',
      model: 'Which models this agent uses, and what happens when one fails.',
      chat: 'How this agent talks to you and handles what you send it.',
      memory: 'What this agent remembers between chats, and how it handles long ones.',
      voice: 'How this agent listens and talks back.',
      tools: 'Turn off any tools this agent shouldn’t use.',
      skills: 'Choose the skills this agent can use.',
      connectors: 'Choose the apps and MCP servers this agent can use.',
      plugins: 'Choose the plugins this agent can use.',
      messaging: 'Channels where you can talk to this agent outside Hermes.',
      schedules: 'Tasks this agent runs on a schedule, and where it sends the results.',
      safety: 'When this agent asks first, and what it can see and reach.',
      browser: 'Saved logins this agent can use, and how its browser behaves.',
      providers: 'Accounts and keys this agent uses to reach models.',
      workspace: 'Where this agent works and how its code runs.',
      limits: 'Where this agent’s commands run, and how far a run can go.'
    }
  },
  fr: {
    general: 'Général',
    access: 'Accès',
    runtime: 'Exécution',
    limits: 'Exécution et limites',
    browser: 'Navigateur et connexions',
    providers: 'Fournisseurs et clés',
    safety: 'Sécurité et approbations',
    invalid: 'Ce réglage ne fait pas partie de cet agent. Choisissez une section ci-dessous.',
    soulDescription: 'Texte brut ou Markdown. Les modifications s’appliquent aux nouveaux chats.',
    characters: 'Caractères',
    voiceGroups: { conversation: 'Conversation', listening: 'Écoute', speaking: 'Parole' },
    voiceAdvancedDescription: 'Enregistrement, transcription et options des fournisseurs',
    voiceModeDescription:
      'Chained : transcription, agent, puis synthèse vocale. GPT-Live : un modèle vocal en direct qui confie les tâches à l’agent.',
    descriptions: {
      general: 'Le nom, la couleur et l’emplacement d’exécution de cet agent.',
      instructions:
        'Les instructions système et la personnalité de cet agent. Chaque nouveau chat commence avec elles.',
      model: 'Les modèles utilisés par cet agent et les solutions de secours en cas d’échec.',
      chat: 'Comment cet agent vous répond et traite ce que vous lui envoyez.',
      memory: 'Ce que cet agent retient entre les chats et comment il gère les longues conversations.',
      voice: 'Comment cet agent écoute et répond à voix haute.',
      tools: 'Désactivez les outils que cet agent ne doit pas utiliser.',
      skills: 'Choisissez les compétences que cet agent peut utiliser.',
      connectors: 'Choisissez les applications et les serveurs MCP que cet agent peut utiliser.',
      plugins: 'Choisissez les extensions que cet agent peut utiliser.',
      messaging: 'Les canaux où vous pouvez parler à cet agent en dehors de Hermes.',
      schedules: 'Les tâches planifiées de cet agent et la destination de leurs résultats.',
      safety: 'Quand cet agent demande votre accord et ce à quoi il peut accéder.',
      browser: 'Les connexions enregistrées utilisables par cet agent et le comportement de son navigateur.',
      providers: 'Les comptes et les clés utilisés par cet agent pour accéder aux modèles.',
      workspace: 'Où cet agent travaille et comment son code s’exécute.',
      limits: 'Où les commandes de cet agent s’exécutent et les limites de chaque exécution.'
    }
  },
  de: {
    general: 'Allgemein',
    access: 'Zugriff',
    runtime: 'Laufzeit',
    limits: 'Laufzeit & Limits',
    browser: 'Browser & Anmeldungen',
    providers: 'Anbieter & Schlüssel',
    safety: 'Sicherheit & Freigaben',
    invalid: 'Diese Einstellung gehört nicht zu diesem Agenten. Wähle unten einen Bereich.',
    soulDescription: 'Klartext oder Markdown. Änderungen gelten für neue Chats.',
    characters: 'Zeichen',
    voiceGroups: { conversation: 'Gespräch', listening: 'Zuhören', speaking: 'Sprechen' },
    voiceAdvancedDescription: 'Aufnahme, Transkription und Anbieteroptionen',
    voiceModeDescription:
      'Chained: Spracherkennung, Agent, dann Sprachausgabe. GPT-Live: ein Live-Sprachmodell, das Aufgaben an den Agenten übergibt.',
    descriptions: {
      general: 'Name, Farbe und Ausführungsort dieses Agenten.',
      instructions: 'Systemanweisungen und Persönlichkeit dieses Agenten. Jeder neue Chat beginnt damit.',
      model: 'Welche Modelle dieser Agent verwendet und was passiert, wenn eines ausfällt.',
      chat: 'Wie dieser Agent antwortet und mit deinen Nachrichten umgeht.',
      memory: 'Was sich dieser Agent zwischen Chats merkt und wie er lange Gespräche verarbeitet.',
      voice: 'Wie dieser Agent zuhört und spricht.',
      tools: 'Deaktiviere Werkzeuge, die dieser Agent nicht verwenden soll.',
      skills: 'Wähle die Fähigkeiten, die dieser Agent verwenden darf.',
      connectors: 'Wähle die Apps und MCP-Server, die dieser Agent verwenden darf.',
      plugins: 'Wähle die Plugins, die dieser Agent verwenden darf.',
      messaging: 'Kanäle, über die du außerhalb von Hermes mit diesem Agenten sprechen kannst.',
      schedules: 'Geplante Aufgaben dieses Agenten und die Ziele für ihre Ergebnisse.',
      safety: 'Wann dieser Agent zuerst nachfragt und worauf er zugreifen kann.',
      browser: 'Gespeicherte Anmeldungen für diesen Agenten und das Verhalten seines Browsers.',
      providers: 'Konten und Schlüssel, mit denen dieser Agent auf Modelle zugreift.',
      workspace: 'Wo dieser Agent arbeitet und wie sein Code ausgeführt wird.',
      limits: 'Wo die Befehle dieses Agenten laufen und wie weit ein Lauf gehen darf.'
    }
  },
  es: {
    general: 'General',
    access: 'Acceso',
    runtime: 'Ejecución',
    limits: 'Ejecución y límites',
    browser: 'Navegador e inicios de sesión',
    providers: 'Proveedores y claves',
    safety: 'Seguridad y aprobaciones',
    invalid: 'Este ajuste no pertenece a este agente. Elige una sección a continuación.',
    soulDescription: 'Texto sin formato o Markdown. Los cambios se aplican a los chats nuevos.',
    characters: 'Caracteres',
    voiceGroups: { conversation: 'Conversación', listening: 'Escucha', speaking: 'Habla' },
    voiceAdvancedDescription: 'Grabación, transcripción y opciones del proveedor',
    voiceModeDescription:
      'Chained: transcripción, agente y síntesis de voz. GPT-Live: un modelo de voz en directo que delega las tareas en el agente.',
    descriptions: {
      general: 'El nombre, el color y dónde se ejecuta este agente.',
      instructions:
        'Las instrucciones del sistema y la personalidad de este agente. Cada chat nuevo comienza con ellas.',
      model: 'Qué modelos usa este agente y qué ocurre cuando uno falla.',
      chat: 'Cómo te responde este agente y procesa lo que le envías.',
      memory: 'Qué recuerda este agente entre chats y cómo gestiona las conversaciones largas.',
      voice: 'Cómo escucha y habla este agente.',
      tools: 'Desactiva las herramientas que este agente no deba usar.',
      skills: 'Elige las habilidades que puede usar este agente.',
      connectors: 'Elige las aplicaciones y los servidores MCP que puede usar este agente.',
      plugins: 'Elige los complementos que puede usar este agente.',
      messaging: 'Canales donde puedes hablar con este agente fuera de Hermes.',
      schedules: 'Las tareas programadas de este agente y dónde envía los resultados.',
      safety: 'Cuándo pide permiso este agente y a qué puede acceder.',
      browser: 'Las sesiones guardadas que puede usar este agente y cómo se comporta su navegador.',
      providers: 'Las cuentas y las claves que usa este agente para acceder a los modelos.',
      workspace: 'Dónde trabaja este agente y cómo se ejecuta su código.',
      limits: 'Dónde se ejecutan los comandos de este agente y hasta dónde puede llegar cada ejecución.'
    }
  },
  ja: {
    general: '一般',
    access: 'アクセス',
    runtime: '実行環境',
    limits: '実行環境と制限',
    browser: 'ブラウザーとログイン',
    providers: 'プロバイダーとキー',
    safety: '安全性と承認',
    invalid: 'この設定はこのエージェントの設定ではありません。下のセクションを選択してください。',
    soulDescription: 'プレーンテキストまたはMarkdown。変更は新しいチャットに適用されます。',
    characters: '文字数',
    voiceGroups: { conversation: '会話', listening: '聞き取り', speaking: '読み上げ' },
    voiceAdvancedDescription: '録音、文字起こし、プロバイダーの設定',
    voiceModeDescription:
      'Chained：音声をテキストに変換し、エージェントが処理して読み上げます。GPT-Live：リアルタイム音声モデルが実際のタスクをエージェントに渡します。',
    descriptions: {
      general: 'このエージェントの名前、色、実行場所。',
      instructions: 'このエージェントのシステムプロンプトとペルソナ。新しいチャットは毎回これを基に始まります。',
      model: 'このエージェントが使用するモデルと、失敗した場合の動作。',
      chat: 'このエージェントの応答方法と、送信した内容の扱い方。',
      memory: 'チャット間で記憶する内容と、長い会話の扱い方。',
      voice: 'このエージェントの音声入力と応答の方法。',
      tools: 'このエージェントに使わせたくないツールを無効にします。',
      skills: 'このエージェントが使用できるスキルを選びます。',
      connectors: 'このエージェントが使用できるアプリとMCPサーバーを選びます。',
      plugins: 'このエージェントが使用できるプラグインを選びます。',
      messaging: 'Hermesの外でこのエージェントと会話できるチャンネル。',
      schedules: 'このエージェントの定期実行タスクと、結果の送信先。',
      safety: 'このエージェントが事前に確認する条件と、アクセスできる範囲。',
      browser: 'このエージェントが利用できる保存済みログインと、ブラウザーの動作。',
      providers: 'モデルへのアクセスに使用するアカウントとキー。',
      workspace: 'このエージェントの作業場所と、コードの実行方法。',
      limits: 'このエージェントのコマンドの実行場所と、実行の制限。'
    }
  },
  zh: {
    general: '常规',
    access: '访问',
    runtime: '运行环境',
    limits: '运行环境与限制',
    browser: '浏览器与登录',
    providers: '提供商与密钥',
    safety: '安全与审批',
    invalid: '此设置不属于此智能体。请选择下方的设置分区。',
    soulDescription: '纯文本或 Markdown。更改会应用于新对话。',
    characters: '字符数',
    voiceGroups: { conversation: '对话', listening: '聆听', speaking: '朗读' },
    voiceAdvancedDescription: '录音、转录和提供商选项',
    voiceModeDescription: 'Chained：语音转文字、智能体处理、文字转语音。GPT-Live：实时语音模型将实际任务交给智能体。',
    descriptions: {
      general: '此智能体的名称、颜色和运行位置。',
      instructions: '此智能体的系统提示词与角色设定。每次新对话都会以此为基础。',
      model: '此智能体使用的模型，以及模型失败时的处理方式。',
      chat: '此智能体如何回复你，以及如何处理你发送的内容。',
      memory: '此智能体在对话之间记住什么，以及如何处理长对话。',
      voice: '此智能体如何聆听并用语音回复。',
      tools: '关闭此智能体不应使用的工具。',
      skills: '选择此智能体可以使用的技能。',
      connectors: '选择此智能体可以使用的应用和 MCP 服务器。',
      plugins: '选择此智能体可以使用的插件。',
      messaging: '在 Hermes 之外与此智能体交流的渠道。',
      schedules: '此智能体定时运行的任务，以及结果的发送位置。',
      safety: '此智能体何时先征求同意，以及可以访问的范围。',
      browser: '此智能体可用的已保存登录信息，以及浏览器的行为。',
      providers: '此智能体访问模型时使用的账户和密钥。',
      workspace: '此智能体的工作位置，以及代码的运行方式。',
      limits: '此智能体命令的运行位置，以及每次运行的限制。'
    }
  },
  'zh-hant': {
    general: '一般',
    access: '存取',
    runtime: '執行環境',
    limits: '執行環境與限制',
    browser: '瀏覽器與登入',
    providers: '供應商與金鑰',
    safety: '安全與核准',
    invalid: '此設定不屬於此代理。請選擇下方的設定區段。',
    soulDescription: '純文字或 Markdown。變更會套用於新對話。',
    characters: '字元數',
    voiceGroups: { conversation: '對話', listening: '聆聽', speaking: '朗讀' },
    voiceAdvancedDescription: '錄音、轉錄和供應商選項',
    voiceModeDescription: 'Chained：語音轉文字、代理處理、文字轉語音。GPT-Live：即時語音模型將實際工作交給代理。',
    descriptions: {
      general: '此代理的名稱、顏色與執行位置。',
      instructions: '此代理的系統提示詞與角色設定。每次新對話都會以此為基礎。',
      model: '此代理使用的模型，以及模型失敗時的處理方式。',
      chat: '此代理如何回覆你，以及如何處理你傳送的內容。',
      memory: '此代理在對話之間記住什麼，以及如何處理長對話。',
      voice: '此代理如何聆聽並以語音回覆。',
      tools: '關閉此代理不應使用的工具。',
      skills: '選擇此代理可以使用的技能。',
      connectors: '選擇此代理可以使用的應用程式與 MCP 伺服器。',
      plugins: '選擇此代理可以使用的外掛。',
      messaging: '在 Hermes 之外與此代理交流的管道。',
      schedules: '此代理定時執行的工作，以及結果的傳送位置。',
      safety: '此代理何時先徵求同意，以及可以存取的範圍。',
      browser: '此代理可用的已儲存登入資訊，以及瀏覽器的行為。',
      providers: '此代理存取模型時使用的帳戶與金鑰。',
      workspace: '此代理的工作位置，以及程式碼的執行方式。',
      limits: '此代理命令的執行位置，以及每次執行的限制。'
    }
  },
  ru: {
    general: 'Общие',
    access: 'Доступ',
    runtime: 'Среда выполнения',
    limits: 'Среда выполнения и ограничения',
    browser: 'Браузер и вход',
    providers: 'Провайдеры и ключи',
    safety: 'Безопасность и разрешения',
    invalid: 'Эта настройка не относится к этому агенту. Выберите раздел ниже.',
    soulDescription: 'Обычный текст или Markdown. Изменения применяются к новым чатам.',
    characters: 'Символы',
    voiceGroups: { conversation: 'Разговор', listening: 'Распознавание', speaking: 'Озвучивание' },
    voiceAdvancedDescription: 'Запись, распознавание речи и настройки провайдера',
    voiceModeDescription:
      'Chained: распознавание речи, агент, затем озвучивание. GPT-Live: голосовая модель в реальном времени передаёт задачи агенту.',
    descriptions: {
      general: 'Имя, цвет и место запуска этого агента.',
      instructions: 'Системные инструкции и роль этого агента. Каждый новый чат начинается с них.',
      model: 'Какие модели использует агент и что происходит при сбое одной из них.',
      chat: 'Как этот агент отвечает и обрабатывает ваши сообщения.',
      memory: 'Что этот агент запоминает между чатами и как обрабатывает длинные разговоры.',
      voice: 'Как этот агент слушает и отвечает голосом.',
      tools: 'Отключите инструменты, которые этот агент не должен использовать.',
      skills: 'Выберите навыки, доступные этому агенту.',
      connectors: 'Выберите приложения и серверы MCP, доступные этому агенту.',
      plugins: 'Выберите плагины, доступные этому агенту.',
      messaging: 'Каналы для общения с этим агентом за пределами Hermes.',
      schedules: 'Задачи этого агента по расписанию и места отправки результатов.',
      safety: 'Когда этот агент сначала спрашивает разрешение и к чему имеет доступ.',
      browser: 'Сохранённые входы, доступные этому агенту, и поведение его браузера.',
      providers: 'Учётные записи и ключи для доступа этого агента к моделям.',
      workspace: 'Где работает этот агент и как выполняется его код.',
      limits: 'Где выполняются команды этого агента и каковы пределы каждого запуска.'
    }
  },
  ar: {
    general: 'عام',
    access: 'الوصول',
    runtime: 'بيئة التشغيل',
    limits: 'بيئة التشغيل والحدود',
    browser: 'المتصفح وتسجيل الدخول',
    providers: 'المزوّدون والمفاتيح',
    safety: 'الأمان والموافقات',
    invalid: 'هذا الإعداد ليس جزءًا من هذا الوكيل. اختر قسمًا أدناه.',
    soulDescription: 'نص عادي أو Markdown. تُطبّق التغييرات على المحادثات الجديدة.',
    characters: 'عدد الأحرف',
    voiceGroups: { conversation: 'المحادثة', listening: 'الاستماع', speaking: 'التحدث' },
    voiceAdvancedDescription: 'خيارات التسجيل والتفريغ ومزود الخدمة',
    voiceModeDescription:
      'Chained: تحويل الكلام إلى نص، ثم الوكيل، ثم تحويل النص إلى كلام. GPT-Live: نموذج صوتي مباشر يسلّم المهام الفعلية إلى الوكيل.',
    descriptions: {
      general: 'اسم هذا الوكيل ولونه ومكان تشغيله.',
      instructions: 'تعليمات النظام وشخصية هذا الوكيل. تبدأ كل محادثة جديدة بها.',
      model: 'النماذج التي يستخدمها هذا الوكيل وما يحدث عند فشل أحدها.',
      chat: 'كيف يرد هذا الوكيل عليك ويتعامل مع ما ترسله.',
      memory: 'ما يتذكره هذا الوكيل بين المحادثات وكيف يتعامل مع المحادثات الطويلة.',
      voice: 'كيف يستمع هذا الوكيل ويرد صوتيًا.',
      tools: 'عطّل الأدوات التي لا ينبغي لهذا الوكيل استخدامها.',
      skills: 'اختر المهارات التي يمكن لهذا الوكيل استخدامها.',
      connectors: 'اختر التطبيقات وخوادم MCP التي يمكن لهذا الوكيل استخدامها.',
      plugins: 'اختر الإضافات التي يمكن لهذا الوكيل استخدامها.',
      messaging: 'القنوات التي يمكنك التحدث فيها مع هذا الوكيل خارج Hermes.',
      schedules: 'المهام المجدولة لهذا الوكيل والأماكن التي يرسل إليها النتائج.',
      safety: 'متى يطلب هذا الوكيل الإذن أولًا وما يمكنه الوصول إليه.',
      browser: 'تسجيلات الدخول المحفوظة المتاحة لهذا الوكيل وكيف يعمل متصفحه.',
      providers: 'الحسابات والمفاتيح التي يستخدمها هذا الوكيل للوصول إلى النماذج.',
      workspace: 'مكان عمل هذا الوكيل وكيفية تشغيل شفرته.',
      limits: 'مكان تنفيذ أوامر هذا الوكيل وحدود كل تشغيل.'
    }
  }
} satisfies Record<BundledLocale, EditorCopy>

const duplicate = {
  en: 'Duplicate',
  fr: 'Dupliquer',
  de: 'Duplizieren',
  es: 'Duplicar',
  ja: '複製',
  zh: '复制',
  'zh-hant': '複製',
  ru: 'Дублировать',
  ar: 'تكرار'
} satisfies Record<BundledLocale, string>

export function useAgentSettingsEditorCopy() {
  const { locale } = useI18n()
  const bundled = isBundledLocale(locale) ? locale : 'en'

  return { ...copy[bundled], duplicate: duplicate[bundled] }
}
