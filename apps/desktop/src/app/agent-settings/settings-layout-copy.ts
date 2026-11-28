import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

type Row = readonly [title: string, description: string]

/** Agent › Settings page layout copy (Figma 895:*): card group names, the one-line summary under
 *  each Advanced disclosure, and the titles/descriptions of the primary rows drawn on each page. */
export interface AgentSettingsLayoutCopy {
  groups: { responses: string; memory: string; longChats: string; project: string; execution: string }
  summaries: Record<
    'model' | 'chat' | 'memory' | 'workspace' | 'limits' | 'safety' | 'browser' | 'providers' | 'tools',
    string
  >
  rows: Record<string, Row>
}

const COPY: Record<BundledLocale, AgentSettingsLayoutCopy> = {
  en: {
    groups: {
      responses: 'Responses',
      memory: 'Memory',
      longChats: 'Long chats',
      project: 'Project',
      execution: 'Execution'
    },
    summaries: {
      model: 'Context window, subagents and multi-model setups',
      chat: 'Attachments and timezone',
      memory: 'Budgets and compression tuning',
      workspace: 'Repository discovery, shell environment and file limits',
      limits: 'Retries, timeouts and output limits',
      safety: 'Timeouts, allowlists and recovery',
      browser: 'Browser profile and private network access',
      providers: 'Custom endpoints and environment variables',
      tools: 'API keys and endpoints for the services some toolsets call'
    },
    rows: {
      'display.personality': ['Personality', 'The default style for new chats. You can still steer tone in any chat.'],
      'display.show_reasoning': ['Show reasoning', 'How the model’s thinking appears, when it shares it.'],
      'memory.memory_enabled': ['Persistent memory', 'Save durable memories that help future chats.'],
      'memory.user_profile_enabled': ['User profile', 'Keep a compact profile of your preferences.'],
      'memory.provider': ['Memory provider', 'Where memories are stored.'],
      'compression.enabled': ['Auto-compression', 'Summarize older context when a chat gets long.'],
      'stt.enabled': ['Speech to text', 'Turn your voice messages into text.'],
      'stt.provider': ['Speech-to-text provider', 'Transcribes what you say.'],
      'voice.auto_tts': ['Read responses aloud', 'Speak this agent’s replies automatically.'],
      'tts.provider': ['Text-to-speech provider', 'Turns replies into speech.'],
      'terminal.cwd': ['Working directory', 'Default folder for this agent’s tool and terminal work.'],
      'code_execution.mode': ['Code execution mode', 'How strictly code stays inside the current project.'],
      'terminal.backend': ['Execution backend', 'Where terminal commands run.'],
      'agent.max_turns': ['Max agent steps', 'Most tool-calling turns before a run stops.'],
      vault: ['Passwords & logins', 'Credentials the browser agent can use.']
    }
  },
  fr: {
    groups: {
      responses: 'Réponses',
      memory: 'Mémoire',
      longChats: 'Longues discussions',
      project: 'Projet',
      execution: 'Exécution'
    },
    summaries: {
      model: 'Fenêtre de contexte, sous-agents et configurations multi-modèles',
      chat: 'Pièces jointes et fuseau horaire',
      memory: 'Budgets et réglage de la compression',
      workspace: 'Découverte des dépôts, environnement du shell et limites de fichiers',
      limits: 'Nouvelles tentatives, délais et limites de sortie',
      safety: 'Délais, listes d’autorisation et récupération',
      browser: 'Profil de navigateur et accès au réseau privé',
      providers: 'Points de terminaison personnalisés et variables d’environnement',
      tools: 'Clés API et points de terminaison des services utilisés par certains outils'
    },
    rows: {
      'display.personality': [
        'Personnalité',
        'Le style par défaut des nouvelles discussions. Vous pouvez toujours ajuster le ton.'
      ],
      'display.show_reasoning': [
        'Afficher le raisonnement',
        'Comment la réflexion du modèle s’affiche quand il la partage.'
      ],
      'memory.memory_enabled': [
        'Mémoire persistante',
        'Enregistre des souvenirs durables utiles aux prochaines discussions.'
      ],
      'memory.user_profile_enabled': ['Profil utilisateur', 'Garde un profil compact de vos préférences.'],
      'memory.provider': ['Fournisseur de mémoire', 'Où les souvenirs sont stockés.'],
      'compression.enabled': ['Compression automatique', 'Résume l’ancien contexte quand une discussion s’allonge.'],
      'stt.enabled': ['Reconnaissance vocale', 'Transforme vos messages vocaux en texte.'],
      'stt.provider': ['Fournisseur de reconnaissance vocale', 'Transcrit ce que vous dites.'],
      'voice.auto_tts': ['Lire les réponses à voix haute', 'Lit automatiquement les réponses de cet agent.'],
      'tts.provider': ['Fournisseur de synthèse vocale', 'Transforme les réponses en voix.'],
      'terminal.cwd': ['Dossier de travail', 'Dossier par défaut pour les outils et le terminal de cet agent.'],
      'code_execution.mode': ['Mode d’exécution du code', 'À quel point le code reste dans le projet actuel.'],
      'terminal.backend': ['Environnement d’exécution', 'Où les commandes du terminal s’exécutent.'],
      'agent.max_turns': ['Étapes maximales', 'Nombre maximal de tours d’outils avant l’arrêt d’une exécution.'],
      vault: ['Mots de passe et identifiants', 'Les identifiants que l’agent de navigation peut utiliser.']
    }
  },
  de: {
    groups: {
      responses: 'Antworten',
      memory: 'Gedächtnis',
      longChats: 'Lange Chats',
      project: 'Projekt',
      execution: 'Ausführung'
    },
    summaries: {
      model: 'Kontextfenster, Subagenten und Multi-Modell-Setups',
      chat: 'Anhänge und Zeitzone',
      memory: 'Budgets und Komprimierung',
      workspace: 'Repository-Suche, Shell-Umgebung und Dateilimits',
      limits: 'Wiederholungen, Zeitlimits und Ausgabegrenzen',
      safety: 'Zeitlimits, Freigabelisten und Wiederherstellung',
      browser: 'Browserprofil und Zugriff aufs private Netzwerk',
      providers: 'Eigene Endpunkte und Umgebungsvariablen',
      tools: 'API-Schlüssel und Endpunkte für Dienste, die manche Toolsets aufrufen'
    },
    rows: {
      'display.personality': [
        'Persönlichkeit',
        'Der Standardstil neuer Chats. Den Ton kannst du in jedem Chat anpassen.'
      ],
      'display.show_reasoning': ['Denken anzeigen', 'Wie das Denken des Modells erscheint, wenn es es teilt.'],
      'memory.memory_enabled': ['Dauerhaftes Gedächtnis', 'Speichert Erinnerungen, die künftigen Chats helfen.'],
      'memory.user_profile_enabled': ['Benutzerprofil', 'Hält ein kompaktes Profil deiner Vorlieben.'],
      'memory.provider': ['Gedächtnisanbieter', 'Wo Erinnerungen gespeichert werden.'],
      'compression.enabled': ['Automatische Komprimierung', 'Fasst älteren Kontext zusammen, wenn ein Chat lang wird.'],
      'stt.enabled': ['Sprache zu Text', 'Wandelt deine Sprachnachrichten in Text um.'],
      'stt.provider': ['Spracherkennungsanbieter', 'Transkribiert, was du sagst.'],
      'voice.auto_tts': ['Antworten vorlesen', 'Liest die Antworten dieses Agenten automatisch vor.'],
      'tts.provider': ['Sprachausgabeanbieter', 'Wandelt Antworten in Sprache um.'],
      'terminal.cwd': ['Arbeitsverzeichnis', 'Standardordner für die Tool- und Terminalarbeit dieses Agenten.'],
      'code_execution.mode': ['Codeausführungsmodus', 'Wie streng Code im aktuellen Projekt bleibt.'],
      'terminal.backend': ['Ausführungsumgebung', 'Wo Terminalbefehle laufen.'],
      'agent.max_turns': ['Maximale Agentenschritte', 'Höchstzahl an Tool-Runden, bevor ein Lauf stoppt.'],
      vault: ['Passwörter & Anmeldungen', 'Zugangsdaten, die der Browser-Agent verwenden kann.']
    }
  },
  es: {
    groups: {
      responses: 'Respuestas',
      memory: 'Memoria',
      longChats: 'Chats largos',
      project: 'Proyecto',
      execution: 'Ejecución'
    },
    summaries: {
      model: 'Ventana de contexto, subagentes y configuraciones multimodelo',
      chat: 'Adjuntos y zona horaria',
      memory: 'Presupuestos y ajuste de la compresión',
      workspace: 'Descubrimiento de repositorios, entorno del shell y límites de archivos',
      limits: 'Reintentos, tiempos de espera y límites de salida',
      safety: 'Tiempos de espera, listas permitidas y recuperación',
      browser: 'Perfil del navegador y acceso a la red privada',
      providers: 'Endpoints personalizados y variables de entorno',
      tools: 'Claves API y endpoints de los servicios que usan algunas herramientas'
    },
    rows: {
      'display.personality': [
        'Personalidad',
        'El estilo predeterminado de los chats nuevos. Puedes ajustar el tono en cualquier chat.'
      ],
      'display.show_reasoning': ['Mostrar razonamiento', 'Cómo aparece el razonamiento del modelo cuando lo comparte.'],
      'memory.memory_enabled': ['Memoria persistente', 'Guarda recuerdos duraderos que ayudan en chats futuros.'],
      'memory.user_profile_enabled': ['Perfil de usuario', 'Mantiene un perfil breve de tus preferencias.'],
      'memory.provider': ['Proveedor de memoria', 'Dónde se guardan los recuerdos.'],
      'compression.enabled': ['Compresión automática', 'Resume el contexto antiguo cuando un chat se alarga.'],
      'stt.enabled': ['Voz a texto', 'Convierte tus mensajes de voz en texto.'],
      'stt.provider': ['Proveedor de voz a texto', 'Transcribe lo que dices.'],
      'voice.auto_tts': ['Leer respuestas en voz alta', 'Lee automáticamente las respuestas de este agente.'],
      'tts.provider': ['Proveedor de texto a voz', 'Convierte las respuestas en voz.'],
      'terminal.cwd': [
        'Directorio de trabajo',
        'Carpeta predeterminada para las herramientas y la terminal de este agente.'
      ],
      'code_execution.mode': [
        'Modo de ejecución de código',
        'Qué tan estrictamente el código se queda en el proyecto actual.'
      ],
      'terminal.backend': ['Entorno de ejecución', 'Dónde se ejecutan los comandos de terminal.'],
      'agent.max_turns': [
        'Pasos máximos del agente',
        'Máximo de turnos con herramientas antes de detener una ejecución.'
      ],
      vault: ['Contraseñas e inicios de sesión', 'Credenciales que puede usar el agente del navegador.']
    }
  },
  ja: {
    groups: {
      responses: '応答',
      memory: 'メモリー',
      longChats: '長いチャット',
      project: 'プロジェクト',
      execution: '実行'
    },
    summaries: {
      model: 'コンテキストウィンドウ、サブエージェント、複数モデルの構成',
      chat: '添付ファイルとタイムゾーン',
      memory: '上限と圧縮の調整',
      workspace: 'リポジトリの検出、シェル環境、ファイルの上限',
      limits: '再試行、タイムアウト、出力の上限',
      safety: 'タイムアウト、許可リスト、復元',
      browser: 'ブラウザープロフィールとプライベートネットワークへのアクセス',
      providers: 'カスタムエンドポイントと環境変数',
      tools: '一部のツールセットが呼び出すサービスの API キーとエンドポイント'
    },
    rows: {
      'display.personality': ['パーソナリティ', '新しいチャットの既定の話し方です。チャットごとに調整もできます。'],
      'display.show_reasoning': ['思考を表示', 'モデルが共有したときの思考の表示方法です。'],
      'memory.memory_enabled': ['永続メモリー', '今後のチャットに役立つ記憶を保存します。'],
      'memory.user_profile_enabled': ['ユーザープロフィール', 'あなたの好みを簡潔なプロフィールに保ちます。'],
      'memory.provider': ['メモリープロバイダー', '記憶の保存先です。'],
      'compression.enabled': ['自動圧縮', 'チャットが長くなったら古い文脈を要約します。'],
      'stt.enabled': ['音声入力', '音声メッセージをテキストにします。'],
      'stt.provider': ['音声認識プロバイダー', '話した内容を文字起こしします。'],
      'voice.auto_tts': ['応答を読み上げる', 'このエージェントの返信を自動で読み上げます。'],
      'tts.provider': ['音声合成プロバイダー', '返信を音声にします。'],
      'terminal.cwd': ['作業ディレクトリ', 'このエージェントのツールとターミナルの既定フォルダーです。'],
      'code_execution.mode': ['コード実行モード', 'コードを現在のプロジェクト内にどれだけ厳しく留めるかです。'],
      'terminal.backend': ['実行環境', 'ターミナルのコマンドを実行する場所です。'],
      'agent.max_turns': ['最大ステップ数', '実行を止めるまでのツール呼び出しの最大回数です。'],
      vault: ['パスワードとログイン', 'ブラウザエージェントが使える認証情報です。']
    }
  },
  zh: {
    groups: { responses: '回复', memory: '记忆', longChats: '长对话', project: '项目', execution: '执行' },
    summaries: {
      model: '上下文窗口、子智能体和多模型配置',
      chat: '附件和时区',
      memory: '预算与压缩调整',
      workspace: '仓库发现、Shell 环境和文件限制',
      limits: '重试、超时和输出限制',
      safety: '超时、允许列表和恢复',
      browser: '浏览器配置文件和私有网络访问',
      providers: '自定义端点和环境变量',
      tools: '部分工具集调用的服务所需的 API 密钥和端点'
    },
    rows: {
      'display.personality': ['个性', '新对话的默认风格。你仍可在任何对话中调整语气。'],
      'display.show_reasoning': ['显示推理', '模型分享思考时的显示方式。'],
      'memory.memory_enabled': ['持久记忆', '保存对以后对话有帮助的长期记忆。'],
      'memory.user_profile_enabled': ['用户画像', '保留一份简洁的偏好画像。'],
      'memory.provider': ['记忆提供方', '记忆的存储位置。'],
      'compression.enabled': ['自动压缩', '对话变长时总结较早的上下文。'],
      'stt.enabled': ['语音转文字', '把你的语音消息转成文字。'],
      'stt.provider': ['语音识别提供方', '转写你说的话。'],
      'voice.auto_tts': ['朗读回复', '自动朗读此智能体的回复。'],
      'tts.provider': ['语音合成提供方', '把回复转成语音。'],
      'terminal.cwd': ['工作目录', '此智能体工具和终端工作的默认文件夹。'],
      'code_execution.mode': ['代码执行模式', '代码限制在当前项目内的严格程度。'],
      'terminal.backend': ['执行后端', '终端命令运行的位置。'],
      'agent.max_turns': ['最大步数', '一次运行停止前的最多工具调用轮数。'],
      vault: ['密码与登录', '浏览器智能体可以使用的凭据。']
    }
  },
  'zh-hant': {
    groups: { responses: '回覆', memory: '記憶', longChats: '長對話', project: '專案', execution: '執行' },
    summaries: {
      model: '內容視窗、子智慧體與多模型設定',
      chat: '附件與時區',
      memory: '預算與壓縮調整',
      workspace: '儲存庫探索、Shell 環境與檔案限制',
      limits: '重試、逾時與輸出限制',
      safety: '逾時、允許清單與復原',
      browser: '瀏覽器設定檔與私人網路存取',
      providers: '自訂端點與環境變數',
      tools: '部分工具集呼叫的服務所需的 API 金鑰與端點'
    },
    rows: {
      'display.personality': ['個性', '新對話的預設風格。你仍可在任何對話中調整語氣。'],
      'display.show_reasoning': ['顯示推理', '模型分享思考時的顯示方式。'],
      'memory.memory_enabled': ['持久記憶', '儲存對日後對話有幫助的長期記憶。'],
      'memory.user_profile_enabled': ['使用者輪廓', '保留一份精簡的偏好輪廓。'],
      'memory.provider': ['記憶提供者', '記憶的儲存位置。'],
      'compression.enabled': ['自動壓縮', '對話變長時摘要較早的內容。'],
      'stt.enabled': ['語音轉文字', '將你的語音訊息轉成文字。'],
      'stt.provider': ['語音辨識提供者', '轉錄你說的話。'],
      'voice.auto_tts': ['朗讀回覆', '自動朗讀此智慧體的回覆。'],
      'tts.provider': ['語音合成提供者', '將回覆轉成語音。'],
      'terminal.cwd': ['工作目錄', '此智慧體工具與終端機工作的預設資料夾。'],
      'code_execution.mode': ['程式碼執行模式', '程式碼限制在目前專案內的嚴格程度。'],
      'terminal.backend': ['執行後端', '終端機指令執行的位置。'],
      'agent.max_turns': ['最大步數', '一次執行停止前的最多工具呼叫輪數。'],
      vault: ['密碼與登入', '瀏覽器智慧體可以使用的憑證。']
    }
  },
  ru: {
    groups: {
      responses: 'Ответы',
      memory: 'Память',
      longChats: 'Длинные чаты',
      project: 'Проект',
      execution: 'Выполнение'
    },
    summaries: {
      model: 'Окно контекста, субагенты и многомодельные настройки',
      chat: 'Вложения и часовой пояс',
      memory: 'Лимиты и настройка сжатия',
      workspace: 'Поиск репозиториев, окружение оболочки и лимиты файлов',
      limits: 'Повторы, тайм-ауты и лимиты вывода',
      safety: 'Тайм-ауты, списки разрешений и восстановление',
      browser: 'Профиль браузера и доступ к частной сети',
      providers: 'Свои эндпоинты и переменные окружения',
      tools: 'API-ключи и эндпоинты сервисов, которые вызывают некоторые наборы инструментов'
    },
    rows: {
      'display.personality': ['Характер', 'Стиль новых чатов по умолчанию. Тон можно менять в любом чате.'],
      'display.show_reasoning': [
        'Показывать рассуждения',
        'Как отображаются размышления модели, когда она ими делится.'
      ],
      'memory.memory_enabled': ['Постоянная память', 'Сохраняет воспоминания, полезные в будущих чатах.'],
      'memory.user_profile_enabled': ['Профиль пользователя', 'Хранит краткий профиль ваших предпочтений.'],
      'memory.provider': ['Провайдер памяти', 'Где хранятся воспоминания.'],
      'compression.enabled': ['Автосжатие', 'Сжимает старый контекст, когда чат становится длинным.'],
      'stt.enabled': ['Речь в текст', 'Превращает голосовые сообщения в текст.'],
      'stt.provider': ['Провайдер распознавания речи', 'Расшифровывает то, что вы говорите.'],
      'voice.auto_tts': ['Читать ответы вслух', 'Автоматически озвучивает ответы этого агента.'],
      'tts.provider': ['Провайдер синтеза речи', 'Превращает ответы в речь.'],
      'terminal.cwd': ['Рабочая папка', 'Папка по умолчанию для инструментов и терминала этого агента.'],
      'code_execution.mode': ['Режим выполнения кода', 'Насколько строго код остаётся внутри текущего проекта.'],
      'terminal.backend': ['Среда выполнения', 'Где выполняются команды терминала.'],
      'agent.max_turns': ['Максимум шагов', 'Сколько раундов с инструментами до остановки запуска.'],
      vault: ['Пароли и входы', 'Учётные данные, которые может использовать браузерный агент.']
    }
  },
  ar: {
    groups: {
      responses: 'الردود',
      memory: 'الذاكرة',
      longChats: 'المحادثات الطويلة',
      project: 'المشروع',
      execution: 'التنفيذ'
    },
    summaries: {
      model: 'نافذة السياق والوكلاء الفرعيون وإعدادات النماذج المتعددة',
      chat: 'المرفقات والمنطقة الزمنية',
      memory: 'الحدود وضبط الضغط',
      workspace: 'اكتشاف المستودعات وبيئة الصدفة وحدود الملفات',
      limits: 'إعادة المحاولة والمهلات وحدود المخرجات',
      safety: 'المهلات وقوائم السماح والاسترداد',
      browser: 'ملف المتصفح والوصول إلى الشبكة الخاصة',
      providers: 'نقاط نهاية مخصصة ومتغيرات البيئة',
      tools: 'مفاتيح API ونقاط النهاية للخدمات التي تستدعيها بعض مجموعات الأدوات'
    },
    rows: {
      'display.personality': ['الشخصية', 'الأسلوب الافتراضي للمحادثات الجديدة. يمكنك تعديل النبرة في أي محادثة.'],
      'display.show_reasoning': ['إظهار التفكير', 'كيف يظهر تفكير النموذج عندما يشاركه.'],
      'memory.memory_enabled': ['ذاكرة دائمة', 'يحفظ ذكريات دائمة تفيد المحادثات القادمة.'],
      'memory.user_profile_enabled': ['ملف المستخدم', 'يحتفظ بملف موجز لتفضيلاتك.'],
      'memory.provider': ['مزوّد الذاكرة', 'مكان حفظ الذكريات.'],
      'compression.enabled': ['الضغط التلقائي', 'يلخص السياق الأقدم عندما تطول المحادثة.'],
      'stt.enabled': ['تحويل الكلام إلى نص', 'يحوّل رسائلك الصوتية إلى نص.'],
      'stt.provider': ['مزوّد التعرف على الكلام', 'يفرّغ ما تقوله نصًا.'],
      'voice.auto_tts': ['قراءة الردود بصوت عالٍ', 'ينطق ردود هذا الوكيل تلقائيًا.'],
      'tts.provider': ['مزوّد تحويل النص إلى كلام', 'يحوّل الردود إلى كلام.'],
      'terminal.cwd': ['مجلد العمل', 'المجلد الافتراضي لأدوات هذا الوكيل والطرفية.'],
      'code_execution.mode': ['وضع تنفيذ الشيفرة', 'مدى صرامة بقاء الشيفرة داخل المشروع الحالي.'],
      'terminal.backend': ['بيئة التنفيذ', 'مكان تشغيل أوامر الطرفية.'],
      'agent.max_turns': ['أقصى خطوات للوكيل', 'أقصى عدد لجولات الأدوات قبل توقف التشغيل.'],
      vault: ['كلمات المرور وتسجيلات الدخول', 'بيانات الاعتماد التي يمكن لوكيل المتصفح استخدامها.']
    }
  }
}

export function useAgentSettingsLayoutCopy(): AgentSettingsLayoutCopy {
  const { locale } = useI18n()

  return COPY[isBundledLocale(locale) ? locale : 'en']
}
