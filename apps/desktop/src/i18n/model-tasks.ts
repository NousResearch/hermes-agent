// Auxiliary model task labels and consent hints, shared by the locale manifests.
export const MODEL_TASK_COPY = {
  en: {
    vision: { label: 'Vision', hint: 'Image analysis' },
    compression: { label: 'Compression', hint: 'Context compaction' },
    skills_hub: { label: 'Skills hub', hint: 'Skill search' },
    approval: { label: 'Approval', hint: 'Smart auto-approve' },
    mcp: { label: 'MCP', hint: 'MCP tool routing' },
    title_generation: { label: 'Title gen', hint: 'Session titles' },
    review: { label: 'Review', hint: '/review reviewer subagent' },
    triage_specifier: { label: 'Triage specifier', hint: 'Kanban spec fleshing' },
    kanban_decomposer: { label: 'Kanban decomposer', hint: 'Task decomposition' },
    profile_describer: { label: 'Profile describer', hint: 'Auto profile descriptions' },
    background_review: {
      label: 'Background review',
      hint: 'Opt-in memory and skill learning after turns; uses model tokens'
    },
    curator: { label: 'Curator', hint: 'Skill-usage review' }
  },
  zh: {
    vision: { label: '视觉', hint: '图片分析' },
    compression: { label: '压缩', hint: '上下文压缩' },
    skills_hub: { label: '技能中心', hint: '技能搜索' },
    approval: { label: '审批', hint: '智能自动批准' },
    mcp: { label: 'MCP', hint: 'MCP 工具路由' },
    title_generation: { label: '标题生成', hint: '会话标题' },
    review: { label: '评审', hint: '/review 评审子智能体' },
    triage_specifier: { label: '分类指定', hint: '看板任务规格补全' },
    kanban_decomposer: { label: '看板分解', hint: '任务拆解' },
    profile_describer: { label: '配置描述', hint: '自动生成配置描述' },
    background_review: { label: '后台复盘', hint: '开启后在对话回合结束时整理记忆和技能，会消耗模型 token' },
    curator: { label: '维护器', hint: '技能使用审查' }
  },
  zhHant: {
    vision: { label: '視覺', hint: '圖片分析' },
    compression: { label: '壓縮', hint: '上下文壓縮' },
    skills_hub: { label: '技能中心', hint: '技能搜尋' },
    approval: { label: '核准', hint: '智慧自動核准' },
    mcp: { label: 'MCP', hint: 'MCP 工具路由' },
    title_generation: { label: '標題生成', hint: '工作階段標題' },
    review: { label: '評審', hint: '/review 評審子代理' },
    triage_specifier: { label: '分類指定', hint: '看板任務規格補全' },
    kanban_decomposer: { label: '看板分解', hint: '任務拆解' },
    profile_describer: { label: '設定檔描述', hint: '自動生成設定檔描述' },
    background_review: { label: '背景回顧', hint: '選擇啟用後在對話回合結束時整理記憶和技能，會消耗模型 token' },
    curator: { label: '策展器', hint: '技能使用審查' }
  },
  ja: {
    vision: { label: 'ビジョン', hint: '画像分析' },
    compression: { label: '圧縮', hint: 'コンテキストの圧縮' },
    skills_hub: { label: 'スキルハブ', hint: 'スキル検索' },
    approval: { label: '承認', hint: 'スマート自動承認' },
    mcp: { label: 'MCP', hint: 'MCP ツールルーティング' },
    title_generation: { label: 'タイトル生成', hint: 'セッションタイトル' },
    review: { label: 'レビュー', hint: '/review レビューサブエージェント' },
    triage_specifier: { label: 'トリアージ指定', hint: 'カンバン仕様の具体化' },
    kanban_decomposer: { label: 'カンバン分解', hint: 'タスク分解' },
    profile_describer: { label: 'プロファイル記述', hint: 'プロファイル概要の自動生成' },
    background_review: {
      label: 'バックグラウンドレビュー',
      hint: '任意で有効化する会話後の記憶とスキル学習。モデルトークンを使用'
    },
    curator: { label: 'キュレーター', hint: 'スキル使用レビュー' }
  },
  ru: {
    vision: { label: 'Зрение', hint: 'Анализ изображений' },
    compression: { label: 'Сжатие', hint: 'Компрессия контекста' },
    skills_hub: { label: 'Хаб навыков', hint: 'Поиск навыков' },
    approval: { label: 'Одобрение', hint: 'Умное авто-одобрение' },
    mcp: { label: 'MCP', hint: 'Маршрутизация MCP-инструментов' },
    title_generation: { label: 'Ген. заголовка', hint: 'Заголовки сеансов' },
    background_review: {
      label: 'Фоновый обзор',
      hint: 'Обучение памяти и навыкам после ответа по выбору; расходует токены модели'
    },
    curator: { label: 'Куратор', hint: 'Просмотр использования навыков' }
  },
  de: {
    vision: {
      label: 'Sehen',
      hint: 'Bildanalyse'
    },
    compression: {
      label: 'Kompression',
      hint: 'Kontext-Verdichtung'
    },
    skills_hub: {
      label: 'Skills-Hub',
      hint: 'Skill-Suche'
    },
    approval: {
      label: 'Freigabe',
      hint: 'Intelligente Auto-Freigabe'
    },
    mcp: {
      label: 'MCP',
      hint: 'MCP-Tool-Routing'
    },
    title_generation: {
      label: 'Titel-Generierung',
      hint: 'Session-Titel'
    },
    review: {
      label: 'Review',
      hint: '/review Bewertungs-Subagent'
    },
    triage_specifier: {
      label: 'Triage-Spezifizierer',
      hint: 'Kanban-Spezifikation ausarbeiten'
    },
    kanban_decomposer: {
      label: 'Kanban-Zerleger',
      hint: 'Aufgaben zerlegen'
    },
    profile_describer: {
      label: 'Profil-Beschreiber',
      hint: 'Automatische Profilbeschreibungen'
    },
    background_review: {
      label: 'Hintergrundprüfung',
      hint: 'Optionales Lernen von Erinnerungen und Skills nach Antworten; verbraucht Modell-Tokens'
    },
    curator: {
      label: 'Kurator',
      hint: 'Skill-Nutzungs-Review'
    }
  },
  fr: {
    vision: {
      label: 'Vision',
      hint: "Analyse d'image"
    },
    compression: {
      label: 'Compression',
      hint: 'Compaction de contexte'
    },
    skills_hub: {
      label: 'Hub de skills',
      hint: 'Recherche de skills'
    },
    approval: {
      label: 'Approbation',
      hint: 'Auto-approbation intelligente'
    },
    mcp: {
      label: 'MCP',
      hint: "Routage d'outils MCP"
    },
    title_generation: {
      label: 'Génération de titre',
      hint: 'Titres de session'
    },
    review: {
      label: 'Révision',
      hint: 'Sous-agent de révision /review'
    },
    triage_specifier: {
      label: 'Précision du triage',
      hint: 'Détail des spécifications Kanban'
    },
    kanban_decomposer: {
      label: 'Décomposition Kanban',
      hint: 'Décomposition des tâches'
    },
    profile_describer: {
      label: 'Description de profil',
      hint: 'Descriptions automatiques des profils'
    },
    background_review: {
      label: 'Revue en arrière-plan',
      hint: 'Apprentissage optionnel des souvenirs et skills après les échanges ; consomme des tokens'
    },
    curator: {
      label: 'Curateur',
      hint: "Revue d'utilisation des skills"
    }
  },
  es: {
    vision: {
      label: 'Visión',
      hint: 'Análisis de imágenes'
    },
    compression: {
      label: 'Compresión',
      hint: 'Compactación de contexto'
    },
    skills_hub: {
      label: 'Hub de skills',
      hint: 'Búsqueda de skills'
    },
    approval: {
      label: 'Aprobación',
      hint: 'Aprobación automática inteligente'
    },
    mcp: {
      label: 'MCP',
      hint: 'Enrutamiento de herramientas MCP'
    },
    title_generation: {
      label: 'Generación de títulos',
      hint: 'Títulos de sesión'
    },
    review: {
      label: 'Revisión',
      hint: 'subagente revisor de /review'
    },
    triage_specifier: {
      label: 'Especificador de triaje',
      hint: 'Detalle de especificaciones de Kanban'
    },
    kanban_decomposer: {
      label: 'Descomponedor de Kanban',
      hint: 'Descomposición de tareas'
    },
    profile_describer: {
      label: 'Descriptor de perfiles',
      hint: 'Descripciones automáticas de perfiles'
    },
    background_review: {
      label: 'Revisión en segundo plano',
      hint: 'Aprendizaje opcional de memoria y skills tras cada turno; consume tokens del modelo'
    },
    curator: {
      label: 'Curador',
      hint: 'Revisión de uso de skills'
    }
  },
  ar: {
    vision: {
      label: 'الرؤية',
      hint: 'تحليل الصور'
    },
    compression: {
      label: 'الضغط',
      hint: 'ضغط السياق'
    },
    skills_hub: {
      label: 'مركز المهارات',
      hint: 'بحث المهارات'
    },
    approval: {
      label: 'الموافقة',
      hint: 'موافقة تلقائية ذكية'
    },
    mcp: {
      label: 'MCP',
      hint: 'توجيه أدوات MCP'
    },
    title_generation: {
      label: 'توليد العناوين',
      hint: 'عناوين الجلسات'
    },
    review: {
      label: 'المراجعة',
      hint: 'وكيل المراجعة الفرعي /review'
    },
    triage_specifier: {
      label: 'محدد الفرز',
      hint: 'توضيح مواصفات كانبان'
    },
    kanban_decomposer: {
      label: 'مفكك كانبان',
      hint: 'تفكيك المهام'
    },
    profile_describer: {
      label: 'واصف الملف الشخصي',
      hint: 'أوصاف ملفات شخصية تلقائية'
    },
    background_review: {
      label: 'مراجعة الخلفية',
      hint: 'تعلم الذاكرة والمهارات بعد الردود عند التفعيل؛ يستهلك رموز النموذج'
    },
    curator: {
      label: 'المنسّق',
      hint: 'مراجعة استخدام المهارات'
    }
  }
} as const
