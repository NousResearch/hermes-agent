package com.nousresearch.hermesagent.ui.i18n

import androidx.compose.runtime.staticCompositionLocalOf

data class HermesStrings(
    val language: AppLanguage,
    val alphaBadge: String,
    val sectionHermes: String,
    val sectionAccounts: String,
    val sectionPortal: String,
    val sectionDevice: String,
    val sectionSettings: String,
    val subtitleHermes: String,
    val subtitleAccounts: String,
    val subtitlePortal: String,
    val subtitleDevice: String,
    val subtitleSettings: String,
    val runtimeSetupAndOnboarding: String,
    val openPageActions: String,
    val hermesLogoDescription: String,
    val settingsNewHereTitle: String,
    val settingsHelpStart: String,
    val settingsHelpAccounts: String,
    val appLanguageTitle: String,
    val appLanguageDescription: String,
    val onDeviceInferenceTitle: String,
    val onDeviceInferenceDescription: String,
    val llamaCppLabel: String,
    val llamaCppDescription: String,
    val liteRtLmLabel: String,
    val liteRtLmDescription: String,
    val noCompatibleLocalModel: String,
    val chatTitle: String,
    val openHistory: String,
    val history: String,
    val newChat: String,
    val backToChat: String,
    val clearConversation: String,
    val speakLastReply: String,
    val welcomeToHermes: String,
    val welcomeDescription: String,
    val accounts: String,
    val settings: String,
    val messageHermes: String,
    val send: String,
    val authIntro: String,
    val corr3xtAuthBaseUrl: String,
    val saveAuthUrl: String,
    val refresh: String,
    val pendingCorr3xtSignIn: String,
    val signIn: String,
    val signOut: String,
    val reconnect: String,
    val hermesProviderPrefix: String,
    val portalTitle: String,
    val portalEmbeddedDescription: String,
    val fullScreenPortal: String,
    val minimizePortal: String,
    val openExternally: String,
    val refreshPortal: String,
    val localDownloadsTitle: String,
    val localDownloadsDescription: String,
    val dataSaverModeTitle: String,
    val dataSaverModeDescription: String,
    val huggingFaceTokenOptional: String,
    val saveToken: String,
    val refreshDownloads: String,
    val repoIdOrDirectUrl: String,
    val filePathInsideRepo: String,
    val revision: String,
    val runtimeTarget: String,
    val inspect: String,
    val download: String,
    val downloadManagerTitle: String,
    val noLocalModelDownloadsYet: String,
    val preferredLocalModel: String,
    val setPreferred: String,
    val remove: String,
) {
    fun currentProviderProfile(providerLabel: String): String {
        return when (language) {
            AppLanguage.CHINESE -> "当前提供商配置：$providerLabel"
            AppLanguage.SPANISH -> "Perfil actual del proveedor: $providerLabel"
            AppLanguage.GERMAN -> "Aktuelles Anbieterprofil: $providerLabel"
            AppLanguage.PORTUGUESE -> "Perfil atual do provedor: $providerLabel"
            AppLanguage.FRENCH -> "Profil fournisseur actuel : $providerLabel"
            AppLanguage.ENGLISH -> "Current provider profile: $providerLabel"
        }
    }

    fun chatCommandsTip(isListening: Boolean): String {
        if (isListening) {
            return when (language) {
                AppLanguage.CHINESE -> "正在聆听…"
                AppLanguage.SPANISH -> "Escuchando…"
                AppLanguage.GERMAN -> "Hört zu…"
                AppLanguage.PORTUGUESE -> "Ouvindo…"
                AppLanguage.FRENCH -> "Écoute…"
                AppLanguage.ENGLISH -> "Listening…"
            }
        }
        return when (language) {
            AppLanguage.CHINESE -> "提示：/help 会显示原生命令"
            AppLanguage.SPANISH -> "Consejo: /help muestra los comandos nativos"
            AppLanguage.GERMAN -> "Tipp: /help zeigt die nativen Befehle"
            AppLanguage.PORTUGUESE -> "Dica: /help mostra os comandos nativos"
            AppLanguage.FRENCH -> "Astuce : /help affiche les commandes natives"
            AppLanguage.ENGLISH -> "Tip: /help shows native chat commands"
        }
    }

    fun providerLabel(): String = when (language) {
        AppLanguage.CHINESE -> "提供商"
        AppLanguage.SPANISH -> "Proveedor"
        AppLanguage.GERMAN -> "Anbieter"
        AppLanguage.PORTUGUESE -> "Provedor"
        AppLanguage.FRENCH -> "Fournisseur"
        AppLanguage.ENGLISH -> "Provider"
    }

    fun baseUrlLabel(): String = when (language) {
        AppLanguage.CHINESE -> "基础 URL"
        AppLanguage.SPANISH -> "URL base"
        AppLanguage.GERMAN -> "Basis-URL"
        AppLanguage.PORTUGUESE -> "URL base"
        AppLanguage.FRENCH -> "URL de base"
        AppLanguage.ENGLISH -> "Base URL"
    }

    fun modelLabel(): String = when (language) {
        AppLanguage.CHINESE -> "模型"
        AppLanguage.SPANISH -> "Modelo"
        AppLanguage.GERMAN -> "Modell"
        AppLanguage.PORTUGUESE -> "Modelo"
        AppLanguage.FRENCH -> "Modèle"
        AppLanguage.ENGLISH -> "Model"
    }

    fun apiKeyLabel(): String = when (language) {
        AppLanguage.CHINESE -> "API 密钥 / 令牌"
        AppLanguage.SPANISH -> "Clave API / token"
        AppLanguage.GERMAN -> "API-Schlüssel / Token"
        AppLanguage.PORTUGUESE -> "Chave API / token"
        AppLanguage.FRENCH -> "Clé API / jeton"
        AppLanguage.ENGLISH -> "API key / token"
    }

    fun saveLabel(): String = when (language) {
        AppLanguage.CHINESE -> "保存"
        AppLanguage.SPANISH -> "Guardar"
        AppLanguage.GERMAN -> "Speichern"
        AppLanguage.PORTUGUESE -> "Salvar"
        AppLanguage.FRENCH -> "Enregistrer"
        AppLanguage.ENGLISH -> "Save"
    }

    fun providerDirectCallHelp(): String = when (language) {
        AppLanguage.CHINESE -> "选择 Hermes 要直接调用的提供商。提供商密钥或令牌在这里保存；应用账户登录请使用账户页面。"
        AppLanguage.SPANISH -> "Elige el proveedor al que Hermes llamará directamente. Guarda aquí claves o tokens de proveedor; usa Cuentas para iniciar sesión en la app."
        AppLanguage.GERMAN -> "Wähle den Anbieter, den Hermes direkt aufrufen soll. Speichere Anbieter-Schlüssel oder Tokens hier; nutze Konten für die App-Anmeldung."
        AppLanguage.PORTUGUESE -> "Escolha o provedor que o Hermes vai chamar diretamente. Salve chaves ou tokens de provedor aqui; use Contas para login no app."
        AppLanguage.FRENCH -> "Choisissez le fournisseur que Hermes doit appeler directement. Enregistrez ici les clés ou jetons fournisseur ; utilisez Comptes pour la connexion à l’application."
        AppLanguage.ENGLISH -> "Choose the provider you want Hermes to call directly. Save provider keys or tokens here; use Accounts for app sign-in."
    }

    fun defaultBaseUrlSummary(providerLabel: String, defaultBaseUrl: String): String = when (language) {
        AppLanguage.CHINESE -> "$providerLabel 的默认地址：$defaultBaseUrl"
        AppLanguage.SPANISH -> "URL predeterminada para $providerLabel: $defaultBaseUrl"
        AppLanguage.GERMAN -> "Standard-URL für $providerLabel: $defaultBaseUrl"
        AppLanguage.PORTUGUESE -> "URL padrão para $providerLabel: $defaultBaseUrl"
        AppLanguage.FRENCH -> "URL par défaut pour $providerLabel : $defaultBaseUrl"
        AppLanguage.ENGLISH -> "Default for $providerLabel: $defaultBaseUrl"
    }

    fun suggestedModelSummary(modelHint: String): String = when (language) {
        AppLanguage.CHINESE -> "建议模型：$modelHint"
        AppLanguage.SPANISH -> "Modelo sugerido: $modelHint"
        AppLanguage.GERMAN -> "Vorgeschlagenes Modell: $modelHint"
        AppLanguage.PORTUGUESE -> "Modelo sugerido: $modelHint"
        AppLanguage.FRENCH -> "Modèle suggéré : $modelHint"
        AppLanguage.ENGLISH -> "Suggested model: $modelHint"
    }

    fun modelSelectionTitle(): String = when (language) {
        AppLanguage.CHINESE -> "模型选择"
        AppLanguage.SPANISH -> "Selección de modelo"
        AppLanguage.GERMAN -> "Modellauswahl"
        AppLanguage.PORTUGUESE -> "Seleção de modelo"
        AppLanguage.FRENCH -> "Sélection du modèle"
        AppLanguage.ENGLISH -> "Model selection"
    }

    fun modelSelectionDescription(): String = when (language) {
        AppLanguage.CHINESE -> "从提供商建议、Gemma 4、Gemma 3 和 Gemma 3n 本地模型中选择，或直接输入自定义模型 ID。"
        AppLanguage.SPANISH -> "Elige entre la sugerencia del proveedor, modelos locales Gemma 4, Gemma 3 y Gemma 3n, o escribe un ID de modelo personalizado."
        AppLanguage.GERMAN -> "Wähle den Anbietervorschlag, lokale Gemma-4-, Gemma-3- und Gemma-3n-Modelle oder gib eine eigene Modell-ID ein."
        AppLanguage.PORTUGUESE -> "Escolha entre a sugestão do provedor, modelos locais Gemma 4, Gemma 3 e Gemma 3n, ou digite um ID de modelo personalizado."
        AppLanguage.FRENCH -> "Choisissez la suggestion du fournisseur, des modèles locaux Gemma 4, Gemma 3 et Gemma 3n, ou saisissez un ID de modèle personnalisé."
        AppLanguage.ENGLISH -> "Choose a provider suggestion, first-class local Gemma 4, Gemma 3, and Gemma 3n models, or type a custom model ID."
    }

    fun addImage(): String = when (language) {
        AppLanguage.CHINESE -> "添加图片"
        AppLanguage.SPANISH -> "Añadir imagen"
        AppLanguage.GERMAN -> "Bild hinzufügen"
        AppLanguage.PORTUGUESE -> "Adicionar imagem"
        AppLanguage.FRENCH -> "Ajouter une image"
        AppLanguage.ENGLISH -> "Add image"
    }

    fun removeAttachment(): String = when (language) {
        AppLanguage.CHINESE -> "移除附件"
        AppLanguage.SPANISH -> "Quitar adjunto"
        AppLanguage.GERMAN -> "Anhang entfernen"
        AppLanguage.PORTUGUESE -> "Remover anexo"
        AppLanguage.FRENCH -> "Retirer la pièce jointe"
        AppLanguage.ENGLISH -> "Remove attachment"
    }

    fun attachedImages(count: Int): String = when (language) {
        AppLanguage.CHINESE -> "已附加 $count 张图片"
        AppLanguage.SPANISH -> "$count imagen(es) adjunta(s)"
        AppLanguage.GERMAN -> "$count Bild(er) angehängt"
        AppLanguage.PORTUGUESE -> "$count imagem(ns) anexada(s)"
        AppLanguage.FRENCH -> "$count image(s) jointe(s)"
        AppLanguage.ENGLISH -> "$count image(s) attached"
    }

    fun retryHermes(): String = when (language) {
        AppLanguage.CHINESE -> "重试 Hermes"
        AppLanguage.SPANISH -> "Reintentar Hermes"
        AppLanguage.GERMAN -> "Hermes erneut versuchen"
        AppLanguage.PORTUGUESE -> "Tentar Hermes novamente"
        AppLanguage.FRENCH -> "Réessayer Hermes"
        AppLanguage.ENGLISH -> "Retry Hermes"
    }

    fun gettingStartedTitle(): String = when (language) {
        AppLanguage.CHINESE -> "开始使用"
        AppLanguage.SPANISH -> "Primeros pasos"
        AppLanguage.GERMAN -> "Erste Schritte"
        AppLanguage.PORTUGUESE -> "Primeiros passos"
        AppLanguage.FRENCH -> "Premiers pas"
        AppLanguage.ENGLISH -> "Getting started"
    }

    fun gettingStartedStep(index: Int): String = when (index) {
        1 -> when (language) {
            AppLanguage.CHINESE -> "1. 账户：使用 Corr3xt 登录邮箱、电话或 Google；提供商密钥在设置中配置。"
            AppLanguage.SPANISH -> "1. Cuentas: inicia sesión por Corr3xt con correo, teléfono o Google; configura claves de proveedores en Ajustes."
            AppLanguage.GERMAN -> "1. Konten: Melde dich per Corr3xt mit E-Mail, Telefon oder Google an; Anbieter-Schlüssel richtest du in Einstellungen ein."
            AppLanguage.PORTUGUESE -> "1. Contas: entre pelo Corr3xt com e-mail, telefone ou Google; configure chaves de provedores nas Configurações."
            AppLanguage.FRENCH -> "1. Comptes : connectez-vous via Corr3xt avec e-mail, téléphone ou Google ; configurez les clés fournisseur dans Paramètres."
            AppLanguage.ENGLISH -> "1. Accounts: sign in through Corr3xt with email, phone, or Google; configure provider keys in Settings."
        }
        2 -> when (language) {
            AppLanguage.CHINESE -> "2. 设置：选择提供商，确认基础 URL/模型，并保存 API 密钥或令牌。"
            AppLanguage.SPANISH -> "2. Ajustes: elige proveedor, confirma la URL base/modelo y guarda la clave API o token."
            AppLanguage.GERMAN -> "2. Einstellungen: Wähle Anbieter, prüfe Basis-URL/Modell und speichere den API-Schlüssel oder Token."
            AppLanguage.PORTUGUESE -> "2. Configurações: escolha provedor, confirme URL base/modelo e salve a chave API ou token."
            AppLanguage.FRENCH -> "2. Réglages : choisissez le fournisseur, vérifiez l’URL de base/le modèle et enregistrez la clé API ou le jeton."
            AppLanguage.ENGLISH -> "2. Settings: choose a provider, confirm the base URL/model, and save your API key or token."
        }
        3 -> when (language) {
            AppLanguage.CHINESE -> "3. 设备：如果希望 Hermes 直接编辑真实手机文件，请授予共享文件夹访问权限。"
            AppLanguage.SPANISH -> "3. Equipo: concede acceso a carpeta compartida si quieres que Hermes edite archivos móviles reales."
            AppLanguage.GERMAN -> "3. Gerät: Erteile Freigabeordner-Zugriff, wenn Hermes echte mobile Dateien direkt bearbeiten soll."
            AppLanguage.PORTUGUESE -> "3. Aparelho: conceda acesso à pasta compartilhada se quiser que Hermes edite arquivos móveis reais."
            AppLanguage.FRENCH -> "3. Appareil : accordez l’accès au dossier partagé pour que Hermes modifie de vrais fichiers mobiles."
            AppLanguage.ENGLISH -> "3. Device: grant shared-folder access if you want Hermes to edit real mobile files directly."
        }
        else -> when (language) {
            AppLanguage.CHINESE -> "4. Hermes 聊天：运行时就绪后，可使用语音输入、聊天命令或齿轮按钮执行页面操作。"
            AppLanguage.SPANISH -> "4. Chat Hermes: usa voz, comandos de chat o el botón de engranaje cuando el runtime esté listo."
            AppLanguage.GERMAN -> "4. Hermes-Chat: Nutze Spracheingabe, Chat-Befehle oder das Zahnrad, sobald die Runtime bereit ist."
            AppLanguage.PORTUGUESE -> "4. Chat Hermes: use voz, comandos de chat ou o botão de engrenagem quando o runtime estiver pronto."
            AppLanguage.FRENCH -> "4. Chat Hermes : utilisez la voix, les commandes ou le bouton engrenage quand le runtime est prêt."
            AppLanguage.ENGLISH -> "4. Hermes chat: use voice input, chat commands, or the cog button for page-specific actions once the runtime is ready."
        }
    }

    fun apiKeyHelp(): String = when (language) {
        AppLanguage.CHINESE -> "粘贴所选提供商的 API 密钥或访问令牌，然后点保存以重启本地 Hermes 后端并应用新配置。"
        AppLanguage.SPANISH -> "Pega la clave API o token de acceso del proveedor seleccionado y pulsa Guardar para reiniciar el backend local de Hermes con la nueva configuración."
        AppLanguage.GERMAN -> "Füge den API-Schlüssel oder Zugriffstoken für den gewählten Anbieter ein und tippe auf Speichern, um das lokale Hermes-Backend mit der neuen Konfiguration neu zu starten."
        AppLanguage.PORTUGUESE -> "Cole a chave API ou token de acesso do provedor selecionado e toque em Salvar para reiniciar o backend local do Hermes com a nova configuração."
        AppLanguage.FRENCH -> "Collez la clé API ou le jeton d’accès du fournisseur sélectionné puis appuyez sur Enregistrer pour redémarrer le backend local Hermes avec la nouvelle configuration."
        AppLanguage.ENGLISH -> "Paste the API key or access token for the selected provider, then tap Save to restart the local Hermes backend with the new config."
    }

    fun openProviderKeyPage(providerLabel: String): String = when (language) {
        AppLanguage.CHINESE -> "打开 $providerLabel 设置页面"
        AppLanguage.SPANISH -> "Abrir página de configuración de $providerLabel"
        AppLanguage.GERMAN -> "$providerLabel-Einrichtungsseite öffnen"
        AppLanguage.PORTUGUESE -> "Abrir página de configuração do $providerLabel"
        AppLanguage.FRENCH -> "Ouvrir la page de configuration $providerLabel"
        AppLanguage.ENGLISH -> "Open $providerLabel setup page"
    }

    fun copyProviderSetupUrl(): String = when (language) {
        AppLanguage.CHINESE -> "复制设置链接"
        AppLanguage.SPANISH -> "Copiar URL de configuración"
        AppLanguage.GERMAN -> "Setup-URL kopieren"
        AppLanguage.PORTUGUESE -> "Copiar URL de configuração"
        AppLanguage.FRENCH -> "Copier l’URL de configuration"
        AppLanguage.ENGLISH -> "Copy setup URL"
    }

    fun importSavedProviderCredential(): String = when (language) {
        AppLanguage.CHINESE -> "使用已保存的 Hermes 凭据"
        AppLanguage.SPANISH -> "Usar credencial Hermes guardada"
        AppLanguage.GERMAN -> "Gespeicherte Hermes-Zugangsdaten nutzen"
        AppLanguage.PORTUGUESE -> "Usar credencial Hermes salva"
        AppLanguage.FRENCH -> "Utiliser l’identifiant Hermes enregistré"
        AppLanguage.ENGLISH -> "Use saved Hermes credential"
    }

    fun copyAuthSignInUrl(): String = when (language) {
        AppLanguage.CHINESE -> "复制登录链接"
        AppLanguage.SPANISH -> "Copiar URL de inicio de sesión"
        AppLanguage.GERMAN -> "Anmelde-URL kopieren"
        AppLanguage.PORTUGUESE -> "Copiar URL de login"
        AppLanguage.FRENCH -> "Copier l’URL de connexion"
        AppLanguage.ENGLISH -> "Copy sign-in URL"
    }

    fun authCopiedSignInUrl(): String = when (language) {
        AppLanguage.CHINESE -> "已复制登录链接。"
        AppLanguage.SPANISH -> "URL de inicio de sesión copiada."
        AppLanguage.GERMAN -> "Anmelde-URL kopiert."
        AppLanguage.PORTUGUESE -> "URL de login copiada."
        AppLanguage.FRENCH -> "URL de connexion copiée."
        AppLanguage.ENGLISH -> "Copied sign-in URL."
    }

    fun toolProfileTitle(): String = when (language) {
        AppLanguage.CHINESE -> "Android Alpha 工具配置"
        AppLanguage.SPANISH -> "Perfil de herramientas Android alpha"
        AppLanguage.GERMAN -> "Android-Alpha-Werkzeugprofil"
        AppLanguage.PORTUGUESE -> "Perfil de ferramentas Android alpha"
        AppLanguage.FRENCH -> "Profil d’outils Android alpha"
        AppLanguage.ENGLISH -> "Android alpha Tool Profile"
    }

    fun toolProfileEnabledSummary(tools: String): String = when (language) {
        AppLanguage.CHINESE -> "已启用：$tools"
        AppLanguage.SPANISH -> "Habilitadas: $tools"
        AppLanguage.GERMAN -> "Aktiviert: $tools"
        AppLanguage.PORTUGUESE -> "Ativadas: $tools"
        AppLanguage.FRENCH -> "Activés : $tools"
        AppLanguage.ENGLISH -> "Enabled: $tools"
    }

    fun toolProfileLinuxSummary(): String = when (language) {
        AppLanguage.CHINESE -> "Hermes 现在在 Android 应用中内置了本地 Linux 命令套件，因此 terminal/process 可以执行真实 CLI 命令，而共享文件夹工具处理文档编辑。"
        AppLanguage.SPANISH -> "Hermes ahora incluye una suite local de comandos Linux dentro de la app Android, así que terminal/process pueden ejecutar CLI reales mientras las herramientas de carpeta compartida gestionan ediciones directas de documentos."
        AppLanguage.GERMAN -> "Hermes enthält jetzt eine lokale Linux-Befehlssuite in der Android-App, sodass terminal/process echte CLI-Befehle ausführen können, während die Freigabeordner-Werkzeuge direkte Dokumentbearbeitungen übernehmen."
        AppLanguage.PORTUGUESE -> "O Hermes agora inclui uma suíte local de comandos Linux dentro do app Android, então terminal/process podem executar comandos CLI reais enquanto as ferramentas de pasta compartilhada fazem edições diretas em documentos."
        AppLanguage.FRENCH -> "Hermes inclut maintenant une suite locale de commandes Linux dans l’application Android, afin que terminal/process puissent exécuter de vraies commandes CLI tandis que les outils de dossier partagé gèrent les modifications directes des documents."
        AppLanguage.ENGLISH -> "Hermes now has a local Linux command suite in the Android app, so terminal/process can execute real CLI commands while shared-folder tools handle direct document edits."
    }

    fun toolProfileAccessibilitySummary(): String = when (language) {
        AppLanguage.CHINESE -> "启用 Hermes 无障碍服务后，可通过 android_ui_snapshot + android_ui_action 使用无障碍定位。"
        AppLanguage.SPANISH -> "La orientación por accesibilidad está disponible mediante android_ui_snapshot + android_ui_action después de activar el servicio de accesibilidad de Hermes."
        AppLanguage.GERMAN -> "Accessibility-Targeting ist über android_ui_snapshot + android_ui_action verfügbar, nachdem du den Hermes-Barrierefreiheitsdienst aktiviert hast."
        AppLanguage.PORTUGUESE -> "A segmentação por acessibilidade fica disponível com android_ui_snapshot + android_ui_action depois que você ativa o serviço de acessibilidade do Hermes."
        AppLanguage.FRENCH -> "Le ciblage par accessibilité est disponible via android_ui_snapshot + android_ui_action après activation du service d’accessibilité Hermes."
        AppLanguage.ENGLISH -> "Accessibility targeting is available through android_ui_snapshot + android_ui_action after you enable the Hermes accessibility service."
    }

    fun toolProfileCommandSuiteSummary(): String = when (language) {
        AppLanguage.CHINESE -> "Android 命令套件会解压到应用私有前缀，并通过 terminal/process 暴露，延续 Hermes 在 Termux 中相同风格的本地 CLI 用法。"
        AppLanguage.SPANISH -> "La suite de comandos Android se extrae a un prefijo privado de la app y se expone mediante terminal/process, manteniendo el mismo estilo de uso CLI local que Hermes ya soporta en Termux."
        AppLanguage.GERMAN -> "Die Android-Befehlssuite wird in ein app-privates Präfix entpackt und über terminal/process bereitgestellt, im selben lokalen CLI-Stil, den Hermes bereits in Termux unterstützt."
        AppLanguage.PORTUGUESE -> "A suíte de comandos Android é extraída para um prefixo privado do app e exposta por terminal/process, no mesmo estilo de uso CLI local que o Hermes já suporta no Termux."
        AppLanguage.FRENCH -> "La suite de commandes Android est extraite dans un préfixe privé à l’application et exposée via terminal/process, dans le même style d’utilisation CLI locale déjà pris en charge par Hermes dans Termux."
        AppLanguage.ENGLISH -> "The Android command suite is extracted into an app-private prefix and exposed through terminal/process for the same style of local CLI usage Hermes already supports in Termux."
    }

    fun toolProfileExcludedSummary(blocked: String): String = when (language) {
        AppLanguage.CHINESE -> "移动运行时中仍排除：$blocked"
        AppLanguage.SPANISH -> "Aún excluido del runtime móvil: $blocked"
        AppLanguage.GERMAN -> "Im mobilen Runtime weiterhin ausgeschlossen: $blocked"
        AppLanguage.PORTUGUESE -> "Ainda excluído do runtime móvel: $blocked"
        AppLanguage.FRENCH -> "Toujours exclus du runtime mobile : $blocked"
        AppLanguage.ENGLISH -> "Still excluded from the mobile runtime: $blocked"
    }

    fun deviceGuideTitle(): String = when (language) {
        AppLanguage.CHINESE -> "如何使用这个 alpha 版本"
        AppLanguage.SPANISH -> "Cómo usar esta alpha"
        AppLanguage.GERMAN -> "So verwendest du diese Alpha"
        AppLanguage.PORTUGUESE -> "Como usar esta alpha"
        AppLanguage.FRENCH -> "Comment utiliser cette alpha"
        AppLanguage.ENGLISH -> "How to use this alpha"
    }

    fun deviceGuideStep(index: Int): String = when (index) {
        1 -> when (language) {
            AppLanguage.CHINESE -> "1. Hermes 现在在 Android 应用内自带本地 Linux 命令套件。先让 Hermes 调用 android_device_status，再使用 terminal/process 执行完整 CLI。"
            AppLanguage.SPANISH -> "1. Hermes ahora incluye una suite local de comandos Linux dentro de la app Android. Pídele a Hermes que llame primero a android_device_status y luego usa terminal/process para la ejecución CLI completa."
            AppLanguage.GERMAN -> "1. Hermes bringt jetzt eine lokale Linux-Befehlssuite in der Android-App mit. Lass Hermes zuerst android_device_status aufrufen und nutze dann terminal/process für vollständige CLI-Ausführung."
            AppLanguage.PORTUGUESE -> "1. O Hermes agora inclui uma suíte local de comandos Linux dentro do app Android. Peça ao Hermes para chamar primeiro android_device_status e depois use terminal/process para execução CLI completa."
            AppLanguage.FRENCH -> "1. Hermes embarque maintenant une suite locale de commandes Linux dans l’application Android. Demandez d’abord à Hermes d’appeler android_device_status, puis utilisez terminal/process pour l’exécution CLI complète."
            AppLanguage.ENGLISH -> "1. Hermes now ships a local Linux command suite inside the Android app. Ask Hermes to call android_device_status first, then use terminal/process for full CLI execution."
        }
        2 -> when (language) {
            AppLanguage.CHINESE -> "2. 如果你想让 Hermes 原地编辑真实文件，请通过 Android 原生选择器授予共享文件夹访问权限。"
            AppLanguage.SPANISH -> "2. Concede una carpeta compartida desde el selector nativo de Android si quieres que Hermes edite los archivos reales en su ubicación."
            AppLanguage.GERMAN -> "2. Gewähre einen freigegebenen Ordner über den nativen Android-Auswahldialog, wenn Hermes echte Dateien direkt am Ort bearbeiten soll."
            AppLanguage.PORTUGUESE -> "2. Conceda uma pasta compartilhada no seletor nativo do Android se quiser que o Hermes edite os arquivos reais no lugar."
            AppLanguage.FRENCH -> "2. Accordez un dossier partagé via le sélecteur natif Android si vous voulez que Hermes modifie directement les vrais fichiers."
            AppLanguage.ENGLISH -> "2. Grant a shared folder from Android's native picker if you want Hermes to edit the real files in place with android_shared_folder_list/read/write."
        }
        3 -> when (language) {
            AppLanguage.CHINESE -> "3. 只有在需要草稿副本或暂存文件时，才把文件导入工作区。"
            AppLanguage.SPANISH -> "3. Importa archivos al espacio de trabajo solo cuando quieras copias temporales o archivos de preparación."
            AppLanguage.GERMAN -> "3. Importiere Dateien nur dann in den Arbeitsbereich, wenn du Entwurfs- oder Staging-Kopien brauchst."
            AppLanguage.PORTUGUESE -> "3. Importe arquivos para o espaço de trabalho apenas quando quiser cópias temporárias ou de preparação."
            AppLanguage.FRENCH -> "3. Importez des fichiers dans l’espace de travail uniquement si vous voulez des copies temporaires ou de préparation."
            AppLanguage.ENGLISH -> "3. Import files into the workspace only when you want scratch copies or staging files."
        }
        4 -> when (language) {
            AppLanguage.CHINESE -> "4. 如果你希望 Hermes 检查可见 UI 并触发更精确的操作，请启用 Hermes 无障碍服务。"
            AppLanguage.SPANISH -> "4. Activa la accesibilidad de Hermes si quieres que inspeccione la UI visible y lance acciones más precisas además de Inicio, Atrás, Recientes, Notificaciones y Ajustes rápidos."
            AppLanguage.GERMAN -> "4. Aktiviere die Hermes-Barrierefreiheit, wenn Hermes die sichtbare UI prüfen und gezielte Aktionen zusätzlich zu Start, Zurück, Letzte Apps, Benachrichtigungen und Schnelleinstellungen auslösen soll."
            AppLanguage.PORTUGUESE -> "4. Ative a acessibilidade do Hermes se quiser que ele inspecione a UI visível e acione ações mais precisas além de Início, Voltar, Recentes, Notificações e Ajustes rápidos."
            AppLanguage.FRENCH -> "4. Activez l’accessibilité Hermes si vous voulez qu’il inspecte l’interface visible et déclenche des actions ciblées en plus de Accueil, Retour, Récents, Notifications et Réglages rapides."
            AppLanguage.ENGLISH -> "4. Enable Hermes accessibility if you want Hermes to inspect the visible UI and trigger targeted actions in addition to Home / Back / Recents / Notifications / Quick settings."
        }
        else -> ""
    }

    fun deviceWorkspacePath(workspacePath: String): String = when (language) {
        AppLanguage.CHINESE -> "工作区路径：$workspacePath"
        AppLanguage.SPANISH -> "Ruta del espacio de trabajo: $workspacePath"
        AppLanguage.GERMAN -> "Arbeitsbereichspfad: $workspacePath"
        AppLanguage.PORTUGUESE -> "Caminho do espaço de trabalho: $workspacePath"
        AppLanguage.FRENCH -> "Chemin de l’espace de travail : $workspacePath"
        AppLanguage.ENGLISH -> "Workspace path: $workspacePath"
    }

    fun operatorStandbyTitle(): String = when (language) {
        AppLanguage.CHINESE -> "操作员待命"
        AppLanguage.SPANISH -> "Operador en espera"
        AppLanguage.GERMAN -> "Operator-Standby"
        AppLanguage.PORTUGUESE -> "Operador em espera"
        AppLanguage.FRENCH -> "Opérateur en veille"
        AppLanguage.ENGLISH -> "Operator standby"
    }

    fun operatorStandbyStatus(ready: Boolean, enabledCount: Int, externalCount: Int): String = when (language) {
        AppLanguage.CHINESE -> if (ready) "已启用 $enabledCount 个自动化，其中 $externalCount 个可由外部广播触发。" else "没有已启用的自动化待运行。"
        AppLanguage.SPANISH -> if (ready) "$enabledCount automatizaciones habilitadas; $externalCount aceptan disparo externo." else "No hay automatizaciones habilitadas esperando ejecución."
        AppLanguage.GERMAN -> if (ready) "$enabledCount Automationen aktiviert; $externalCount nehmen externe Broadcasts an." else "Keine aktivierten Automationen warten auf Ausführung."
        AppLanguage.PORTUGUESE -> if (ready) "$enabledCount automações ativadas; $externalCount aceitam acionamento externo." else "Nenhuma automação ativada aguardando execução."
        AppLanguage.FRENCH -> if (ready) "$enabledCount automatisations activées ; $externalCount acceptent un déclenchement externe." else "Aucune automatisation activée en attente d’exécution."
        AppLanguage.ENGLISH -> if (ready) "$enabledCount enabled automations; $externalCount accept external broadcast dispatch." else "No enabled automations are waiting for dispatch."
    }

    fun operatorStandbyRunHistory(count: Int): String = when (language) {
        AppLanguage.CHINESE -> "最近运行：$count"
        AppLanguage.SPANISH -> "Ejecuciones recientes: $count"
        AppLanguage.GERMAN -> "Letzte Ausführungen: $count"
        AppLanguage.PORTUGUESE -> "Execuções recentes: $count"
        AppLanguage.FRENCH -> "Exécutions récentes : $count"
        AppLanguage.ENGLISH -> "Recent runs: $count"
    }

    fun operatorStandbyRemoteDispatch(count: Int): String = when (language) {
        AppLanguage.CHINESE -> "远程待命自动化：$count"
        AppLanguage.SPANISH -> "Automatizaciones de espera remota: $count"
        AppLanguage.GERMAN -> "Remote-Standby-Automationen: $count"
        AppLanguage.PORTUGUESE -> "Automações de espera remota: $count"
        AppLanguage.FRENCH -> "Automatisations de veille distante : $count"
        AppLanguage.ENGLISH -> "Remote standby automations: $count"
    }

    fun operatorStandbyLastDispatch(taskName: String, source: String, channel: String): String {
        val cleanTask = taskName.ifBlank {
            when (language) {
                AppLanguage.CHINESE -> "远程任务"
                AppLanguage.SPANISH -> "tarea remota"
                AppLanguage.GERMAN -> "Remote-Aufgabe"
                AppLanguage.PORTUGUESE -> "tarefa remota"
                AppLanguage.FRENCH -> "tâche distante"
                AppLanguage.ENGLISH -> "remote task"
            }
        }
        val cleanSource = source.ifBlank {
            when (language) {
                AppLanguage.CHINESE -> "远程"
                AppLanguage.SPANISH -> "remoto"
                AppLanguage.GERMAN -> "remote"
                AppLanguage.PORTUGUESE -> "remoto"
                AppLanguage.FRENCH -> "distant"
                AppLanguage.ENGLISH -> "remote"
            }
        }
        val cleanChannel = channel.ifBlank {
            when (language) {
                AppLanguage.CHINESE -> "待命"
                AppLanguage.SPANISH -> "espera"
                AppLanguage.GERMAN -> "Standby"
                AppLanguage.PORTUGUESE -> "espera"
                AppLanguage.FRENCH -> "veille"
                AppLanguage.ENGLISH -> "standby"
            }
        }
        return when (language) {
            AppLanguage.CHINESE -> "上次远程调度：$cleanTask，经由 $cleanSource/$cleanChannel"
            AppLanguage.SPANISH -> "Último despacho remoto: $cleanTask por $cleanSource/$cleanChannel"
            AppLanguage.GERMAN -> "Letzte Remote-Dispatch: $cleanTask über $cleanSource/$cleanChannel"
            AppLanguage.PORTUGUESE -> "Último despacho remoto: $cleanTask por $cleanSource/$cleanChannel"
            AppLanguage.FRENCH -> "Dernier dispatch distant : $cleanTask via $cleanSource/$cleanChannel"
            AppLanguage.ENGLISH -> "Last remote dispatch: $cleanTask via $cleanSource/$cleanChannel"
        }
    }

    fun operatorStandbyLastRun(label: String, success: Boolean?, result: String): String {
        val status = when (success) {
            true -> when (language) {
                AppLanguage.CHINESE -> "成功"
                AppLanguage.SPANISH -> "correcto"
                AppLanguage.GERMAN -> "erfolgreich"
                AppLanguage.PORTUGUESE -> "sucesso"
                AppLanguage.FRENCH -> "réussi"
                AppLanguage.ENGLISH -> "success"
            }
            false -> when (language) {
                AppLanguage.CHINESE -> "失败"
                AppLanguage.SPANISH -> "fallo"
                AppLanguage.GERMAN -> "fehlgeschlagen"
                AppLanguage.PORTUGUESE -> "falha"
                AppLanguage.FRENCH -> "échec"
                AppLanguage.ENGLISH -> "failed"
            }
            null -> when (language) {
                AppLanguage.CHINESE -> "未运行"
                AppLanguage.SPANISH -> "sin ejecutar"
                AppLanguage.GERMAN -> "nicht ausgeführt"
                AppLanguage.PORTUGUESE -> "sem execução"
                AppLanguage.FRENCH -> "non exécuté"
                AppLanguage.ENGLISH -> "not run"
            }
        }
        val cleanLabel = label.ifBlank {
            when (language) {
                AppLanguage.CHINESE -> "自动化"
                AppLanguage.SPANISH -> "automatización"
                AppLanguage.GERMAN -> "Automation"
                AppLanguage.PORTUGUESE -> "automação"
                AppLanguage.FRENCH -> "automatisation"
                AppLanguage.ENGLISH -> "automation"
            }
        }
        val cleanResult = result.take(180)
        return when (language) {
            AppLanguage.CHINESE -> if (cleanResult.isBlank()) "上次运行：$cleanLabel ($status)" else "上次运行：$cleanLabel ($status) - $cleanResult"
            AppLanguage.SPANISH -> if (cleanResult.isBlank()) "Última ejecución: $cleanLabel ($status)" else "Última ejecución: $cleanLabel ($status) - $cleanResult"
            AppLanguage.GERMAN -> if (cleanResult.isBlank()) "Letzte Ausführung: $cleanLabel ($status)" else "Letzte Ausführung: $cleanLabel ($status) - $cleanResult"
            AppLanguage.PORTUGUESE -> if (cleanResult.isBlank()) "Última execução: $cleanLabel ($status)" else "Última execução: $cleanLabel ($status) - $cleanResult"
            AppLanguage.FRENCH -> if (cleanResult.isBlank()) "Dernière exécution : $cleanLabel ($status)" else "Dernière exécution : $cleanLabel ($status) - $cleanResult"
            AppLanguage.ENGLISH -> if (cleanResult.isBlank()) "Last run: $cleanLabel ($status)" else "Last run: $cleanLabel ($status) - $cleanResult"
        }
    }

    fun portalLoadingStatus(loggedIn: Boolean): String = when (language) {
        AppLanguage.CHINESE -> if (loggedIn) "已登录 Nous Portal" else "正在加载嵌入式 Portal 预览"
        AppLanguage.SPANISH -> if (loggedIn) "Sesión iniciada en Nous Portal" else "Cargando la vista previa incrustada del portal"
        AppLanguage.GERMAN -> if (loggedIn) "Bei Nous Portal angemeldet" else "Eingebettete Portal-Vorschau wird geladen"
        AppLanguage.PORTUGUESE -> if (loggedIn) "Sessão iniciada no Nous Portal" else "Carregando a prévia incorporada do portal"
        AppLanguage.FRENCH -> if (loggedIn) "Connecté à Nous Portal" else "Chargement de l’aperçu intégré du portail"
        AppLanguage.ENGLISH -> if (loggedIn) "Signed in to Nous Portal" else "Loading the embedded portal preview"
    }

    fun portalFallbackStatus(error: String): String = when (language) {
        AppLanguage.CHINESE -> "使用默认 Nous Portal URL（$error）"
        AppLanguage.SPANISH -> "Usando la URL predeterminada de Nous Portal ($error)"
        AppLanguage.GERMAN -> "Standard-URL von Nous Portal wird verwendet ($error)"
        AppLanguage.PORTUGUESE -> "Usando a URL padrão do Nous Portal ($error)"
        AppLanguage.FRENCH -> "URL Nous Portal par défaut utilisée ($error)"
        AppLanguage.ENGLISH -> "Using default Nous Portal URL ($error)"
    }

    fun authNotSignedIn(): String = when (language) {
        AppLanguage.CHINESE -> "未登录"
        AppLanguage.SPANISH -> "Sin iniciar sesión"
        AppLanguage.GERMAN -> "Nicht angemeldet"
        AppLanguage.PORTUGUESE -> "Sem sessão iniciada"
        AppLanguage.FRENCH -> "Non connecté"
        AppLanguage.ENGLISH -> "Not signed in"
    }

    fun cancelPendingSignIn(): String = when (language) {
        AppLanguage.CHINESE -> "取消等待中的登录"
        AppLanguage.SPANISH -> "Cancelar inicio de sesión pendiente"
        AppLanguage.GERMAN -> "Ausstehende Anmeldung abbrechen"
        AppLanguage.PORTUGUESE -> "Cancelar login pendente"
        AppLanguage.FRENCH -> "Annuler la connexion en attente"
        AppLanguage.ENGLISH -> "Cancel pending sign-in"
    }

    fun authGlobalStatusDefault(): String = when (language) {
        AppLanguage.CHINESE -> "已准备好使用已配置的 Corr3xt 应用登录 URL；提供商访问请在设置中使用安全 API 密钥或令牌。"
        AppLanguage.SPANISH -> "La URL Corr3xt configurada está lista para el inicio de sesión de la app; los proveedores usan claves API o tokens seguros en Ajustes."
        AppLanguage.GERMAN -> "Die konfigurierte Corr3xt-URL ist für die App-Anmeldung bereit; Anbieter nutzen sichere API-Schlüssel oder Tokens in Einstellungen."
        AppLanguage.PORTUGUESE -> "A URL Corr3xt configurada está pronta para login no app; provedores usam chaves API ou tokens seguros nas Configurações."
        AppLanguage.FRENCH -> "L’URL Corr3xt configurée est prête pour la connexion à l’application ; les fournisseurs utilisent des clés API ou jetons sécurisés dans Paramètres."
        AppLanguage.ENGLISH -> "Configured Corr3xt app sign-in URL is ready; providers use secure API keys or tokens in Settings."
    }

    fun authConfigureCorr3xtFirst(): String = when (language) {
        AppLanguage.CHINESE -> "请先配置可访问的 Corr3xt URL 以启用应用登录；提供商访问请在设置中使用安全 API 密钥或令牌。"
        AppLanguage.SPANISH -> "Configura una URL Corr3xt accesible para activar el inicio de sesión de la app; los proveedores usan claves API o tokens seguros en Ajustes."
        AppLanguage.GERMAN -> "Konfiguriere zuerst eine erreichbare Corr3xt-URL für die App-Anmeldung; Anbieter nutzen sichere API-Schlüssel oder Tokens in Einstellungen."
        AppLanguage.PORTUGUESE -> "Configure uma URL Corr3xt acessível para ativar o login no app; provedores usam chaves API ou tokens seguros nas Configurações."
        AppLanguage.FRENCH -> "Configurez d’abord une URL Corr3xt joignable pour activer la connexion à l’application ; les fournisseurs utilisent des clés API ou jetons sécurisés dans Paramètres."
        AppLanguage.ENGLISH -> "Configure a reachable Corr3xt URL to enable app sign-in; providers use secure API keys or tokens in Settings."
    }

    fun authWaitingCallback(label: String): String = when (language) {
        AppLanguage.CHINESE -> "正在等待 $label 的 Corr3xt 回调"
        AppLanguage.SPANISH -> "Esperando el callback de Corr3xt para $label"
        AppLanguage.GERMAN -> "Warte auf Corr3xt-Callback für $label"
        AppLanguage.PORTUGUESE -> "Aguardando o callback do Corr3xt para $label"
        AppLanguage.FRENCH -> "En attente du callback Corr3xt pour $label"
        AppLanguage.ENGLISH -> "Waiting for Corr3xt callback for $label"
    }

    fun authConnectedMethods(count: Int): String = when (language) {
        AppLanguage.CHINESE -> "已连接 $count 个登录方式"
        AppLanguage.SPANISH -> "$count métodos de inicio conectados"
        AppLanguage.GERMAN -> "$count Anmeldemethoden verbunden"
        AppLanguage.PORTUGUESE -> "$count métodos de login conectados"
        AppLanguage.FRENCH -> "$count méthodes de connexion connectées"
        AppLanguage.ENGLISH -> "$count sign-in methods connected"
    }

    fun authNoBrowser(): String = when (language) {
        AppLanguage.CHINESE -> "无法打开 Corr3xt：没有可用浏览器"
        AppLanguage.SPANISH -> "No se puede abrir Corr3xt: no hay navegador disponible"
        AppLanguage.GERMAN -> "Corr3xt konnte nicht geöffnet werden: kein Browser verfügbar"
        AppLanguage.PORTUGUESE -> "Não foi possível abrir o Corr3xt: nenhum navegador disponível"
        AppLanguage.FRENCH -> "Impossible d’ouvrir Corr3xt : aucun navigateur disponible"
        AppLanguage.ENGLISH -> "Unable to open Corr3xt: no browser is available"
    }

    fun authTryAgain(): String = when (language) {
        AppLanguage.CHINESE -> "无法打开 Corr3xt。请检查认证 URL 后重试。"
        AppLanguage.SPANISH -> "No se pudo abrir Corr3xt. Revisa la URL de autenticación e inténtalo de nuevo."
        AppLanguage.GERMAN -> "Corr3xt konnte nicht geöffnet werden. Prüfe die Auth-URL und versuche es erneut."
        AppLanguage.PORTUGUESE -> "Não foi possível abrir o Corr3xt. Verifique a URL de autenticação e tente novamente."
        AppLanguage.FRENCH -> "Impossible d’ouvrir Corr3xt. Vérifiez l’URL d’authentification puis réessayez."
        AppLanguage.ENGLISH -> "Unable to open Corr3xt. Check the auth URL and try again."
    }

    fun authCheckingCorr3xt(label: String): String = when (language) {
        AppLanguage.CHINESE -> "正在检查 $label 的 Corr3xt 登录页面…"
        AppLanguage.SPANISH -> "Comprobando la página de inicio Corr3xt para $label…"
        AppLanguage.GERMAN -> "Corr3xt-Anmeldeseite für $label wird geprüft…"
        AppLanguage.PORTUGUESE -> "Verificando a página de login Corr3xt para $label…"
        AppLanguage.FRENCH -> "Vérification de la page de connexion Corr3xt pour $label…"
        AppLanguage.ENGLISH -> "Checking Corr3xt sign-in page for $label…"
    }

    fun authHostCouldNotBeResolved(host: String): String = when (language) {
        AppLanguage.CHINESE -> "无法解析 Corr3xt 登录主机 $host。请使用可访问的登录 URL，或在设置中用 API 密钥配置此提供商。"
        AppLanguage.SPANISH -> "No se pudo resolver el host de inicio Corr3xt $host. Usa una URL de autenticación accesible o configura este proveedor con una clave API en Ajustes."
        AppLanguage.GERMAN -> "Der Corr3xt-Anmeldehost $host konnte nicht aufgelöst werden. Verwende eine erreichbare Auth-URL oder konfiguriere diesen Anbieter in den Einstellungen mit einem API-Schlüssel."
        AppLanguage.PORTUGUESE -> "Não foi possível resolver o host de login Corr3xt $host. Use uma URL de autenticação acessível ou configure este provedor com uma chave API nas Configurações."
        AppLanguage.FRENCH -> "Impossible de résoudre l’hôte de connexion Corr3xt $host. Utilisez une URL d’authentification accessible ou configurez ce fournisseur avec une clé API dans Paramètres."
        AppLanguage.ENGLISH -> "Corr3xt auth host $host could not be resolved. Use a reachable auth URL or configure this provider with an API key in Settings."
    }

    fun authPageCouldNotBeReached(errorName: String): String = when (language) {
        AppLanguage.CHINESE -> "无法访问 Corr3xt 登录页面：$errorName。请使用可访问的登录 URL，或在设置中用 API 密钥配置此提供商。"
        AppLanguage.SPANISH -> "No se pudo abrir la página de inicio Corr3xt: $errorName. Usa una URL de autenticación accesible o configura este proveedor con una clave API en Ajustes."
        AppLanguage.GERMAN -> "Die Corr3xt-Anmeldeseite konnte nicht erreicht werden: $errorName. Verwende eine erreichbare Auth-URL oder konfiguriere diesen Anbieter in den Einstellungen mit einem API-Schlüssel."
        AppLanguage.PORTUGUESE -> "Não foi possível acessar a página de login Corr3xt: $errorName. Use uma URL de autenticação acessível ou configure este provedor com uma chave API nas Configurações."
        AppLanguage.FRENCH -> "Impossible d’atteindre la page de connexion Corr3xt : $errorName. Utilisez une URL d’authentification accessible ou configurez ce fournisseur avec une clé API dans Paramètres."
        AppLanguage.ENGLISH -> "Corr3xt auth page could not be reached: $errorName. Use a reachable auth URL or configure this provider with an API key in Settings."
    }

    fun authAppSignInHostCouldNotBeResolved(host: String): String = when (language) {
        AppLanguage.CHINESE -> "无法解析 Corr3xt 应用登录主机 $host。在设置可访问的 Corr3xt URL 前，应用登录不可用；运行时提供商请在设置中使用安全 API 密钥或令牌。"
        AppLanguage.SPANISH -> "No se pudo resolver el host de inicio de sesión Corr3xt $host. El inicio de sesión de la app no está disponible hasta configurar una URL Corr3xt accesible; los proveedores de runtime usan claves API o tokens seguros en Ajustes."
        AppLanguage.GERMAN -> "Der Corr3xt-App-Anmeldehost $host konnte nicht aufgelöst werden. Die App-Anmeldung ist nicht verfügbar, bis eine erreichbare Corr3xt-URL gesetzt ist; Runtime-Anbieter nutzen sichere API-Schlüssel oder Tokens in Einstellungen."
        AppLanguage.PORTUGUESE -> "Não foi possível resolver o host de login Corr3xt $host. O login do app fica indisponível até configurar uma URL Corr3xt acessível; provedores de runtime usam chaves API ou tokens seguros nas Configurações."
        AppLanguage.FRENCH -> "Impossible de résoudre l’hôte de connexion Corr3xt $host. La connexion à l’application est indisponible tant qu’une URL Corr3xt joignable n’est pas définie ; les fournisseurs runtime utilisent des clés API ou jetons sécurisés dans Paramètres."
        AppLanguage.ENGLISH -> "Corr3xt app sign-in host $host could not be resolved. App sign-in is unavailable until a reachable Corr3xt URL is set; runtime providers use secure API keys or tokens in Settings."
    }

    fun authAppSignInPageCouldNotBeReached(errorName: String): String = when (language) {
        AppLanguage.CHINESE -> "无法访问 Corr3xt 应用登录页面：$errorName。在设置可访问的 Corr3xt URL 前，应用登录不可用；运行时提供商请在设置中使用安全 API 密钥或令牌。"
        AppLanguage.SPANISH -> "No se pudo abrir la página de inicio de sesión Corr3xt: $errorName. El inicio de sesión de la app no está disponible hasta configurar una URL Corr3xt accesible; los proveedores de runtime usan claves API o tokens seguros en Ajustes."
        AppLanguage.GERMAN -> "Die Corr3xt-App-Anmeldeseite konnte nicht erreicht werden: $errorName. Die App-Anmeldung ist nicht verfügbar, bis eine erreichbare Corr3xt-URL gesetzt ist; Runtime-Anbieter nutzen sichere API-Schlüssel oder Tokens in Einstellungen."
        AppLanguage.PORTUGUESE -> "Não foi possível acessar a página de login Corr3xt: $errorName. O login do app fica indisponível até configurar uma URL Corr3xt acessível; provedores de runtime usam chaves API ou tokens seguros nas Configurações."
        AppLanguage.FRENCH -> "Impossible d’atteindre la page de connexion Corr3xt : $errorName. La connexion à l’application est indisponible tant qu’une URL Corr3xt joignable n’est pas définie ; les fournisseurs runtime utilisent des clés API ou jetons sécurisés dans Paramètres."
        AppLanguage.ENGLISH -> "Corr3xt app sign-in page could not be reached: $errorName. App sign-in is unavailable until a reachable Corr3xt URL is set; runtime providers use secure API keys or tokens in Settings."
    }

    fun authApiKeyFallbackAvailable(label: String): String = when (language) {
        AppLanguage.CHINESE -> "可改用 $label 的安全 API 密钥设置继续。"
        AppLanguage.SPANISH -> "Puedes continuar con la configuración segura de clave API para $label."
        AppLanguage.GERMAN -> "Du kannst stattdessen mit der sicheren API-Schlüssel-Einrichtung für $label fortfahren."
        AppLanguage.PORTUGUESE -> "Você pode continuar com a configuração segura por chave API para $label."
        AppLanguage.FRENCH -> "Vous pouvez continuer avec la configuration sécurisée par clé API pour $label."
        AppLanguage.ENGLISH -> "You can continue with secure API-key setup for $label."
    }

    fun authApiKeyFallbackTitle(): String = when (language) {
        AppLanguage.CHINESE -> "改用 API 密钥"
        AppLanguage.SPANISH -> "Usar clave API"
        AppLanguage.GERMAN -> "API-Schlüssel verwenden"
        AppLanguage.PORTUGUESE -> "Usar chave API"
        AppLanguage.FRENCH -> "Utiliser une clé API"
        AppLanguage.ENGLISH -> "Use API key instead"
    }

    fun authApiKeyFallbackDescription(label: String): String = when (language) {
        AppLanguage.CHINESE -> "Hermes 会预选 $label，密钥会保存在 Android 加密存储中，并同步到本地 Python 运行时环境。"
        AppLanguage.SPANISH -> "Hermes preseleccionará $label, guardará la clave en el almacenamiento cifrado de Android y la sincronizará con el runtime local de Python."
        AppLanguage.GERMAN -> "Hermes wählt $label vor, speichert den Schlüssel verschlüsselt unter Android und synchronisiert ihn mit der lokalen Python-Runtime."
        AppLanguage.PORTUGUESE -> "O Hermes vai pré-selecionar $label, salvar a chave no armazenamento criptografado do Android e sincronizá-la com o runtime Python local."
        AppLanguage.FRENCH -> "Hermes présélectionnera $label, enregistrera la clé dans le stockage chiffré Android et la synchronisera avec le runtime Python local."
        AppLanguage.ENGLISH -> "Hermes will preselect $label, save the key in Android encrypted storage, and sync it into the local Python runtime."
    }

    fun useApiKeyInSettings(): String = when (language) {
        AppLanguage.CHINESE -> "在设置中使用 API 密钥"
        AppLanguage.SPANISH -> "Usar clave API en Ajustes"
        AppLanguage.GERMAN -> "API-Schlüssel in Einstellungen nutzen"
        AppLanguage.PORTUGUESE -> "Usar chave API nas Configurações"
        AppLanguage.FRENCH -> "Utiliser une clé API dans Paramètres"
        AppLanguage.ENGLISH -> "Use API key in Settings"
    }

    fun setUpApiKeyFor(label: String): String = when (language) {
        AppLanguage.CHINESE -> "设置 $label API 密钥"
        AppLanguage.SPANISH -> "Configurar clave API de $label"
        AppLanguage.GERMAN -> "$label-API-Schlüssel einrichten"
        AppLanguage.PORTUGUESE -> "Configurar chave API do $label"
        AppLanguage.FRENCH -> "Configurer la clé API $label"
        AppLanguage.ENGLISH -> "Set up $label API key"
    }

    fun authApiKeySetupReady(label: String): String = when (language) {
        AppLanguage.CHINESE -> "$label 已准备好使用安全 API 密钥设置。请在设置中粘贴密钥并保存。"
        AppLanguage.SPANISH -> "$label está listo para configuración segura con clave API. Pega la clave en Ajustes y guarda."
        AppLanguage.GERMAN -> "$label ist für die sichere API-Schlüssel-Einrichtung bereit. Füge den Schlüssel in den Einstellungen ein und speichere."
        AppLanguage.PORTUGUESE -> "$label está pronto para configuração segura por chave API. Cole a chave nas Configurações e salve."
        AppLanguage.FRENCH -> "$label est prêt pour une configuration sécurisée par clé API. Collez la clé dans Paramètres puis enregistrez."
        AppLanguage.ENGLISH -> "$label is ready for secure API-key setup. Paste the key in Settings and save."
    }

    fun authCanceled(): String = when (language) {
        AppLanguage.CHINESE -> "已取消等待中的 Corr3xt 登录"
        AppLanguage.SPANISH -> "Inicio de sesión Corr3xt pendiente cancelado"
        AppLanguage.GERMAN -> "Ausstehende Corr3xt-Anmeldung abgebrochen"
        AppLanguage.PORTUGUESE -> "Login Corr3xt pendente cancelado"
        AppLanguage.FRENCH -> "Connexion Corr3xt en attente annulée"
        AppLanguage.ENGLISH -> "Canceled pending Corr3xt sign-in"
    }

    fun authDescription(methodId: String, fallback: String): String {
        return when (methodId) {
            "email" -> when (language) {
                AppLanguage.CHINESE -> "通过 Corr3xt 使用邮箱链接或密码流程登录应用。"
                AppLanguage.SPANISH -> "Inicia sesión en la app mediante Corr3xt usando un enlace por correo o un flujo con contraseña."
                AppLanguage.GERMAN -> "Melde dich über Corr3xt mit einem E-Mail-Link oder Passwort-Flow in der App an."
                AppLanguage.PORTUGUESE -> "Entre no app pelo Corr3xt usando um link por e-mail ou fluxo com senha."
                AppLanguage.FRENCH -> "Connectez-vous à l’application via Corr3xt avec un lien e-mail ou un flux par mot de passe."
                AppLanguage.ENGLISH -> fallback
            }
            "google" -> when (language) {
                AppLanguage.CHINESE -> "通过 Corr3xt 使用 Google 账户登录应用。"
                AppLanguage.SPANISH -> "Inicia sesión en la app con una cuenta de Google mediante Corr3xt."
                AppLanguage.GERMAN -> "Melde dich über Corr3xt mit einem Google-Konto in der App an."
                AppLanguage.PORTUGUESE -> "Entre no app com uma conta Google pelo Corr3xt."
                AppLanguage.FRENCH -> "Connectez-vous à l’application avec un compte Google via Corr3xt."
                AppLanguage.ENGLISH -> fallback
            }
            "phone" -> when (language) {
                AppLanguage.CHINESE -> "通过 Corr3xt 使用短信或手机验证流程登录应用。"
                AppLanguage.SPANISH -> "Inicia sesión en la app con un flujo de SMS o verificación por teléfono mediante Corr3xt."
                AppLanguage.GERMAN -> "Melde dich über Corr3xt mit einem SMS- oder Telefonverifizierungsfluss in der App an."
                AppLanguage.PORTUGUESE -> "Entre no app com um fluxo de SMS ou verificação por telefone via Corr3xt."
                AppLanguage.FRENCH -> "Connectez-vous à l’application via Corr3xt avec un flux SMS ou de vérification téléphonique."
                AppLanguage.ENGLISH -> fallback
            }
            "chatgpt" -> when (language) {
                AppLanguage.CHINESE -> "粘贴 ChatGPT Web 访问令牌并同步到 Hermes Android。"
                AppLanguage.SPANISH -> "Pega un token de acceso de ChatGPT Web y sincronízalo con Hermes Android."
                AppLanguage.GERMAN -> "Füge ein ChatGPT-Web-Zugriffstoken ein und synchronisiere es mit Hermes Android."
                AppLanguage.PORTUGUESE -> "Cole um token de acesso do ChatGPT Web e sincronize-o com o Hermes Android."
                AppLanguage.FRENCH -> "Collez un jeton d’accès ChatGPT Web et synchronisez-le avec Hermes Android."
                AppLanguage.ENGLISH -> fallback
            }
            "claude" -> when (language) {
                AppLanguage.CHINESE -> "使用 Anthropic / Claude API 密钥进行 Hermes Android 远程模型调用。"
                AppLanguage.SPANISH -> "Usa una clave API de Anthropic / Claude para llamadas remotas de Hermes Android."
                AppLanguage.GERMAN -> "Nutze einen Anthropic-/Claude-API-Schlüssel für Hermes-Android-Remote-Modellaufrufe."
                AppLanguage.PORTUGUESE -> "Use uma chave API Anthropic / Claude para chamadas remotas do Hermes Android."
                AppLanguage.FRENCH -> "Utilisez une clé API Anthropic / Claude pour les appels de modèle distants Hermes Android."
                AppLanguage.ENGLISH -> fallback
            }
            "gemini" -> when (language) {
                AppLanguage.CHINESE -> "使用 Google AI Studio / Gemini API 密钥进行 Hermes Android 远程模型调用。"
                AppLanguage.SPANISH -> "Usa una clave API de Google AI Studio / Gemini para llamadas remotas de Hermes Android."
                AppLanguage.GERMAN -> "Nutze einen Google-AI-Studio-/Gemini-API-Schlüssel für Hermes-Android-Remote-Modellaufrufe."
                AppLanguage.PORTUGUESE -> "Use uma chave API Google AI Studio / Gemini para chamadas remotas do Hermes Android."
                AppLanguage.FRENCH -> "Utilisez une clé API Google AI Studio / Gemini pour les appels de modèle distants Hermes Android."
                AppLanguage.ENGLISH -> fallback
            }
            "qwen" -> when (language) {
                AppLanguage.CHINESE -> "使用 Qwen Cloud / DashScope API 密钥进行 Hermes Android 远程模型调用。"
                AppLanguage.SPANISH -> "Usa una clave API de Qwen Cloud / DashScope para llamadas remotas de Hermes Android."
                AppLanguage.GERMAN -> "Nutze einen Qwen-Cloud-/DashScope-API-Schlüssel für Hermes-Android-Remote-Modellaufrufe."
                AppLanguage.PORTUGUESE -> "Use uma chave API Qwen Cloud / DashScope para chamadas remotas do Hermes Android."
                AppLanguage.FRENCH -> "Utilisez une clé API Qwen Cloud / DashScope pour les appels de modèle distants Hermes Android."
                AppLanguage.ENGLISH -> fallback
            }
            "qwen-oauth" -> when (language) {
                AppLanguage.CHINESE -> "复用已有的 Qwen OAuth / Qwen Chat 令牌；新的 Qwen OAuth 登录已于 2026-04-15 停用，新设置请使用 Qwen Cloud。"
                AppLanguage.SPANISH -> "Reutiliza un token existente de Qwen OAuth / Qwen Chat; los inicios de sesión nuevos con Qwen OAuth se discontinuaron el 2026-04-15, así que usa Qwen Cloud para una configuración nueva."
                AppLanguage.GERMAN -> "Verwende einen vorhandenen Qwen-OAuth-/Qwen-Chat-Token; neue Qwen-OAuth-Anmeldungen wurden am 2026-04-15 eingestellt, nutze für neue Einrichtung Qwen Cloud."
                AppLanguage.PORTUGUESE -> "Reutilize um token Qwen OAuth / Qwen Chat existente; novos logins Qwen OAuth foram descontinuados em 2026-04-15, então use Qwen Cloud para nova configuração."
                AppLanguage.FRENCH -> "Réutilisez un jeton Qwen OAuth / Qwen Chat existant ; les nouvelles connexions Qwen OAuth ont été arrêtées le 2026-04-15, utilisez Qwen Cloud pour une nouvelle configuration."
                AppLanguage.ENGLISH -> fallback
            }
            "zai" -> when (language) {
                AppLanguage.CHINESE -> "使用 Z.AI / GLM API 密钥进行 Hermes Android 远程模型调用。"
                AppLanguage.SPANISH -> "Usa una clave API de Z.AI / GLM para llamadas remotas de Hermes Android."
                AppLanguage.GERMAN -> "Nutze einen Z.AI-/GLM-API-Schlüssel für Hermes-Android-Remote-Modellaufrufe."
                AppLanguage.PORTUGUESE -> "Use uma chave API Z.AI / GLM para chamadas remotas do Hermes Android."
                AppLanguage.FRENCH -> "Utilisez une clé API Z.AI / GLM pour les appels de modèle distants Hermes Android."
                AppLanguage.ENGLISH -> fallback
            }
            else -> fallback
        }
    }

    fun authRefreshDescription(): String = when (language) {
        AppLanguage.CHINESE -> "重新加载本地 Corr3xt 与提供商登录状态。"
        AppLanguage.SPANISH -> "Vuelve a cargar el estado local de Corr3xt y de los proveedores."
        AppLanguage.GERMAN -> "Lädt den lokalen Corr3xt- und Anbieter-Anmeldestatus neu."
        AppLanguage.PORTUGUESE -> "Recarrega o estado local do Corr3xt e dos provedores."
        AppLanguage.FRENCH -> "Recharge l’état local de Corr3xt et des fournisseurs."
        AppLanguage.ENGLISH -> "Reload local Corr3xt and provider auth status."
    }

    fun authCancelPendingDescription(): String = when (language) {
        AppLanguage.CHINESE -> "停止等待当前的 Corr3xt 回调。"
        AppLanguage.SPANISH -> "Deja de esperar el callback actual de Corr3xt."
        AppLanguage.GERMAN -> "Beendet das Warten auf den aktuellen Corr3xt-Callback."
        AppLanguage.PORTUGUESE -> "Para de aguardar o callback atual do Corr3xt."
        AppLanguage.FRENCH -> "Arrête d’attendre le callback Corr3xt en cours."
        AppLanguage.ENGLISH -> "Stop waiting for the current Corr3xt callback."
    }

    fun authWaitingCallbackFor(label: String): String = when (language) {
        AppLanguage.CHINESE -> "正在等待 $label 的 Corr3xt 回调。"
        AppLanguage.SPANISH -> "Esperando el callback de Corr3xt para $label."
        AppLanguage.GERMAN -> "Warte auf den Corr3xt-Callback für $label."
        AppLanguage.PORTUGUESE -> "Aguardando o callback do Corr3xt para $label."
        AppLanguage.FRENCH -> "En attente du callback Corr3xt pour $label."
        AppLanguage.ENGLISH -> "Waiting for Corr3xt callback for $label."
    }

    fun localDownloadsExampleGuidance(): String = when (language) {
        AppLanguage.CHINESE -> "输入任意 Hugging Face 仓库、hf:// 仓库、仓库页面 URL、resolve URL 或直接文件 URL。Hermes 会优先尝试推断与当前运行时匹配的文件；如果仓库里没有明显的 GGUF / LiteRT-LM 文件，就会退回到另一个看起来像模型工件的文件，并把最终是否可运行交给所选后端决定。若想固定具体文件，可填写仓库内文件路径。示例：GGUF 可用 `Qwen/Qwen2.5-1.5B-Instruct-GGUF`；LiteRT-LM 可用 `litert-community/Phi-4-mini-instruct`。"
        AppLanguage.SPANISH -> "Introduce cualquier repo de Hugging Face, un repo hf://, la URL de la página del repo, una URL resolve o una URL directa al archivo. Hermes intentará priorizar un archivo nativo del runtime cuando pueda inferirlo; si el repo no expone un GGUF / LiteRT-LM claro, hará fallback a otro artefacto que parezca de modelo y dejará que el backend elegido decida si puede cargarlo. Si quieres fijar un archivo exacto, completa la ruta interna del repo. Ejemplos: GGUF `Qwen/Qwen2.5-1.5B-Instruct-GGUF`; LiteRT-LM `litert-community/Phi-4-mini-instruct`."
        AppLanguage.GERMAN -> "Gib ein beliebiges Hugging-Face-Repo, ein hf://-Repo, eine Repo-Seiten-URL, eine Resolve-URL oder eine direkte Datei-URL ein. Hermes bevorzugt nach Möglichkeit eine runtime-native Datei; wenn das Repo kein klares GGUF / LiteRT-LM-Artefakt enthält, fällt Hermes auf eine andere modellartige Datei zurück und überlässt dem gewählten Backend die endgültige Kompatibilitätsentscheidung. Wenn du eine bestimmte Datei erzwingen willst, trage den Pfad im Repo ein. Beispiele: GGUF `Qwen/Qwen2.5-1.5B-Instruct-GGUF`; LiteRT-LM `litert-community/Phi-4-mini-instruct`."
        AppLanguage.PORTUGUESE -> "Insira qualquer repositório do Hugging Face, um repositório hf://, a URL da página do repositório, uma URL resolve ou uma URL direta do arquivo. O Hermes tenta priorizar um arquivo nativo do runtime quando consegue inferi-lo; se o repositório não expuser um GGUF / LiteRT-LM claro, ele faz fallback para outro artefato com cara de modelo e deixa o backend escolhido decidir se consegue carregá-lo. Se quiser fixar um arquivo exato, preencha o caminho interno do repositório. Exemplos: GGUF `Qwen/Qwen2.5-1.5B-Instruct-GGUF`; LiteRT-LM `litert-community/Phi-4-mini-instruct`."
        AppLanguage.FRENCH -> "Saisissez n’importe quel dépôt Hugging Face, un dépôt hf://, l’URL de la page du dépôt, une URL resolve ou une URL directe de fichier. Hermes essaie de privilégier un fichier natif pour le runtime lorsqu’il peut l’inférer ; si le dépôt n’expose pas clairement un artefact GGUF / LiteRT-LM, Hermes se rabat sur un autre artefact ressemblant à un modèle et laisse le backend choisi décider s’il peut le charger. Si vous voulez forcer un fichier précis, renseignez le chemin du fichier dans le dépôt. Exemples : GGUF `Qwen/Qwen2.5-1.5B-Instruct-GGUF` ; LiteRT-LM `litert-community/Phi-4-mini-instruct`."
        AppLanguage.ENGLISH -> "Enter any Hugging Face repo, hf:// repo, repo page URL, resolve URL, or direct file URL. Hermes will try to prefer a runtime-native file when it can infer one; if the repo does not expose a clear GGUF / LiteRT-LM artifact, Hermes falls back to another likely model artifact and lets the selected backend decide whether it can load it. If you want to pin an exact file, fill in the repo file path. Examples: GGUF `Qwen/Qwen2.5-1.5B-Instruct-GGUF`; LiteRT-LM `litert-community/Phi-4-mini-instruct`."
    }

    fun downloadManagerReliabilityDescription(): String = when (language) {
        AppLanguage.CHINESE -> "意外断线会由 Android DownloadManager 安全处理。如果手机在下载过程中关机，Hermes 会在重启后重新加载已保存的进度。若移动数据一直暂停，请打开系统下载界面，或使用下方按钮在允许移动数据 / 漫游后重新开始。"
        AppLanguage.SPANISH -> "Android DownloadManager maneja con seguridad las pérdidas de conexión inesperadas. Si el teléfono se apaga a mitad de la descarga, Hermes volverá a cargar el progreso guardado al reiniciarse. Si los datos móviles siguen pausados, abre la pantalla de descargas del sistema o reinicia la descarga abajo permitiendo datos móviles / roaming."
        AppLanguage.GERMAN -> "Unerwartete Verbindungsabbrüche werden vom Android-Downloadmanager sicher behandelt. Wenn sich das Telefon mitten im Download ausschaltet, lädt Hermes den gespeicherten Fortschritt nach dem Neustart erneut. Falls mobile Daten weiter pausiert bleiben, öffne die System-Downloads oder starte den Download unten mit erlaubten mobilen Daten / Roaming neu."
        AppLanguage.PORTUGUESE -> "Perdas inesperadas de conexão são tratadas com segurança pelo Android DownloadManager. Se o telefone desligar no meio do download, o Hermes recarrega o progresso salvo após reiniciar. Se os dados móveis continuarem pausados, abra a tela de downloads do sistema ou reinicie abaixo permitindo dados móveis / roaming."
        AppLanguage.FRENCH -> "Les pertes de connexion inattendues sont gérées en toute sécurité par Android DownloadManager. Si le téléphone s’éteint pendant le téléchargement, Hermes recharge la progression enregistrée après le redémarrage. Si les données mobiles restent bloquées, ouvrez l’écran de téléchargements système ou relancez ci-dessous en autorisant les données mobiles / l’itinérance."
        AppLanguage.ENGLISH -> "Unexpected connection loss is handled safely by Android DownloadManager. If the phone shuts down mid-download, Hermes reloads the saved progress after restart. If mobile data stays paused, open the system Downloads screen or restart below with mobile data / roaming allowed."
    }

    fun localDownloadStatusLabel(status: String): String {
        return when (status.trim().lowercase()) {
            "queued" -> when (language) {
                AppLanguage.CHINESE -> "排队中"
                AppLanguage.SPANISH -> "En cola"
                AppLanguage.GERMAN -> "In Warteschlange"
                AppLanguage.PORTUGUESE -> "Na fila"
                AppLanguage.FRENCH -> "En file d’attente"
                AppLanguage.ENGLISH -> "Queued"
            }
            "downloading" -> when (language) {
                AppLanguage.CHINESE -> "下载中"
                AppLanguage.SPANISH -> "Descargando"
                AppLanguage.GERMAN -> "Wird heruntergeladen"
                AppLanguage.PORTUGUESE -> "Baixando"
                AppLanguage.FRENCH -> "Téléchargement"
                AppLanguage.ENGLISH -> "Downloading"
            }
            "paused" -> when (language) {
                AppLanguage.CHINESE -> "已暂停"
                AppLanguage.SPANISH -> "Pausado"
                AppLanguage.GERMAN -> "Pausiert"
                AppLanguage.PORTUGUESE -> "Pausado"
                AppLanguage.FRENCH -> "En pause"
                AppLanguage.ENGLISH -> "Paused"
            }
            "completed" -> when (language) {
                AppLanguage.CHINESE -> "已完成"
                AppLanguage.SPANISH -> "Completado"
                AppLanguage.GERMAN -> "Abgeschlossen"
                AppLanguage.PORTUGUESE -> "Concluído"
                AppLanguage.FRENCH -> "Terminé"
                AppLanguage.ENGLISH -> "Completed"
            }
            "failed" -> when (language) {
                AppLanguage.CHINESE -> "失败"
                AppLanguage.SPANISH -> "Falló"
                AppLanguage.GERMAN -> "Fehlgeschlagen"
                AppLanguage.PORTUGUESE -> "Falhou"
                AppLanguage.FRENCH -> "Échec"
                AppLanguage.ENGLISH -> "Failed"
            }
            "missing" -> when (language) {
                AppLanguage.CHINESE -> "缺失"
                AppLanguage.SPANISH -> "Falta"
                AppLanguage.GERMAN -> "Fehlt"
                AppLanguage.PORTUGUESE -> "Ausente"
                AppLanguage.FRENCH -> "Manquant"
                AppLanguage.ENGLISH -> "Missing"
            }
            else -> status
        }
    }

    fun localDownloadStatusLine(runtimeFlavor: String, status: String): String {
        return "$runtimeFlavor · ${localDownloadStatusLabel(status)}"
    }

    fun restartOnMobileData(): String = when (language) {
        AppLanguage.CHINESE -> "通过移动数据重新开始"
        AppLanguage.SPANISH -> "Reiniciar con datos móviles"
        AppLanguage.GERMAN -> "Über mobile Daten neu starten"
        AppLanguage.PORTUGUESE -> "Reiniciar com dados móveis"
        AppLanguage.FRENCH -> "Redémarrer via les données mobiles"
        AppLanguage.ENGLISH -> "Restart on mobile data"
    }

    fun openSystemDownloads(): String = when (language) {
        AppLanguage.CHINESE -> "打开系统下载"
        AppLanguage.SPANISH -> "Abrir descargas del sistema"
        AppLanguage.GERMAN -> "System-Downloads öffnen"
        AppLanguage.PORTUGUESE -> "Abrir downloads do sistema"
        AppLanguage.FRENCH -> "Ouvrir les téléchargements système"
        AppLanguage.ENGLISH -> "Open system Downloads"
    }

    fun quickLocalModelsTitle(): String = when (language) {
        AppLanguage.CHINESE -> "一键本地模型"
        AppLanguage.SPANISH -> "Modelos locales con un toque"
        AppLanguage.GERMAN -> "Lokale Modelle mit einem Tipp"
        AppLanguage.PORTUGUESE -> "Modelos locais com um toque"
        AppLanguage.FRENCH -> "Modèles locaux en un geste"
        AppLanguage.ENGLISH -> "One-tap local models"
    }

    fun quickLocalModelsDescription(): String = when (language) {
        AppLanguage.CHINESE -> "选择已验证的移动模型。Hermes 会下载、设为首选，并在文件准备好后自动启动本地运行时。"
        AppLanguage.SPANISH -> "Elige un modelo móvil validado. Hermes lo descarga, lo marca como preferido e inicia el runtime local cuando el archivo está listo."
        AppLanguage.GERMAN -> "Wähle ein validiertes Mobilmodell. Hermes lädt es, markiert es als bevorzugt und startet die lokale Laufzeit, sobald die Datei bereit ist."
        AppLanguage.PORTUGUESE -> "Escolha um modelo móvel validado. O Hermes baixa, marca como preferido e inicia o runtime local quando o arquivo estiver pronto."
        AppLanguage.FRENCH -> "Choisissez un modèle mobile validé. Hermes le télécharge, le marque comme préféré et démarre le runtime local dès que le fichier est prêt."
        AppLanguage.ENGLISH -> "Choose a validated mobile model. Hermes downloads it, marks it preferred, and starts the local runtime when the file is ready."
    }

    fun detectedModelCatalogTitle(): String = when (language) {
        AppLanguage.CHINESE -> "已检测模型目录"
        AppLanguage.SPANISH -> "Catálogo de modelos detectados"
        AppLanguage.GERMAN -> "Erkannter Modellkatalog"
        AppLanguage.PORTUGUESE -> "Catálogo de modelos detectados"
        AppLanguage.FRENCH -> "Catalogue de modèles détectés"
        AppLanguage.ENGLISH -> "Detected model catalog"
    }

    fun detectedModelCatalogDescription(): String = when (language) {
        AppLanguage.CHINESE -> "从已签名的 Cloudflare 目录选择一个模型。Hermes 会验证签名，然后通过 Hugging Face 下载所选文件。"
        AppLanguage.SPANISH -> "Elige un modelo del catálogo firmado de Cloudflare. Hermes verifica la firma y descarga el archivo seleccionado desde Hugging Face."
        AppLanguage.GERMAN -> "Wähle ein Modell aus dem signierten Cloudflare-Katalog. Hermes prüft die Signatur und lädt die ausgewählte Datei von Hugging Face."
        AppLanguage.PORTUGUESE -> "Escolha um modelo do catálogo assinado da Cloudflare. O Hermes verifica a assinatura e baixa o arquivo selecionado pelo Hugging Face."
        AppLanguage.FRENCH -> "Choisissez un modèle dans le catalogue Cloudflare signé. Hermes vérifie la signature puis télécharge le fichier choisi depuis Hugging Face."
        AppLanguage.ENGLISH -> "Choose a model from the signed Cloudflare catalog. Hermes verifies the signature, then downloads the selected file from Hugging Face."
    }

    fun detectedModelDropdownPlaceholder(): String = when (language) {
        AppLanguage.CHINESE -> "选择检测到的模型"
        AppLanguage.SPANISH -> "Elegir modelo detectado"
        AppLanguage.GERMAN -> "Erkanntes Modell wählen"
        AppLanguage.PORTUGUESE -> "Escolher modelo detectado"
        AppLanguage.FRENCH -> "Choisir un modèle détecté"
        AppLanguage.ENGLISH -> "Choose detected model"
    }

    fun refreshCatalog(): String = when (language) {
        AppLanguage.CHINESE -> "刷新目录"
        AppLanguage.SPANISH -> "Actualizar catálogo"
        AppLanguage.GERMAN -> "Katalog aktualisieren"
        AppLanguage.PORTUGUESE -> "Atualizar catálogo"
        AppLanguage.FRENCH -> "Actualiser le catalogue"
        AppLanguage.ENGLISH -> "Refresh catalog"
    }

    fun downloadAndStart(): String = when (language) {
        AppLanguage.CHINESE -> "下载并启动"
        AppLanguage.SPANISH -> "Descargar e iniciar"
        AppLanguage.GERMAN -> "Herunterladen und starten"
        AppLanguage.PORTUGUESE -> "Baixar e iniciar"
        AppLanguage.FRENCH -> "Télécharger et démarrer"
        AppLanguage.ENGLISH -> "Download and start"
    }

    fun useAndStart(): String = when (language) {
        AppLanguage.CHINESE -> "使用并启动"
        AppLanguage.SPANISH -> "Usar e iniciar"
        AppLanguage.GERMAN -> "Verwenden und starten"
        AppLanguage.PORTUGUESE -> "Usar e iniciar"
        AppLanguage.FRENCH -> "Utiliser et démarrer"
        AppLanguage.ENGLISH -> "Use and start"
    }

    fun startRuntime(): String = when (language) {
        AppLanguage.CHINESE -> "启动运行时"
        AppLanguage.SPANISH -> "Iniciar runtime"
        AppLanguage.GERMAN -> "Laufzeit starten"
        AppLanguage.PORTUGUESE -> "Iniciar runtime"
        AppLanguage.FRENCH -> "Démarrer le runtime"
        AppLanguage.ENGLISH -> "Start runtime"
    }

    fun remoteFallbackTitle(): String = when (language) {
        AppLanguage.CHINESE -> "远程备用"
        AppLanguage.SPANISH -> "Respaldo remoto"
        AppLanguage.GERMAN -> "Remote-Fallback"
        AppLanguage.PORTUGUESE -> "Fallback remoto"
        AppLanguage.FRENCH -> "Secours distant"
        AppLanguage.ENGLISH -> "Remote fallback"
    }

    fun remoteFallbackDescription(): String = when (language) {
        AppLanguage.CHINESE -> "本地模型不可用时，Hermes 可以使用远程 OpenAI 兼容提供商。点一个提供商即可填入常用默认值；设置会打开官方密钥或登录页面。"
        AppLanguage.SPANISH -> "Cuando no haya un modelo local disponible, Hermes puede usar un proveedor remoto compatible con OpenAI. Toca un proveedor para rellenar valores comunes; la configuración abre la página oficial de claves o inicio de sesión."
        AppLanguage.GERMAN -> "Wenn kein lokales Modell verfügbar ist, kann Hermes einen OpenAI-kompatiblen Remote-Anbieter nutzen. Tippe auf einen Anbieter, um Standardwerte einzutragen; die Einrichtung öffnet die offizielle Schlüssel- oder Anmeldeseite."
        AppLanguage.PORTUGUESE -> "Quando não houver modelo local disponível, o Hermes pode usar um provedor remoto compatível com OpenAI. Toque em um provedor para preencher padrões comuns; a configuração abre a página oficial de chaves ou login."
        AppLanguage.FRENCH -> "Quand aucun modèle local n’est disponible, Hermes peut utiliser un fournisseur distant compatible OpenAI. Touchez un fournisseur pour remplir les valeurs courantes ; la configuration ouvre la page officielle de clés ou de connexion."
        AppLanguage.ENGLISH -> "When no local model is available, Hermes can use a remote OpenAI-compatible provider. Tap a provider to fill common defaults; setup opens the official key or sign-in page."
    }

    fun remoteOnly(): String = when (language) {
        AppLanguage.CHINESE -> "仅远程"
        AppLanguage.SPANISH -> "Solo remoto"
        AppLanguage.GERMAN -> "Nur remote"
        AppLanguage.PORTUGUESE -> "Somente remoto"
        AppLanguage.FRENCH -> "Distant uniquement"
        AppLanguage.ENGLISH -> "Remote only"
    }

    fun authBaseUrlMustBeValid(): String = when (language) {
        AppLanguage.CHINESE -> "Corr3xt 基础 URL 必须是有效的 http(s) 地址"
        AppLanguage.SPANISH -> "La URL base de Corr3xt debe ser una URL http(s) válida"
        AppLanguage.GERMAN -> "Die Corr3xt-Basis-URL muss eine gültige http(s)-URL sein"
        AppLanguage.PORTUGUESE -> "A URL base do Corr3xt deve ser uma URL http(s) válida"
        AppLanguage.FRENCH -> "L’URL de base Corr3xt doit être une URL http(s) valide"
        AppLanguage.ENGLISH -> "Corr3xt base URL must be a valid http(s) URL"
    }

    fun authSavedBaseUrl(): String = when (language) {
        AppLanguage.CHINESE -> "已保存 Corr3xt 基础 URL"
        AppLanguage.SPANISH -> "URL base de Corr3xt guardada"
        AppLanguage.GERMAN -> "Corr3xt-Basis-URL gespeichert"
        AppLanguage.PORTUGUESE -> "URL base do Corr3xt salva"
        AppLanguage.FRENCH -> "URL de base Corr3xt enregistrée"
        AppLanguage.ENGLISH -> "Saved Corr3xt base URL"
    }

    fun authOpenedCorr3xt(label: String): String = when (language) {
        AppLanguage.CHINESE -> "已打开 Corr3xt 进行 $label 登录。如果浏览器卡住，请复制登录链接并粘贴到其他浏览器。"
        AppLanguage.SPANISH -> "Corr3xt abierto para iniciar sesión con $label. Si el navegador se queda bloqueado, copia la URL de inicio de sesión y pégala en otro navegador."
        AppLanguage.GERMAN -> "Corr3xt für die Anmeldung mit $label geöffnet. Wenn der Browser hängen bleibt, kopiere die Anmelde-URL und füge sie in einem anderen Browser ein."
        AppLanguage.PORTUGUESE -> "Corr3xt aberto para login com $label. Se o navegador travar, copie a URL de login e cole em outro navegador."
        AppLanguage.FRENCH -> "Corr3xt ouvert pour la connexion avec $label. Si le navigateur se bloque, copiez l’URL de connexion et collez-la dans un autre navigateur."
        AppLanguage.ENGLISH -> "Opened Corr3xt for $label sign-in. If your browser stalls, copy the sign-in URL and paste it into another browser."
    }

    fun languageSwitchedTo(label: String): String = when (language) {
        AppLanguage.CHINESE -> "界面语言已切换为 $label"
        AppLanguage.SPANISH -> "Idioma cambiado a $label"
        AppLanguage.GERMAN -> "Sprache auf $label umgestellt"
        AppLanguage.PORTUGUESE -> "Idioma alterado para $label"
        AppLanguage.FRENCH -> "Langue changée en $label"
        AppLanguage.ENGLISH -> "Language switched to $label"
    }

    fun authSignedInWith(label: String): String = when (language) {
        AppLanguage.CHINESE -> "已通过 $label 登录"
        AppLanguage.SPANISH -> "Sesión iniciada con $label"
        AppLanguage.GERMAN -> "Angemeldet mit $label"
        AppLanguage.PORTUGUESE -> "Sessão iniciada com $label"
        AppLanguage.FRENCH -> "Connecté avec $label"
        AppLanguage.ENGLISH -> "Signed in with $label"
    }
}

val LocalHermesStrings = staticCompositionLocalOf { hermesStringsFor(AppLanguage.ENGLISH) }

fun hermesStringsFor(language: AppLanguage): HermesStrings {
    return when (language) {
        AppLanguage.CHINESE -> HermesStrings(
            language = language,
            alphaBadge = "ALPHA",
            sectionHermes = "Hermes",
            sectionAccounts = "账户",
            sectionPortal = "Portal",
            sectionDevice = "设备",
            sectionSettings = "设置",
            subtitleHermes = "聊天、命令与语音",
            subtitleAccounts = "Corr3xt 登录与提供商访问",
            subtitlePortal = "Portal 预览与浏览器回退",
            subtitleDevice = "文件、Linux 套件与手机控制",
            subtitleSettings = "运行时提供商与 API 配置",
            runtimeSetupAndOnboarding = "运行时设置与引导",
            openPageActions = "打开页面操作",
            hermesLogoDescription = "Hermes 标志",
            settingsNewHereTitle = "首次使用？",
            settingsHelpStart = "如果你已经有 API 密钥，请先从 OpenRouter 或其他 API 提供商开始。",
            settingsHelpAccounts = "如果你想使用邮箱、电话或 Google 的 Corr3xt 应用登录流程，请使用账户页面；提供商密钥保留在设置中。",
            appLanguageTitle = "应用语言",
            appLanguageDescription = "轻点旗帜即可立即保存并切换应用语言。",
            onDeviceInferenceTitle = "端侧推理",
            onDeviceInferenceDescription = "选择一个本地推理后端，让 Hermes 在手机上运行模型。",
            llamaCppLabel = "llama.cpp (GGUF)",
            llamaCppDescription = "使用嵌入式 Linux 套件和 GGUF 模型运行本地代理。",
            liteRtLmLabel = "LiteRT-LM",
            liteRtLmDescription = "使用 Google 的 LiteRT-LM Android 运行时加载 .litertlm 模型。",
            noCompatibleLocalModel = "尚未选择兼容的本地模型。请先下载并设为首选模型。",
            chatTitle = "Hermes 聊天",
            openHistory = "打开历史记录",
            history = "历史记录",
            newChat = "新聊天",
            backToChat = "返回聊天",
            clearConversation = "清空对话",
            speakLastReply = "朗读上一条回复",
            welcomeToHermes = "欢迎使用 Hermes",
            welcomeDescription = "可使用聊天、语音输入，或 /help、/history、/provider、/signin 等原生命令。",
            accounts = "账户",
            settings = "设置",
            messageHermes = "向 Hermes 发送消息",
            send = "发送",
            authIntro = "Corr3xt 用于应用登录；提供商访问使用设置中的安全 API 密钥或令牌。",
            corr3xtAuthBaseUrl = "Corr3xt 认证基础 URL",
            saveAuthUrl = "保存认证 URL",
            refresh = "刷新",
            pendingCorr3xtSignIn = "等待中的 Corr3xt 登录",
            signIn = "登录",
            signOut = "退出登录",
            reconnect = "重新连接",
            hermesProviderPrefix = "Hermes 提供商",
            portalTitle = "Nous Portal",
            portalEmbeddedDescription = "该页面现在会自动加载嵌入式 Portal。使用右上角按钮全屏或还原，必要时回退到浏览器。",
            fullScreenPortal = "Portal 全屏",
            minimizePortal = "还原 Portal",
            openExternally = "在外部打开",
            refreshPortal = "刷新 Portal",
            localDownloadsTitle = "Hugging Face 本地模型下载",
            localDownloadsDescription = "直接把完整模型文件下载到手机，使用 Android 系统下载管理器保存进度，并在断网或重启后安全恢复。",
            dataSaverModeTitle = "省流模式",
            dataSaverModeDescription = "启用后，大型模型下载会等待 Wi‑Fi / 非计费网络，以尽量减少移动数据使用。",
            huggingFaceTokenOptional = "Hugging Face 令牌（可选）",
            saveToken = "保存令牌",
            refreshDownloads = "刷新下载",
            repoIdOrDirectUrl = "仓库 ID 或直接 URL",
            filePathInsideRepo = "仓库内文件路径",
            revision = "版本",
            runtimeTarget = "运行目标",
            inspect = "检查",
            download = "下载",
            downloadManagerTitle = "下载管理器",
            noLocalModelDownloadsYet = "还没有本地模型下载。",
            preferredLocalModel = "首选本地模型",
            setPreferred = "设为首选",
            remove = "移除",
        )
        AppLanguage.SPANISH -> HermesStrings(
            language = language,
            alphaBadge = "ALPHA",
            sectionHermes = "Hermes",
            sectionAccounts = "Cuentas",
            sectionPortal = "Portal",
            sectionDevice = "Dispositivo",
            sectionSettings = "Ajustes",
            subtitleHermes = "Chat, comandos y voz",
            subtitleAccounts = "Inicio de sesión Corr3xt y acceso a proveedores",
            subtitlePortal = "Vista previa del portal y apertura en navegador",
            subtitleDevice = "Archivos, suite Linux y controles del teléfono",
            subtitleSettings = "Proveedor de runtime y configuración de API",
            runtimeSetupAndOnboarding = "Configuración del runtime y bienvenida",
            openPageActions = "Abrir acciones de la página",
            hermesLogoDescription = "Logo de Hermes",
            settingsNewHereTitle = "¿Nuevo aquí?",
            settingsHelpStart = "Empieza con OpenRouter u otro proveedor con API si ya tienes una clave.",
            settingsHelpAccounts = "Usa Cuentas para flujos Corr3xt de la app con correo, teléfono o Google; las claves de proveedores quedan en Ajustes.",
            appLanguageTitle = "Idioma de la app",
            appLanguageDescription = "Toca una bandera para guardar y cambiar el idioma al instante.",
            onDeviceInferenceTitle = "Inferencia en el dispositivo",
            onDeviceInferenceDescription = "Elige un backend local para que Hermes ejecute modelos en el teléfono.",
            llamaCppLabel = "llama.cpp (GGUF)",
            llamaCppDescription = "Ejecuta el agente local con la suite Linux integrada y modelos GGUF.",
            liteRtLmLabel = "LiteRT-LM",
            liteRtLmDescription = "Carga modelos .litertlm con el runtime Android de LiteRT-LM de Google.",
            noCompatibleLocalModel = "Aún no hay un modelo local compatible seleccionado. Descárgalo y márcalo como preferido primero.",
            chatTitle = "Chat de Hermes",
            openHistory = "Abrir historial",
            history = "Historial",
            newChat = "Nuevo chat",
            backToChat = "Volver al chat",
            clearConversation = "Borrar conversación",
            speakLastReply = "Leer la última respuesta",
            welcomeToHermes = "Bienvenido a Hermes",
            welcomeDescription = "Usa el chat, la voz o comandos nativos como /help, /history, /provider y /signin.",
            accounts = "Cuentas",
            settings = "Ajustes",
            messageHermes = "Enviar mensaje a Hermes",
            send = "Enviar",
            authIntro = "Corr3xt se usa para iniciar sesión en la app; los proveedores usan claves API o tokens seguros en Ajustes.",
            corr3xtAuthBaseUrl = "URL base de autenticación Corr3xt",
            saveAuthUrl = "Guardar URL de autenticación",
            refresh = "Actualizar",
            pendingCorr3xtSignIn = "Inicio de sesión Corr3xt pendiente",
            signIn = "Iniciar sesión",
            signOut = "Cerrar sesión",
            reconnect = "Reconectar",
            hermesProviderPrefix = "Proveedor de Hermes",
            portalTitle = "Nous Portal",
            portalEmbeddedDescription = "El portal incrustado ahora se carga automáticamente aquí. Usa el botón superior derecho para maximizar o minimizar la vista previa, o abre el navegador si hace falta.",
            fullScreenPortal = "Portal a pantalla completa",
            minimizePortal = "Minimizar portal",
            openExternally = "Abrir fuera",
            refreshPortal = "Actualizar portal",
            localDownloadsTitle = "Descargas locales de modelos desde Hugging Face",
            localDownloadsDescription = "Descarga archivos completos del modelo al teléfono, conserva el progreso en el gestor de descargas de Android y reanuda con seguridad tras cortes de red o reinicios.",
            dataSaverModeTitle = "Modo ahorro de datos",
            dataSaverModeDescription = "Cuando está activo, las descargas grandes esperan Wi‑Fi / redes no medidas para minimizar el uso de datos móviles.",
            huggingFaceTokenOptional = "Token de Hugging Face (opcional)",
            saveToken = "Guardar token",
            refreshDownloads = "Actualizar descargas",
            repoIdOrDirectUrl = "ID del repositorio o URL directa",
            filePathInsideRepo = "Ruta del archivo dentro del repo",
            revision = "Revisión",
            runtimeTarget = "Objetivo de runtime",
            inspect = "Inspeccionar",
            download = "Descargar",
            downloadManagerTitle = "Gestor de descargas",
            noLocalModelDownloadsYet = "Todavía no hay descargas locales de modelos.",
            preferredLocalModel = "Modelo local preferido",
            setPreferred = "Marcar preferido",
            remove = "Eliminar",
        )
        AppLanguage.GERMAN -> HermesStrings(
            language = language,
            alphaBadge = "ALPHA",
            sectionHermes = "Hermes",
            sectionAccounts = "Konten",
            sectionPortal = "Portal",
            sectionDevice = "Gerät",
            sectionSettings = "Einstellungen",
            subtitleHermes = "Chat, Befehle und Sprache",
            subtitleAccounts = "Corr3xt-Anmeldung und Anbieterzugang",
            subtitlePortal = "Portal-Vorschau und Browser-Fallback",
            subtitleDevice = "Dateien, Linux-Suite und Telefonsteuerung",
            subtitleSettings = "Runtime-Anbieter und API-Konfiguration",
            runtimeSetupAndOnboarding = "Runtime-Einrichtung und Onboarding",
            openPageActions = "Seitenaktionen öffnen",
            hermesLogoDescription = "Hermes-Logo",
            settingsNewHereTitle = "Neu hier?",
            settingsHelpStart = "Beginne mit OpenRouter oder einem anderen API-Anbieter, wenn du bereits einen Schlüssel hast.",
            settingsHelpAccounts = "Nutze Konten für Corr3xt-App-Anmeldungen mit E-Mail, Telefon oder Google; Anbieter-Schlüssel bleiben in den Einstellungen.",
            appLanguageTitle = "App-Sprache",
            appLanguageDescription = "Tippe auf eine Flagge, um die Sprache sofort zu speichern und zu wechseln.",
            onDeviceInferenceTitle = "On-Device-Inferenz",
            onDeviceInferenceDescription = "Wähle ein lokales Backend, damit Hermes Modelle direkt auf dem Telefon ausführt.",
            llamaCppLabel = "llama.cpp (GGUF)",
            llamaCppDescription = "Führe den lokalen Agenten mit der eingebetteten Linux-Suite und GGUF-Modellen aus.",
            liteRtLmLabel = "LiteRT-LM",
            liteRtLmDescription = "Lade .litertlm-Modelle mit Googles LiteRT-LM-Android-Runtime.",
            noCompatibleLocalModel = "Noch kein kompatibles lokales Modell ausgewählt. Bitte zuerst herunterladen und als bevorzugt markieren.",
            chatTitle = "Hermes-Chat",
            openHistory = "Verlauf öffnen",
            history = "Verlauf",
            newChat = "Neuer Chat",
            backToChat = "Zurück zum Chat",
            clearConversation = "Unterhaltung leeren",
            speakLastReply = "Letzte Antwort vorlesen",
            welcomeToHermes = "Willkommen bei Hermes",
            welcomeDescription = "Nutze Chat, Spracheingabe oder native Befehle wie /help, /history, /provider und /signin.",
            accounts = "Konten",
            settings = "Einstellungen",
            messageHermes = "Hermes Nachricht senden",
            send = "Senden",
            authIntro = "Corr3xt wird für die App-Anmeldung genutzt; Anbieter verwenden sichere API-Schlüssel oder Tokens in den Einstellungen.",
            corr3xtAuthBaseUrl = "Corr3xt-Auth-Basis-URL",
            saveAuthUrl = "Auth-URL speichern",
            refresh = "Aktualisieren",
            pendingCorr3xtSignIn = "Ausstehende Corr3xt-Anmeldung",
            signIn = "Anmelden",
            signOut = "Abmelden",
            reconnect = "Neu verbinden",
            hermesProviderPrefix = "Hermes-Anbieter",
            portalTitle = "Nous Portal",
            portalEmbeddedDescription = "Das eingebettete Portal wird jetzt automatisch geladen. Nutze die Schaltfläche oben rechts zum Maximieren oder Minimieren oder wechsle bei Bedarf in den Browser.",
            fullScreenPortal = "Portal im Vollbild",
            minimizePortal = "Portal minimieren",
            openExternally = "Extern öffnen",
            refreshPortal = "Portal aktualisieren",
            localDownloadsTitle = "Lokale Modell-Downloads von Hugging Face",
            localDownloadsDescription = "Lade komplette Modelldateien direkt auf das Telefon, speichere den Fortschritt im Android-Downloadmanager und setze sicher nach Netzverlust oder Neustart fort.",
            dataSaverModeTitle = "Datensparmodus",
            dataSaverModeDescription = "Wenn aktiviert, warten große Downloads auf Wi‑Fi / ungedrosselte Netze, damit nur minimale mobile Daten verwendet werden.",
            huggingFaceTokenOptional = "Hugging Face Token (optional)",
            saveToken = "Token speichern",
            refreshDownloads = "Downloads aktualisieren",
            repoIdOrDirectUrl = "Repo-ID oder direkte URL",
            filePathInsideRepo = "Dateipfad im Repo",
            revision = "Revision",
            runtimeTarget = "Runtime-Ziel",
            inspect = "Prüfen",
            download = "Herunterladen",
            downloadManagerTitle = "Downloadmanager",
            noLocalModelDownloadsYet = "Noch keine lokalen Modell-Downloads.",
            preferredLocalModel = "Bevorzugtes lokales Modell",
            setPreferred = "Bevorzugen",
            remove = "Entfernen",
        )
        AppLanguage.PORTUGUESE -> HermesStrings(
            language = language,
            alphaBadge = "ALPHA",
            sectionHermes = "Hermes",
            sectionAccounts = "Contas",
            sectionPortal = "Portal",
            sectionDevice = "Dispositivo",
            sectionSettings = "Configurações",
            subtitleHermes = "Chat, comandos e voz",
            subtitleAccounts = "Login Corr3xt e acesso a provedores",
            subtitlePortal = "Prévia do portal e fallback no navegador",
            subtitleDevice = "Arquivos, suíte Linux e controles do telefone",
            subtitleSettings = "Provedor de runtime e configuração de API",
            runtimeSetupAndOnboarding = "Configuração do runtime e introdução",
            openPageActions = "Abrir ações da página",
            hermesLogoDescription = "Logo do Hermes",
            settingsNewHereTitle = "Novo por aqui?",
            settingsHelpStart = "Comece com OpenRouter ou outro provedor de API se você já tiver uma chave.",
            settingsHelpAccounts = "Use Contas para fluxos Corr3xt do app com e-mail, telefone ou Google; chaves de provedores ficam nas Configurações.",
            appLanguageTitle = "Idioma do app",
            appLanguageDescription = "Toque em uma bandeira para salvar e trocar o idioma imediatamente.",
            onDeviceInferenceTitle = "Inferência no dispositivo",
            onDeviceInferenceDescription = "Escolha um backend local para que o Hermes execute modelos no telefone.",
            llamaCppLabel = "llama.cpp (GGUF)",
            llamaCppDescription = "Execute o agente local com a suíte Linux integrada e modelos GGUF.",
            liteRtLmLabel = "LiteRT-LM",
            liteRtLmDescription = "Carregue modelos .litertlm com o runtime Android LiteRT-LM do Google.",
            noCompatibleLocalModel = "Ainda não existe um modelo local compatível selecionado. Baixe e marque um como preferido primeiro.",
            chatTitle = "Chat Hermes",
            openHistory = "Abrir histórico",
            history = "Histórico",
            newChat = "Novo chat",
            backToChat = "Voltar ao chat",
            clearConversation = "Limpar conversa",
            speakLastReply = "Ler última resposta",
            welcomeToHermes = "Bem-vindo ao Hermes",
            welcomeDescription = "Use o chat, entrada por voz ou comandos nativos como /help, /history, /provider e /signin.",
            accounts = "Contas",
            settings = "Configurações",
            messageHermes = "Mensagem para Hermes",
            send = "Enviar",
            authIntro = "O Corr3xt é usado para login no app; provedores usam chaves API ou tokens seguros nas Configurações.",
            corr3xtAuthBaseUrl = "URL base de autenticação Corr3xt",
            saveAuthUrl = "Salvar URL de autenticação",
            refresh = "Atualizar",
            pendingCorr3xtSignIn = "Login Corr3xt pendente",
            signIn = "Entrar",
            signOut = "Sair",
            reconnect = "Reconectar",
            hermesProviderPrefix = "Provedor Hermes",
            portalTitle = "Nous Portal",
            portalEmbeddedDescription = "O portal incorporado agora carrega automaticamente aqui. Use o botão no canto superior direito para maximizar ou minimizar a prévia, ou abra no navegador se precisar.",
            fullScreenPortal = "Portal em tela cheia",
            minimizePortal = "Minimizar portal",
            openExternally = "Abrir externamente",
            refreshPortal = "Atualizar portal",
            localDownloadsTitle = "Downloads locais de modelos do Hugging Face",
            localDownloadsDescription = "Baixe arquivos completos de modelos diretamente para o telefone, mantenha o progresso no gerenciador de downloads do Android e retome com segurança após queda de rede ou reinício.",
            dataSaverModeTitle = "Modo economia de dados",
            dataSaverModeDescription = "Quando ativado, downloads grandes aguardam Wi‑Fi / rede não tarifada para reduzir o uso de dados móveis.",
            huggingFaceTokenOptional = "Token do Hugging Face (opcional)",
            saveToken = "Salvar token",
            refreshDownloads = "Atualizar downloads",
            repoIdOrDirectUrl = "ID do repositório ou URL direta",
            filePathInsideRepo = "Caminho do arquivo no repositório",
            revision = "Revisão",
            runtimeTarget = "Alvo do runtime",
            inspect = "Inspecionar",
            download = "Baixar",
            downloadManagerTitle = "Gerenciador de downloads",
            noLocalModelDownloadsYet = "Ainda não há downloads locais de modelos.",
            preferredLocalModel = "Modelo local preferido",
            setPreferred = "Definir preferido",
            remove = "Remover",
        )
        AppLanguage.FRENCH -> HermesStrings(
            language = language,
            alphaBadge = "ALPHA",
            sectionHermes = "Hermes",
            sectionAccounts = "Comptes",
            sectionPortal = "Portal",
            sectionDevice = "Appareil",
            sectionSettings = "Réglages",
            subtitleHermes = "Chat, commandes et voix",
            subtitleAccounts = "Connexion Corr3xt et accès aux fournisseurs",
            subtitlePortal = "Aperçu du portail et ouverture navigateur",
            subtitleDevice = "Fichiers, suite Linux et contrôles du téléphone",
            subtitleSettings = "Fournisseur de runtime et configuration API",
            runtimeSetupAndOnboarding = "Configuration du runtime et accueil",
            openPageActions = "Ouvrir les actions de la page",
            hermesLogoDescription = "Logo Hermes",
            settingsNewHereTitle = "Nouveau ici ?",
            settingsHelpStart = "Commencez avec OpenRouter ou un autre fournisseur API si vous avez déjà une clé.",
            settingsHelpAccounts = "Utilisez Comptes pour les flux Corr3xt de l’application avec e-mail, téléphone ou Google ; les clés fournisseur restent dans Paramètres.",
            appLanguageTitle = "Langue de l’application",
            appLanguageDescription = "Touchez un drapeau pour enregistrer et changer la langue immédiatement.",
            onDeviceInferenceTitle = "Inférence sur l’appareil",
            onDeviceInferenceDescription = "Choisissez un backend local pour que Hermes exécute des modèles sur le téléphone.",
            llamaCppLabel = "llama.cpp (GGUF)",
            llamaCppDescription = "Exécutez l’agent local avec la suite Linux intégrée et des modèles GGUF.",
            liteRtLmLabel = "LiteRT-LM",
            liteRtLmDescription = "Chargez des modèles .litertlm avec le runtime Android LiteRT-LM de Google.",
            noCompatibleLocalModel = "Aucun modèle local compatible n’est encore sélectionné. Téléchargez-en un puis marquez-le comme préféré.",
            chatTitle = "Chat Hermes",
            openHistory = "Ouvrir l’historique",
            history = "Historique",
            newChat = "Nouveau chat",
            backToChat = "Retour au chat",
            clearConversation = "Effacer la conversation",
            speakLastReply = "Lire la dernière réponse",
            welcomeToHermes = "Bienvenue dans Hermes",
            welcomeDescription = "Utilisez le chat, la voix ou des commandes natives comme /help, /history, /provider et /signin.",
            accounts = "Comptes",
            settings = "Réglages",
            messageHermes = "Message à Hermes",
            send = "Envoyer",
            authIntro = "Corr3xt sert à la connexion à l’application ; les fournisseurs utilisent des clés API ou jetons sécurisés dans Paramètres.",
            corr3xtAuthBaseUrl = "URL de base d’authentification Corr3xt",
            saveAuthUrl = "Enregistrer l’URL d’authentification",
            refresh = "Actualiser",
            pendingCorr3xtSignIn = "Connexion Corr3xt en attente",
            signIn = "Se connecter",
            signOut = "Se déconnecter",
            reconnect = "Reconnecter",
            hermesProviderPrefix = "Fournisseur Hermes",
            portalTitle = "Nous Portal",
            portalEmbeddedDescription = "Le portail intégré se charge maintenant automatiquement ici. Utilisez le bouton en haut à droite pour agrandir ou réduire l’aperçu, ou ouvrez le navigateur si nécessaire.",
            fullScreenPortal = "Portal plein écran",
            minimizePortal = "Réduire le portal",
            openExternally = "Ouvrir à l’extérieur",
            refreshPortal = "Actualiser le portal",
            localDownloadsTitle = "Téléchargements locaux de modèles depuis Hugging Face",
            localDownloadsDescription = "Téléchargez des fichiers de modèle complets directement sur le téléphone, conservez la progression dans le gestionnaire de téléchargements Android et reprenez en toute sécurité après une perte réseau ou un redémarrage.",
            dataSaverModeTitle = "Mode économie de données",
            dataSaverModeDescription = "Lorsqu’il est activé, les gros téléchargements attendent le Wi‑Fi / un réseau non limité afin de minimiser les données mobiles.",
            huggingFaceTokenOptional = "Jeton Hugging Face (optionnel)",
            saveToken = "Enregistrer le jeton",
            refreshDownloads = "Actualiser les téléchargements",
            repoIdOrDirectUrl = "ID du dépôt ou URL directe",
            filePathInsideRepo = "Chemin du fichier dans le dépôt",
            revision = "Révision",
            runtimeTarget = "Cible du runtime",
            inspect = "Inspecter",
            download = "Télécharger",
            downloadManagerTitle = "Gestionnaire de téléchargements",
            noLocalModelDownloadsYet = "Aucun téléchargement local de modèle pour l’instant.",
            preferredLocalModel = "Modèle local préféré",
            setPreferred = "Définir comme préféré",
            remove = "Supprimer",
        )
        AppLanguage.ENGLISH -> HermesStrings(
            language = language,
            alphaBadge = "ALPHA",
            sectionHermes = "Hermes",
            sectionAccounts = "Accounts",
            sectionPortal = "Portal",
            sectionDevice = "Device",
            sectionSettings = "Settings",
            subtitleHermes = "Chat, commands, and voice",
            subtitleAccounts = "Corr3xt sign-in and provider access",
            subtitlePortal = "Portal preview and browser fallback",
            subtitleDevice = "Files, Linux suite, and phone controls",
            subtitleSettings = "Runtime provider and API configuration",
            runtimeSetupAndOnboarding = "Runtime setup and onboarding",
            openPageActions = "Open page actions",
            hermesLogoDescription = "Hermes logo",
            settingsNewHereTitle = "New here?",
            settingsHelpStart = "Start with OpenRouter or another API provider if you already have a key.",
            settingsHelpAccounts = "Use Accounts for Corr3xt app sign-in with email, phone, or Google; keep provider keys in Settings.",
            appLanguageTitle = "App language",
            appLanguageDescription = "Tap a flag to save and switch the app language immediately.",
            onDeviceInferenceTitle = "On-device inference",
            onDeviceInferenceDescription = "Choose a local backend so Hermes can run models directly on the phone.",
            llamaCppLabel = "llama.cpp (GGUF)",
            llamaCppDescription = "Run the local agent with the embedded Linux suite and GGUF models.",
            liteRtLmLabel = "LiteRT-LM",
            liteRtLmDescription = "Load .litertlm models with Google’s LiteRT-LM Android runtime.",
            noCompatibleLocalModel = "No compatible local model is selected yet. Download one and mark it as preferred first.",
            chatTitle = "Hermes Chat",
            openHistory = "Open history",
            history = "History",
            newChat = "New chat",
            backToChat = "Back to chat",
            clearConversation = "Clear conversation",
            speakLastReply = "Speak last reply",
            welcomeToHermes = "Welcome to Hermes",
            welcomeDescription = "Use chat for normal prompts, voice input, or native app commands like /help, /history, /provider, and /signin.",
            accounts = "Accounts",
            settings = "Settings",
            messageHermes = "Message Hermes",
            send = "Send",
            authIntro = "Corr3xt is used for app sign-in; providers use secure API keys or tokens in Settings.",
            corr3xtAuthBaseUrl = "Corr3xt auth base URL",
            saveAuthUrl = "Save auth URL",
            refresh = "Refresh",
            pendingCorr3xtSignIn = "Pending Corr3xt sign-in",
            signIn = "Sign in",
            signOut = "Sign out",
            reconnect = "Reconnect",
            hermesProviderPrefix = "Hermes provider",
            portalTitle = "Nous Portal",
            portalEmbeddedDescription = "The embedded portal now auto-loads on this page. Use the top-right full screen button to maximize or minimize the preview, or fall back to the browser if verification gets stuck.",
            fullScreenPortal = "Full screen portal",
            minimizePortal = "Minimize portal",
            openExternally = "Open externally",
            refreshPortal = "Refresh portal",
            localDownloadsTitle = "Hugging Face local model downloads",
            localDownloadsDescription = "Download full model files directly to the phone, keep progress in Android’s system download manager, and resume safely after network loss or a phone restart.",
            dataSaverModeTitle = "Data saver mode",
            dataSaverModeDescription = "When enabled, large model downloads wait for Wi‑Fi / unmetered connectivity so Hermes uses only minimal mobile data.",
            huggingFaceTokenOptional = "Hugging Face token (optional)",
            saveToken = "Save token",
            refreshDownloads = "Refresh downloads",
            repoIdOrDirectUrl = "Repo ID or direct URL",
            filePathInsideRepo = "File path inside repo",
            revision = "Revision",
            runtimeTarget = "Runtime target",
            inspect = "Inspect",
            download = "Download",
            downloadManagerTitle = "Download manager",
            noLocalModelDownloadsYet = "No local model downloads yet.",
            preferredLocalModel = "Preferred local model",
            setPreferred = "Set preferred",
            remove = "Remove",
        )
    }
}
