package com.mobilefork.hermesagent

import android.app.Activity
import android.app.Application
import android.app.Instrumentation
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.Intents.intending
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.mobilefork.hermesagent.backend.BackendKind
import com.mobilefork.hermesagent.backend.HermesRuntimeManager
import com.mobilefork.hermesagent.backend.OnDeviceBackendManager
import com.mobilefork.hermesagent.auth.OpenRouterLoopbackOAuthServer
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.AuthSessionStore
import com.mobilefork.hermesagent.data.ConversationStore
import com.mobilefork.hermesagent.data.LocalModelDownloadStore
import com.mobilefork.hermesagent.data.StoredConversationAttachment
import com.mobilefork.hermesagent.data.StoredConversationMessage
import com.mobilefork.hermesagent.device.HermesProviderSetupWebActivity
import com.mobilefork.hermesagent.models.HermesModelDownloadManager
import com.mobilefork.hermesagent.ui.boot.BootUiState
import com.mobilefork.hermesagent.ui.settings.LocalModelDownloadsSection
import com.mobilefork.hermesagent.ui.settings.LocalModelDownloadsViewModel
import com.mobilefork.hermesagent.ui.shell.AppShellScreen
import com.mobilefork.hermesagent.ui.theme.HermesTheme
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.hamcrest.Description
import org.hamcrest.Matcher
import org.hamcrest.TypeSafeMatcher
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketException
import java.util.Collections
import java.util.concurrent.atomic.AtomicBoolean

@RunWith(AndroidJUnit4::class)
class DeepAppUiVisualInstrumentedTest {
    @get:Rule
    val composeRule = createComposeRule()

    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesRuntimeManager.stop()
        OnDeviceBackendManager.stopAll()
    }

    @Test
    fun settingsOneTapModelNavigationAndTranslationFlowCapturesScreenshots() {
        LocalModelDownloadStore(app).apply {
            saveDownloads(emptyList())
            setPreferredDownloadId("")
        }
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "visual-ui-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithText("Hermes Fork Chat").assertIsDisplayed()
        capture("01-hermes-chat")
        composeRule.onNodeWithTag("HermesChatInput").performTextInput("Describe the attached image and then summarize the phone status.")
        capture("02-hermes-typing")

        composeRule.onNodeWithTag("HermesNavSettings").performClick()
        composeRule.onAllNodesWithText("Settings")[0].assertIsDisplayed()
        capture("03-settings")
        composeRule.onNodeWithText("Theme and chat layout").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("ChatDisplayExpanded").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("CardShape-square").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("SaveAppearanceButton").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Check setup").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("LiteRtLmMtpMode-auto").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("LiteRtLmMtpMode-enabled").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("LiteRtLmMtpMode-disabled").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("One-tap local models").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Qwen3.5 0.8B Q4_K_M (GGUF)").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Gemma 4 E2B (LiteRT-LM)").performScrollTo().assertIsDisplayed()
        capture("04-one-tap-models")

        composeRule.onNodeWithText("🇪🇸 Español").performScrollTo().performClick()
        assertTrue(composeRule.onAllNodesWithText("Idioma de la app").fetchSemanticsNodes().isNotEmpty())
        composeRule.onNodeWithText("Modelos locales con un toque").performScrollTo()
        assertTrue(composeRule.onAllNodesWithText("Descargar e iniciar").fetchSemanticsNodes().isNotEmpty())
        capture("05-settings-spanish")

        composeRule.onNodeWithTag("HermesNavAccounts").performClick()
        composeRule.onAllNodesWithText("Cuentas")[0].assertIsDisplayed()
        capture("06-accounts-spanish")

        composeRule.onNodeWithTag("HermesNavDevice").performClick()
        composeRule.onAllNodesWithText("Dispositivo")[0].assertIsDisplayed()
        capture("07-device-spanish")

        composeRule.onNodeWithTag("HermesNavNousPortal").performClick()
        assertTrue(composeRule.onAllNodesWithText("Portal del proveedor").fetchSemanticsNodes().isNotEmpty())
        capture("08-portal-spanish")
    }

    @Test
    fun chatInputAcceptsHumanLikeTypingWithoutLosingComposerControls() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
                chatDisplayMode = "compact",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "human-like-typing-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        val promptChunks = listOf(
            "Set up a tiny demo app, ",
            "check the endpoint, ",
            "then report Qwen and Gemma status.",
        )
        composeRule.onNodeWithTag("HermesChatInput").performClick()
        promptChunks.forEach { chunk ->
            composeRule.onNodeWithTag("HermesChatInput").performTextInput(chunk)
            Thread.sleep(45L)
        }

        composeRule.onNodeWithText(
            "Set up a tiny demo app, check the endpoint, then report Qwen and Gemma status.",
        ).assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMicButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatSendButton").assertIsDisplayed()
        capture("13-human-like-typing")
    }

    @Test
    fun localModelImportButtonImportsPhoneFileAndMarksPreferredModel() {
        val sourceDir = File(app.filesDir, "hermes-home/import-fixtures").apply { mkdirs() }
        val sourceFile = File(sourceDir, "hermes-import-button-test.gguf").apply {
            writeText("HERMES_IMPORT_BUTTON_TEST")
        }
        val sourceUri = Uri.fromFile(sourceFile)
        val importedFile = HermesModelDownloadManager.modelsDirectory(app).resolve(sourceFile.name).apply { delete() }
        LocalModelDownloadStore(app).apply {
            saveDownloads(emptyList())
            setPreferredDownloadId("")
        }

        val viewModel = LocalModelDownloadsViewModel(app)
        val openDocumentLaunched = AtomicBoolean(false)
        composeRule.setContent {
            HermesTheme {
                LocalModelDownloadsSection(
                    dataSaverMode = false,
                    offlineAirplaneMode = true,
                    onDataSaverModeChange = {},
                    selectedBackend = BackendKind.LLAMA_CPP.persistedValue,
                    onRuntimeFlavorSelected = {},
                    onCompletedDownloadReady = {},
                    importModelClickOverride = {
                        openDocumentLaunched.set(true)
                        HermesModelDownloadManager.importLocalModelFile(
                            context = app,
                            store = LocalModelDownloadStore(app),
                            sourceUri = sourceUri,
                        )
                    },
                    viewModel = viewModel,
                )
            }
        }

        try {
            composeRule.onNodeWithTag("HermesImportModelButton").assertIsDisplayed().performClick()
            composeRule.waitUntil(timeoutMillis = 20_000) {
                val store = LocalModelDownloadStore(app)
                openDocumentLaunched.get() &&
                    store.loadDownloads().firstOrNull { it.id == store.preferredDownloadId() }?.destinationFileName == sourceFile.name
            }

            val store = LocalModelDownloadStore(app)
            val preferred = store.loadDownloads().firstOrNull { it.id == store.preferredDownloadId() }
            assertEquals(sourceFile.name, preferred?.destinationFileName)
            assertEquals("GGUF", preferred?.runtimeFlavor)
            assertEquals(sourceFile.length(), preferred?.totalBytes)
            assertTrue("Expected imported model copy at ${importedFile.absolutePath}", importedFile.isFile)
            assertEquals(sourceFile.readText(), importedFile.readText())
        } finally {
            importedFile.delete()
            sourceFile.delete()
        }
    }

    @Test
    fun compactChatModeCollapsesPromptAndExpandedModeToggleWorks() {
        val conversationStore = ConversationStore(app)
        conversationStore.clearAll()
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
                chatDisplayMode = "compact",
            )
        )
        val seededConversation = conversationStore.createNewConversation("Compact layout validation")
        val now = System.currentTimeMillis()
        conversationStore.upsertMessage(
            seededConversation.sessionId,
            StoredConversationMessage(
                id = "compact-user",
                role = "user",
                content = "/help\nUse the camera, file attachment, and voice input.",
                createdAtEpochMs = now,
                attachments = listOf(
                    StoredConversationAttachment(
                        uri = "content://hermes-test/attachment.png",
                        displayName = "attachment.png",
                        mimeType = "image/png",
                        sizeBytes = 1024L,
                    ),
                ),
            ),
        )
        conversationStore.upsertMessage(
            seededConversation.sessionId,
            StoredConversationMessage(
                id = "compact-assistant",
                role = "assistant",
                content = "Available app commands include /help, /history, /provider, and /signin. Camera, image upload, voice input, tool calls, skills, and agent actions are highlighted.",
                createdAtEpochMs = now + 1_000L,
            ),
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "compact-chat-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithTag("HermesChatHistoryButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatPageActionsButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatDisplayToggle").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesFloatingActionButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMicButton").assertIsDisplayed()
        capture("09-compact-floating-icon")

        composeRule.onNodeWithTag("HermesCompactChatTurn").performScrollTo()
        composeRule.onNodeWithText("Available app commands", substring = true).performScrollTo()

        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").performClick()
        composeRule.onNodeWithTag("HermesChatComposerActions").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatAttachImageButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatCameraButton").assertIsDisplayed()
        capture("10-compact-action-tray")
        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").performClick()
        composeRule.waitForIdle()

        composeRule.onNodeWithTag("HermesChatDisplayToggle").performClick()
        composeRule.waitUntil(timeoutMillis = 5_000) {
            AppSettingsStore(app).load().chatDisplayMode == "expanded"
        }
        assertEquals("expanded", AppSettingsStore(app).load().chatDisplayMode)
        composeRule.waitUntil(timeoutMillis = 5_000) {
            composeRule.onAllNodesWithText("You").fetchSemanticsNodes().isNotEmpty()
        }
        composeRule.onNodeWithText("You").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatHistoryButton").performClick()
        composeRule.onNodeWithText("Conversation history").assertIsDisplayed()
    }

    @Test
    fun compactControlsRemainReachableOnNarrowScreens() {
        val conversationStore = ConversationStore(app)
        conversationStore.clearAll()
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
                chatDisplayMode = "compact",
            )
        )
        val seededConversation = conversationStore.createNewConversation("Narrow controls validation")
        val now = System.currentTimeMillis()
        conversationStore.upsertMessage(
            seededConversation.sessionId,
            StoredConversationMessage(
                id = "narrow-user",
                role = "user",
                content = "/help",
                createdAtEpochMs = now,
            ),
        )
        conversationStore.upsertMessage(
            seededConversation.sessionId,
            StoredConversationMessage(
                id = "narrow-assistant",
                role = "assistant",
                content = "Available app commands include camera, image upload, voice input, tool calls, skills, and agent actions.",
                createdAtEpochMs = now + 1_000L,
            ),
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "narrow-controls-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithTag("HermesChatHistoryButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatPageActionsButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatDisplayToggle").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesFloatingActionButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMicButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatSendButton").assertIsDisplayed()

        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").performClick()
        composeRule.onNodeWithTag("HermesChatComposerActions").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatAttachImageButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatCameraButton").assertIsDisplayed()
        capture("11-narrow-controls")
    }

    @Test
    fun ultraNarrowComposerControlsRemainReachableOnTinyScreens() {
        org.junit.Assume.assumeTrue(
            "Run with a screen width below 220dp to exercise the ultra-narrow composer.",
            app.resources.configuration.screenWidthDp < 220,
        )
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
                chatDisplayMode = "compact",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "ultra-narrow-controls-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithTag("HermesChatInput").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatComposerUltraNarrowControls").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatMicButton").assertIsDisplayed()
        composeRule.onNodeWithTag("HermesChatSendButton").assertIsDisplayed()

        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").performClick()
        composeRule.onNodeWithTag("HermesChatComposerActions").assertIsDisplayed()
        capture("14-ultra-narrow-controls")
        composeRule.onNodeWithTag("HermesChatMoreInputActionsButton").performClick()
        composeRule.waitForIdle()

        composeRule.onNodeWithTag("HermesChatInput").performClick()
        composeRule.onNodeWithTag("HermesChatInput").performTextInput("tiny phone check")
        composeRule.onNodeWithTag("HermesChatInput").assertTextContains("tiny phone check")
        assertTrue(
            "Bottom navigation should collapse while the keyboard is open on tiny screens",
            composeRule.onAllNodesWithTag("HermesNavSettings").fetchSemanticsNodes().isEmpty(),
        )
        assertTrue(
            "Floating chat icon should collapse before it can overlap tiny composer controls",
            composeRule.onAllNodesWithTag("HermesFloatingActionButton").fetchSemanticsNodes().isEmpty(),
        )
        capture("15-ultra-narrow-keyboard")
    }

    @Test
    fun customEndpointDebugPreviewNormalizesPastedUrlInSettings() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "custom",
                baseUrl = "localhost:11434/v1/chat/completions?debug=true",
                model = "qwen-local",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "custom-endpoint-preview-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithTag("HermesNavSettings").performClick()
        composeRule.onNodeWithTag("HermesEndpointDebugPreview").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText(
            "Hermes will try: http://localhost:11434/v1/chat/completions",
        ).performScrollTo().assertIsDisplayed()
        capture("12-custom-endpoint-debug-preview")
    }

    @Test
    fun signinOpenRouterCommandOpensOpenRouterOAuthPage() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "signin-openrouter-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        val openRouterOAuthOpened = AtomicBoolean(false)
        val openRouterOAuthIntent = object : TypeSafeMatcher<Intent>() {
            override fun describeTo(description: Description) {
                description.appendText("OpenRouter OAuth browser intent")
            }

            override fun matchesSafely(intent: Intent): Boolean {
                val chooserTarget = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
                val uri = chooserTarget?.data ?: return false
                val callbackUrl = Uri.parse(uri.getQueryParameter("callback_url").orEmpty())
                val matches = intent.action == Intent.ACTION_CHOOSER &&
                    chooserTarget.action == Intent.ACTION_VIEW &&
                    chooserTarget.`package` == null &&
                    uri.scheme == "https" &&
                    uri.host == "openrouter.ai" &&
                    uri.path == "/auth" &&
                    uri.getQueryParameter("code_challenge_method") == "S256" &&
                    callbackUrl.scheme == "http" &&
                    callbackUrl.host == "localhost" &&
                    callbackUrl.port == OpenRouterLoopbackOAuthServer.DEFAULT_PORT &&
                    callbackUrl.path == "/hermes/openrouter/callback" &&
                    callbackUrl.getQueryParameter("method") == "openrouter"
                if (matches) {
                    openRouterOAuthOpened.set(true)
                }
                return matches
            }
        }
        Intents.init()
        try {
            intending(openRouterOAuthIntent).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, null))

            composeRule.onNodeWithTag("HermesChatInput").performTextInput("/signin openrouter")
            composeRule.onNodeWithText("Send").performClick()

            composeRule.waitUntil(timeoutMillis = 10_000) { openRouterOAuthOpened.get() }
            val pending = AuthSessionStore(app).loadPendingRequest()
            assertEquals("openrouter", pending?.methodId)
            assertEquals("openrouter-oauth", pending?.authProvider)
            assertEquals("S256", pending?.codeChallengeMethod)
            assertTrue(pending?.codeVerifier.orEmpty().isNotBlank())
            assertTrue(pending?.startUrl.orEmpty().startsWith("https://openrouter.ai/auth"))
        } finally {
            Intents.release()
        }
    }

    @Test
    fun signinQwenCommandOpensSetupPageAndReloadsSettingsProviderProfile() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "signin-qwen-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        val qwenSetupOpened = AtomicBoolean(false)
        val qwenSetupIntent = providerSetupOpenFor(Uri.parse("https://docs.qwencloud.com/developer-guides/administration/api-keys")) {
            qwenSetupOpened.set(true)
        }
        Intents.init()
        try {
            intending(qwenSetupIntent).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, null))

            composeRule.onNodeWithTag("HermesChatInput").performTextInput("/signin qwen")
            composeRule.onNodeWithText("Send").performClick()

            composeRule.waitUntil(timeoutMillis = 10_000) { qwenSetupOpened.get() }
            composeRule.waitUntil(timeoutMillis = 10_000) {
                AppSettingsStore(app).load().provider == "alibaba"
            }

            val settings = AppSettingsStore(app).load()
            assertEquals("alibaba", settings.provider)
            assertEquals("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", settings.baseUrl)
            assertEquals("qwen3.6-plus", settings.model)
        } finally {
            Intents.release()
        }
    }

    @Test
    fun signinOpenAiCommandOpensOpenAiSetupPageAndReloadsSettingsProviderProfile() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "signin-openai-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        val openAiSetupOpened = AtomicBoolean(false)
        val openAiSetupIntent = providerSetupOpenFor(
            Uri.parse("https://platform.openai.com/settings/organization/api-keys")
        ) {
            openAiSetupOpened.set(true)
        }
        Intents.init()
        try {
            intending(openAiSetupIntent).respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK, null))

            composeRule.onNodeWithTag("HermesChatInput").performTextInput("/signin openai")
            composeRule.onNodeWithText("Send").performClick()

            composeRule.waitUntil(timeoutMillis = 10_000) { openAiSetupOpened.get() }
            composeRule.waitUntil(timeoutMillis = 10_000) {
                AppSettingsStore(app).load().provider == "openai"
            }

            val settings = AppSettingsStore(app).load()
            assertEquals("openai", settings.provider)
            assertEquals("https://api.openai.com/v1", settings.baseUrl)
            assertEquals("gpt-4.1", settings.model)
        } finally {
            Intents.release()
        }
    }

    @Test
    fun accountsRuntimeProvidersExposeDirectSetupUrls() {
        AppSettingsStore(app).save(
            AppSettings(
                provider = "openrouter",
                baseUrl = "https://openrouter.ai/api/v1",
                model = "anthropic/claude-sonnet-4",
                onDeviceBackend = BackendKind.NONE.persistedValue,
                languageTag = "en",
            )
        )

        composeRule.setContent {
            AppShellScreen(
                bootUiState = BootUiState(
                    status = "Hermes backend is ready",
                    ready = true,
                    probeResult = "accounts-provider-setup-url-test",
                    baseUrl = "http://127.0.0.1:15436/v1",
                ),
                onRetryHermes = {},
            )
        }

        composeRule.onNodeWithTag("HermesNavAccounts").performClick()
        composeRule.onAllNodesWithText("Accounts")[0].assertIsDisplayed()
        composeRule.onNodeWithText("Qwen OAuth (legacy)").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderCredential-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderSaveCredential-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithText("Set up Qwen OAuth (legacy) API key").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderOpenSetup-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderCopySetup-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderCheckSetup-qwen-oauth").performScrollTo().assertIsDisplayed()
        composeRule.onNodeWithTag("AuthProviderCopySetup-qwen-oauth").performClick()
        composeRule.onNodeWithText("Copied Qwen OAuth (legacy) setup URL and 5 alternate official pages.").performScrollTo().assertIsDisplayed()
    }

    @Test
    fun corr3xtSignInRejectsReachableHostWithoutOAuthStartRoute() {
        val server = TestHttpServer { target -> if (target == "/") 200 else 404 }
        try {
            AppSettingsStore(app).save(
                AppSettings(
                    provider = "openrouter",
                    baseUrl = "https://openrouter.ai/api/v1",
                    model = "anthropic/claude-sonnet-4",
                    corr3xtBaseUrl = "http://127.0.0.1:${server.port}",
                    onDeviceBackend = BackendKind.NONE.persistedValue,
                    languageTag = "en",
                )
            )

            composeRule.setContent {
                AppShellScreen(
                    bootUiState = BootUiState(
                        status = "Hermes backend is ready",
                        ready = true,
                        probeResult = "corr3xt-route-test",
                        baseUrl = "http://127.0.0.1:15436/v1",
                    ),
                    onRetryHermes = {},
                )
            }

            composeRule.onNodeWithTag("HermesNavAccounts").performClick()
            composeRule.onAllNodesWithText("Accounts")[0].assertIsDisplayed()
            composeRule.onNodeWithTag("AuthSignIn-phone").performScrollTo().performClick()
            composeRule.waitUntil(timeoutMillis = 15_000) {
                composeRule.onAllNodesWithText(
                    "Corr3xt app sign-in page could not be reached: HTTP 404",
                    substring = true,
                ).fetchSemanticsNodes().isNotEmpty()
            }
            assertTrue(server.seenRequests().contains("/oauth/start"))
        } finally {
            server.close()
        }
    }

    private fun capture(name: String) {
        val outputDir = File(app.filesDir, "hermes-ui-visuals").apply { mkdirs() }
        val outputFile = File(outputDir, "$name.png")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val bitmap = takeVisibleScreenshot(instrumentation)
        if (bitmap != null) {
            assertTrue("Hermes UI screenshot $name appears blank", screenshotHasVisibleContent(bitmap))
            FileOutputStream(outputFile).use { output ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)
            }
            bitmap.recycle()
            return
        }
        val descriptor = instrumentation.uiAutomation.executeShellCommand("screencap -p")
        android.os.ParcelFileDescriptor.AutoCloseInputStream(descriptor).use { input ->
            FileOutputStream(outputFile).use { output ->
                input.copyTo(output)
            }
        }
        assertTrue("Failed to capture Hermes UI screenshot $name", outputFile.length() > 0L)
    }

    private fun takeVisibleScreenshot(instrumentation: Instrumentation): Bitmap? {
        for (attempt in 1..3) {
            composeRule.waitForIdle()
            instrumentation.waitForIdleSync()
            Thread.sleep(150L)
            val candidate = instrumentation.uiAutomation.takeScreenshot() ?: continue
            if (screenshotHasVisibleContent(candidate) || attempt == 3) {
                return candidate
            }
            candidate.recycle()
        }
        return null
    }

    private fun screenshotHasVisibleContent(bitmap: Bitmap): Boolean {
        val stepX = maxOf(1, bitmap.width / 48)
        val stepY = maxOf(1, bitmap.height / 48)
        var visibleSamples = 0
        for (y in 0 until bitmap.height step stepY) {
            for (x in 0 until bitmap.width step stepX) {
                val pixel = bitmap.getPixel(x, y)
                if (Color.alpha(pixel) == 0) continue
                val red = Color.red(pixel)
                val green = Color.green(pixel)
                val blue = Color.blue(pixel)
                val high = maxOf(red, green, blue)
                val low = minOf(red, green, blue)
                if (high > 42 || high - low > 14) {
                    visibleSamples += 1
                }
            }
        }
        return visibleSamples > 64
    }

    private fun providerSetupOpenFor(uri: Uri, onMatch: (() -> Unit)? = null): Matcher<Intent> {
        return object : TypeSafeMatcher<Intent>() {
            override fun describeTo(description: Description) {
                description.appendText("Hermes provider setup intent for ").appendValue(uri)
            }

            override fun matchesSafely(intent: Intent): Boolean {
                val chooserTarget = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
                val matches = (
                    intent.action == Intent.ACTION_CHOOSER &&
                        chooserTarget?.action == Intent.ACTION_VIEW &&
                        chooserTarget.data == uri &&
                        chooserTarget.`package` == null
                    ) ||
                    (
                        intent.component?.className == HermesProviderSetupWebActivity::class.java.name &&
                            intent.getStringExtra(PROVIDER_SETUP_URL_EXTRA) == uri.toString()
                        )
                if (matches) {
                    onMatch?.invoke()
                }
                return matches
            }
        }
    }

    private companion object {
        private const val PROVIDER_SETUP_URL_EXTRA = "com.mobilefork.hermesagent.PROVIDER_SETUP_URL"
    }

    private class TestHttpServer(private val responseCodeForTarget: (String) -> Int) : AutoCloseable {
        private val serverSocket = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))
        private val requests = Collections.synchronizedList(mutableListOf<String>())
        private val thread = Thread {
            try {
                while (!serverSocket.isClosed) {
                    val socket = serverSocket.accept()
                    socket.use { client ->
                        val reader = BufferedReader(InputStreamReader(client.getInputStream()))
                        val requestLine = reader.readLine().orEmpty()
                        val target = requestLine.split(" ").getOrNull(1).orEmpty()
                        requests.add(target)
                        val code = responseCodeForTarget(target)
                        val reason = when (code) {
                            200 -> "OK"
                            400 -> "Bad Request"
                            404 -> "Not Found"
                            else -> "Status"
                        }
                        client.getOutputStream().write(
                            "HTTP/1.1 $code $reason\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                .toByteArray(Charsets.UTF_8)
                        )
                    }
                }
            } catch (_: SocketException) {
                // Closing the server socket stops the test server.
            }
        }.apply {
            isDaemon = true
            start()
        }

        val port: Int
            get() = serverSocket.localPort

        fun seenRequests(): List<String> = requests.toList()

        override fun close() {
            serverSocket.close()
            thread.join(1_000)
        }
    }
}
