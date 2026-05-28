package com.mobilefork.hermesagent.ui.chat

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.text.format.DateFormat
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.R
import com.mobilefork.hermesagent.data.ProviderPresets
import com.mobilefork.hermesagent.ui.auth.AuthViewModel
import com.mobilefork.hermesagent.ui.i18n.LocalHermesStrings
import com.mobilefork.hermesagent.ui.settings.SettingsViewModel
import com.mobilefork.hermesagent.ui.shell.AppSection
import com.mobilefork.hermesagent.ui.shell.ShellActionItem
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun ChatScreen(
    modifier: Modifier = Modifier,
    viewModel: ChatViewModel = viewModel(),
    settingsViewModel: SettingsViewModel,
    authViewModel: AuthViewModel,
    onNavigateToSection: (AppSection) -> Unit,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
    onOpenContextActions: (() -> Unit)? = null,
) {
    val uiState by viewModel.uiState.collectAsState()
    val settingsState by settingsViewModel.uiState.collectAsState()
    val strings = LocalHermesStrings.current
    val context = LocalContext.current
    val listState = rememberLazyListState()
    val scrollScope = rememberCoroutineScope()
    val showScrollToBottom by remember {
        derivedStateOf {
            val totalItems = listState.layoutInfo.totalItemsCount
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
            totalItems > 0 && lastVisible < totalItems - 2
        }
    }
    var ttsController by remember(context) { mutableStateOf<HermesTtsController?>(null) }
    var composerActionMenuOpen by rememberSaveable { mutableStateOf(false) }

    DisposableEffect(context) {
        onDispose {
            ttsController?.shutdown()
        }
    }

    val speechLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        viewModel.setListening(false)
        if (result.resultCode != Activity.RESULT_OK) {
            viewModel.setStatus(strings.voiceInputCanceled())
            return@rememberLauncherForActivityResult
        }
        val transcript = SpeechInputController.extractBestResult(result.data)
        if (transcript.isNullOrBlank()) {
            viewModel.setStatus(strings.noSpeechCaptured())
        } else {
            viewModel.applyVoiceInput(transcript)
        }
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            viewModel.setListening(true)
            runCatching {
                speechLauncher.launch(SpeechInputController.buildIntent())
            }.getOrElse {
                viewModel.setListening(false)
                viewModel.setStatus(strings.voiceRecognitionUnavailable())
            }
        } else {
            viewModel.setListening(false)
            viewModel.setStatus(strings.microphonePermissionRequired())
        }
    }
    val imageLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            viewModel.attachImage(uri.toString())
        }
    }
    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        if (bitmap == null) {
            viewModel.setStatus(strings.cameraCaptureCanceled())
        } else {
            runCatching {
                persistCameraPreview(context, bitmap)
            }.onSuccess { uri ->
                viewModel.attachImage(uri.toString())
            }.onFailure { error ->
                viewModel.setStatus(strings.cameraAttachFailed(error.message ?: "unknown error"))
            }
        }
    }

    fun speak(text: String): Boolean {
        val controller = ttsController ?: HermesTtsController(context).also { ttsController = it }
        val worked = controller.speak(text)
        if (!worked) {
            viewModel.setStatus(strings.speechPlaybackNotReady())
        }
        return worked
    }

    fun startVoiceInput() {
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
        if (granted) {
            viewModel.setListening(true)
            try {
                speechLauncher.launch(SpeechInputController.buildIntent())
            } catch (_: ActivityNotFoundException) {
                viewModel.setListening(false)
                viewModel.setStatus(strings.voiceRecognitionUnavailable())
            }
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    fun applyProvider(providerId: String): Boolean {
        val preset = ProviderPresets.find(providerId) ?: return false
        settingsViewModel.updateProvider(preset.id)
        settingsViewModel.updateBaseUrl(preset.baseUrl)
        settingsViewModel.updateModel(preset.modelHint)
        settingsViewModel.save()
        return true
    }

    fun applyModel(modelName: String): Boolean {
        if (modelName.isBlank()) return false
        settingsViewModel.updateModel(modelName)
        settingsViewModel.save()
        return true
    }

    fun startAuthMethod(methodId: String): Boolean {
        val supported = setOf("openrouter", "openai", "chatgpt", "claude", "gemini", "qwen", "qwen-coding-plan", "qwen-oauth", "zai", "google", "email", "phone")
        if (methodId !in supported) return false
        return authViewModel.startAuth(methodId)
    }

    val shellActions = remember(strings.language, uiState.isShowingHistory, uiState.messages, uiState.activeConversationTitle) {
        if (uiState.isShowingHistory) {
            listOf(
                ShellActionItem(
                    label = strings.newChat.ifBlank { "New chat" },
                    description = strings.newChatActionDescription(),
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::startNewConversation,
                ),
                ShellActionItem(
                    label = strings.backToChat.ifBlank { "Back to chat" },
                    description = strings.backToChatActionDescription(),
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::hideHistory,
                ),
            )
        } else {
            listOf(
                ShellActionItem(
                    label = strings.history.ifBlank { "History" },
                    description = strings.historyActionDescription(),
                    iconRes = R.drawable.ic_action_history,
                    onClick = viewModel::showHistory,
                ),
                ShellActionItem(
                    label = strings.newChat.ifBlank { "New chat" },
                    description = strings.newChatInlineActionDescription(),
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::startNewConversation,
                ),
                ShellActionItem(
                    label = strings.clearConversation.ifBlank { "Clear conversation" },
                    description = strings.clearConversationActionDescription(),
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = viewModel::clearCurrentConversation,
                ),
                ShellActionItem(
                    label = strings.speakLastReply.ifBlank { "Speak last reply" },
                    description = strings.speakLastReplyActionDescription(),
                    iconRes = R.drawable.ic_action_speaker,
                    onClick = { speak(viewModel.latestAssistantReply()) },
                ),
            )
        }
    }

    SideEffect {
        onContextActionsChanged(shellActions)
    }

    LaunchedEffect(uiState.messages.size, uiState.isShowingHistory, settingsState.chatDisplayMode) {
        if (!uiState.isShowingHistory && uiState.messages.isNotEmpty()) {
            val targetIndex = if (settingsState.chatDisplayMode == "compact") {
                buildChatTurns(uiState.messages).lastIndex
            } else {
                uiState.messages.lastIndex
            }
            listState.animateScrollToItem(targetIndex.coerceAtLeast(0))
        }
    }

    fun handleSend() {
        val input = uiState.input.trim()
        if (input.isEmpty() && uiState.attachments.isEmpty()) return
        if (input.isEmpty()) {
            viewModel.sendMessage()
            return
        }
        val commandResult = ChatCommandRouter.execute(
            rawInput = input,
            host = ChatCommandHost(
                openHistory = viewModel::showHistory,
                newConversation = viewModel::startNewConversation,
                clearConversation = viewModel::clearCurrentConversation,
                navigateToSection = onNavigateToSection,
                applyProvider = ::applyProvider,
                applyModel = ::applyModel,
                startAuthMethod = ::startAuthMethod,
                speakLastReply = { speak(viewModel.latestAssistantReply()) },
            ),
            strings = strings,
        )
        if (commandResult.handled) {
            viewModel.consumeCommandResult(input, commandResult.feedback)
        } else {
            viewModel.sendMessage()
        }
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            BoxWithConstraints(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                val tinyVerticalViewport = maxHeight < 360.dp
                val tinyHorizontalViewport = maxWidth < 260.dp
                val tinyRuntimeViewport = tinyVerticalViewport || tinyHorizontalViewport
                val contentPadding = if (tinyRuntimeViewport) {
                    PaddingValues(horizontal = 4.dp, vertical = 4.dp)
                } else {
                    PaddingValues(horizontal = 12.dp, vertical = 8.dp)
                }
                val contentSpacing = if (tinyRuntimeViewport) 4.dp else 8.dp
                val showFloatingActionIcon = !uiState.isShowingHistory &&
                    uiState.messages.isNotEmpty() && !composerActionMenuOpen &&
                    !tinyVerticalViewport &&
                    !tinyHorizontalViewport
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .widthIn(max = 960.dp)
                        .padding(contentPadding),
                    verticalArrangement = Arrangement.spacedBy(contentSpacing),
                ) {
                if (!tinyVerticalViewport) {
                    ChatHeaderCard(
                        title = uiState.activeConversationTitle,
                        chatDisplayMode = settingsState.chatDisplayMode,
                        onOpenHistory = viewModel::showHistory,
                        onToggleDisplayMode = {
                            settingsViewModel.updateChatDisplayMode(
                                if (settingsState.chatDisplayMode == "compact") "expanded" else "compact",
                            )
                        },
                        onOpenActions = if (shellActions.isNotEmpty() && onOpenContextActions != null) {
                            {
                                onContextActionsChanged(shellActions)
                                onOpenContextActions()
                            }
                        } else {
                            null
                        },
                    )
                }
                if (uiState.status.isNotBlank() && !tinyRuntimeViewport) {
                    StatusBanner(text = uiState.status)
                }
                if (uiState.error.isNotBlank()) {
                    StatusBanner(text = uiState.error, isError = true)
                }
                if (uiState.isShowingHistory) {
                    ConversationHistoryList(
                        summaries = uiState.conversationSummaries,
                        onOpenConversation = viewModel::openConversation,
                        onStartNew = viewModel::startNewConversation,
                        modifier = Modifier.weight(1f),
                    )
                } else if (uiState.messages.isEmpty()) {
                    LazyColumn(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        contentPadding = PaddingValues(top = 24.dp, bottom = 8.dp),
                    ) {
                        item {
                            EmptyChatHint(
                                onNewChat = viewModel::startNewConversation,
                                onOpenAccounts = { onNavigateToSection(AppSection.Accounts) },
                                onOpenSettings = { onNavigateToSection(AppSection.Settings) },
                                onSignalQuickAction = { action -> viewModel.sendQuickPrompt(action.prompt) },
                            )
                        }
                    }
                } else {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                    ) {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            contentPadding = PaddingValues(bottom = 8.dp),
                        ) {
                            if (settingsState.chatDisplayMode == "expanded") {
                                itemsIndexed(uiState.messages, key = { _, message -> message.id }) { index, message ->
                                    val previous = uiState.messages.getOrNull(index - 1)
                                    ChatBubble(
                                        message = message,
                                        showTimestamp = previous == null ||
                                            minuteBucket(previous.createdAtEpochMs) != minuteBucket(message.createdAtEpochMs),
                                        keywordHighlightingEnabled = settingsState.keywordHighlightingEnabled,
                                        onSpeak = { speak(message.content) },
                                    )
                                }
                            } else {
                                val turns = buildChatTurns(uiState.messages)
                                items(turns, key = { it.id }) { turn ->
                                    CompactChatTurn(
                                        turn = turn,
                                        keywordHighlightingEnabled = settingsState.keywordHighlightingEnabled,
                                        onSpeak = { message -> speak(message.content) },
                                    )
                                }
                            }
                        }
                        if (showScrollToBottom) {
                            Surface(
                                modifier = Modifier
                                    .align(Alignment.BottomEnd)
                                    .padding(end = 8.dp, bottom = 8.dp)
                                    .size(38.dp)
                                    .clickable {
                                        scrollScope.launch {
                                            listState.animateScrollToItem((listState.layoutInfo.totalItemsCount - 1).coerceAtLeast(0))
                                        }
                                    },
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.18f),
                                shape = MaterialTheme.shapes.large,
                                tonalElevation = 1.dp,
                            ) {
                                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                    Text("↓", color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.titleMedium)
                                }
                            }
                        }
                    }
                }
                // The Activity uses adjustResize; adding imePadding here double-lifts the composer on phones.
                ChatComposer(
                    modifier = Modifier.fillMaxWidth(),
                    input = uiState.input,
                    attachments = uiState.attachments,
                    isSending = uiState.isSending,
                    isListening = uiState.isListening,
                    onInputChange = viewModel::updateInput,
                    onAttachImage = { imageLauncher.launch(arrayOf("image/*")) },
                    onCaptureImage = { cameraLauncher.launch(null) },
                    onRemoveAttachment = viewModel::removeAttachment,
                    onMic = ::startVoiceInput,
                    onSend = ::handleSend,
                    onActionMenuExpandedChange = { composerActionMenuOpen = it },
                    onSignalQuickAction = { action -> viewModel.sendQuickPrompt(action.prompt) },
                )
            }
                if (showFloatingActionIcon) {
                    HermesFloatingActionIcon(
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .padding(end = 16.dp, bottom = 92.dp),
                        onClick = {
                            if (shellActions.isNotEmpty() && onOpenContextActions != null) {
                                onContextActionsChanged(shellActions)
                                onOpenContextActions()
                            } else {
                                viewModel.showHistory()
                            }
                        },
                    )
                }
        }
    }
}
}

@Composable
private fun ChatHeaderCard(
    title: String,
    chatDisplayMode: String,
    onOpenHistory: () -> Unit,
    onToggleDisplayMode: () -> Unit,
    onOpenActions: (() -> Unit)? = null,
) {
    val strings = LocalHermesStrings.current
    val displayTitle = if (title.equals("New chat", ignoreCase = true)) strings.newChat else title
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shape = MaterialTheme.shapes.medium,
        tonalElevation = 1.dp,
    ) {
        BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
            val narrowHeader = maxWidth < 360.dp
            if (narrowHeader) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 8.dp, vertical = 7.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_nav_hermes),
                            contentDescription = strings.sectionHermes,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(22.dp),
                        )
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = strings.chatTitle.ifBlank { "Hermes Chat" },
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.SemiBold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                            Text(
                                text = displayTitle,
                                style = MaterialTheme.typography.labelSmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                        ChatHeaderHistoryButton(onOpenHistory = onOpenHistory)
                        if (onOpenActions != null) {
                            ChatHeaderPageActionsButton(onOpenActions = onOpenActions)
                        }
                    }
                    ChatHeaderDisplayModeButton(
                        chatDisplayMode = chatDisplayMode,
                        onToggleDisplayMode = onToggleDisplayMode,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            } else {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_nav_hermes),
                        contentDescription = strings.sectionHermes,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp),
                    )
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = strings.chatTitle.ifBlank { "Hermes Chat" },
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            text = displayTitle,
                            style = MaterialTheme.typography.labelSmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        ChatHeaderHistoryButton(onOpenHistory = onOpenHistory)
                        ChatHeaderDisplayModeButton(
                            chatDisplayMode = chatDisplayMode,
                            onToggleDisplayMode = onToggleDisplayMode,
                        )
                        if (onOpenActions != null) {
                            ChatHeaderPageActionsButton(onOpenActions = onOpenActions)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatHeaderHistoryButton(onOpenHistory: () -> Unit) {
    val strings = LocalHermesStrings.current
    IconButton(
        onClick = onOpenHistory,
        modifier = Modifier
            .size(40.dp)
            .testTag("HermesChatHistoryButton"),
    ) {
        Icon(
            painter = painterResource(id = R.drawable.ic_action_history),
            contentDescription = strings.openHistory.ifBlank { "Open history" },
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(20.dp),
        )
    }
}

@Composable
private fun ChatHeaderDisplayModeButton(
    chatDisplayMode: String,
    onToggleDisplayMode: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    Button(
        onClick = onToggleDisplayMode,
        modifier = modifier
            .heightIn(min = 36.dp)
            .testTag("HermesChatDisplayToggle"),
        shape = MaterialTheme.shapes.small,
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.16f),
            contentColor = MaterialTheme.colorScheme.primary,
        ),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 5.dp),
    ) {
        Text(
            text = strings.chatDisplayModeLabel(chatDisplayMode),
            style = MaterialTheme.typography.labelMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun ChatHeaderPageActionsButton(onOpenActions: () -> Unit) {
    val strings = LocalHermesStrings.current
    IconButton(
        onClick = onOpenActions,
        modifier = Modifier
            .size(40.dp)
            .testTag("HermesChatPageActionsButton"),
    ) {
        Icon(
            painter = painterResource(id = R.drawable.ic_action_cog),
            contentDescription = strings.openPageActions.ifBlank { "Open page actions" },
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(20.dp),
        )
    }
}

@Composable
private fun HermesFloatingActionIcon(
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    Surface(
        modifier = modifier
            .size(52.dp)
            .testTag("HermesFloatingActionButton")
            .clip(RoundedCornerShape(18.dp))
            .clickable(onClick = onClick),
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(18.dp),
        tonalElevation = 5.dp,
        shadowElevation = 4.dp,
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.linearGradient(
                        listOf(
                            MaterialTheme.colorScheme.primary.copy(alpha = 0.18f),
                            MaterialTheme.colorScheme.tertiary.copy(alpha = 0.10f),
                            MaterialTheme.colorScheme.surface,
                        ),
                    ),
                )
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.primary.copy(alpha = 0.22f),
                    shape = RoundedCornerShape(18.dp),
                ),
            contentAlignment = Alignment.Center,
        ) {
            Image(
                painter = painterResource(id = R.drawable.ic_hermes_floating_mark),
                contentDescription = strings.openPageActions.ifBlank { "Open Hermes actions" },
                modifier = Modifier.size(34.dp),
                contentScale = ContentScale.Fit,
            )
        }
    }
}

@Composable
private fun StatusBanner(text: String, isError: Boolean = false) {
    val strings = LocalHermesStrings.current
    val displayText = if (isError) text else strings.chatStatusText(text)
    val endpointStatus = isEndpointStatusText(displayText)
    val indicatorColor = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = if (isError) MaterialTheme.colorScheme.error.copy(alpha = 0.14f) else MaterialTheme.colorScheme.secondaryContainer,
        shape = MaterialTheme.shapes.medium,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Top,
        ) {
            if (endpointStatus) {
                Box(
                    modifier = Modifier
                        .padding(top = 3.dp)
                        .size(8.dp)
                        .clip(RoundedCornerShape(50))
                        .background(indicatorColor),
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                if (endpointStatus) {
                    Text(
                        text = strings.endpointStatusIndicatorLabel(),
                        color = indicatorColor,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                Text(
                    text = displayText,
                    color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSecondaryContainer,
                    style = MaterialTheme.typography.bodySmall,
                )
                if (endpointStatus && isError) {
                    Text(
                        text = strings.endpointStatusTroubleshootingHint(),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }
    }
}

private fun isEndpointStatusText(text: String): Boolean {
    val lower = text.lowercase()
    return listOf("endpoint", "stream", "sse", "http", "non-stream", "[done]", "base url", "model name").any { token ->
        lower.contains(token)
    } || (lower.contains("hermes is") && lower.contains(" via "))
}

@Composable
private fun EmptyChatHint(
    onNewChat: () -> Unit,
    onOpenAccounts: () -> Unit,
    onOpenSettings: () -> Unit,
    onSignalQuickAction: (SignalIntelligenceQuickAction) -> Unit,
) {
    val strings = LocalHermesStrings.current
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.medium,
        tonalElevation = 1.dp,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = strings.welcomeToHermes.ifBlank { "Welcome to Hermes" },
                style = MaterialTheme.typography.titleSmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = strings.welcomeDescription,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
            SignalIntelligenceQuickActionGrid(
                enabled = true,
                onSignalQuickAction = onSignalQuickAction,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    onClick = onNewChat,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(horizontal = 6.dp, vertical = 8.dp),
                ) {
                    Text(
                        text = strings.newChat.ifBlank { "New chat" },
                        style = MaterialTheme.typography.labelMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Button(
                    onClick = onOpenAccounts,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(horizontal = 6.dp, vertical = 8.dp),
                ) {
                    Text(
                        text = strings.accounts.ifBlank { "Accounts" },
                        style = MaterialTheme.typography.labelMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Button(
                    onClick = onOpenSettings,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(horizontal = 6.dp, vertical = 8.dp),
                ) {
                    Text(
                        text = strings.settings.ifBlank { "Settings" },
                        style = MaterialTheme.typography.labelMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }
    }
}

@Composable
private fun ChatBubble(
    message: ChatUiMessage,
    showTimestamp: Boolean = true,
    keywordHighlightingEnabled: Boolean,
    onSpeak: () -> Unit,
) {
    val isUser = message.role == "user"
    val containerColor = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    val contentColor = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant
    val strings = LocalHermesStrings.current
    val roleLabel = if (isUser) strings.userRoleLabel() else "Hermes"
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val bubbleMaxWidth = if (maxWidth < 760.dp) maxWidth * 0.88f else 640.dp
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
        ) {
            Surface(
                modifier = Modifier.widthIn(max = bubbleMaxWidth),
                color = containerColor,
                shape = RoundedCornerShape(
                    topStart = 22.dp,
                    topEnd = 22.dp,
                    bottomStart = if (isUser) 22.dp else 8.dp,
                    bottomEnd = if (isUser) 8.dp else 22.dp,
                ),
                tonalElevation = 1.dp,
            ) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(roleLabel, style = MaterialTheme.typography.labelLarge, color = contentColor)
                        if (showTimestamp) {
                            QuietMetaText(
                                text = DateFormat.format("HH:mm", message.createdAtEpochMs).toString(),
                                color = contentColor,
                            )
                        }
                    }
                    HighlightedMessageText(
                        text = message.content.ifBlank { "…" },
                        color = contentColor,
                        keywordHighlightingEnabled = keywordHighlightingEnabled,
                    )
                    AttachmentPreviewColumn(attachments = message.attachments, contentColor = contentColor)
                    if (!isUser && hasToolActivity(message.content)) {
                        CompactActivityRow(content = message.content, contentColor = contentColor)
                    }
                    if (!isUser && message.content.isNotBlank()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.End,
                        ) {
                            IconButton(onClick = onSpeak) {
                                Icon(
                                    painter = painterResource(id = R.drawable.ic_action_speaker),
                                    contentDescription = strings.speakReply(),
                                    tint = contentColor,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun CompactChatTurn(
    turn: ChatTurn,
    keywordHighlightingEnabled: Boolean,
    onSpeak: (ChatUiMessage) -> Unit,
) {
    var promptExpanded by rememberSaveable(turn.id) { mutableStateOf(false) }
    val userMessage = turn.userMessage
    val strings = LocalHermesStrings.current
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("HermesCompactChatTurn"),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.medium,
        tonalElevation = 1.dp,
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (userMessage != null) {
                CompactPromptHeader(
                    message = userMessage,
                    expanded = promptExpanded,
                    keywordHighlightingEnabled = keywordHighlightingEnabled,
                    onToggle = { promptExpanded = !promptExpanded },
                )
            }
            if (turn.assistantMessages.isEmpty()) {
                QuietMetaText(text = strings.hermesPreparingReply(), color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                turn.assistantMessages.forEachIndexed { index, assistantMessage ->
                    if (index == 0) {
                        Text(
                            text = "Hermes",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    HighlightedMessageText(
                        text = assistantMessage.content.ifBlank { "…" },
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        keywordHighlightingEnabled = keywordHighlightingEnabled,
                    )
                    AttachmentPreviewColumn(
                        attachments = assistantMessage.attachments,
                        contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    if (hasToolActivity(assistantMessage.content)) {
                        CompactActivityRow(
                            content = assistantMessage.content,
                            contentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            val metaMessage = turn.assistantMessages.lastOrNull() ?: userMessage
            if (metaMessage != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    QuietMetaText(
                        text = DateFormat.format("HH:mm", metaMessage.createdAtEpochMs).toString(),
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    if (turn.assistantMessages.any { it.content.isNotBlank() }) {
                        IconButton(
                            onClick = { turn.assistantMessages.lastOrNull { it.content.isNotBlank() }?.let(onSpeak) },
                            modifier = Modifier.size(34.dp),
                        ) {
                            Icon(
                                painter = painterResource(id = R.drawable.ic_action_speaker),
                                contentDescription = strings.speakReply(),
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp),
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun CompactPromptHeader(
    message: ChatUiMessage,
    expanded: Boolean,
    keywordHighlightingEnabled: Boolean,
    onToggle: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    val label = strings.compactPromptLabel(expanded)
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .testTag("HermesCompactPromptHeader"),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.78f),
        shape = MaterialTheme.shapes.medium,
        tonalElevation = 0.dp,
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (message.attachments.isNotEmpty()) {
                        QuietMetaText(
                            text = strings.attachmentCount(message.attachments.size),
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    Text(if (expanded) "▲" else "▼", color = MaterialTheme.colorScheme.primary)
                }
            }
            if (expanded) {
                HighlightedMessageText(
                    text = message.content.ifBlank { strings.attachmentOnlyPrompt() },
                    color = MaterialTheme.colorScheme.onSurface,
                    keywordHighlightingEnabled = keywordHighlightingEnabled,
                )
                AttachmentPreviewColumn(attachments = message.attachments, contentColor = MaterialTheme.colorScheme.onSurface)
            } else {
                Text(
                    text = shortPromptPreview(message.content),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.86f),
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun QuietMetaText(text: String, color: androidx.compose.ui.graphics.Color) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = color.copy(alpha = 0.64f),
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
    )
}

@Composable
private fun HighlightedMessageText(
    text: String,
    color: androidx.compose.ui.graphics.Color,
    keywordHighlightingEnabled: Boolean,
) {
    if (!keywordHighlightingEnabled || text.isBlank()) {
        Text(text = text, color = color, style = MaterialTheme.typography.bodyMedium)
        return
    }
    val pattern = remember {
        Regex(
            pattern = """(?i)(/help|/history|/provider|/signin|camera|file attachment|image upload|voice input|native app commands?|skills?|tool calls?|agent actions?)""",
        )
    }
    val highlighted = buildAnnotatedString {
        var cursor = 0
        pattern.findAll(text).forEach { match ->
            if (match.range.first > cursor) {
                append(text.substring(cursor, match.range.first))
            }
            val start = length
            append(match.value)
            addStyle(
                SpanStyle(
                    color = MaterialTheme.colorScheme.primary,
                    background = MaterialTheme.colorScheme.primary.copy(alpha = 0.14f),
                    fontWeight = FontWeight.SemiBold,
                ),
                start,
                length,
            )
            cursor = match.range.last + 1
        }
        if (cursor < text.length) {
            append(text.substring(cursor))
        }
    }
    Text(text = highlighted, color = color, style = MaterialTheme.typography.bodyMedium)
}

@Composable
private fun AttachmentPreviewColumn(
    attachments: List<ChatAttachment>,
    contentColor: androidx.compose.ui.graphics.Color,
) {
    var selectedAttachment by remember { mutableStateOf<ChatAttachment?>(null) }
    if (attachments.isEmpty()) return
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        attachments.forEach { attachment ->
            AttachmentPreview(
                attachment = attachment,
                contentColor = contentColor,
                onOpen = { selectedAttachment = attachment },
            )
        }
    }
    selectedAttachment?.let { attachment ->
        FullscreenAttachmentDialog(attachment = attachment, onDismiss = { selectedAttachment = null })
    }
}

@Composable
private fun AttachmentPreview(
    attachment: ChatAttachment,
    contentColor: androidx.compose.ui.graphics.Color,
    onOpen: () -> Unit,
) {
    val image = rememberAttachmentBitmap(attachment)
    val strings = LocalHermesStrings.current
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        Surface(
            modifier = Modifier
                .fillMaxWidth(0.7f)
                .heightIn(min = 112.dp, max = 280.dp)
                .clickable(onClick = onOpen)
                .testTag("HermesChatImagePreview"),
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f),
            shape = MaterialTheme.shapes.medium,
        ) {
            if (image != null) {
                Image(
                    bitmap = image,
                    contentDescription = attachment.displayName,
                    modifier = Modifier.fillMaxWidth(),
                    contentScale = ContentScale.Crop,
                )
            } else {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_action_image),
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                    )
                    Text(attachment.displayName, color = contentColor, style = MaterialTheme.typography.bodySmall)
                    QuietMetaText(text = attachment.mimeType.ifBlank { strings.genericAttachmentLabel() }, color = contentColor)
                }
            }
        }
    }
}

@Composable
private fun FullscreenAttachmentDialog(
    attachment: ChatAttachment,
    onDismiss: () -> Unit,
) {
    val image = rememberAttachmentBitmap(attachment)
    val strings = LocalHermesStrings.current
    Dialog(onDismissRequest = onDismiss) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.surface,
            shape = MaterialTheme.shapes.large,
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = attachment.displayName,
                        modifier = Modifier.weight(1f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    IconButton(onClick = onDismiss) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_action_close),
                            contentDescription = strings.removeAttachment(),
                            tint = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
                if (image != null) {
                    Image(
                        bitmap = image,
                        contentDescription = attachment.displayName,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 560.dp),
                        contentScale = ContentScale.Fit,
                    )
                } else {
                    Text(strings.attachmentPreviewUnavailable(), style = MaterialTheme.typography.bodyMedium)
                }
            }
        }
    }
}

@Composable
private fun rememberAttachmentBitmap(attachment: ChatAttachment): ImageBitmap? {
    val context = LocalContext.current
    return remember(attachment.uri) {
        runCatching {
            context.contentResolver.openInputStream(Uri.parse(attachment.uri))?.use { stream ->
                BitmapFactory.decodeStream(stream)?.asImageBitmap()
            }
        }.getOrNull()
    }
}

private fun persistCameraPreview(context: Context, bitmap: Bitmap): Uri {
    val directory = File(context.cacheDir, "hermes-camera").apply { mkdirs() }
    val file = File(directory, "camera-${System.currentTimeMillis()}.jpg")
    file.outputStream().use { output ->
        require(bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output)) {
            "Unable to encode camera image"
        }
    }
    return Uri.fromFile(file)
}

@Composable
private fun CompactActivityRow(
    content: String,
    contentColor: androidx.compose.ui.graphics.Color,
) {
    var expanded by rememberSaveable(content.take(64)) { mutableStateOf(false) }
    val strings = LocalHermesStrings.current
    val diagnosticCards = remember(content) { extractDiagnosticCards(content) }
    val visibleDiagnosticCards = diagnosticCardsForActivityPreview(diagnosticCards, expanded)
    val hiddenDiagnosticCardCount = hiddenDiagnosticCardCountForActivityPreview(diagnosticCards, expanded)
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { expanded = !expanded },
        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.10f),
        shape = MaterialTheme.shapes.small,
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = strings.activityToolContext(),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
                Text(if (expanded) strings.hideLabel() else strings.detailsLabel(), style = MaterialTheme.typography.labelSmall, color = contentColor.copy(alpha = 0.72f))
            }
            visibleDiagnosticCards.forEach { card ->
                DiagnosticSummaryCard(
                    card = card,
                    expanded = expanded,
                    contentColor = contentColor,
                )
            }
            if (hiddenDiagnosticCardCount > 0) {
                Text(
                    text = strings.moreCards(hiddenDiagnosticCardCount),
                    style = MaterialTheme.typography.labelSmall,
                    color = contentColor.copy(alpha = 0.62f),
                )
            }
            if (expanded) {
                Text(
                    text = content.take(360),
                    style = MaterialTheme.typography.bodySmall,
                    color = contentColor.copy(alpha = 0.72f),
                )
            }
        }
    }
}

@Composable
private fun DiagnosticSummaryCard(
    card: DiagnosticCardSummary,
    expanded: Boolean,
    contentColor: androidx.compose.ui.graphics.Color,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = card.title,
                modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            if (card.rowCount > 0) {
                Text(
                    text = "${card.rowCount}",
                    style = MaterialTheme.typography.labelSmall,
                    color = contentColor.copy(alpha = 0.62f),
                )
            }
        }
        Text(
            text = card.body,
            style = MaterialTheme.typography.bodySmall,
            color = contentColor.copy(alpha = 0.78f),
            maxLines = if (expanded) 3 else 2,
            overflow = TextOverflow.Ellipsis,
        )
        if (expanded && card.rows.isNotEmpty()) {
            DiagnosticMiniGraph(card = card, contentColor = contentColor)
        }
    }
}

@Composable
private fun DiagnosticMiniGraph(
    card: DiagnosticCardSummary,
    contentColor: androidx.compose.ui.graphics.Color,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 2.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        card.rows.take(8).forEach { row ->
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = row.label,
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.labelSmall,
                        color = contentColor.copy(alpha = 0.84f),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        text = row.valueLabel,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                        maxLines = 1,
                    )
                }
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(contentColor.copy(alpha = 0.12f)),
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(row.fraction.coerceIn(0.05f, 1f))
                            .height(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.82f)),
                    )
                }
                Text(
                    text = row.detail,
                    style = MaterialTheme.typography.labelSmall,
                    color = contentColor.copy(alpha = 0.58f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

private fun hasToolActivity(content: String): Boolean {
    val lower = content.lowercase()
    return "tool" in lower ||
        "terminal_tool" in lower ||
        "android_system_tool" in lower ||
        "file_write_tool" in lower ||
        "\"cards\"" in lower ||
        "wifi_scan" in lower ||
        "bluetooth_scan" in lower ||
        "radio_signal_status" in lower ||
        "sensor_snapshot" in lower
}

@Composable
private fun ConversationHistoryList(
    summaries: List<ChatConversationSummary>,
    onOpenConversation: (String) -> Unit,
    onStartNew: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(strings.conversationHistoryTitle(), style = MaterialTheme.typography.headlineSmall)
            Button(onClick = onStartNew) {
                Text(strings.newChat.ifBlank { "New chat" })
            }
        }
        if (summaries.isEmpty()) {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.surfaceVariant,
                shape = MaterialTheme.shapes.large,
            ) {
                Text(
                    text = strings.noConversationHistory(),
                    modifier = Modifier.padding(16.dp),
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                items(summaries, key = { it.id }) { summary ->
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.large,
                        onClick = { onOpenConversation(summary.id) },
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Text(summary.title, style = MaterialTheme.typography.titleMedium)
                            if (summary.preview.isNotBlank()) {
                                Text(summary.preview, style = MaterialTheme.typography.bodySmall)
                            }
                            Text(
                                text = "${summary.updatedLabel} · ${strings.messageCount(summary.messageCount)}",
                                style = MaterialTheme.typography.labelMedium,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatComposer(
    modifier: Modifier = Modifier,
    input: String,
    attachments: List<ChatAttachment>,
    isSending: Boolean,
    isListening: Boolean,
    onInputChange: (String) -> Unit,
    onAttachImage: () -> Unit,
    onCaptureImage: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onMic: () -> Unit,
    onSend: () -> Unit,
    onActionMenuExpandedChange: (Boolean) -> Unit,
    onSignalQuickAction: (SignalIntelligenceQuickAction) -> Unit,
) {
    val strings = LocalHermesStrings.current
    var actionMenuOpen by rememberSaveable { mutableStateOf(false) }
    val actionMenuScrollState = rememberScrollState()
    val compactActionButtonColors = ButtonDefaults.buttonColors(
        containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.14f),
        contentColor = MaterialTheme.colorScheme.primary,
        disabledContainerColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f),
        disabledContentColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
    )
    LaunchedEffect(actionMenuOpen) {
        onActionMenuExpandedChange(actionMenuOpen)
    }
    DisposableEffect(Unit) {
        onDispose { onActionMenuExpandedChange(false) }
    }
    Surface(
        modifier = modifier,
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(24.dp),
        tonalElevation = 2.dp,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            if (actionMenuOpen) {
                BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
                    val ultraNarrowActionMenu = maxWidth < 220.dp
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = if (ultraNarrowActionMenu) 64.dp else 220.dp)
                            .testTag("HermesChatComposerActions"),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Column(
                            modifier = Modifier
                                .verticalScroll(actionMenuScrollState)
                                .padding(8.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            if (ultraNarrowActionMenu) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(3.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    UltraNarrowComposerIconButton(
                                        iconRes = R.drawable.ic_action_image,
                                        contentDescription = strings.attachImage(),
                                        active = false,
                                        onClick = {
                                            actionMenuOpen = false
                                            onAttachImage()
                                        },
                                        testTag = "HermesChatAttachImageButton",
                                        modifier = Modifier.weight(1f),
                                    )
                                    UltraNarrowComposerIconButton(
                                        iconRes = R.drawable.ic_action_image,
                                        contentDescription = strings.camera(),
                                        active = false,
                                        onClick = {
                                            actionMenuOpen = false
                                            onCaptureImage()
                                        },
                                        testTag = "HermesChatCameraButton",
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                            } else {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Button(
                                        onClick = {
                                            actionMenuOpen = false
                                            onAttachImage()
                                        },
                                        modifier = Modifier
                                            .weight(1f)
                                            .heightIn(min = 36.dp)
                                            .testTag("HermesChatAttachImageButton"),
                                        shape = MaterialTheme.shapes.small,
                                        colors = compactActionButtonColors,
                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                                    ) {
                                        Icon(
                                            painter = painterResource(id = R.drawable.ic_action_image),
                                            contentDescription = null,
                                            modifier = Modifier.size(16.dp),
                                        )
                                        Spacer(modifier = Modifier.size(6.dp))
                                        Text(
                                            text = strings.attachImage(),
                                            style = MaterialTheme.typography.labelMedium,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                        )
                                    }
                                    Button(
                                        onClick = {
                                            actionMenuOpen = false
                                            onCaptureImage()
                                        },
                                        modifier = Modifier
                                            .weight(1f)
                                            .heightIn(min = 36.dp)
                                            .testTag("HermesChatCameraButton"),
                                        shape = MaterialTheme.shapes.small,
                                        colors = compactActionButtonColors,
                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                                    ) {
                                        Icon(
                                            painter = painterResource(id = R.drawable.ic_action_image),
                                            contentDescription = null,
                                            modifier = Modifier.size(16.dp),
                                        )
                                        Spacer(modifier = Modifier.size(6.dp))
                                        Text(
                                            text = strings.camera(),
                                            style = MaterialTheme.typography.labelMedium,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                        )
                                    }
                                }
                                SignalIntelligenceQuickActionGrid(
                                    compact = true,
                                    enabled = !isSending && input.isBlank() && attachments.isEmpty(),
                                    onSignalQuickAction = { action ->
                                        actionMenuOpen = false
                                        onSignalQuickAction(action)
                                    },
                                )
                                QuietMetaText(text = strings.chatCommandsTip(isListening), color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                    }
                }
            }
            if (attachments.isNotEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("HermesChatAttachments"),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text(strings.attachedImages(attachments.size), style = MaterialTheme.typography.bodySmall)
                    attachments.forEach { attachment ->
                        Surface(
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = MaterialTheme.shapes.small,
                        ) {
                            Row(
                                modifier = Modifier.padding(start = 10.dp, end = 4.dp, top = 4.dp, bottom = 4.dp),
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Text(attachment.displayName, style = MaterialTheme.typography.labelMedium)
                                IconButton(onClick = { onRemoveAttachment(attachment.uri) }, modifier = Modifier.size(28.dp)) {
                                    Icon(
                                        painter = painterResource(id = R.drawable.ic_action_close),
                                        contentDescription = strings.removeAttachment(),
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(16.dp),
                                    )
                                }
                            }
                        }
                    }
                }
            }
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("HermesChatComposerFrame"),
            ) {
                val ultraNarrowComposer = maxWidth < 220.dp
                val stackedComposer = maxWidth < 340.dp
                if (stackedComposer) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("HermesChatComposerCompact"),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        ComposerInputField(
                            input = input,
                            onInputChange = onInputChange,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        if (ultraNarrowComposer) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("HermesChatComposerUltraNarrowControls"),
                                horizontalArrangement = Arrangement.spacedBy(3.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                UltraNarrowComposerIconButton(
                                    iconRes = R.drawable.ic_nav_settings,
                                    contentDescription = strings.moreInputActions(),
                                    active = actionMenuOpen,
                                    onClick = { actionMenuOpen = !actionMenuOpen },
                                    testTag = "HermesChatMoreInputActionsButton",
                                    modifier = Modifier.weight(1f),
                                )
                                UltraNarrowComposerIconButton(
                                    iconRes = R.drawable.ic_action_mic,
                                    contentDescription = strings.voiceInputLabel(),
                                    active = isListening,
                                    onClick = onMic,
                                    testTag = "HermesChatMicButton",
                                    modifier = Modifier.weight(1f),
                                )
                                UltraNarrowComposerSendButton(
                                    input = input,
                                    attachments = attachments,
                                    isSending = isSending,
                                    onSend = onSend,
                                    modifier = Modifier.weight(1f),
                                )
                            }
                        } else {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .testTag("HermesChatComposerRow"),
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                ComposerActionsButton(
                                    actionMenuOpen = actionMenuOpen,
                                    onToggle = { actionMenuOpen = !actionMenuOpen },
                                )
                                ComposerMicButton(
                                    isListening = isListening,
                                    onMic = onMic,
                                )
                                Spacer(modifier = Modifier.weight(1f))
                                ChatSendButton(
                                    input = input,
                                    attachments = attachments,
                                    isSending = isSending,
                                    onSend = onSend,
                                )
                            }
                        }
                    }
                } else {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("HermesChatComposerRow"),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        ComposerActionsButton(
                            actionMenuOpen = actionMenuOpen,
                            onToggle = { actionMenuOpen = !actionMenuOpen },
                        )
                        ComposerInputField(
                            input = input,
                            onInputChange = onInputChange,
                            modifier = Modifier.weight(1f),
                        )
                        ComposerMicButton(
                            isListening = isListening,
                            onMic = onMic,
                        )
                        ChatSendButton(
                            input = input,
                            attachments = attachments,
                            isSending = isSending,
                            onSend = onSend,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ComposerActionsButton(
    actionMenuOpen: Boolean,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    IconButton(
        onClick = onToggle,
        modifier = modifier
            .heightIn(min = 40.dp)
            .testTag("HermesChatMoreInputActionsButton"),
    ) {
        Icon(
            painter = painterResource(id = R.drawable.ic_nav_settings),
            contentDescription = strings.moreInputActions(),
            tint = if (actionMenuOpen) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(22.dp),
        )
    }
}

@Composable
private fun ComposerInputField(
    input: String,
    onInputChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    OutlinedTextField(
        value = input,
        onValueChange = onInputChange,
        modifier = modifier
            .heightIn(max = 112.dp)
            .testTag("HermesChatInput"),
        shape = MaterialTheme.shapes.large,
        placeholder = {
            Text(
                text = strings.messageHermes.ifBlank { "Message Hermes" },
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        maxLines = 4,
    )
}

@Composable
private fun ComposerMicButton(
    isListening: Boolean,
    onMic: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    IconButton(
        onClick = onMic,
        modifier = modifier
            .heightIn(min = 40.dp)
            .testTag("HermesChatMicButton"),
    ) {
        Icon(
            painter = painterResource(id = R.drawable.ic_action_mic),
            contentDescription = strings.voiceInputLabel(),
            tint = if (isListening) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(22.dp),
        )
    }
}

@Composable
private fun SignalIntelligenceQuickActionGrid(
    enabled: Boolean,
    compact: Boolean = false,
    onSignalQuickAction: (SignalIntelligenceQuickAction) -> Unit,
) {
    val buttonColors = if (compact) {
        ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
            contentColor = MaterialTheme.colorScheme.primary,
            disabledContainerColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f),
            disabledContentColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.38f),
        )
    } else {
        ButtonDefaults.buttonColors()
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("HermesSignalQuickActions"),
        verticalArrangement = Arrangement.spacedBy(if (compact) 4.dp else 6.dp),
    ) {
        Text(
            text = LocalHermesStrings.current.signalIntelligence(),
            style = if (compact) MaterialTheme.typography.labelMedium else MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.primary,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        SIGNAL_INTELLIGENCE_QUICK_ACTIONS.chunked(2).forEach { rowActions ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(if (compact) 6.dp else 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                rowActions.forEach { action ->
                    Button(
                        onClick = { onSignalQuickAction(action) },
                        enabled = enabled,
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = if (compact) 34.dp else 40.dp)
                            .testTag("HermesSignalQuickAction_${action.id}"),
                        shape = MaterialTheme.shapes.small,
                        colors = buttonColors,
                        contentPadding = PaddingValues(
                            horizontal = if (compact) 6.dp else 8.dp,
                            vertical = if (compact) 4.dp else 6.dp,
                        ),
                    ) {
                        Icon(
                            painter = painterResource(id = action.iconRes),
                            contentDescription = null,
                            modifier = Modifier.size(if (compact) 14.dp else 16.dp),
                        )
                        Text(
                            text = " ${action.label}",
                            style = if (compact) MaterialTheme.typography.labelSmall else MaterialTheme.typography.labelMedium,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
                if (rowActions.size == 1) {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun ChatSendButton(
    input: String,
    attachments: List<ChatAttachment>,
    isSending: Boolean,
    onSend: () -> Unit,
    modifier: Modifier = Modifier.widthIn(min = 64.dp, max = 88.dp),
) {
    val strings = LocalHermesStrings.current
    Box(modifier = modifier) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Button(
                onClick = onSend,
                enabled = !isSending && (input.isNotBlank() || attachments.isNotEmpty()),
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("HermesChatSendButton"),
                shape = RoundedCornerShape(28.dp),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            ) {
                Text(
                    text = if (isSending) "…" else strings.send.ifBlank { "Send" },
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun UltraNarrowComposerIconButton(
    iconRes: Int,
    contentDescription: String,
    active: Boolean,
    onClick: () -> Unit,
    testTag: String,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier
            .height(32.dp)
            .clip(RoundedCornerShape(14.dp))
            .clickable(onClick = onClick)
            .testTag(testTag),
        color = if (active) {
            MaterialTheme.colorScheme.secondary.copy(alpha = 0.22f)
        } else {
            MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)
        },
        shape = RoundedCornerShape(14.dp),
    ) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Icon(
                painter = painterResource(id = iconRes),
                contentDescription = contentDescription,
                tint = if (active) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(16.dp),
            )
        }
    }
}

@Composable
private fun UltraNarrowComposerSendButton(
    input: String,
    attachments: List<ChatAttachment>,
    isSending: Boolean,
    onSend: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    val enabled = !isSending && (input.isNotBlank() || attachments.isNotEmpty())
    Surface(
        modifier = modifier
            .height(32.dp)
            .clip(RoundedCornerShape(14.dp))
            .clickable(enabled = enabled, onClick = onSend)
            .testTag("HermesChatSendButton"),
        color = if (enabled) {
            MaterialTheme.colorScheme.primary
        } else {
            MaterialTheme.colorScheme.onSurface.copy(alpha = 0.10f)
        },
        shape = RoundedCornerShape(14.dp),
    ) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
                text = if (isSending) "…" else strings.send.ifBlank { "Send" },
                color = if (enabled) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.labelSmall,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
