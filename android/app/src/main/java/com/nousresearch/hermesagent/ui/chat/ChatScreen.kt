package com.nousresearch.hermesagent.ui.chat

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.text.format.DateFormat
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
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
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.ui.auth.AuthViewModel
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings
import com.nousresearch.hermesagent.ui.settings.SettingsViewModel
import com.nousresearch.hermesagent.ui.shell.AppSection
import com.nousresearch.hermesagent.ui.shell.ShellActionItem
import kotlinx.coroutines.launch

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

    DisposableEffect(context) {
        onDispose {
            ttsController?.shutdown()
        }
    }

    val speechLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        viewModel.setListening(false)
        if (result.resultCode != Activity.RESULT_OK) {
            viewModel.setStatus("Voice input canceled")
            return@rememberLauncherForActivityResult
        }
        val transcript = SpeechInputController.extractBestResult(result.data)
        if (transcript.isNullOrBlank()) {
            viewModel.setStatus("No speech was captured")
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
                viewModel.setStatus("Voice recognition is not available on this device")
            }
        } else {
            viewModel.setListening(false)
            viewModel.setStatus("Microphone permission is required for voice input")
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

    fun speak(text: String): Boolean {
        val controller = ttsController ?: HermesTtsController(context).also { ttsController = it }
        val worked = controller.speak(text)
        if (!worked) {
            viewModel.setStatus("Speech playback is not ready yet")
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
                viewModel.setStatus("Voice recognition is not available on this device")
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
                    description = "Start a fresh Hermes conversation.",
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::startNewConversation,
                ),
                ShellActionItem(
                    label = strings.backToChat.ifBlank { "Back to chat" },
                    description = "Return to the active conversation.",
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::hideHistory,
                ),
            )
        } else {
            listOf(
                ShellActionItem(
                    label = strings.history.ifBlank { "History" },
                    description = "Browse previous Hermes conversations.",
                    iconRes = R.drawable.ic_action_history,
                    onClick = viewModel::showHistory,
                ),
                ShellActionItem(
                    label = strings.newChat.ifBlank { "New chat" },
                    description = "Start a fresh conversation without leaving Hermes.",
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::startNewConversation,
                ),
                ShellActionItem(
                    label = strings.clearConversation.ifBlank { "Clear conversation" },
                    description = "Remove the current conversation and start clean.",
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = viewModel::clearCurrentConversation,
                ),
                ShellActionItem(
                    label = strings.speakLastReply.ifBlank { "Speak last reply" },
                    description = "Play the latest assistant reply out loud.",
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
        )
        if (commandResult.handled) {
            viewModel.consumeCommandResult(input, commandResult.feedback)
        } else {
            viewModel.sendMessage()
        }
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .widthIn(max = 960.dp)
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                        .imePadding(),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
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
                if (uiState.status.isNotBlank()) {
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
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                        contentPadding = PaddingValues(vertical = 8.dp),
                    ) {
                        item {
                            EmptyChatHint(
                                onNewChat = viewModel::startNewConversation,
                                onOpenAccounts = { onNavigateToSection(AppSection.Accounts) },
                                onOpenSettings = { onNavigateToSection(AppSection.Settings) },
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
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                            contentPadding = PaddingValues(bottom = 12.dp),
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
                                    .size(42.dp)
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
                ChatComposer(
                    input = uiState.input,
                    attachments = uiState.attachments,
                    isSending = uiState.isSending,
                    isListening = uiState.isListening,
                    onInputChange = viewModel::updateInput,
                    onAttachImage = { imageLauncher.launch(arrayOf("image/*")) },
                    onRemoveAttachment = viewModel::removeAttachment,
                    onMic = ::startVoiceInput,
                    onSend = ::handleSend,
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
        color = MaterialTheme.colorScheme.primaryContainer,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                painter = painterResource(id = R.drawable.ic_nav_hermes),
                contentDescription = strings.sectionHermes,
                tint = MaterialTheme.colorScheme.primary,
            )
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = strings.chatTitle.ifBlank { "Hermes Chat" },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = displayTitle,
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onOpenHistory) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_action_history),
                        contentDescription = strings.openHistory.ifBlank { "Open history" },
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
                Button(
                    onClick = onToggleDisplayMode,
                    modifier = Modifier.testTag("HermesChatDisplayToggle"),
                    shape = MaterialTheme.shapes.small,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.16f),
                        contentColor = MaterialTheme.colorScheme.primary,
                    ),
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                ) {
                    Text(
                        text = if (chatDisplayMode == "compact") "Compact" else "Expanded",
                        style = MaterialTheme.typography.labelMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                if (onOpenActions != null) {
                    IconButton(onClick = onOpenActions) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_action_cog),
                            contentDescription = strings.openPageActions.ifBlank { "Open page actions" },
                            tint = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusBanner(text: String, isError: Boolean = false) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = if (isError) MaterialTheme.colorScheme.error.copy(alpha = 0.14f) else MaterialTheme.colorScheme.secondaryContainer,
        shape = MaterialTheme.shapes.medium,
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(12.dp),
            color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSecondaryContainer,
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun EmptyChatHint(
    onNewChat: () -> Unit,
    onOpenAccounts: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 1.dp,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                text = strings.welcomeToHermes.ifBlank { "Welcome to Hermes" },
                style = MaterialTheme.typography.titleMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Text(strings.welcomeDescription, style = MaterialTheme.typography.bodyMedium)
            Button(onClick = onNewChat, modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = strings.newChat.ifBlank { "New chat" },
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenAccounts, modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = strings.accounts.ifBlank { "Accounts" },
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Button(onClick = onOpenSettings, modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = strings.settings.ifBlank { "Settings" },
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
    val roleLabel = if (isUser) "You" else "Hermes"
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            modifier = Modifier.widthIn(max = 320.dp),
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
                modifier = Modifier.padding(14.dp),
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
                                contentDescription = "Speak reply",
                                tint = contentColor,
                            )
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
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("HermesCompactChatTurn"),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 1.dp,
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
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
                QuietMetaText(text = "Hermes is preparing a reply", color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                                contentDescription = "Speak reply",
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
    val label = if (expanded) "Your full prompt" else "Your prompt"
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
                            text = "${message.attachments.size} attachment${if (message.attachments.size == 1) "" else "s"}",
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    Text(if (expanded) "▲" else "▼", color = MaterialTheme.colorScheme.primary)
                }
            }
            if (expanded) {
                HighlightedMessageText(
                    text = message.content.ifBlank { "Attachment-only prompt" },
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
                    QuietMetaText(text = attachment.mimeType.ifBlank { "attachment" }, color = contentColor)
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
                            contentDescription = "Close image preview",
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
                    Text("Preview is not available for this attachment.", style = MaterialTheme.typography.bodyMedium)
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

@Composable
private fun CompactActivityRow(
    content: String,
    contentColor: androidx.compose.ui.graphics.Color,
) {
    var expanded by rememberSaveable(content.take(64)) { mutableStateOf(false) }
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
                    text = "Activity: tool context",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
                Text(if (expanded) "Hide" else "Details", style = MaterialTheme.typography.labelSmall, color = contentColor.copy(alpha = 0.72f))
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

private fun hasToolActivity(content: String): Boolean {
    val lower = content.lowercase()
    return "tool" in lower || "terminal_tool" in lower || "android_system_tool" in lower || "file_write_tool" in lower
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
            Text("Conversation history", style = MaterialTheme.typography.headlineSmall)
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
                    text = "No conversation history yet. Start a new Hermes chat to create one.",
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
                                text = "${summary.updatedLabel} · ${summary.messageCount} messages",
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
    input: String,
    attachments: List<ChatAttachment>,
    isSending: Boolean,
    isListening: Boolean,
    onInputChange: (String) -> Unit,
    onAttachImage: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onMic: () -> Unit,
    onSend: () -> Unit,
) {
    val strings = LocalHermesStrings.current
    var actionMenuOpen by rememberSaveable { mutableStateOf(false) }
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 2.dp,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (actionMenuOpen) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("HermesChatComposerActions"),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = MaterialTheme.shapes.medium,
                ) {
                    Row(
                        modifier = Modifier.padding(8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Button(
                            onClick = {
                                actionMenuOpen = false
                                onAttachImage()
                            },
                            modifier = Modifier.testTag("HermesChatAttachImageButton"),
                        ) {
                            Icon(
                                painter = painterResource(id = R.drawable.ic_action_image),
                                contentDescription = null,
                                modifier = Modifier.size(18.dp),
                            )
                            Text(" Image")
                        }
                        QuietMetaText(text = strings.chatCommandsTip(isListening), color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            if (attachments.isNotEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("HermesChatAttachments"),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
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
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("HermesChatComposerRow"),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = { actionMenuOpen = !actionMenuOpen }) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_nav_settings),
                        contentDescription = "More input actions",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(22.dp),
                    )
                }
                OutlinedTextField(
                    value = input,
                    onValueChange = onInputChange,
                    modifier = Modifier
                        .weight(1f)
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
                IconButton(onClick = onMic) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_action_mic),
                        contentDescription = "Voice input",
                        tint = if (isListening) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(22.dp),
                    )
                }
                Button(
                    onClick = onSend,
                    enabled = !isSending && (input.isNotBlank() || attachments.isNotEmpty()),
                    modifier = Modifier.testTag("HermesChatSendButton"),
                ) {
                    Text(
                        text = if (isSending) "…" else strings.send.ifBlank { "Send" },
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
            if (!actionMenuOpen) {
                QuietMetaText(text = strings.chatCommandsTip(isListening), color = MaterialTheme.colorScheme.onSurface)
            }
        }
    }
}
