package com.nousresearch.hermesagent.ui.chat

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.text.format.DateFormat
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.ui.auth.AuthViewModel
import com.nousresearch.hermesagent.ui.settings.SettingsViewModel
import com.nousresearch.hermesagent.ui.shell.AppSection
import com.nousresearch.hermesagent.ui.shell.ShellActionItem

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
    val context = LocalContext.current
    val listState = rememberLazyListState()
    val ttsController = remember(context) { HermesTtsController(context) }

    DisposableEffect(ttsController) {
        onDispose { ttsController.shutdown() }
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

    fun speak(text: String): Boolean {
        val worked = ttsController.speak(text)
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
        val supported = setOf("chatgpt", "claude", "gemini", "google", "email", "phone")
        if (methodId !in supported) return false
        authViewModel.startAuth(methodId)
        return true
    }

    val shellActions = remember(uiState.isShowingHistory, uiState.messages, uiState.activeConversationTitle) {
        if (uiState.isShowingHistory) {
            listOf(
                ShellActionItem(
                    label = "New chat",
                    description = "Start a fresh Hermes conversation.",
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::startNewConversation,
                ),
                ShellActionItem(
                    label = "Back to chat",
                    description = "Return to the active conversation.",
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::hideHistory,
                ),
            )
        } else {
            listOf(
                ShellActionItem(
                    label = "History",
                    description = "Browse previous Hermes conversations.",
                    iconRes = R.drawable.ic_action_history,
                    onClick = viewModel::showHistory,
                ),
                ShellActionItem(
                    label = "New chat",
                    description = "Start a fresh conversation without leaving Hermes.",
                    iconRes = R.drawable.ic_nav_hermes,
                    onClick = viewModel::startNewConversation,
                ),
                ShellActionItem(
                    label = "Clear conversation",
                    description = "Remove the current conversation and start clean.",
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = viewModel::clearCurrentConversation,
                ),
                ShellActionItem(
                    label = "Speak last reply",
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

    LaunchedEffect(uiState.messages.size, uiState.isShowingHistory) {
        if (!uiState.isShowingHistory && uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.lastIndex)
        }
    }

    fun handleSend() {
        val input = uiState.input.trim()
        if (input.isEmpty()) return
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
                    onOpenHistory = viewModel::showHistory,
                    onOpenActions = onOpenContextActions,
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
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        contentAlignment = Alignment.Center,
                    ) {
                        EmptyChatHint(
                            onNewChat = viewModel::startNewConversation,
                            onOpenAccounts = { onNavigateToSection(AppSection.Accounts) },
                            onOpenSettings = { onNavigateToSection(AppSection.Settings) },
                        )
                    }
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                        contentPadding = PaddingValues(bottom = 12.dp),
                    ) {
                        items(uiState.messages, key = { it.id }) { message ->
                            ChatBubble(
                                message = message,
                                onSpeak = { speak(message.content) },
                            )
                        }
                    }
                }
                ChatComposer(
                    input = uiState.input,
                    isSending = uiState.isSending,
                    isListening = uiState.isListening,
                    onInputChange = viewModel::updateInput,
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
    onOpenHistory: () -> Unit,
    onOpenActions: (() -> Unit)? = null,
) {
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
                contentDescription = "Hermes",
                tint = MaterialTheme.colorScheme.primary,
            )
            Column(modifier = Modifier.weight(1f)) {
                Text("Hermes Chat", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(title, style = MaterialTheme.typography.bodySmall)
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onOpenHistory) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_action_history),
                        contentDescription = "Open history",
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
                if (onOpenActions != null) {
                    IconButton(onClick = onOpenActions) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_action_cog),
                            contentDescription = "Open page actions",
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
            Text("Welcome to Hermes", style = MaterialTheme.typography.titleMedium)
            Text("Use chat for normal prompts, voice input, or native app commands like /help, /history, /provider, and /signin.")
            Button(onClick = onNewChat, modifier = Modifier.fillMaxWidth()) {
                Text("New chat")
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenAccounts, modifier = Modifier.weight(1f)) {
                    Text("Accounts")
                }
                Button(onClick = onOpenSettings, modifier = Modifier.weight(1f)) {
                    Text("Settings")
                }
            }
        }
    }
}

@Composable
private fun ChatBubble(
    message: ChatUiMessage,
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
                    Text(
                        text = DateFormat.format("HH:mm", message.createdAtEpochMs).toString(),
                        style = MaterialTheme.typography.labelSmall,
                        color = contentColor.copy(alpha = 0.72f),
                    )
                }
                Text(text = message.content.ifBlank { "…" }, color = contentColor)
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
private fun ConversationHistoryList(
    summaries: List<ChatConversationSummary>,
    onOpenConversation: (String) -> Unit,
    onStartNew: () -> Unit,
    modifier: Modifier = Modifier,
) {
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
                Text("New chat")
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
    isSending: Boolean,
    isListening: Boolean,
    onInputChange: (String) -> Unit,
    onMic: () -> Unit,
    onSend: () -> Unit,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shape = MaterialTheme.shapes.extraLarge,
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onMic) {
                Icon(
                    painter = painterResource(id = R.drawable.ic_action_mic),
                    contentDescription = "Voice input",
                    tint = if (isListening) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(22.dp),
                )
            }
            OutlinedTextField(
                value = input,
                onValueChange = onInputChange,
                modifier = Modifier.weight(1f),
                label = { Text("Message Hermes") },
                maxLines = 5,
                supportingText = {
                    Text(
                        if (isListening) "Listening…" else "Tip: /help shows native chat commands",
                        textAlign = TextAlign.Start,
                    )
                },
            )
            Button(onClick = onSend, enabled = !isSending) {
                Text(if (isSending) "…" else "Send")
            }
        }
    }
}
