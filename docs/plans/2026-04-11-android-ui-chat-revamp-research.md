# Android UI + Chat Revamp Research Notes

> For Hermes: use subagent-driven-development to execute the implementation plan task-by-task after this research note and the companion implementation plan are reviewed.

Goal: redesign the Hermes Android app into a more mobile-native experience with bottom navigation, custom vector icons, safe-area-aware layout, context-aware page actions, automatic Nous Portal loading, better chat presentation, voice input, TTS playback, and native chat-side handling for common Hermes app commands.

Date: 2026-04-11
Branch context: feat/termux-install-path

---

## Sources reviewed

### Device screenshots
Latest screenshots inspected from:
- `/storage/emulated/0/Pictures/Screenshots/Screenshot_20260411_230606.jpg`
- `/storage/emulated/0/Pictures/Screenshots/Screenshot_20260411_230616.jpg`
- `/storage/emulated/0/Pictures/Screenshots/Screenshot_20260411_230623.jpg`
- `/storage/emulated/0/Pictures/Screenshots/Screenshot_20260411_230625.jpg`
- `/storage/emulated/0/Pictures/Screenshots/Screenshot_20260411_222016.jpg`

### App files reviewed
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/api/HermesApiClient.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/api/HermesSseClient.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/data/ConversationStore.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeManager.kt`
- `android/app/src/main/AndroidManifest.xml`
- `android/app/build.gradle.kts`

---

## Current state

### 1. App shell/navigation
Current shell is a vertical `Column` with:
- branded top header (`HermesBrandBar`)
- `TabRow` for five sections
- section body underneath

Problems confirmed by screenshots:
- tab labels wrap badly on small screens (`Hermes`, `Accounts`, `Settings` split across lines)
- header + tabs consume too much vertical space
- no modern bottom navigation or mobile-safe context action affordance
- no unified `Scaffold` with system-bar-safe bottom layout

### 2. Bottom inset handling
Some screens use `navigationBarsPadding`, but not through one central shell.
Problems confirmed by screenshots:
- content/cards sit too close to Android system navigation buttons
- chat composer feels clipped/crowded near the bottom edge
- there is no persistent bottom navigation bar with proper inset handling

### 3. Nous Portal UX
Current Nous Portal page is external-first but still exposes a large block of page-specific buttons inline:
- Refresh
- Open externally
- Try embedded preview
- Reload preview
- Scroll to top

Problems confirmed by screenshots:
- page feels like a prototype/debug surface instead of product UI
- embedded preview is optional, not automatic
- actions are always visible instead of contextual
- no single page-level floating action affordance
- the screen still uses too much vertical chrome before the actual preview

### 4. Chat UX
Current chat screen is extremely bare:
- simple `LazyColumn`
- each item is just `ROLE` + plain text
- no bubble styling
- no timestamps
- no assistant actions
- no history list
- no voice input
- no TTS playback
- no suggestions / quick actions / command affordances

`ChatViewModel` currently:
- keeps messages only in memory inside `ChatUiState`
- streams assistant text from SSE
- uses `ConversationStore.currentSessionId()`
- does not persist visible chat transcripts or titles
- does not support local slash/app commands

`ConversationStore` currently only stores a single session id.
There is no conversation metadata model, no per-session title, and no chat history browser.

### 5. Voice / speech
Current Android project contains no use of:
- `SpeechRecognizer`
- `RecognizerIntent`
- `TextToSpeech`

Manifest does not yet request:
- `android.permission.RECORD_AUDIO`

### 6. Native chat command support
Current app chat sends user text directly to `/v1/chat/completions`.
There is no native intercept layer for app commands such as:
- new chat
- open history
- switch page
- show auth/accounts
- provider/model shortcuts
- speak latest reply
- clear chat
- refresh portal

### 7. Backend/API surface
Current Android app talks to the local API server through:
- `HermesApiClient`
- `HermesSseClient`

The Android app has session IDs, but it does not currently expose a native session/history list UI.
The app already has enough local persistence to add this without server-side protocol changes as a first step.

---

## Design conclusions

### A. Replace top tabs with bottom navigation
The screenshots make this mandatory.
Recommended approach:
- central `Scaffold`
- branded compact top bar
- `NavigationBar` at the bottom
- custom vector icons for each section
- selected state only for the current page
- `navigationBarsPadding()` at the bottom bar level

Why:
- prevents wrapped top labels
- creates a mobile-native layout
- keeps page switching anchored near the thumb zone

### B. Introduce a page-aware floating action button + modal sheet/sidebar
The user wants a small settings/cog button above the phone bar which opens a context-sensitive sidebar.
Recommended approach:
- bottom-right `FloatingActionButton`
- a `ModalBottomSheet` (preferred over a left drawer on phones)
- content dynamically derived from `AppSection`

Reason for `ModalBottomSheet` instead of a classic side drawer:
- more thumb-friendly on phones
- easier to anchor above the bottom nav/system bar
- better fit for contextual, page-specific actions
- still satisfies the “sidebar” requirement functionally in an Android-native way

Per-page examples:
- Hermes: History, New chat, Speak last reply, Clear conversation, Accounts, Settings
- Nous Portal: Open externally, Refresh page, Retry embedded preview, Scroll to top
- Device: Refresh device state, Grant folder, Import file, Open accessibility settings
- Accounts: Refresh auth state, Cancel pending sign-in
- Settings: no FAB / no sheet by default

### C. Make Nous Portal auto-load by default, but keep browser fallback
The current optional embedded preview toggle is too defensive and feels broken.
Recommended behavior:
- page opens with embedded portal loading automatically
- page still exposes browser fallback via contextual action sheet
- top-of-page explanatory copy should be much shorter
- keep HTTP/error handling and clear fallback messaging

### D. Preserve external-first reality without making the UI feel broken
The portal may still hit verification/cookies/mobile issues.
The UI should therefore:
- automatically try to load the embedded view
- show load state and error state cleanly
- keep “Open externally” in the page actions sheet
- avoid a wall of inline debug/utility buttons

### E. Chat needs a product-level redesign, not incremental text tweaks
Minimum acceptable upgrade:
- bubble layout with clear user vs assistant styling
- timestamps / metadata kept subtle
- sticky bottom composer with safe-area padding
- mic button for speech input
- speaker button on assistant replies for TTS
- quick actions and command hints
- conversation history list
- local command parsing before sending to the runtime

### F. History can be implemented app-side first
Instead of depending immediately on a new backend endpoint, first persist local chat sessions in Android:
- session ID
- title derived from first user turn
- updatedAt
- message list snapshot

This is enough to ship:
- a history sheet/page
- resume prior conversations
- “new chat” action

### G. Native chat commands should target app functionality first
First-pass local commands should include:
- `/help`
- `/new`
- `/history`
- `/accounts`
- `/settings`
- `/device`
- `/portal`
- `/clear`
- `/provider <id>`
- `/model <name>`
- `/speak last`

These should be handled by the app before sending raw text to the Hermes runtime.
This satisfies the user requirement that Hermes auth/commands feel natively supported in the app chat.

### H. Voice input should use platform speech APIs, not Python/Whisper first
For Android alpha, the fastest reliable path is:
- `SpeechRecognizer` or `RecognizerIntent` for voice-to-text
- runtime permission for RECORD_AUDIO
- fallback error text in chat status if unavailable

This avoids shipping a heavier local ASR stack immediately.

### I. TTS should use Android `TextToSpeech`
This is native, lightweight, and good enough for alpha.
Each assistant bubble can expose a speaker icon.
The chat sheet can also provide “Speak last reply”.

---

## Recommended architecture

### New shell model
Create a central shell state model:
- current section
- whether contextual actions are available
- whether action sheet is open
- current sheet action items

Use:
- `Scaffold`
- compact top bar
- bottom `NavigationBar`
- per-page content in body
- page-aware FAB
- `ModalBottomSheet`

### New chat state model additions
Need persisted conversations with messages and metadata.
Add data classes like:
- `StoredConversation`
- `StoredMessage`
- `ConversationSummary`

Add store methods:
- list summaries
- create new session
- append/update messages
- load session by id
- clear session
- clear all

### New voice/TTS adapters
Add thin Android wrappers:
- `SpeechToTextController` / `SpeechInputLauncher`
- `HermesTtsController`

### New page actions model
Add shell-wide action definitions:
- id
- label
- icon
- visibility by page
- callback target

---

## Risks

1. Compose bottom sheets + WebView + safe area interactions can be fiddly on small screens.
2. `SpeechRecognizer` lifecycle on Android needs careful cleanup.
3. `TextToSpeech` must be initialized once and released correctly.
4. Local history snapshots can drift if streaming messages are not persisted incrementally.
5. Native command parsing must not eat normal user text accidentally.
6. Portal auto-load may still be degraded by remote anti-bot checks, so the external fallback must remain excellent.

---

## Recommended implementation order

1. Shell + navigation + safe area refactor
2. Contextual action sheet/FAB
3. Portal page redesign on top of new shell
4. Chat history persistence
5. Chat bubble UI + composer redesign
6. Voice input
7. TTS playback
8. Native command handling
9. Validation + release

---

## Non-goals for the first pass

- full server-side session/history browsing protocol
- full multimodal attachments/media upload pipeline
- full portal-native replacement UI
- full voice call / streaming audio UX
- deep RAG/search over chat history

The first pass should aim for a mobile-native, polished alpha that is immediately more usable and extensible.
