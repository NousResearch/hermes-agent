# Android UI + Chat Revamp Implementation Plan

> For Hermes: use subagent-driven-development to implement this plan task-by-task.

Goal: revamp the Hermes Android app into a mobile-native experience with bottom navigation, custom vector icons, safe-area-aware layout, automatic Nous Portal loading, context-aware page actions, richer Hermes chat UX, conversation history, voice-to-text, TTS playback, and native app-command handling in chat.

Architecture: replace the current `TabRow` shell with a `Scaffold`-based mobile shell, drive page-specific actions through a shared contextual bottom sheet, keep portal and device flows integrated, and upgrade the chat surface with persisted local conversations plus native Android speech/TTS helpers.

Tech stack: Jetpack Compose Material3, Android platform SpeechRecognizer/TextToSpeech, existing Hermes local API server + SSE client, SharedPreferences-backed local stores, existing Chaquopy bridge.

---

## Task 1: Add shell/navigation models and contextual action definitions

Objective: centralize page metadata, bottom navigation labels/icons, and context-aware actions.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/ShellModels.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt`
- Test: `tests/hermes_android/test_android_shell_navigation.py`

Step 1: Create `ShellModels.kt` with:
- `AppSection` metadata
- `ShellActionItem`
- `ShellActionKind`
- custom vector icon resource name references per section

Step 2: Replace the current `TabRow` assumptions so `AppShell.kt` reads from the new models.

Step 3: Add tests asserting:
- no `TabRow` remains in the shell
- bottom navigation is present
- contextual action model exists

Step 4: Run:
- `source .venv/bin/activate && python -m pytest tests/hermes_android/test_android_shell_navigation.py -q`

Step 5: Commit.

---

## Task 2: Create custom vector icons for bottom navigation and shell actions

Objective: add functional custom SVG/vector icon resources for Hermes, Accounts, Portal, Device, Settings, History, Refresh, External, Voice, and Speaker.

Files:
- Create under `android/app/src/main/res/drawable/`:
  - `ic_shell_hermes.xml`
  - `ic_shell_accounts.xml`
  - `ic_shell_portal.xml`
  - `ic_shell_device.xml`
  - `ic_shell_settings.xml`
  - `ic_action_history.xml`
  - `ic_action_refresh.xml`
  - `ic_action_external.xml`
  - `ic_action_mic.xml`
  - `ic_action_speaker.xml`
  - `ic_action_cog.xml`
- Test: `tests/hermes_android/test_android_branding.py`

Step 1: Add clean vector drawables matching the Hermes palette.
Step 2: Keep them simple and alpha-appropriate.
Step 3: Extend branding tests to assert the new icon assets exist.
Step 4: Run targeted branding tests.
Step 5: Commit.

---

## Task 3: Replace top tabs with a mobile-safe bottom navigation shell

Objective: fix wrapped labels and ensure bottom navigation respects safe insets.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/theme/HermesTheme.kt`
- Test: `tests/hermes_android/test_android_shell_navigation.py`

Step 1: Convert shell to `Scaffold`.
Step 2: Add compact top brand bar.
Step 3: Add `NavigationBar` at the bottom with icons + labels.
Step 4: Apply `navigationBarsPadding()` to the bottom bar and safe padding to content.
Step 5: Ensure body content is padded with `innerPadding` from `Scaffold`.
Step 6: Add/adjust tests for bottom navigation and safe-area-aware shell structure.
Step 7: Run targeted tests.
Step 8: Commit.

---

## Task 4: Add contextual floating action button and bottom sheet

Objective: create the user-requested bottom-right cog above the system bar and make its actions depend on the current page.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/ContextActionSheet.kt`
- Test: `tests/hermes_android/test_android_shell_navigation.py`

Step 1: Add a FAB with the custom cog icon.
Step 2: Show it only when the current page exposes contextual actions.
Step 3: Add `ModalBottomSheet` content derived from `AppSection`.
Step 4: Wire initial actions:
- Hermes: history, new chat, clear chat, speak last reply, accounts, settings
- Nous Portal: open externally, refresh page, scroll to top
- Device: refresh, grant folder, import file, open accessibility settings
- Accounts: refresh auth state, cancel pending sign-in when applicable
- Settings: no contextual actions / FAB hidden

Step 5: Add tests for page-aware action visibility.
Step 6: Run targeted tests.
Step 7: Commit.

---

## Task 5: Redesign Nous Portal page for automatic loading

Objective: make portal loading feel automatic and remove always-visible debug-style controls.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt`
- Modify: shell action wiring in `AppShell.kt`
- Test: `tests/hermes_android/test_android_onboarding_and_portal.py`

Step 1: Load the embedded WebView automatically when the page opens.
Step 2: Remove the current always-visible utility button cluster from the main page body.
Step 3: Keep browser fallback and refresh actions in the contextual action sheet.
Step 4: Keep a short page header/status, loading indicator, and error state.
Step 5: Preserve WebView hardening:
- cookies
- third-party cookies
- JS/DOM storage
- user agent
- HTTP/error handling
Step 6: Add/update tests to assert:
- portal auto-load path exists
- actions are contextual instead of always inline
- external fallback remains available
Step 7: Run targeted portal tests.
Step 8: Commit.

---

## Task 6: Create persisted Android conversation history models/store

Objective: support native chat history and session browsing.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/data/ConversationHistoryStore.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/data/ConversationStore.kt`
- Test: `tests/hermes_android/test_android_conversation_history.py`

Step 1: Define stored message/session metadata.
Step 2: Persist:
- session id
- title
- created/updated timestamps
- message snapshots
- last assistant message
Step 3: Add methods to:
- create new conversation
- list conversation summaries
- load conversation by id
- append/update messages during streaming
- clear one or all conversations
Step 4: Add tests for session creation, update, listing, and reload.
Step 5: Run targeted tests.
Step 6: Commit.

---

## Task 7: Upgrade chat UI into a mobile-native conversation screen

Objective: replace role-label text blocks with polished message bubbles and a sticky bottom composer.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt`
- Create if helpful: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatComponents.kt`
- Test: `tests/hermes_android/test_android_chat_ui.py`

Step 1: Add bubble-style message rows with user vs assistant alignment.
Step 2: Add subtle timestamps/metadata.
Step 3: Make the composer sticky and safe-area aware.
Step 4: Replace the plain OutlinedTextField row with:
- mic icon
- message field
- send button
- optional shortcut chips
Step 5: Add assistant bubble action row with speaker icon.
Step 6: Add tests for new chat layout and action affordances.
Step 7: Run targeted tests.
Step 8: Commit.

---

## Task 8: Add conversation history sheet/page for Hermes chat

Objective: let users browse prior conversations from the Hermes page action sheet.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ConversationHistorySheet.kt`
- Modify: `ChatViewModel.kt`
- Modify: `AppShell.kt`
- Test: `tests/hermes_android/test_android_chat_ui.py`

Step 1: Add a history sheet listing conversation titles + updated times.
Step 2: Allow selecting a prior conversation to reload it.
Step 3: Add new chat / clear chat actions.
Step 4: Wire Hermes page contextual action “History”.
Step 5: Run targeted tests.
Step 6: Commit.

---

## Task 9: Add voice-to-text input

Objective: support native voice input in the Hermes chat composer.

Files:
- Modify: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/SpeechInputController.kt`
- Modify: `ChatScreen.kt`
- Modify: `ChatViewModel.kt`
- Test: `tests/hermes_android/test_android_chat_voice.py`

Step 1: Add `RECORD_AUDIO` permission to the manifest.
Step 2: Add SpeechRecognizer/RecognizerIntent helper.
Step 3: Add a mic button in the composer.
Step 4: Feed recognized text into the input box.
Step 5: Add graceful error/status handling when recognition is unavailable.
Step 6: Add tests for UI presence and controller integration seams.
Step 7: Run targeted tests.
Step 8: Commit.

---

## Task 10: Add TTS playback for assistant replies

Objective: let users play assistant replies with a speaker button.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/HermesTtsController.kt`
- Modify: `ChatScreen.kt`
- Modify: `ChatViewModel.kt`
- Test: `tests/hermes_android/test_android_chat_voice.py`

Step 1: Add a TextToSpeech controller.
Step 2: Expose “speak reply” from each assistant bubble.
Step 3: Add Hermes contextual action “Speak last reply”.
Step 4: Add stop/replace behavior for consecutive playback.
Step 5: Run targeted tests.
Step 6: Commit.

---

## Task 11: Add native chat-side Hermes commands

Objective: make app-side commands/auth/navigation work natively inside the Hermes chat input.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatCommandRouter.kt`
- Modify: `ChatViewModel.kt`
- Modify: `AppShell.kt`
- Test: `tests/hermes_android/test_android_chat_commands.py`

Step 1: Intercept slash commands before sending text to the runtime.
Step 2: Support at least:
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
Step 3: Commands should produce native app effects and/or system-style assistant messages in the chat feed.
Step 4: Keep non-command text flowing to the Hermes runtime unchanged.
Step 5: Add tests.
Step 6: Commit.

---

## Task 12: Refine page-specific content spacing and safe areas

Objective: make every page respect bottom system buttons and not collide with the new bottom nav/FAB.

Files:
- Modify: `AuthScreen.kt`
- Modify: `SettingsScreen.kt`
- Modify: `DeviceScreen.kt`
- Modify: `ChatScreen.kt`
- Modify: `NousPortalScreen.kt`
- Test: `tests/hermes_android/test_android_shell_navigation.py`

Step 1: audit body padding on all pages
Step 2: remove duplicated top/bottom chrome where the new shell already handles it
Step 3: ensure body content plus FAB plus nav bar fit on small phones
Step 4: add/adjust tests for safe-area aware screens
Step 5: commit

---

## Task 13: Update Android docs and onboarding copy

Objective: reflect the new mobile-first navigation and capabilities.

Files:
- Modify: `android/README.md`
- Modify: `website/docs/getting-started/android-app.md`
- Modify any relevant in-app copy in `AppShell.kt`, `DeviceScreen.kt`, `SettingsScreen.kt`

Step 1: document bottom navigation and contextual actions
Step 2: document voice input/TTS/history/native commands
Step 3: keep onboarding concise for first-time users
Step 4: commit

---

## Task 14: Validation

Run targeted suites while building:
- `source .venv/bin/activate && python -m pytest tests/hermes_android -q`
- `source .venv/bin/activate && python -m pytest tests/gateway/test_api_server_android_toolset.py -q`

Run packaging/runtime validation:
- `cd android && export PYTHON_FOR_BUILD="$(command -v python3.11 || command -v python3 || command -v python)" && ./gradlew :app:installDebugPythonRequirements`

Run full validation before shipping:
- `source .venv/bin/activate && python -m pytest tests/ -q`

---

## Task 15: Commit, push, and ship next alpha

Commands:
- `git add ...`
- `git commit -m "feat(android): revamp mobile shell and chat experience"`
- `git push --force-with-lease fork feat/termux-install-path`
- `gh run list --repo adybag14-cyber/hermes-agent --branch feat/termux-install-path --limit 5`
- `gh release create v0.0.1-alpha.6 --repo adybag14-cyber/hermes-agent --target feat/termux-install-path --title "Hermes Android v0.0.1-alpha.6" --notes "..."`

---

## Notes

- Prefer bottom sheet over side drawer for the contextual action experience on phones.
- The request says “sidebar”; the implementation can still satisfy the intent with a mobile-native sheet opened from the bottom-right cog.
- Keep the portal browser fallback excellent even if the embedded page still occasionally struggles.
- The first pass should optimize for usability and extension points, not for a perfect final visual system.
