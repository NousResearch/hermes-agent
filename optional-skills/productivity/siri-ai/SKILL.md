---
name: siri-ai
description: "Interact with Siri AI's text chat via desktop automation."
version: 1.0.0
author: Damian Cavanagh (dhcav)
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [siri, Siri AI, macOS, Apple, desktop-automation]
---

# Siri AI

Interact with Siri AI (macOS Sequoia+) through its text chat UI using the `computer_use` desktop automation tool. Siri AI has a full text-based conversation interface — type questions, send them, and read responses. Voice input via the microphone button is also supported.

## When to Use

- User asks Siri AI to answer a question
- User wants Siri to draft text, emails, or notes
- User wants Siri to perform a task (check calendar, set alarms, control smart home)
- User asks "what does Siri think about X"
- User wants to interact with Siri via voice (requires microphone hardware)

## When NOT to Use

- User wants to control Siri via HomePod/AirPods (not accessible programmatically)
- User asks for something that requires authentication (banking, private data)
- No microphone available and user insists on voice input

## Prerequisites

- **macOS Sequoia** (or later) with Siri AI app installed
- **`computer_use` tool** available (cua-driver desktop automation)
- Siri AI app running (usually starts with macOS; verify with `ps aux | grep "Siri AI"`)
- **Microphone** (built-in, headset, or external) for voice input
- **Full Disk Access** for the terminal/app (System Settings → Privacy & Security → Full Disk Access)
- **Microphone permission** granted to the terminal app (System Settings → Privacy & Security → Microphone)
- Siri configured as your assistant (System Settings → Apple Intelligence & Siri → Siri Voice / Default Actions)

## Procedure

All steps use the `computer_use` tool. Siri AI (macOS Sequoia) has a text chat UI:
- A sidebar listing conversations by date
- A main chat area (empty when no conversation is active)
- A text input field at the bottom (placeholder: "Ask Siri")
- A microphone button (for voice input — see Voice Interaction below)

**App name is "Siri"** — `list_apps` returns `name: 'Siri'` even though the app bundle is `Siri AI.app`. The macOS process is `com.apple.campo`.

### Text Input

#### Step 1: Discover Siri's Window

```python
# Find the window — list_windows is reliable
call computer_use: action='list_windows', app='Siri'
```

The result includes `pid`, `window_id`, and `off_screen`. Use `pid` and `window_id` in subsequent capture calls.

If no window exists, press Space (macOS Siri hotkey) to activate the UI:
```python
call computer_use: action='focus_app', app='Siri', raise_window=true
call computer_use: action='key', keys='space'
```

#### Step 2: Type a Message

```python
call computer_use:
  action='type'
  app='Siri'
  text='Your question here'
  delivery_mode='background'
```

**Do NOT click "New Conversation"** — just type directly into the input field. Siri AI automatically creates a new conversation.

Use `delivery_mode='background'` first. Escalate to `foreground` only if text doesn't appear.

#### Step 3: Send the Message

```python
call computer_use: action='key', app='Siri', keys='return'
```

#### Step 4: Read the Response

```python
call computer_use: action='wait', seconds=5
call computer_use:
  action='capture'
  mode='som'
  pid=<pid_from_list_windows>
  window_id=<window_id_from_list_windows>
```

Wait 3–5 seconds for short answers, 5–10 for complex ones. Always re-capture before reading.

**Reading responses:** The sidebar shows the conversation title + preview. The main area contains the full response. Use the screenshot image to visually confirm — element labels only show truncated previews.

### Voice Interaction

Siri AI has a microphone button in the UI. Use this for hands-free voice input.

#### Step 1: Focus Siri

```python
call computer_use: action='focus_app', app='Siri', raise_window=true
```

If no window, press Space:
```python
call computer_use: action='key', keys='space'
```

#### Step 2: Click the Microphone Button

Capture to find the mic element:
```python
call computer_use: action='capture', mode='som', app='Siri'
```

Look for the microphone icon in the input area (an `AXButton` with a mic role or icon). Click it:
```python
call computer_use: action='click', app='Siri', element=<mic_element_index>
```

#### Step 3: Speak

Speak your question into the microphone. Wait for Siri to stop recording (mic icon changes state).

```python
call computer_use: action='wait', seconds=8
```

#### Step 4: Read the Response

```python
call computer_use: action='capture', mode='som', app='Siri'
```

#### Step 5: Cancel Voice Mode

Exit Siri's voice mode at any time:
```python
call computer_use: action='key', app='Siri', keys='escape'
```

### Quick Reference

| Action | Command |
|---|---|
| Discover window | `list_windows`, app='Siri' |
| Activate UI (no window) | `focus_app`, app='Siri' + `key`, keys='space' |
| Type message | `type`, app='Siri', text='...', delivery_mode='background' |
| Send message | `key`, app='Siri', keys='return' |
| Read response | `wait`, seconds=5 → `capture`, mode='som' |
| Start voice mode | `capture` → `click` mic element |
| Speak | user speaks → `wait`, seconds=8 → `capture` |
| Exit voice mode | `key`, app='Siri', keys='escape' |

## Pitfalls

- **"No window matched"** — Siri may be in background. Press Space to activate the Siri orb/UI.
- **Text not appearing** — The input field may need focus. Click the input area first, then type.
- **Response not showing** — Wait longer; Siri takes 3-10 seconds to generate. Capture again.
- **Microphone not working** — Check system permissions (System Settings → Privacy & Security → Microphone) and ensure Siri is enabled.
- **Siri says it can't help** — This is Siri's policy limit; not a tool issue.

## Verification

Confirm the interaction succeeded by:
1. **Response visible** — re-capture the window and verify Siri's response appears in the main chat area (not just the sidebar preview).
2. **New conversation in sidebar** — the sidebar should show a new entry under "Today" with Siri's response as the preview text.
3. **Input field cleared** — the text input field should show the placeholder "Ask Siri" after the response.

If the response is missing or incomplete, repeat the capture step. Siri may still be generating a long response.

## Examples

### Example: Ask Siri a Question

User: "What does Siri think about the weather today?"

```python
# 1. Discover window
call computer_use: action='list_windows', app='Siri'

# 2. Focus
call computer_use: action='focus_app', app='Siri', raise_window=true

# 3. Type
call computer_use: action='type', app='Siri', text='What is the weather today?'

# 4. Send
call computer_use: action='key', app='Siri', keys='return'

# 5. Wait and read
call computer_use: action='wait', seconds=5
call computer_use: action='capture', mode='som', app='Siri'
```

### Example: Voice Input

User: "Ask Siri the weather via voice"

```python
# 1. Focus Siri
call computer_use: action='focus_app', app='Siri', raise_window=true

# 2. Capture to find microphone
call computer_use: action='capture', mode='som', app='Siri'
# Look for the mic icon element (e.g., #42 AXButton 'Microphone')

# 3. Click mic to start listening
call computer_use: action='click', app='Siri', element=42

# 4. User speaks: "What's the weather today?"
# Wait for Siri to process
call computer_use: action='wait', seconds=8

# 5. Read response
call computer_use: action='capture', mode='som', app='Siri'
```

### Example: Multi-turn Conversation

For follow-ups within the same conversation, simply type the next message — the conversation context is already active in Siri AI.
