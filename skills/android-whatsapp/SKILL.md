---
name: android-whatsapp
description: Send WhatsApp messages on Android
version: 1.0.0
metadata:
  hermes:
    tags: [android, whatsapp, messaging]
    category: android
---

# Sending a WhatsApp Message

## When to Use
User asks to send a WhatsApp message to a contact.

## Procedure

1. android_open_app("com.whatsapp")
2. android_wait(text="Chats")
3. To open existing chat: android_tap_text("<contact name>")
4. To new message: android_tap_text("New chat") → android_type("<contact name>") → tap match
5. android_tap_text("Type a message")  (or the message input area)
6. android_type("<message text>")
7. STOP — confirm with user before sending
8. android_tap_text("Send")  OR  android_press_key("enter")

## Pitfalls
- WhatsApp's message input node class is "android.widget.EditText"
- After typing, read the screen to verify text is correct before sending
- Group chats may have different UI — read_screen to identify layout
- Media messages require different flow — report to user if they want to send media
