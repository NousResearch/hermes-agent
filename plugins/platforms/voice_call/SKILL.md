# Voice calls

You can place and manage real phone calls with the `voice_call` tool.

## When to call someone

Only place calls the user explicitly asked for (or pre-authorized, e.g.
"call me when the build finishes"). Calls ring a real phone and cost real
money. Never call numbers the user hasn't given you.

## Modes

- **notify** — deliver one spoken message and hang up. Best for alerts and
  reminders: `{"action": "initiate_call", "to_number": "+1...", "message":
  "Your deployment finished successfully.", "mode": "notify"}`
- **conversation** — stay on the line. Use `continue_call` to say something
  and wait for the reply, `speak_to_user` to talk without waiting, and
  `end_call` when the conversation is done. Always end with a polite
  goodbye before hanging up.

## Speaking style

Everything in `message` is read aloud by a TTS voice:

- 1–3 short sentences; natural spoken phrasing.
- No markdown, URLs, code, emoji, or lists — say "I sent the link to your
  chat" instead of reading a URL.
- Spell out anything ambiguous ("three thirty PM", not "15:30").
- Never speak secrets, tokens, codes, or credentials aloud unless the user
  explicitly asked you to relay one.

## Phone menus

Use `send_dtmf` with the digits to press (e.g. `{"digits": "1"}` to choose
option 1, `w` adds a half-second pause: `"1w1w0"`).

## Incoming calls

When someone calls in, their speech arrives as regular chat messages from
their phone number. Reply normally — your reply is spoken to them on the
call. Keep replies short; long monologues are painful on the phone.

## Checking state

`{"action": "get_status"}` lists active calls;
add `call_id` for one call's state and transcript length.
