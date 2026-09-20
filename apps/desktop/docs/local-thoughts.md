# Local thoughts in Quick Entry

Quick Entry can save an unfinished thought without sending a prompt or creating
a session. This is an ADHD-friendly capture aid: there is no required title,
category, or task to decide before writing something down.

1. Open Quick Entry with your configured shortcut.
2. Type a fragment and choose **Save thought** (Cmd+S on Mac, Ctrl+S elsewhere).
   Saving works while the gateway is offline, once Desktop has identified the
   current connection and profile.
3. Open **Saved thoughts**, then **Use in input** to return to one. This does not
   send it anywhere. If a draft is already present, **Append to draft** keeps both
   texts. Enter sends through the existing chat path when connected; Shift+Enter
   adds a line break.

Escape and dismissing the window preserve the draft. Drafts also remain
recoverable after an Enter handoff because the existing chat bridge has no
receipt confirming delivery. A persisted notice identifies the attempted handoff.
Enter will not repeat it until you edit the text or choose **Allow another send**;
check the chat first. A main-process rejection is shown without clearing the text.
Forwarding to the main chat window is not proof that the backend accepted it.
A failed local write keeps the text in the window; retry before quitting Desktop.

## Storage and profile changes

Thoughts and drafts live on this device in Desktop's user-data directory, under
`thoughts/`. They are scoped to the connection and profile shown in Quick Entry,
including when the connection points to a remote host. Saving does not call the
gateway or a model. An ambiguous legacy connection cannot use local capture;
ordinary Quick Entry chat remains available.

Profile renames move the local inbox. If that move fails, Desktop shows where
the retained folder can be recovered. Profile deletion retires the inbox before
sending the deletion request so a new profile with the same name starts empty.
A confirmed refusal restores the inbox; an uncertain response retains the
retired folder and reports its recovery location. Retired folders are not
listed in Quick Entry or removed automatically.

This first slice provides capture and retrieval. It does not add Start my day,
reminders, automatic categorization, sync, or a saved-thought deletion UI.
