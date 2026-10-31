# Zoho Mail

Base: `https://mail.zoho.<dc>/api`. Every path starts with `accounts/<accountId>`; get the numeric id from `zoho_api.py mail-accounts`.

## Find and read

```
mail-accounts                                    # accountId, send-as addresses with display names
mail-folders --account <id>                      # numeric folderId per folder
request GET mail accounts/<id>/messages/view --query folderId=<folderId> --query start=1 --query limit=50
request GET mail accounts/<id>/messages/search --query searchKey=subject:invoice --query limit=20
request GET mail accounts/<id>/folders/<folderId>/messages/<messageId>/content
request GET mail accounts/<id>/folders/<folderId>/messages/<messageId>/attachmentinfo --query includeInline=true
```

- `folderId` must be the number, not a name like `INBOX`.
- `/messages/view` accepts `folderId`, `start`, `limit`. Extra parameters such as `sortOrder` return `EXTRA_PARAM_FOUND`; sort in Python by `receivedTime` (epoch milliseconds).
- `/messages/search` takes only `searchKey` and `limit`. Use a field prefix: `subject:`, `from:`, `to:`, `entire:`. A bare keyword returns HTTP 400. One term per query.
- The listing returns subjects HTML escaped (`&#39;`). Unescape before comparing.
- Content is HTML. Screenshots pasted into a mail are inline parts; `attachmentinfo?includeInline=true` lists them and `.../attachments/<attachmentId>` downloads one.
- Paths that do not exist and only waste a try: `folders/<id>/messages` (list), `messages/<id>?includeContent=true`, `messages/drafts`.

## Send a new mail

```
mail-send --account <id> --from you@example.com --to someone@example.org --subject "Offer" --html-file offer.html
```

Without `--send` this is a dry run. `--draft` stores it in Drafts. `--from` must appear under `send_as` in `mail-accounts`, otherwise `INVALID_SENDER_ADDRESS`. Aliases send with the display name configured for them; check it in `mail-accounts` before mailing customers.

## Reply in the thread

```
mail-reply --account <id> --message <messageId> --from you@example.com --to someone@example.org --subject "Re: Offer" --html-file answer.html --bcc you@example.com
```

This calls `POST accounts/<id>/messages/<messageId>` with `"action": "reply"`, the only way the API links the answer to the original thread. What does not work:

- `POST accounts/<id>/messages` with `Re:` in the subject: arrives as a separate mail in Zoho.
- An `inReplyTo` or `references` field on that call: ignored, or rejected with `EXTRA_KEY_FOUND_IN_JSON`.

There is no draft mode for threaded replies. Show the user the text, keep the dry run output, and run with `--send` after approval.

The `--bcc` to the sending address puts a copy into the inbox so the conversation is visible there too.

## Attachments

Two steps: upload, then reference the upload in the message body.

1. Upload (multipart, field name `attach`): `POST accounts/<id>/messages/attachments?uploadType=multipart&isInline=false`. The response `data` is a list of `{storeName, attachmentName, attachmentPath}`.
2. Send or save the draft as JSON with `"attachments": [ ...those objects... ]` as a real array.

Building the message itself as multipart fails with `INVALID_INPUTSTREAM` once the body is longer.

## Move and flag

```
request PUT mail accounts/<id>/updatemessage --data '{"mode":"moveMessage","messageId":[<messageId>],"destfolderId":<folderId>}'
```

Needs `ZohoMail.messages.UPDATE`. The REST API has no "mark as read" for messages.

## After sending

The response `data.messageId` proves the request was accepted, not that it arrived. Read the Sent folder (`mail-folders`, then `messages/view` on it) and show the entry to the user.
