# Email Attachments → Drive Shareable Links

Use when the user says "give me the links for those plans/PDFs" after you've
located email attachments — deliver Drive links instead of (or in addition to)
inline MEDIA files, so they can forward to a group (landowners, vendors, etc.).

## Steps

1. Download attachments via Gmail API:

```python
import sys, base64
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')  # resolve first!
resp = gmail.users().messages().attachments().get(
    userId='me', messageId=MSG_ID, id=ATT_ID).execute()
raw = base64.urlsafe_b64decode(resp['data'] + '=' * (-len(resp['data']) % 4))
# VERIFY: len(raw) == the size field from the message metadata
with open('/tmp/out.pdf', 'wb') as f:
    f.write(raw)
```

2. Upload binary to Drive. `files().create()` does NOT take a `media_path`
   kwarg — it raises `TypeError: Got an unexpected keyword argument media_path`.
   Use `MediaFileUpload` from `googleapiclient.http`:

```python
from googleapiclient.http import MediaFileUpload

drive = build_service('drive', 'v3', service_name='google-draas')

folder = drive.files().create(body={
    'name': 'Project - Plan Set (date)',
    'mimeType': 'application/vnd.google-apps.folder'}, fields='id,name').execute()

media = MediaFileUpload('/tmp/out.pdf', mimetype='application/pdf', resumable=False)
f = drive.files().create(body={'name': 'plan.pdf', 'parents': [folder['id']]},
                         media_body=media, fields='id,name,webViewLink').execute()
# shareable with anyone holding the link:
drive.permissions().create(fileId=f['id'],
                           body={'type': 'anyone', 'role': 'reader'}).execute()
print(f['webViewLink'])
print('https://drive.google.com/drive/folders/' + folder['id'])
```

## Notes / pitfalls

- Always `gws_resolve_account()` first for the right `service_name`. For a
  session user like Bharat that is usually `google-draas` (their own token),
  but never assume.
- Creating a Google-native folder via Drive API metadata (no media) is fine
  without MediaFileUpload; only real files need it.
- `anyone/reader` permission makes the link work for external recipients —
  explicitly state to the user that the files are "anyone with link can view"
  so they know it forwards outside the org.
- Pair with the attachment-size check (base64 length vs metadata `size`) —
  wrong decoder silently truncates (pitfall 15 in SKILL.md).
- Return BOTH the folder link (all files in one) and per-file links; users
  preparing a presentation/meeting paste the folder link into a deck.