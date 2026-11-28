# Zoho WorkDrive

Base: `https://workdrive.zoho.<dc>/api/v1`. The API speaks JSON:API: send and accept `application/vnd.api+json`. `Accept: application/json` returns HTTP 415 with an empty body. The helper sets the headers.

## Find your way in

```
request GET workdrive users/me                          # data.id = user id, attributes.preferred_team_id
request GET workdrive users/<userId>/privatespace       # data[0].id = personal space folder id
request GET workdrive teams/<teamId>/teamfolders         # team folders (no users.READ needed)
wd-list <folderId>                                      # contents: id, name, type, size
request GET workdrive files/<fileId>                    # metadata, attributes.parent_id
```

- Search is not usable: `files/search` returns 400 or empty results for files that exist. Walk folders with `wd-list` and filter by name in Python.
- A recently touched file can be found through `users/<userId>/recentfiles`; then follow `parent_id` upward with `files/<id>` to reconstruct its folder path.
- Listing teams (`users/<userId>/teams`) needs the scope `WorkDrive.users.READ` (error `F7007` without it).

## Create a folder

```
wd-mkdir <parentId> "Reports 2026"
```

Raw form: `POST files` with `{"data":{"attributes":{"name":"...","parent_id":"..."},"type":"files"}}`. Keep the returned `data.id` instead of listing the parent again.

## Upload

```
wd-upload <folderId> report.pdf chart.png
```

- Multipart to `POST upload` with form fields `parent_id`, `override-name-exist` and the file in the field `content`. Parameters belong in the form body, not the query string.
- `override-name-exist=true` does not replace the existing file. WorkDrive keeps it, adds a version and renames the new upload to `<name>.Final.<ext>`; the extra dot makes the file unusable for many recipients. The helper always sends `false`; upload revisions under a new name.
- The helper downloads every uploaded file again and compares SHA-256 (`sha256_match`). It also checks that the stored name equals the local name. Only `"ok": true` counts as delivered.
- Large files: upload one after another in one process; the helper raises the timeout for downloads to 600 seconds.

## Download

```
request GET workdrive download/<fileId>      # through the helper only for small text files
```

Use `https://workdrive.zoho.<dc>/api/v1/download/<fileId>`. The `download_url` attribute in file metadata points to a separate download host that rejects the normal OAuth token (`INVALID_OAUTHSCOPE`); do not follow it.

## Rename, move, trash (ask first)

All `PATCH files/<id>` with the full envelope `{"data":{"attributes":{...},"type":"files","id":"<id>"}}`:

- rename: `"name": "new.pdf"`
- move: `"parent_id": "<folderId>"`
- trash: `"status": "51"` (the number as a string; the word `TRASH` returns HTTP 500). Trash is reversible in the web app. Permanent deletion needs `WorkDrive.files.DELETE`.
