# Content workflows

Use IDs, not paths, once an item is resolved. If the current OAuth identity cannot see the target, verify the exact item ID and ask the owner to invite that identity to the intended file, folder, or Hub.

## Browse and create folders

```bash
box folders:get <FOLDER_ID> --json --fields id,name,parent,item_collection
box folders:items <FOLDER_ID> --json --max-items 100 --fields id,name,type
box folders:create <PARENT_ID> "Customer-123" --json --fields id,name,parent
```

Duplicate names in one parent return `409`. Reuse the existing folder ID instead of retrying blindly.

## Verify a shared file or folder

When the current OAuth identity receives a file or folder invite, use the ID from its Box URL if available and fetch that exact item. Do not use an absence from folder `0` as proof that access failed; it is only that identity's root listing. If only a name is known, use Box search to resolve the ID, then fetch the item:

```bash
box search "Quarterly plan" --json --limit 20 --fields id,name,type,parent
box files:get <FILE_ID> --json --fields id,name,parent
box folders:get <FOLDER_ID> --json --fields id,name,parent
```

Use [Box Hubs](hubs.md) for a Hub invite: Hubs are not files or folders and are discovered separately.

## Upload, download, and version files

```bash
box files:upload ./artifact.pdf --parent-id <FOLDER_ID> --json --fields id,name,size
box files:get <FILE_ID> --json --fields id,name,size,sha1,parent
box files:download <FILE_ID> --destination . --save-as local-copy.pdf
box files:versions:upload <FILE_ID> ./updated.pdf --json --fields id,name,sha1
box files:versions:list <FILE_ID> --json
box files:versions:download <FILE_ID> <VERSION_ID> --destination . --save-as older.pdf
```

Download source bytes only when the task truly requires local editing or the user explicitly approves external analysis. Prefer a new version over replacing an unrelated file by name.

## Create native Box Notes

When the user asks for a Box Note, create a native note from Markdown through `box request`; do not substitute an uploaded text file named `.boxnote`. Read [REST API fallback](rest-api.md) for the exact request and verification command. Create it immediately when the destination is explicit or unambiguously the actor's root; otherwise ask which folder to use.

## Rename, tag, and move

```bash
box files:update <FILE_ID> --name "Renamed.pdf" --json --fields id,name
box files:update <FILE_ID> --description "Updated by Hermes" --tags "reviewed,2026" --json
box files:move <FILE_ID> <NEW_PARENT_ID> --json --fields id,name,parent
box folders:move <FOLDER_ID> <NEW_PARENT_ID> --json --fields id,name,parent
```

Read back the item or its parent after every write. Moving a folder moves its contents; confirm broad moves before executing them.

## File descriptions

Treat 255 characters as the safe file-description limit; Box can truncate longer values. Never use a description as a fallback for extracted metadata. Set one only when the user explicitly asks for a description, verify that the complete intended text fits before writing, then fetch the file and compare the returned description with the intended value. Use [Search and AI](search-and-ai.md) to persist extracted results as metadata or a JSON sidecar instead.

## Collaborate and share

```bash
box collaborations:create <FOLDER_ID> folder --role editor --login collaborator@example.com --json
box shared-links:create <FILE_ID> file --access company --json
box shared-links:create <FOLDER_ID> folder --access open --json
```

Use the narrowest collaboration role. Creating or widening a shared link changes access, so require explicit confirmation.

## Navigate without changing permissions

Report these links for items already known to the caller; they do not create a shared link:

- File: `https://app.box.com/file/<FILE_ID>`
- Folder: `https://app.box.com/folder/<FOLDER_ID>`

Include the item ID with the link. If a human cannot open an item visible only to the connected Box account, state that rather than creating a link with broader access.

## Read and write metadata

```bash
box files:metadata:get <FILE_ID> --scope global --template-key properties --json
box files:metadata:create <FILE_ID> --scope global --template-key properties \
  --data invoice_id=INV-001 --json
```

`global.properties` is Box's built-in schema-free metadata instance; no template creation is required. Its values are not a reusable typed enterprise schema and cannot be used by the Metadata Query API. Read all existing metadata instances before writing so unrelated properties are preserved. Use [Search and AI](search-and-ai.md) when metadata must be extracted from document content; do not use a partial, unrelated, or incomplete enterprise template.
