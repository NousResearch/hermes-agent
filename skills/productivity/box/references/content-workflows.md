# Content Workflows

CLI examples run via `terminal`. Read `references/auth-and-setup.md` if the service account lacks access to target folders.

## Upload a file

```bash
box files:upload ./artifact.pdf --parent-id <FOLDER_ID> --json --fields id,name,size
```

- Set destination folder ID first; handle name conflicts explicitly.
- Large files may need chunked upload — see `box files:upload --help`.
- Verify: list parent folder and confirm returned `id`.

Docs: https://developer.box.com/reference/post-files-content/

## Create folders

```bash
box folders:create <PARENT_ID> "Customer-123" --json --fields id,name
```

Persist returned folder IDs. Duplicate names in the same parent return `409`.

## List folder items

```bash
box folders:items <FOLDER_ID> --json --max-items 100 --fields id,name,type
```

Paginate fully for bulk operations — do not assume one page covers all items.

## Download a file

```bash
box files:download <FILE_ID> ./local-copy.pdf
```

Fetch metadata first to confirm the correct file ID:

```bash
box files:get <FILE_ID> --json --fields id,name,size,sha1
```

## Shared links

Confirm with the user before widening access.

```bash
box shared-links:create <FILE_ID> file --access company --json
box shared-links:create <FOLDER_ID> folder --access open --json  # widest — confirm intent
```

## Collaborations

```bash
box collaborations:create <FOLDER_ID> collaborator@example.com editor --json
```

Prefer folder-level collaboration when multiple files share access. Use the narrowest role that satisfies the request.

## Move file or folder

```bash
box files:update <FILE_ID> --parent-id <NEW_PARENT_ID> --json
box folders:update <FOLDER_ID> --parent-id <NEW_PARENT_ID> --json
```

Moving a folder moves all contents. Target folder name conflicts return `409`. For bulk moves see `references/bulk-operations.md`.

## Metadata

```bash
box files:metadata:get <FILE_ID> enterprise properties --json
box files:metadata:create <FILE_ID> enterprise properties --json -d '{"invoice_id":"INV-001"}'
```

Read template definitions before writing. Keep template keys in config, not scattered through code.

## Verification pattern

Every write: read-back with the same actor — list parent folder or get the object and confirm `id` + `name`.
