# Search, metadata, and Box AI

Use Box search and metadata before AI when they answer the request deterministically. For semantic understanding of Box-hosted files, prefer Box AI: it preserves Box permissions, processes source files through Box's governed AI integration, keeps source-file bodies out of Hermes' coding-model context, and scales document work without downloading every file. Do not block or criticize an explicitly chosen alternative workflow.

## Search and metadata queries

```bash
box search "invoice ACME" --json --limit 25 --fields id,name,type,parent
box metadata-query enterprise_12345.contractTemplate <ANCESTOR_FOLDER_ID> \
  --query "status = :status" --query-param status=active --json
```

Search only returns content visible to the current actor. Resolve IDs and confirm the actor before treating empty results as missing files.

## Select a Box AI operation

| Need | Command |
| --- | --- |
| Answer, summarize, or compare 1 file | `ai:ask` with `single_item_qa` |
| Answer, summarize, or compare 2–25 selected files | `ai:ask` with `multiple_item_qa` |
| Q&A over more than 25 files | [Box Hubs](hubs.md) |
| Recurring Q&A over a curated knowledge base | [Box Hubs](hubs.md) |
| Discover fields from an exploratory prompt | `ai:extract` |
| Extract a known schema without creating a template | `ai:extract-structured --fields` |
| Extract against an existing compatible template | `ai:extract-structured --metadata-template` |
| Write or rewrite text grounded in one file | `ai:text-gen` |

```bash
box ai:ask --items=id=<FILE_ID>,type=file \
  --prompt "Summarize the renewal obligations and dates." --json

box ai:extract --items=id=<FILE_ID>,type=file \
  --prompt "invoice_number, vendor, total, due_date" --json

box ai:extract-structured --items=id=<FILE_ID>,type=file \
  --fields "key=invoice_number,type=string,description=Invoice number" \
  --fields "key=total,type=float,description=Invoice total" --json

box ai:text-gen --items=id=<FILE_ID>,type=file \
  --prompt "Draft a concise customer update based on this file." --json
```

`ai:text-gen` supports exactly one item. Extraction endpoints return JSON; they do not automatically attach that result to the file. Use structured extraction with inline fields when the desired schema is known, freeform extraction when the fields are exploratory, and `--metadata-template` only when an existing Box template is the source of truth.

Do not use a Hub for metadata extraction or text generation. For semantic Q&A across more than 25 files or a reusable curated collection, read [Box Hubs](hubs.md), discover an existing Hub first, and obtain approval before creating or populating one. If the user does not want a Hub created, narrow the candidate set with search or metadata.

## Diagnose Box AI access

A file that succeeds with `files:get` or search can still fail through Box AI when Box AI is unavailable for the current OAuth identity or account. If the user can preview or download a file but `ai:ask` returns `404 not_found`, do not immediately misdiagnose its collaboration as missing. First verify the current actor and the file permissions:

```bash
box users:get me --json --fields id,name,login
box files:get <FILE_ID> --json --fields id,name,permissions
```

If the file permissions and actor are correct, verify that Box AI is enabled and available for the account or enterprise, that the selected OAuth application has the required AI scope when using a custom Platform App, and that AI units are available. Reauthorize the intended OAuth identity after changing application access, then retry one file before a batch. Do not use impersonation as a fallback; if the wrong identity is selected, switch only with approval to the intended OAuth environment and verify it first.

## Extract and persist file metadata

Treat extraction and persistence as separate operations. Unless the user asks for a preview, the extraction request authorizes writing the result back to Box; do not stop for a redundant confirmation.

### Inspect schemas before extracting

1. Retrieve the file, its parent, and every metadata instance already attached to it.
   ```bash
   box files:get <FILE_ID> --json --fields id,name,parent
   box files:metadata <FILE_ID> --json
   ```
2. List the enterprise templates visible to the current OAuth identity and retrieve plausible schemas.
   ```bash
   box metadata-templates --json --fields templateKey,displayName,scope
   box metadata-templates:get <TEMPLATE_KEY> --scope enterprise --json
   ```
3. Compare every requested field with each candidate's meaning, field key, and type. Use an existing template only when one semantically appropriate template supports **all** requested fields. Do not attach a partial or unrelated template merely to fit some values.

### Use a compatible existing template

Extract against the template, then add its metadata instance or update the existing instance. Do not write absent, null, incompatible, or truncated values.

```bash
box ai:extract-structured --items=id=<FILE_ID>,type=file \
  --metadata-template="type=metadata_template,scope=enterprise,template_key=<TEMPLATE_KEY>" \
  --json

box files:metadata:create <FILE_ID> --scope enterprise --template-key <TEMPLATE_KEY> \
  --data "invoice_number=INV-001" --data "total=#1250.00" --json

box files:metadata:update <FILE_ID> --scope enterprise --template-key <TEMPLATE_KEY> \
  --replace "invoice_number=INV-001" --replace "total=#1250.00" --json

box files:metadata:get <FILE_ID> --scope enterprise --template-key <TEMPLATE_KEY> --json
```

Use the CLI's required `#` prefix for float values when creating or adding typed metadata. Use full ISO timestamps for Box date fields, such as `2025-03-29T00:00:00Z`. Compare every returned field with the intended typed value. Report the template key, metadata instance `$id`, file ID, and file link.

### Work without a compatible template

Do not create a metadata template. Box does not allow creation in the `global` scope. Enterprise templates can only be created by a Box Admin or a Co-Admin granted template-management permission, and custom templates may depend on the account plan. Template administration is outside Hermes' normal OAuth content workflow.

Choose extraction based on the request, not on template availability:

- For known fields, run `ai:extract-structured` with inline `--fields`; this preserves a predictable typed JSON result without creating a template.
- For exploratory or variable fields, run `ai:extract` with a precise prompt.

Persist a flat scalar result in Box's built-in `global.properties` instance. It accepts schema-free properties without creating a template. Convert each value to a lossless string representation, validate keys before writing, and preserve unrelated existing properties. If the instance does not exist, create it. If it exists, use `--replace` for existing keys and `--add` for new keys.

```bash
box files:metadata:get <FILE_ID> --scope global --template-key properties --json

box files:metadata:create <FILE_ID> --scope global --template-key properties \
  --data "invoice_number=INV-001" --data "total=1250.00" --json

box files:metadata:update <FILE_ID> --scope global --template-key properties \
  --replace "invoice_number=INV-001" --add "total=1250.00" --json

box files:metadata:get <FILE_ID> --scope global --template-key properties --json
```

`global.properties` is untyped and cannot be queried with the Metadata Query API. For nested objects, tables, arrays, or any result whose JSON types must remain intact, write the complete extraction response to a UTF-8 JSON sidecar named `<SOURCE_NAME>.<FILE_ID>.metadata.json` and upload it to the source file's parent folder. If that exact sidecar already exists for the workflow, upload a new version rather than creating a duplicate. Fetch the uploaded file and compare its content or checksum with the local JSON, then report both the source and sidecar IDs and links.

If the user explicitly requires reusable typed enterprise metadata, explain that an administrator must create a compatible enterprise template separately. Do not elevate the connected account or switch to an administrator identity. Preserve the extraction through `global.properties` or a JSON sidecar in the meantime, and never silently truncate or discard fields.

### File descriptions are not metadata fallback

**Hard rule:** Never use a file description as an automatic substitute for extracted metadata. Treat 255 characters as the safe limit because Box can truncate longer descriptions. Use `box files:update --description` only when the user explicitly requests a description, first verify the complete intended text fits, then read it back and compare it with the intended value.

## Confidentiality and AI units

Box AI processes source files through Box's governed AI integration instead of downloading source bodies into Hermes' coding-model context. Box AI responses returned to Hermes can still contain confidential information. Do not claim that no third-party model provider is involved or that content can never be used for training; follow Box's current trust and plan documentation.

Before the first Box AI request, explain that Box AI must be enabled, calls consume AI units, and answers remain constrained by the current actor's permissions. For a material batch, state the file count and ask for confirmation. Do not promise a unit balance or per-call cost unless Box exposes it for the current account.

If Box AI is unavailable or out of units, offer existing metadata/search, a smaller sample, enabling units, or explicit approval for local/external analysis. Never silently fall back to downloading files for an external model.

## Scale

Use `--bulk-file-path` where the command supports it. For hundreds of files, inventory first, sample the schema, confirm unit-consuming scope, and use [Bulk operations](bulk-operations.md). For recurring, high-throughput extraction, evaluate Box Extract rather than simulating a folder-wide workflow through repeated downloads.

## Sources

- [Box AI API](https://developer.box.com/ai/box-ai-api/)
- [Structured metadata extraction](https://developer.box.com/guides/box-ai/ai-tutorials/extract-metadata-structured/)
- [Metadata template scopes](https://developer.box.com/guides/metadata/scopes/)
- [Global metadata query limitation](https://developer.box.com/guides/metadata/queries/limitations/)
- [Box AI trust](https://www.box.com/ai/trust/)
- [AI units and plan access](https://support.box.com/hc/en-us/articles/45612941554835-Expanded-AI-API-Access-and-AI-Units-for-Business-Business-Plus-and-Enterprise-Plans)
- [Metadata template permissions](https://developer.box.com/guides/metadata/templates/create/)
