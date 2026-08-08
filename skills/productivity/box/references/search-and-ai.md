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
| One-off Q&A over 25 files | narrow with search or metadata first |
| Recurring Q&A over a curated knowledge base | [Box Hubs](hubs.md) |
| Preview flexible key-value fields without storing them | `ai:extract` |
| Extract metadata and store it on the file | `ai:extract-structured` |
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

`ai:text-gen` supports exactly one item. Use structured extraction when the schema is known and must be repeatable. Use a metadata template with `--metadata-template` when the Box template is the source of truth.

Do not use a Hub for metadata extraction or text generation. For semantic Q&A across a reusable curated collection, read [Box Hubs](hubs.md). For a one-off request that names more than 25 files, narrow the candidate set before proposing a Hub.

## Diagnose Box AI access

A file that succeeds with `files:get` or search can still fail through Box AI when Box AI is unavailable for the current OAuth identity or account. If the user can preview or download a file but `ai:ask` returns `404 not_found`, do not immediately misdiagnose its collaboration as missing. First verify the current actor and the file permissions:

```bash
box users:get me --json --fields id,name,login
box files:get <FILE_ID> --json --fields id,name,permissions
```

If the file permissions and actor are correct, verify that Box AI is enabled and available for the account or enterprise, that the selected OAuth application has the required AI scope when using a custom Platform App, and that AI units are available. Reauthorize the intended OAuth identity after changing application access, then retry one file before a batch. Do not use impersonation as a fallback; if the wrong identity is selected, switch only with approval to the intended OAuth environment and verify it first.

## Extract and persist file metadata

Treat a request to extract metadata as a structured Box metadata workflow, not a request to merely show an AI response. Unless the user asks to preview only, persist values only when one template can represent every requested field with the correct types.

1. Inspect the file's metadata instances, list the enterprise templates, and retrieve every candidate schema before extracting or writing.
   ```bash
   box files:metadata <FILE_ID> --json
   box metadata-templates --json --fields templateKey,displayName,scope
   box metadata-templates:get <TEMPLATE_KEY> --scope enterprise --json
   ```
2. Compare every requested field to each candidate's field key and type. Select an existing template only when it supports **all** requested fields. Do not attach a semantically unrelated template or one that supports only some fields just to persist a subset.
3. Extract against the selected template. Request only the target file IDs and keep the extraction output in the terminal result rather than downloading the source file.
   ```bash
   box ai:extract-structured --items=id=<FILE_ID>,type=file \
     --metadata-template="type=metadata_template,scope=enterprise,template_key=<TEMPLATE_KEY>" \
     --json
   ```
4. Convert returned values to the selected template's field keys and types. Add a metadata instance when the selected template has none on the file; otherwise replace its extracted fields. Do not write absent, null, incompatible, or truncated values.
   ```bash
   box files:metadata:create <FILE_ID> --scope enterprise --template-key <TEMPLATE_KEY> \
     --data "invoice_number=INV-001" --data "total=#1250.00" --json

   box files:metadata:update <FILE_ID> --scope enterprise --template-key <TEMPLATE_KEY> \
     --replace "invoice_number=INV-001" --replace "total=#1250.00" --json
   ```
   Use the required `#` prefix for float values with `files:metadata:create` or `files:metadata:add`.
5. Retrieve the specific instance and compare every returned field with the intended value. A successful response to the write is not verification. Report the template key, metadata instance ID, file link, and any missing, normalized, or rejected values.
   ```bash
   box files:metadata:get <FILE_ID> --scope enterprise --template-key <TEMPLATE_KEY> --json
   ```

The extraction request authorizes matching per-file metadata writes, so do not ask again after finding a fully compatible existing template. Creating or changing a metadata template is an enterprise-wide schema change: require explicit user approval before doing it. It also requires an OAuth token for a Box Admin or a Co-Admin authorized to create and edit metadata templates. Do not elevate a dedicated Hermes identity for this purpose. If the approved administrator OAuth session is unavailable, leave structured metadata unchanged and ask the administrator to create the template manually or select another supported outcome. If the approved administrator session is available, create a dedicated, semantically appropriate template with stable field keys and correct field types, attach it to the target file, and verify every field. Use full ISO timestamps for date values, such as `2025-03-29T00:00:00Z`.

```bash
box metadata-templates:create --display-name "Invoice extraction" \
  --template-key invoice_extraction \
  --string "Client name" --field-key client_name \
  --number "Invoice amount" --field-key invoice_amount \
  --date "Invoice date" --field-key invoice_date --json
```

If no existing template supports every field and the user does not approve a new one, leave structured metadata unchanged. Explain that the schema is incomplete and ask whether they want a separate sidecar document, an explicitly requested description that fits, or another existing template. Never silently truncate or discard fields.

### File descriptions are not metadata fallback

**Hard rule:** Never use a file description as an automatic substitute for structured metadata. Box file descriptions are limited to 256 characters. Use `box files:update --description` only when the user explicitly requests a description, first verify the complete intended text fits within 256 characters, then read it back and compare it with the intended value. Do not use a description for complete extracted metadata or as a fallback for fields that do not fit a template.

## Confidentiality and AI units

Box AI processes source files through Box's governed AI integration instead of downloading source bodies into Hermes' coding-model context. Box AI responses returned to Hermes can still contain confidential information. Do not claim that no third-party model provider is involved or that content can never be used for training; follow Box's current trust and plan documentation.

Before the first Box AI request, explain that Box AI must be enabled, calls consume AI units, and answers remain constrained by the current actor's permissions. For a material batch, state the file count and ask for confirmation. Do not promise a unit balance or per-call cost unless Box exposes it for the current account.

If Box AI is unavailable or out of units, offer existing metadata/search, a smaller sample, enabling units, or explicit approval for local/external analysis. Never silently fall back to downloading files for an external model.

## Scale

Use `--bulk-file-path` where the command supports it. For hundreds of files, inventory first, sample the schema, confirm unit-consuming scope, and use [Bulk operations](bulk-operations.md). For recurring, high-throughput extraction, evaluate Box Extract rather than simulating a folder-wide workflow through repeated downloads.

## Sources

- [Box AI API](https://developer.box.com/ai/box-ai-api/)
- [Structured metadata extraction](https://developer.box.com/guides/box-ai/ai-tutorials/extract-metadata-structured/)
- [Box AI trust](https://www.box.com/ai/trust/)
- [AI units and plan access](https://support.box.com/hc/en-us/articles/45612941554835-Expanded-AI-API-Access-and-AI-Units-for-Business-Business-Plus-and-Enterprise-Plans)
- [Metadata template permissions](https://developer.box.com/guides/metadata/templates/create/)
