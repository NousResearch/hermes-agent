# Box Hubs

Use a Box Hub for recurring Q&A over a curated knowledge base. A direct Box AI Ask request handles up to 25 selected files; a Hub request sends one `hubs` item and searches the Hub's indexed content. Do not use a Hub for metadata extraction or text generation.

## Check availability and discover an existing Hub

The Box Free Developer Plan includes the Hubs API, Box AI APIs, and a monthly AI-unit allowance for building and testing. Do not apply Box web-app plan wording as a blanket API restriction. The CLI calls the same APIs and does not bypass account entitlements: production availability still depends on the organization's plan and Box configuration.

Before the first Hub AI request, explain that Box AI must be enabled and consumes AI units; do not wait for acknowledgement. Explain that answers only use indexed files the current actor can access. Hubs are not files or folders: never use `folders:items 0` to discover or reject a Hub invitation. Confirm the current actor, then list accessible Hubs before proposing a new one:

```bash
box users:get me --json --fields id,name,login
box hubs --scope all --max-items 1000 --json
box hubs --query "Product" --scope all --sort relevance --json
box hubs:get <HUB_ID> --json
box hubs:items <HUB_ID> --max-items 100 --json
```

For a known Hub URL or ID, run `box hubs:get <HUB_ID>` directly even if the list is empty. Report each Hub as `https://app.box.com/hubs/<HUB_ID>`. Check `is_ai_enabled` before asking a question, then make one bounded Hub Ask request to verify actual API availability. Box AI for Hubs must have been enabled before the Hub was created so Box can index its content. If Hub AI is unavailable, distinguish a disabled feature, a Hub created before AI enablement, indexing delay, missing Hub collaboration, missing access to underlying files, and exhausted AI units; do not silently download source files into Hermes' model context.

## Ask questions across a Hub

Use one Hub item and `single_item_qa`. Request citations so Hermes can report the source files behind an answer. Use `box request` (or the SDK) for Hub Q&A rather than relying on `box ai:ask`, whose installed CLI versions may not accept Hub item types. This uses the Box AI Ask endpoint; the `box-version: 2025.0` header is required for `/hubs` management endpoints, not this request.

```bash
box request /ai/ask -X POST \
  --body '{"mode":"single_item_qa","items":[{"id":"<HUB_ID>","type":"hubs"}],"prompt":"Summarize the approved renewal terms and cite each source.","include_citations":true}' \
  --json
```

State the Hub ID and navigation link with the answer. List cited file IDs, names, and file links when Box returns citations. Treat an answer as bounded by indexed, accessible Hub content; do not claim it searched files that have not indexed or that the actor cannot access.

## Create and populate a Hub

Do not create a Hub automatically. For Q&A over more than 25 files or a reusable curated collection, discover an existing accessible Hub first. If none fits, offer to create a curated Hub and obtain explicit approval before creating or populating it. If the user declines, narrow the one-off scope with search or metadata instead.

After approval, create it, report its link, and verify it:

```bash
box hubs:create "Policy knowledge base" --description "Approved policy reference" --json
box hubs:get <HUB_ID> --json
```

Adding an item curates a reference; it does not move the underlying file or folder. A clearly requested small addition may proceed without a redundant prompt. Confirm before bulk additions or removals, then verify every returned result and read back the Hub items. The API can return partial success for multi-item changes, so do not treat a successful request alone as proof that every item was added.

```bash
box hubs:items:manage <HUB_ID> \
  --add id=<FILE_ID>,type=file --json
box hubs:items:manage <HUB_ID> \
  --add id=<FOLDER_ID>,type=folder --json
box hubs:items <HUB_ID> --max-items 100 --json
```

Without `parent-id`, the CLI adds the item to the first Item List block. To target a specific Item List block, first list pages with `box hubs:document:pages <HUB_ID> --json`, retrieve blocks with `box hubs:document:blocks <HUB_ID> <PAGE_ID> --json`, then pass the returned Item List block ID as `parent-id`.

Confirm before enabling or disabling Hub AI, deleting or copying a Hub, or changing shared access. Verify each change with `box hubs:get`, `box hubs:items`, or `box hubs:collaborations`:

```bash
box hubs:update <HUB_ID> --ai-enabled --json
box hubs:collaborations <HUB_ID> --max-items 100 --json
box hubs:collaborations:create <HUB_ID> --role viewer --user-id <USER_ID> --json
```

## Handle indexing, permissions, and limits

Newly added content usually indexes within minutes but can take up to an hour. Verify the item addition, wait or retry a bounded number of times, and report a retryable indexing state instead of declaring the source absent. Diagnose permissions separately: a successful `box hubs` or `box hubs:get` proves Hub access, not access to every underlying file. Hub answers respect the querying actor's access to underlying files.

Box AI for Hubs has a service limit per Hub and across the enterprise. Box's dedicated Hubs guidance currently documents 20,000 files per Hub; verify current account or product documentation when operating near the boundary. Do not present that number as an immutable guarantee. Only the first 4 MB of a supported document's text representation is indexed. Explain AI-unit use before the first request and confirm a material batch or broad Hub population.

## Sources

- [Box Hubs API overview](https://developer.box.com/guides/hubs-api/)
- [Box Free Developer Plan](https://developer.box.com/guides/getting-started/free-developer-plan/)
- [Box AI Ask API](https://developer.box.com/reference/post-ai-ask/)
- [Ask questions about a Hub](https://developer.box.com/guides/box-ai/ai-tutorials/ask-questions/)
- [Box AI for Hubs](https://support.box.com/hc/en-us/articles/29347206309395-Box-AI-for-Hubs)
- [Box Hubs limits](https://support.box.com/hc/en-us/articles/28323495455123-Box-Hubs-Known-Issues-and-Limitations)
