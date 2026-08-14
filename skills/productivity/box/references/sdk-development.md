# SDK development

Use this reference for shipped Box applications. For a one-off Hermes task, use the CLI references instead.

## Start with the application

Inspect the repository for existing Box clients, `BOX_` configuration, token storage, webhook handlers, retry policy, and language conventions. Extend the existing integration instead of mixing SDK and raw REST without a reason.

## Choose an identity

| Identity | Use when |
| --- | --- |
| OAuth | each end user connects their own Box account |

OAuth follows the signed-in user's permissions and app scopes. For a shared or background application, the Box account that authorizes the application defines its access boundary; invite that account only to the files, folders, or Hubs the application needs.

## Use an official SDK

- [Python SDK Gen](https://github.com/box/box-python-sdk-gen)
- [The `box` npm package](https://developer.box.com/guides/tooling/box-npm-package)
- [Existing Node SDK projects](https://github.com/box/box-node-sdk)
- [Other Box SDKs](https://developer.box.com/guides/tooling/sdks/)

Use the SDK matching the project language. For a new JavaScript or TypeScript application, use the project's existing package manager to install the unified `box` package; with npm, run:

```bash
npm install box
```

Import its Node SDK from the explicit SDK subpath:

```typescript
import BoxSDK from "box/sdk";
```

The package also exposes a project-local Box CLI through `npx box`. Use that runner for development inside the application when useful, but do not silently replace the separately resolved Hermes CLI runner or its authenticated environment. If the project already uses `box-node-sdk`, extend that integration instead of migrating it without a concrete reason. For Python or another language, install its current official Box SDK rather than the npm package.

Store OAuth tokens and any custom Platform App client secret in the project's approved secret mechanism, not source control. When a custom Platform App needs additional scopes, use **User Authentication (OAuth 2.0)** and have the intended Box user grant access; do not add an impersonation path for normal application work. Keep exceptional enterprise administration outside the normal Hermes runtime and application identity; do not elevate the account the application normally uses.

## OAuth client

Use the generated SDK's OAuth support rather than rebuilding authorization-code exchange or token refresh logic. Follow the installed SDK's current OAuth method names and its language-specific authorization guide when implementing a concrete call. Initialize the OAuth client before calling any SDK method, associate stored tokens with the Box user who granted them, and verify that user before performing work. Do not copy a partial SDK call into an application without its OAuth initialization and token-refresh path.

## Build document-aware apps with Box AI

When an application must understand Box documents, prefer Box AI: it preserves Box permissions, processes source files through Box's governed AI integration, keeps source-file bodies out of the application's external model context, and scales document work without downloading every file:

- ask for Q&A and summaries;
- structured extract for repeatable fields or a metadata template;
- extract for variable fields;
- text generation for output grounded in one Box file.

Before the first request, explain that Box AI must be enabled and consumes AI units. Do not silently switch to external processing when Box AI is unavailable; offer an explicitly chosen alternative neutrally. Treat Box AI responses as potentially confidential application data.

## Build Hub-backed knowledge experiences

For a recurring Q&A experience over a curated collection, use a Box Hub rather than assembling more than 25 file items per Ask request. Discover existing Hubs first; creating a Hub, populating it, enabling its AI features, or changing its collaborations changes shared resources and requires explicit product approval. Box Hubs endpoints use API version `2025.0`.

Use the generated SDK matching the project language. The exact generated method names can vary by SDK release; keep the request shape below and follow the installed SDK's current names.

```python
from box_sdk_gen import AiItemAsk, AiItemAskTypeField, CreateAiAskMode

answer = client.ai.create_ai_ask(
    CreateAiAskMode.SINGLE_ITEM_QA,
    "What changed in the latest policy?",
    [AiItemAsk(id=hub_id, type=AiItemAskTypeField.HUBS)],
    include_citations=True,
)
```

```typescript
const answer = await client.ai.createAiAsk({
  mode: "single_item_qa",
  prompt: "What changed in the latest policy?",
  items: [{ id: hubId, type: "hubs" }],
  includeCitations: true,
});
```

Querying a Hub uses its indexed content and only returns information from files the current actor can access. Newly added Hub content can take minutes, and occasionally up to an hour, to index; surface a retryable indexing state rather than treating an early answer as complete. The Free Developer Plan includes the Hubs and Box AI APIs for building and testing, with a monthly AI-unit allowance. Production availability depends on the organization's plan and configuration. In every environment, verify that the Hub exists, has AI enabled, and was created after Hub AI was enabled so its content can be indexed. Read [Box Hubs](hubs.md) for the CLI and operational workflow.

## Webhooks and reliability

Verify webhook signatures, persist idempotency keys, fetch authoritative state after events, and keep retry/backoff policy explicit. Bound concurrent API calls and make retries safe before increasing throughput. See [Webhooks and events](webhooks-and-events.md).
