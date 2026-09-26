# SimplePost MCP reference

The hosted SimplePost MCP endpoint is:

```text
https://app.simplepost.social/mcp
```

It uses OAuth with authorization-server discovery, dynamic client registration, PKCE, token exchange, and refresh. Hermes manages those protocol details. Social account connection, disconnection, and reauthorization remain web-app actions.

## Tool workflow

1. Call `list_accounts` before a write. Use returned `accountId` values; never infer them from platform names or prior text.
2. Check returned trial allowance or posting eligibility. Allowance can be shared by accounts on the same platform, while drafts do not consume it.
3. If media has a public URL, pass it directly in `media`. If Hermes supplies an attachment as a structured file parameter, call `upload_media` and use the returned URL.
4. Call `create_post` with `postingMode: "now"`, `"schedule"`, or `"draft"` when the action and inputs are explicit.
5. Supply a new unique `idempotencyKey` for each newly intended post. Reuse that same key and unchanged content only when retrying an uncertain result.
6. Use `preview_post` for a text and structured preflight, and `show_post_preview` when the user asks to see the rendered post.
7. Use `validate_post` only when the user requests validation or troubleshooting without creating anything.
8. Use `inspect_posts` to list or inspect drafts, scheduled posts, delivered posts, and failed posts.
9. Use `update_scheduled_post` and `discard_scheduled_post` only for drafts or future scheduled posts.
10. After creating or updating a draft or scheduled post, call `show_post_preview` with the returned post ID when a rendered review is useful.
11. Use `get_schedule` for structured schedule data and `show_schedule` for the calendar UI.
12. To quote an earlier post, find the exact record with `inspect_posts` and pass its `id` as `quotePostId`. Schedule a quote after its source when the source is still scheduled.

Do not call tools for unsupported actions. Already-published social posts must be edited or deleted on the destination platform.

## Naming contract

- `list_*`: enumerate collections.
- `get_*`: retrieve text and structured data without UI.
- `validate_*`: check without writing.
- `preview_*`: compute a non-writing preflight without UI.
- `show_*`: render MCP Apps UI with a text fallback.
- `create_*`, `update_*`, `discard_*`, `upload_*`: mutate stored or external state.

Hermes receives the text and structured fallback even when it does not render MCP Apps UI.

## Scheduling

Pass a future ISO 8601 datetime with an offset or `Z`:

```json
{
  "postingMode": "schedule",
  "scheduledFor": "2026-10-03T09:00:00+02:00"
}
```

Never pass a date-only string or a naive local time. Resolve “tomorrow at 9” using the user's timezone before calling the tool.

## Content and account overrides

The root message is the first post. Use `thread` for ordered follow-up segments on X, Bluesky, Threads, and Telegram. Other platforms receive only the root and should return a warning.

Use `accountOverrides` for platform-specific text and `accountOptions`, keyed by connected account ID, for platform-specific settings. Do not rewrite supplied content for each platform unless requested.

## Media

MCP media items use public or SimplePost-managed URLs:

```json
{ "type": "image", "url": "https://cdn.example.com/image.jpg" }
```

Videos may include `thumbnailUrl`. Instagram requires an image or video, and YouTube requires a video.

SimplePost imports public media into managed storage before saving or publishing. If `validate_post` or `preview_post` returns `fittedMedia`, pass those returned items to `create_post` so the checked bytes are reused. Preserve `filename`, `size`, `durationSec`, and `thumbnailUrl` returned by `upload_media`.

Use `upload_media.file` only for a structured file parameter registered by Hermes. Never construct one from a displayed path, filename, file ID, base64 data, or an earlier message. If the structured reference is unavailable and the media has a public URL, retry once with `upload_media.url`; otherwise ask the user to reattach the file or provide a public URL.

## Result handling

For immediate publishing, inspect:

- `summary.overallSuccess`;
- `postingResults[].success`;
- `postingResults[].message` and `error`;
- `postingResults[].threadResults` for threads.

For every created or updated post, also inspect:

- `validation.isValid`, errors, and warnings;
- `post.repostEnabled`, `post.repostDueAt`, and `post.repostStatus`.

Always show the exact content that was previewed, created, scheduled, drafted, edited, or discarded.

Drafts are saved even when `validation.isValid` is false. Report that the draft was saved and separately list what must be fixed. Do not present an invalid draft as ready to publish.

An immediate publish can take minutes. If the result is uncertain, retry only with the same idempotency key and unchanged content. A new key can create a duplicate post.

## TikTok options

TikTok posts default to public when privacy is omitted and the connected account permits it. Override the audience through `accountOptions`, keyed by connected account ID:

```json
{
  "accountOptions": {
    "TIKTOK_ACCOUNT_ID": { "privacyLevel": "SELF_ONLY" }
  }
}
```

Supported values are `PUBLIC_TO_EVERYONE`, `MUTUAL_FOLLOW_FRIENDS`, `FOLLOWER_OF_CREATOR`, and `SELF_ONLY`. Explicit privacy and legacy `visibility` choices take precedence over the default. Existing choices remain when editing without replacement settings.

If the requested audience is unavailable, surface the error and call `get_tiktok_creator_info` to inspect supported choices. Ask for an alternative; never silently fall back.

TikTok supports 1–35 JPEG or WebP photos in the root `media` array. Do not mix images and videos. Useful account-scoped options include:

- `autoAddMusic: true` for TikTok-selected recommended music on a photo Direct Post;
- `publishMode: "draft"` to upload to the TikTok inbox for manual editing and publication;
- `photoCoverIndex` for the zero-based cover image;
- `title` and `description` for photo posts.

Top-level `postingMode: "draft"` saves a SimplePost draft and performs no TikTok upload. For an inbox upload now, use top-level `postingMode: "now"` and account-level `publishMode: "draft"`. Return the tool's message because the user must finish an inbox upload in TikTok.

## Recovery and limits

- OAuth tokens belong in Hermes' credential storage, never in prompts or committed files.
- If authentication expires, use `terminal(command="hermes mcp login simplepost", timeout=315)` and then reload MCP.
- If `hermes mcp test simplepost` fails before login, complete OAuth and repeat the test once.
- Do not retry billing, allowance, validation, or platform-policy failures to bypass them.
- Do not claim engagement analytics from `inspect_posts`; it exposes content and delivery state only.
