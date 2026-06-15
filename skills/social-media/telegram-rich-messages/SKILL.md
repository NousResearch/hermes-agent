---
name: telegram-rich-messages
description: "Use when composing, sending, or implementing Telegram Bot API rich messages: structured Markdown/HTML, rich_message payloads, drafts, tables, media blocks, details, math, and Hermes Telegram formatting caveats."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [telegram, bot-api, rich-messages, rich-formatting, markdown, html, gateway]
    homepage: https://core.telegram.org/bots/api#rich-message-formatting-options
    related_skills: [hermes-agent]
---

# Telegram Rich Messages and Rich Formatting

## Overview

Telegram Bot API 10.1 added **Rich Messages**: structured messages that go beyond classic `sendMessage` formatting. Use them for headings, lists, tables, media blocks, block quotes, pull quotes, collapsible details, footnotes/references, formulas, maps, collages, slideshows, and streamed AI draft previews.

Rich messages are sent with `sendRichMessage` using an `InputRichMessage` object. That object uses **exactly one** of:

- `markdown`: Rich Markdown content.
- `html`: Rich HTML content.

Classic Telegram message formatting still exists (`sendMessage` + `parse_mode=MarkdownV2` or `HTML`). Do not confuse it with rich messages: rich messages have a different payload shape, larger limits, richer block syntax, and different escaping rules.

## When to Use

Use this skill when:

- The user asks to use Telegram's new rich message/formatting features.
- You need to compose Telegram-native structured output with headings, tables, details, math, media blocks, or footnotes.
- You are implementing or reviewing Hermes Telegram gateway support for `sendRichMessage`, `sendRichMessageDraft`, or `editMessageText.rich_message`.
- You need to decide whether a Telegram response should be classic MarkdownV2/HTML or a Bot API rich message.

Do **not** use rich messages for every response. For short chat replies, classic Hermes/Telegram markdown is simpler and more portable. Reach for rich messages when structure meaningfully improves mobile readability or when a Bot API method explicitly needs `rich_message`.

---

## API Surface

### `InputRichMessage`

`InputRichMessage` describes a rich message to send.

Fields:

- `html: string` — rich message content using Rich HTML.
- `markdown: string` — rich message content using Rich Markdown.
- `is_rtl: boolean` — optional; show right-to-left.
- `skip_entity_detection: boolean` — optional; disables automatic detection of URLs, e-mail addresses, username mentions, hashtags, cashtags, bot commands, phone numbers, and bank card numbers.

Rule: **exactly one** of `html` or `markdown` must be used.

### `sendRichMessage`

Use to send a persistent rich message.

Required parameters:

- `chat_id`: target chat ID or username.
- `rich_message`: `InputRichMessage`.

Common optional parameters:

- `message_thread_id` / `direct_messages_topic_id`
- `business_connection_id`
- `disable_notification`
- `protect_content`
- `reply_parameters`
- `reply_markup`
- `allow_paid_broadcast`

If the message contains a media block, the bot must have permission to send that media type in the chat.

### `sendRichMessageDraft`

Use to stream an ephemeral rich preview while generating a final answer.

Required parameters:

- `chat_id`: target private chat.
- `draft_id`: non-zero integer; updates with the same ID animate.
- `rich_message`: partial `InputRichMessage`.

Important: drafts are **temporary 30-second previews**. After generation finishes, call `sendRichMessage` with the complete message to persist it.

`<tg-thinking>` / `RichBlockThinking` is draft-only and cannot appear in persisted messages.

### `editMessageText.rich_message`

Bot API 10.1 allows editing a message into/with a rich message via the `rich_message` parameter to `editMessageText`.

---

## Limits

Rich messages are subject to Telegram limits:

- Up to **32,768 UTF-8 characters** in rich message text, including custom emoji alternative text and formula source.
- Up to **500 blocks**, including nested blocks, list items, ordered list items, table rows, quotation blocks, and details blocks.
- Up to **16 levels** of nested formatting and blocks.
- Up to **50 media attachments** total, including photos, videos, and audio files.
- Up to **20 columns** in a table.

Design for mobile: even if Telegram accepts a huge rich message, shorter sections with details blocks are easier to read.

---

## Rich Markdown Quick Reference

Use this in `rich_message.markdown`.

### Inline formatting

```markdown
**bold text**
__bold text__
*italic text*
_italic text_
~~strikethrough text~~
`inline fixed-width code`
==marked text==
||spoiler||
[inline URL](https://t.me/)
[inline e-mail](mailto:user@example.com)
[inline phone number](tel:+123456789)
[inline mention of a user](tg://user?id=123456789)
![🙂](tg://emoji?id=5368324170671202286)
![22:45 tomorrow](tg://time?unix=1647531900&format=wDT)
$x^2 + y^2$
```

### Blocks

```markdown
# Heading 1
## Heading 2
### Heading 3

Paragraph text.

```python
print("pre-formatted code block")
```

---

- unordered list item
* unordered list item
+ unordered list item
1. ordered list item
2. ordered list item
- [ ] task list item
- [x] completed task list item

>Block quotation started
>
>Block quotation continued
```

### Media blocks

Media must be a separate block and must use HTTP/HTTPS URLs. Telegram determines media type from MIME type and URL.

```markdown
![](https://telegram.org/example/photo.jpg)
![](https://telegram.org/example/video.mp4)
![](https://telegram.org/example/audio.mp3)
![](https://telegram.org/example/audio.ogg)
![](https://telegram.org/example/animation.gif)
![](https://telegram.org/example/photo.jpg "Photo caption")
```

### Tables

```markdown
| Metric | Value |
|:-------|------:|
| Speed | **42** ms |
| Status | ==ready== |
```

Table cells can contain **inline formatting only**. Keep Telegram tables narrow; 2-4 columns usually read best on phones.

### Footnotes/references and math blocks

```markdown
Text with a reference[^note].

[^note]: Footnote with _italic_ and <u>HTML underline</u>.

$$E = mc^2$$

```math
E = mc^2
```
```

### Rich Markdown + HTML together

Rich Markdown is compatible with GitHub Flavored Markdown where possible and can contain supported Rich HTML tags for features that lack Markdown syntax:

```markdown
Intro with <u>underlined text</u>, <sup>superscript</sup>, and $x^2 + y^2$.

<details open><summary>Summary with **bold text**</summary>

### Details heading

- List item with _italic text_
- List item with <tg-spoiler>spoiler</tg-spoiler>

</details>

<tg-collage>
![](https://telegram.org/example/photo.jpg)
![](https://telegram.org/example/video.mp4)
</tg-collage>
```

---

## Rich HTML Quick Reference

Use this in `rich_message.html`. Rich HTML supports more explicit control and is often safer for generated content because you can HTML-escape user-provided text.

### Inline tags

Supported inline tags include:

```html
<b>bold</b>, <strong>bold</strong>
<i>italic</i>, <em>italic</em>
<u>underlined</u>, <ins>underlined</ins>
<s>strikethrough</s>, <strike>strikethrough</strike>, <del>strikethrough</del>
<code>inline code</code>
<mark>marked text</mark>
<sub>subscript</sub>
<sup>superscript</sup>
<tg-spoiler>spoiler</tg-spoiler>
<tg-math>x^2 + y^2</tg-math>
<a href="https://t.me/">inline URL</a>
<a href="mailto:user@example.com">inline e-mail</a>
<a href="tel:+123456789">inline phone</a>
<a href="tg://user?id=123456789">inline mention</a>
<tg-emoji emoji-id="5368324170671202286">🙂</tg-emoji>
<tg-time unix="1647531900" format="wDT">22:45 tomorrow</tg-time>
```

### Structure tags

```html
<h1>Heading 1</h1>
<h2>Heading 2</h2>
<p>Paragraph text</p>
<pre><code class="language-python">print("hello")</code></pre>
<footer>Footer text</footer>
<hr/>
<ul><li>unordered list item</li></ul>
<ol start="3" type="a" reversed><li>ordered list item</li></ol>
<ul>
  <li><input type="checkbox" checked>Checked checkbox</li>
  <li><input type="checkbox">Unchecked checkbox</li>
</ul>
<blockquote>Quote<br>continued<cite>The Author</cite></blockquote>
<aside>Pull quote<cite>The Author</cite></aside>
<details open><summary>Title</summary>Content</details>
<tg-math-block>E = mc^2</tg-math-block>
```

### Media, maps, collages, slideshows

```html
<figure><img src="https://telegram.org/example/photo.jpg" tg-spoiler/><figcaption>Photo caption<cite>Credit</cite></figcaption></figure>
<figure><video src="https://telegram.org/example/video.mp4"></video><figcaption>Video caption</figcaption></figure>
<figure><audio src="https://telegram.org/example/audio.mp3"></audio><figcaption>Audio caption</figcaption></figure>
<tg-map lat="41.9" long="12.5" zoom="14"/>
<tg-collage><img src="https://telegram.org/example/photo.jpg"/><video src="https://telegram.org/example/video.mp4"/></tg-collage>
<tg-slideshow><img src="https://telegram.org/example/photo.jpg"/><video src="https://telegram.org/example/video.mp4"/></tg-slideshow>
```

### Tables

```html
<table bordered striped>
  <caption>Table caption</caption>
  <tr><th>Header 1</th><th>Header 2</th></tr>
  <tr><td align="left">Value</td><td align="right">42</td></tr>
</table>
```

### Anchors and references

```html
<a name="chapter-1"></a>
<a href="#chapter-1">Jump to chapter 1</a>
<tg-reference name="note-1">Referenced text</tg-reference>
<a href="#note-1">Reference</a>
```

Rich HTML named entities supported by Telegram include `&lt;`, `&gt;`, `&amp;`, `&quot;`, `&apos;`, `&nbsp;`, `&hellip;`, `&mdash;`, `&ndash;`, `&lsquo;`, `&rsquo;`, `&ldquo;`, and `&rdquo;`. Numerical HTML entities are supported.

---

## Payload Recipes

### Send Rich Markdown with `curl`

Never print bot tokens in chat or logs. Use environment variables or secret references outside LLM context.

```bash
curl -sS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendRichMessage" \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "chat_id": 123456789,
  "rich_message": {
    "markdown": "# Deployment report\n\n- [x] Build passed\n- [x] Tests passed\n\n| Metric | Value |\n|:--|--:|\n| Duration | **42s** |\n\n<details><summary>Logs</summary>Everything green.</details>",
    "skip_entity_detection": true
  }
}
JSON
```

### Send Rich HTML with Python

```python
import os
import requests

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
chat_id = 123456789
html = """
<h1>Deployment report</h1>
<ul>
  <li><input type="checkbox" checked>Build passed</li>
  <li><input type="checkbox" checked>Tests passed</li>
</ul>
<table bordered striped>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Duration</td><td align="right"><b>42s</b></td></tr>
</table>
<details><summary>Logs</summary><pre>Everything green.</pre></details>
"""

resp = requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendRichMessage",
    json={
        "chat_id": chat_id,
        "rich_message": {"html": html, "skip_entity_detection": True},
    },
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Stream a Thinking Draft, then Persist

```json
{
  "chat_id": 123456789,
  "draft_id": 42,
  "rich_message": {
    "html": "<tg-thinking>Thinking…</tg-thinking><p>Drafting the report.</p>"
  }
}
```

Then call `sendRichMessage` with the final rich message. Do not rely on the draft to remain visible.

---

## Hermes-Specific Guidance

Current Hermes Telegram gateway responses use rich delivery automatically when it materially improves the message: native tables, task lists, headings, collapsible details, math, footnotes/references, rich HTML blocks, or long replies that would otherwise split at Telegram's 4,096-character text limit. Short/plain replies still use the classic Telegram formatting path because it is simpler and renders well for bold, italic, links, inline code, spoilers, and similar lightweight formatting. If Telegram rejects a rich payload or the endpoint is unavailable, Hermes falls back to classic MarkdownV2 without losing the message.

In classic fallback paths:

- Standard Telegram output supports common markdown such as `**bold**`, `*italic*`, `~~strike~~`, `||spoiler||`, inline code, fenced code, headings, links, and media attachments.
- Telegram has no classic Markdown table rendering; Hermes may rewrite pipe tables into readable row groups.
- Rich message-only features such as `<details>`, native tables, collages, slideshows, maps, footnotes, and `sendRichMessageDraft` require explicit Bot API rich message support in the sender path.

When implementing Hermes support:

1. Add a rich-send path that calls Bot API `sendRichMessage` directly or via SDK raw request if the installed `python-telegram-bot` version lacks a helper.
2. Preserve the normal `sendMessage` path as fallback for clients/bot API errors.
3. Introduce an explicit message marker or send option; do not guess that every markdown response should become a rich message.
4. Validate that exactly one of `markdown`/`html` is set.
5. For generated HTML, escape user-controlled text before interpolating into tags.
6. For drafts, call `sendRichMessageDraft` only in private chats and always finish with `sendRichMessage`.
7. Split or degrade messages that exceed rich-message limits.
8. Add tests for rich payload construction, fallback behavior, secret-safe error logging, and media permission errors.

---

## Authoring Rules for Agents

When asked to create a rich Telegram message:

1. **Choose format:** prefer Rich Markdown for content-heavy reports; prefer Rich HTML when precise structure/escaping matters.
2. **Keep it mobile-first:** use headings, short sections, lists, and details blocks instead of a giant wall of text.
3. **Use real tables sparingly:** wide tables are hard to read on mobile; convert to bullets if more than 4 columns.
4. **Disable auto-detection when risky:** set `skip_entity_detection: true` for logs, code, financial numbers, cards, phone-like strings, or generated reports where accidental links are harmful.
5. **Separate media blocks:** media must be standalone blocks and use HTTP/HTTPS URLs.
6. **Escape untrusted text:** especially in Rich HTML; only supported tags/entities are accepted.
7. **Do not expose tokens:** never echo bot tokens, webhook URLs containing tokens, or auth headers.
8. **Verify by sending to a safe test chat** when changing sender code.

---

## Common Pitfalls

1. **Mixing classic MarkdownV2 with Rich Markdown.** Rich Markdown uses GFM-like `**bold**`, `~~strike~~`, tables, details, and inline HTML. Classic MarkdownV2 has stricter escaping and does not support rich blocks.

2. **Sending both `html` and `markdown`.** `InputRichMessage` requires exactly one of them.

3. **Assuming Hermes final responses automatically use `sendRichMessage`.** They may not. Rich message syntax in a normal final response can be flattened or escaped unless the Telegram gateway explicitly sends a rich payload.

4. **Forgetting the draft lifecycle.** `sendRichMessageDraft` is ephemeral; it is not a final message.

5. **Using local file paths for media blocks.** Rich media blocks require HTTP/HTTPS URLs. Use native media attachment upload paths for local files, or host the file first.

6. **Wide or deeply nested structures.** Telegram accepts up to 20 table columns and 16 nesting levels, but mobile users will hate it. Optimize for readability.

7. **Unescaped generated HTML.** Rich HTML only accepts supported tags; raw `<`, `>`, and `&` in user text must be escaped.

8. **Accidental entity detection.** URLs, emails, phone numbers, bot commands, hashtags, cashtags, and card numbers are detected unless `skip_entity_detection` is true.

---

## Verification Checklist

- [ ] Chose `markdown` or `html`, not both.
- [ ] Content is within 32,768 UTF-8 characters.
- [ ] Estimated blocks are under 500 and nesting under 16 levels.
- [ ] Tables are no wider than 20 columns and are readable on mobile.
- [ ] Media blocks use HTTP/HTTPS URLs and are separate blocks.
- [ ] User-controlled HTML text is escaped.
- [ ] `skip_entity_detection` is set intentionally.
- [ ] Drafts are followed by a persistent `sendRichMessage`.
- [ ] Tokens and auth headers are never printed.
- [ ] Sender has fallback behavior if `sendRichMessage` is unavailable or rejected.
