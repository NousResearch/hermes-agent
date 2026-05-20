# Twitter/X Automation Reference

## Prerequisites

Connect your X account at https://inference.sh/settings/connections before using any Twitter operations.

## Available Operations

| Operation | App ID | Input |
|-----------|--------|-------|
| Post tweet | `twitter/post-tweet` | `{"text": "..."}` |
| Post with media | `twitter/post-create` | `{"text": "...", "media": "file.png"}` |
| Like | `twitter/like` | `{"tweet_id": "..."}` |
| Retweet | `twitter/retweet` | `{"tweet_id": "..."}` |
| Delete tweet | `twitter/delete` | `{"tweet_id": "..."}` |
| Get post | `twitter/get-post` | `{"tweet_id": "..."}` |
| Send DM | `twitter/send-dm` | `{"username": "...", "text": "..."}` |
| Follow user | `twitter/follow` | `{"username": "..."}` |

## Workflows

**Visual tweet with AI-generated image:**
```bash
belt app run seedream/4.5 --input '{"prompt": "abstract visualization of neural networks"}'
belt app run twitter/post-create --input '{"text": "Exploring neural network architectures", "media": "output.png"}'
```

**News digest:**
```bash
belt app run tavily/search --input '{"query": "AI news this week"}'
belt app run openrouter/claude-haiku-4.5 --input '{"prompt": "Summarize in 280 chars: <results>"}'
belt app run twitter/post-tweet --input '{"text": "<summary>"}'
```

**Engagement:**
```bash
belt app run twitter/get-post --input '{"tweet_id": "123456789"}'
belt app run twitter/like --input '{"tweet_id": "123456789"}'
belt app run twitter/retweet --input '{"tweet_id": "123456789"}'
```

## Tips

- Media uploads: pass local file paths — the CLI handles upload.
- Tweet text limit: 280 characters standard, 25,000 for Premium.
- Rate limits apply per X API tier. Space out bulk operations.
- For threads, post each tweet individually and reply to the previous tweet's ID.
