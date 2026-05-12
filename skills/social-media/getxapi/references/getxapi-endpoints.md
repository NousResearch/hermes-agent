# getxapi Endpoint Reference

Base URL: `https://api.getxapi.com`
Auth: `Authorization: Bearer <api-key>` (all endpoints)
Post auth: value of the X `auth_token` cookie, passed as an `auth_token` field in the JSON body (create endpoint)

## Account

### GET /account/me
Check credits, email, request count.

```bash
curl -s -H "Authorization: Bearer <your-api-key>" "https://api.getxapi.com/account/me"
```

Response fields: `email`, `credits` (string, e.g. "$7.53"), `requests` (integer).

---

## Search

### GET /twitter/tweet/advanced_search
Search tweets by keyword.

| Param | Required | Notes |
|-------|----------|-------|
| `q` | Yes | Search query. Single-word queries work best. Use URL encoding for multi-word (`docker%20compose`) |
| `product` | No | `Top` (real engagement) or `Latest` (chronological, mostly bot spam). Always use `Top`. |
| `count` | No | Results per page, max ~20 |
| `cursor` | No | Pagination cursor from previous response |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/tweet/advanced_search?q=homelab&product=Top&count=20"
```

Response: `{"tweets": [...]}` or `{"data": {"tweets": [...]}}`. Each tweet has `id`, `text`, `author.userName`, `author.followers`, `isReply`, `createdAt`, `viewCount`, `media[]`.

**Field mapping gotchas:**
- `author.userName` — NOT `screen_name` (v1.1), `username` (v2 user objects), or `author.username`
- `author.followers` — NOT `public_metrics.followers_count`
- `isReply` (boolean) — NOT `in_reply_to_user_id`
- `createdAt` — NOT `created_at`
- `inReplyToId` — original tweet ID this is replying to

**Search operators (limited support):**
- `has:media` — tweets with images/video
- `from:username` — from specific user
- `-filter:retweets` — exclude retweets
- `-filter:replies` — exclude replies
- Combine: `q=homelab%20has:media%20-filter:retweets`

---

## Tweets

### POST /twitter/tweet/create
Post a new tweet or reply.

Body (JSON):
```json
{
  "auth_token": "<your-auth-token>",
  "text": "tweet content here"
}
```

Add `"reply_to_tweet_id": "1234567890"` to the body to post a reply instead of a standalone tweet.

```bash
curl -s -X POST "https://api.getxapi.com/twitter/tweet/create" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"auth_token":"<your-auth-token>","text":"hello"}'
```

Response: tweet object with `id`, `text`, `createdAt`, `author`, `url`.

**Shell quoting:** If text contains apostrophes, write JSON to temp file:
```bash
cat > /tmp/post.json << 'EOF'
{"auth_token":"<your-auth-token>","text":"don't use inline quotes"}
EOF
curl -s -X POST "https://api.getxapi.com/twitter/tweet/create" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d @/tmp/post.json
```

### /twitter/tweet/delete (not implemented)
**DOES NOT EXIST.** Returns 404. Tweets must be deleted manually on x.com. There is no programmatic undo.

---

## User

### GET /twitter/user/tweets
Fetch user's timeline.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes* | Username (e.g. `kextcache`) |
| `userId` | Yes* | User ID alternative to userName |
| `count` | No | Results per page |

*One of `userName` or `userId` is required.

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/tweets?userName=kextcache&count=20"
```

Response: `{"userName": "...", "userId": "...", "tweet_count": N, "has_more": bool, "next_cursor": "...", "tweets": [...]}`

---

## Tweet Detail

### GET /twitter/tweet/detail
Get single tweet by ID.

| Param | Required | Notes |
|-------|----------|-------|
| `id` | Yes | Tweet ID. Use `id=` NOT `tweet_id=` — wrong param returns error. |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/tweet/detail?id=2054014418455839156"
```

Response: `{"data": {"type": "tweet", "id": "...", "text": "...", "author": {...}, "inReplyToId": "...", "createdAt": "...", "media": [...], ...}}`

Key fields in response:
- `data.id` — tweet ID
- `data.text` — full text
- `data.inReplyToId` — original tweet being replied to (empty/null for standalone)
- `data.author.userName` — author handle
- `data.createdAt` — timestamp (format: `Tue May 12 01:42:53 +0000 2026`)
- `data.isReply` — boolean
- `data.viewCount` — impression count
- `data.media` — array of media objects with `media_url_https`

---

## Rate Limits & Credits

| Action | Cost | Rate limit |
|--------|------|------------|
| Search | ~$0.001/req | Per-account X limits apply |
| Post/Reply | $0.002/post | X daily post cap (varies by account age/activity) |
| User timeline | ~$0.001/req | Standard |
| Tweet detail | ~$0.001/req | Standard |
| Account check | Free | Unlimited |

Accounts in warmup (new or low-activity) hit X's daily post cap aggressively. Start at 3-4 posts per run window, scale up over 2-3 weeks.

---

## Image Download for Vision Analysis

Tweet images are at `https://pbs.twimg.com/media/<media_id>?format=jpg&name=small` (or `medium`, `large`). Download before analyzing with vision tools:

```bash
curl -s -o /tmp/tweet_img.jpg \
  "https://pbs.twimg.com/media/HH8jjRZXIAQ7dlA.jpg?format=jpg&name=small"
file /tmp/tweet_img.jpg  # verify it's JPEG, not HTML error page
```
