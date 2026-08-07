# getxapi Endpoint Reference

> **SECURITY**: All curl examples use `<your-api-key>` and `<your-auth-token>`
> placeholders. Never paste real credentials into chat or LLM context.
> Store secrets in `~/.hermes/.env` and reference them via environment variables.


Base URL: `https://api.getxapi.com`
Auth: `Authorization: Bearer <your-api-key>` (all endpoints)
Post auth: value of the X `auth_token` cookie, passed as an `auth_token` field in the JSON body

## Account

### GET /account/me
Free. Rate limit: 30 req/min. Returns account details, credit balance, usage.

```bash
curl -s -H "Authorization: Bearer <your-api-key>" "https://api.getxapi.com/account/me"
```

Response fields: `email`, `name`, `credits_remaining`, `credits_used`, `total_requests`, `created_at`.

### GET /account/payments
Free. Rate limit: 30 req/min. Returns payment history.

```bash
curl -s -H "Authorization: Bearer <your-api-key>" "https://api.getxapi.com/account/payments"
```

Response fields: `{"payments": [{"amount", "credits_added", "status", "created_at"}]}`

---

## Tweets

### GET /twitter/tweet/advanced_search
Cost: $0.001. ~20 tweets/page. Cursor pagination.

| Param | Required | Notes |
|-------|----------|-------|
| `q` | Yes | Search query. Supports operators: `from:user`, `to:user`, `has:media`, `-filter:retweets`, `min_faves:N`, `since:YYYY-MM-DD` |
| `product` | No | `Latest` (default) or `Top` |
| `count` | No | Results per page (default 20, max ~20) |
| `cursor` | No | Pagination cursor |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/tweet/advanced_search?q=from:elonmusk&product=Top"
```

Response fields: `query`, `tweet_count`, `has_more`, `next_cursor`, `tweets` (array of tweet objects).

Tweet object fields: `type`, `id`, `url`, `twitterUrl`, `text`, `source`, like/reply/quote/view counts, `createdAt`, `isReply`, `inReplyToId`, `media[]`, `author{...}`.

### GET /twitter/tweet/detail
Cost: $0.001. Returns a single tweet with full author and media.

| Param | Required | Notes |
|-------|----------|-------|
| `id` | Yes | Tweet ID |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/tweet/detail?id=1234567890123456789"
```

Response wrapped in `data` field: `type`, `id`, `text`, `author{...}`, `inReplyToId`, `createdAt`, `media[]`, `quoted_tweet`.

### GET /twitter/tweet/replies
Cost: $0.001. ~20 replies/page. Cursor pagination.

| Param | Required | Notes |
|-------|----------|-------|
| `id` | Yes | Tweet ID |
| `cursor` | No | Pagination cursor |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/tweet/replies?id=1234567890123456789"
```

Response fields: `{"tweetId", "reply_count", "has_more", "next_cursor", "replies": [...]}`

### GET /twitter/tweet/article
Cost: $0.001. Returns full article/note content. Tweet must be an article tweet.

| Param | Required | Notes |
|-------|----------|-------|
| `id` | Yes | Tweet ID of the article |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/tweet/article?id=1234567890123456789"
```

Response fields: `article` with `title`, `preview_text`, `cover_media_img_url`, `contents` (rich text blocks).

### POST /twitter/tweet/create
Cost: $0.002. Creates a tweet or reply.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token cookie value |
| `text` | Yes | Tweet text |
| `reply_to_tweet_id` | No | Tweet ID to reply to |
| `quote_tweet_url` | No | Full URL of tweet to quote |
| `quote_tweet_id` | No | Tweet ID to quote (resolved to URL) |
| `media_urls` | No | Array of image/video URLs |
| `media_ids` | No | Pre-uploaded Twitter media IDs |
| `media` | No | Array of `{data: base64, type: mime/type}` |
| `proxy` | No | Custom proxy URL |

```bash
curl -s -X POST "https://api.getxapi.com/twitter/tweet/create" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"auth_token":"<your-auth-token>","text":"Hello world"}'
```

Add `"reply_to_tweet_id": "1234567890"` to the body for replies.

### POST /twitter/tweet/favorite
Cost: $0.001. Likes a tweet.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `tweet_id` | Yes | Tweet ID to like |

```bash
curl -s -X POST "https://api.getxapi.com/twitter/tweet/favorite" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"auth_token":"<your-auth-token>","tweet_id":"1234567890123456789"}'
```

### POST /twitter/tweet/retweet
Cost: $0.001. Retweets a tweet.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `tweet_id` | Yes | Tweet ID to retweet |

```bash
curl -s -X POST "https://api.getxapi.com/twitter/tweet/retweet" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"auth_token":"<your-auth-token>","tweet_id":"1234567890123456789"}'
```

### /twitter/tweet/delete (not implemented)
Does not exist. Returns 404. Tweets must be deleted manually on x.com.

---

## Users

### GET /twitter/user/search
Cost: $0.001. ~20 users/page. Searches for users (equivalent to "People" tab).

| Param | Required | Notes |
|-------|----------|-------|
| `q` | Yes | Search query (username, keyword, topic) |
| `cursor` | No | Pagination cursor |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/search?q=exampleKeyword1"
```

### GET /twitter/user/info
Cost: $0.001. Returns full profile for a user.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/info?userName=someuser"
```

Response fields: `id`, `name`, `userName`, `location`, `description`, `protected`, `isVerified`, `isBlueVerified`, `followers`, `following`, `favouritesCount`, `statusesCount`, `mediaCount`, `createdAt`, `profilePicture`, `coverPicture`, `canDm`, `pinnedTweetIds`.

### GET /twitter/user/info_by_id
Cost: $0.001. Same as `/user/info` but by numeric user ID.

| Param | Required | Notes |
|-------|----------|-------|
| `userId` | Yes | Numeric user ID |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/info_by_id?userId=44196397"
```

### GET /twitter/user/user_about
Cost: $0.001. Returns account metadata: creation location, signup source, username change history.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/user_about?userName=someuser"
```

### GET /twitter/user/tweets
Cost: $0.001. ~20 tweets/page. User's profile "Posts" tab.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Conditional | Screen name. Required if `userId` not provided |
| `userId` | Conditional | Numeric ID. Faster — skips username lookup |
| `count` | No | Results per page (default 20) |
| `cursor` | No | Pagination cursor |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/tweets?userName=someuser&count=20"
```

### GET /twitter/user/tweets_and_replies
Cost: $0.001. ~20 results/page. User's profile "Replies" tab.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/tweets_and_replies?userName=someuser"
```

### GET /twitter/user/media
Cost: $0.001. ~20 results/page. User's tweets containing media.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

```bash
curl -s -H "Authorization: Bearer <your-api-key>" \
  "https://api.getxapi.com/twitter/user/media?userName=someuser"
```

### GET /twitter/user/mentions
Cost: $0.001. ~20 tweets/page. Tweets mentioning the user.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

### POST /twitter/user/likes
Cost: $0.001. ~20 tweets/page. User's liked tweets.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `cursor` | No | Pagination cursor |

```bash
curl -s -X POST "https://api.getxapi.com/twitter/user/likes" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"auth_token":"<your-auth-token>"}'
```

### GET /twitter/user/followers
Cost: $0.001. ~200 users/page. REST v1 endpoint — no total cap, full follower list accessible.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

### GET /twitter/user/followers_v2
Cost: $0.001. ~70 users/page. Higher fidelity, capped at ~800 total (matches x.com UI).

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

### GET /twitter/user/following
Cost: $0.001. ~200 users/page. REST v1 — no total cap.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

### GET /twitter/user/following_v2
Cost: $0.001. ~70 users/page. Capped at ~800 total.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

### GET /twitter/user/verified_followers
Cost: $0.001. ~20 verified (blue check) followers/page.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Screen name (without @) |
| `cursor` | No | Pagination cursor |

### POST /twitter/user/followers_you_know
Cost: $0.001. ~20 users/page. Returns followers you know for a target user.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `user_id` | Conditional | Target user ID |
| `user_name` | Conditional | Target username |
| `cursor` | No | Pagination cursor |

### GET /twitter/user/check_follow_relationship
Cost: $0.001. Checks follow/block/mute relationship between two users.

| Param | Required | Notes |
|-------|----------|-------|
| `source_user_name` | Yes | Source screen name |
| `target_user_name` | Yes | Target screen name |

Response fields: `sourceFollowsTarget`, `targetFollowsSource`, `canDm`, `blocking`, `blockedBy`, `muting`.

### POST /twitter/user/home_timeline
Cost: $0.001. ~20 tweets/page. Home timeline feed.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `cursor` | No | Pagination cursor |

### POST /twitter/user/bookmark_search
Cost: $0.001. ~20 results/page. Searches user's bookmarks.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `q` | Yes | Search query |
| `cursor` | No | Pagination cursor |

### GET /twitter/user/affiliates
Cost: $0.001. ~20 users/page. Affiliated accounts of a verified organization.

| Param | Required | Notes |
|-------|----------|-------|
| `userName` | Yes | Organization username |
| `cursor` | No | Pagination cursor |

### POST /twitter/user_login
Cost: $0.001. Returns fresh auth tokens (auth_token, ct0, twid).

**⚠️ SECURITY:** This endpoint accepts highly sensitive credentials — `username`, `password`, and optionally `totp_secret`. Never paste these into chat or agent context. Only use this endpoint locally with credentials stored in `~/.hermes/.env`.

| Field | Required | Notes |
|-------|----------|-------|
| `username` | Yes | X username |
| `password` | Yes | Account password |
| `email` | Yes | Email for verification |
| `totp_secret` | Conditional | TOTP secret for 2FA accounts |
| `proxy` | No | Custom proxy URL |

### POST /twitter/user/update_avatar
Cost: $0.001. Updates profile picture. Provide `image_url` or `image_base64` (not both).

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `image_url` | Conditional | Public image URL |
| `image_base64` | Conditional | Base64-encoded image |

### POST /twitter/user/update_banner
Cost: $0.001. Updates profile banner. Same params as update_avatar.

### POST /twitter/user/update_profile
Cost: $0.001. Updates profile fields (name, description, location, url, colors, birthdate).

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `name` | No | Display name |
| `description` | No | Bio |
| `location` | No | Location |
| `url` | No | Website URL |
| `birthdate_year/month/day` | No | Birthdate |
| `birthdate_visibility` | No | `self`, `mutualfollow`, `followers`, `following`, `public` |

---

## Lists

### GET /twitter/list/members
Cost: $0.001. ~20 users/page. Members of a list.

| Param | Required | Notes |
|-------|----------|-------|
| `listId` | Yes | List ID |
| `cursor` | No | Pagination cursor |

---

## Direct Messages

### POST /twitter/dm/list
Cost: $0.002. ~50 messages/page.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `cursor` | No | Pagination cursor |
| `count` | No | Messages per page (default 50) |

### POST /twitter/dm/send
Cost: $0.002. Sends a DM.

| Field | Required | Notes |
|-------|----------|-------|
| `auth_token` | Yes | X auth_token |
| `recipient_id` | Conditional | Recipient user ID |
| `recipient_username` | Conditional | Recipient username |
| `text` | Yes | Message text |

---

## Pagination

All list endpoints use cursor-based pagination:
- Omit `cursor` for the first page
- Pass `next_cursor` from the response for subsequent pages
- ~20 items/page (followers/following v1: ~200; v2 variants: ~70; DMs: ~50)
- Paginate until `has_more` is `false`
- Followers/following v2 capped at ~800 results; v1 has no cap

## Error Codes

| Status | Meaning | Common Cause |
|--------|---------|-------------|
| 400 | Bad Request | Missing required parameter |
| 401 | Unauthorized | Invalid/missing API key or auth_token |
| 404 | Not Found | User or tweet does not exist |
| 429 | Rate Limited | Account endpoints: exceeded 30 req/min |
| 502 | Bad Gateway | X rejected the mutation |

All errors return JSON: `{"error": "description"}`

## Pricing

| Category | Cost |
|----------|------|
| All read endpoints | $0.001/req |
| Create tweet | $0.002/req |
| DM endpoints | $0.002/req |
| Account endpoints | Free (30 req/min limit) |
