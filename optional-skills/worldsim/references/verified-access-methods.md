# Verified Access Methods — Complete Platform Map (April 2026)

Every method tested from our environment. Use this as the single
source of truth for what works and what doesn't.

## TIER 1 — Full API / Rich Data Access

### Twitter/X ✅✅✅
| Method | Endpoint | Auth | Rate Limit | Returns |
|--------|----------|------|-----------|---------|
| API v2 bearer | api.twitter.com/2/ | Bearer token | 10K tweets/15min | Profiles, tweets, search |
| nitter.cz | web_extract | None | No limit seen | Full timeline (UNRELIABLE — see note below) |
| ThreadReaderApp | web_extract /user/{handle} | None | No limit seen | Historical threads |

#### CRITICAL: X API curl is the gold standard for voice calibration (April 2026)
The BEST voice data source is direct curl to X API v2 with bearer token.
Returns full tweet text + public_metrics per tweet. Always prefer this for
mechanical calibration (word count, caps, punctuation, emoji rate).

```bash
source ~/.dotenv
# 1. Get user ID from handle
curl -s -H "Authorization: Bearer $X_BEARER_TOKEN" \
  "https://api.twitter.com/2/users/by/username/{handle}?user.fields=description,public_metrics,location,created_at"
# 2. Get timeline (30 tweets per page, paginate with meta.next_token)
curl -s -H "Authorization: Bearer $X_BEARER_TOKEN" \
  "https://api.twitter.com/2/users/{user_id}/tweets?max_results=30&tweet.fields=created_at,public_metrics,text&exclude=retweets"
# 3 pages = 90 tweets — enough for fidelity 100 voice calibration
```

NOTE: scripts/x_api.py is BROKEN — imports hermes_tools at top level, can't
run standalone via terminal(). Use direct curl above instead.

#### nitter.cz reliability warning (April 2026)
nitter.cz via web_extract works SOMETIMES but is unreliable:
- Returns 502 Cloudflare errors for /with_replies on some handles
- Returns "User not found" for valid handles (e.g. karan4d exists but nitter says not found)
- Main profile page (/handle) more reliable than /with_replies
- Use as SUPPLEMENT to X API curl, not primary source. If nitter fails, don't retry — use curl.

### Bluesky ✅✅
| Method | Endpoint | Auth | Returns |
|--------|----------|------|---------|
| getProfile | public.api.bsky.app | None | Full profile, stats |
| getAuthorFeed | public.api.bsky.app | None | 50 posts + engagement |
| searchActors | public.api.bsky.app | None | Find handles by name |
| searchPosts | BLOCKED (403) | — | Use searchActors + getAuthorFeed workaround |

### Mastodon ✅✅✅ (FULLY OPEN)
| Method | Endpoint | Auth | Returns |
|--------|----------|------|---------|
| Account lookup | {instance}/api/v1/accounts/lookup?acct={user} | None | Full profile |
| Account statuses | {instance}/api/v1/accounts/{id}/statuses | None | All posts |
| Search | {instance}/api/v2/search?q={query}&type=accounts | None | Account search |
| WebFinger | {instance}/.well-known/webfinger?resource=acct:{user}@{instance} | None | Identity resolution |
| Trending | {instance}/api/v1/trends/tags | None | Trending content |
Key instances: mastodon.social, hachyderm.io, sigmoid.social

### Instagram ✅✅ (CRACKED)
| Method | Endpoint | Auth | Returns |
|--------|----------|------|---------|
| Private Web API | i.instagram.com/api/v1/users/web_profile_info/ | Mobile UA + x-ig-app-id: 936619743392459 | Profile + 12 posts + captions + CDN URLs |
| oEmbed | instagram.com/api/v1/oembed/ | None | Caption + author for individual posts |
| Pixwox | web_extract pixwox.com/profile/{user} | None | 12+ posts, engagement |
| SocialBlade | web_extract socialblade.com/instagram/user/{user} | None | Analytics, follower trends |
| CDN images | scontent-*.cdninstagram.com URLs from API | None | Full-res images → vision_analyze |
| Google index | web_search site:instagram.com | None | Bio, follower count, captions |

### GitHub ✅✅
| Method | Endpoint | Auth | Returns |
|--------|----------|------|---------|
| REST API | api.github.com/users/{user} | None (60 req/hr) | Profile, repos, events, gists |
| Profile README | github.com/{user}/{user} | None | Self-description (voice gold) |

### Reddit ✅✅
| Method | Endpoint | Auth | Returns |
|--------|----------|------|---------|
| JSON API | reddit.com/user/{user}.json | User-Agent header required | Comments, posts, scores |
| Search | reddit.com/r/{sub}/search.json | User-Agent header | Subreddit-specific search |

## TIER 2 — Good Data, Reliable Access

### Facebook ✅✅ (CRACKED — Googlebot UA trick)
| Method | Endpoint | Returns |
|--------|----------|---------|
| Googlebot UA (BEST) | curl facebook.com/{page} with Googlebot UA | OG tags: name, bio/about, likes count (e.g. 121M for zuck), talking_about count, og:image, profile pic |
| Page Plugin embed | plugins/page.php?href=...&tabs=timeline | Name, follower count, numeric page_id |
| Graph /picture | graph.facebook.com/v19.0/{page}/picture?redirect=false | Direct CDN profile pic URL (no auth) |
| web_search | site:facebook.com {name} | Profile snippets from Google index |
| Script: scripts/facebook_api.py — combines all 3 methods |
| NOTE: Works for PUBLIC Pages (businesses, public figures, orgs). Personal profiles behind privacy settings are not accessible. |
| Tested: zuck (121M likes), NVIDIA, Meta, CocaCola, BillGates, BarackObama |

### Threads (Meta) ✅✅ (CRACKED — OG tags DO exist)
| Method | Endpoint | Returns |
|--------|----------|---------|
| Profile OG tags (BEST) | curl -L threads.com/@{user} (NOTE: .com not .net — .net 301 redirects) | display_name, follower_count (e.g. "5.5M"), thread_count, bio, profile_picture_url |
| Post OG tags | curl -L threads.com/@{user}/post/{shortcode} | Full post text, author name, image URL |
| WebFinger | threads.net/.well-known/webfinger?resource=acct:{user}@threads.net | ActivityPub ID, profile URL (works for federated users) |
| IMPORTANT: threads.NET redirects to threads.COM — always use -L flag or go directly to .com |
| Post discovery | web_search site:threads.net @{user} | Find post URLs to then fetch |
| Script: scripts/threads_api.py — profile + post + webfinger extraction |
| Previous test was WRONG about "no OG tags" — they're there, you just need standard curl |
| Tested: zuck (5.5M followers), mosseri, nvidia |

### Medium ✅✅
| Method | Returns |
|--------|---------|
| RSS feed: medium.com/feed/@{user} (curl) | FULL article text, tags, dates — NO AUTH |
| web_extract on profile | Bio, follower count, article list, themes |
| web_extract on articles | Full content (paywall may truncate non-members) |

### Quora ✅✅
| Method | Returns |
|--------|---------|
| web_extract on profile | Bio, credentials, Q&A with direct quotes |
| web_search site:quora.com | Finds profiles and specific answers |
| VOICE VALUE: Opinions in own words, analogies, intellectual identity |

### Goodreads ✅✅ (HIDDEN GEM)
| Method | Returns |
|--------|---------|
| web_extract on user profile | Favorites, reviews in own voice, social graph, reading history |
| web_extract on author page | Bio, books, ratings, notable quotes |
| VOICE VALUE: "You are what you read" — intellectual identity fingerprint |
| Example: Karpathy's Goodreads reveals gaming passion, favorite authors (Feynman, Clarke) |

### Google Scholar ✅✅
| Method | Returns |
|--------|---------|
| web_search + web_extract on profile | Citations, h-index, top papers, co-authors |
| Semantic Scholar API via web_extract | Paper list, citation counts, author ID |
| Endpoint: api.semanticscholar.org/graph/v1/author/search?query={name} |

### Product Hunt ✅
| Method | Returns |
|--------|---------|
| web_extract on producthunt.com/@{user} | Bio, launch history, forum activity |

### HackerNews ✅
| Method | Returns |
|--------|---------|
| Algolia API: hn.algolia.com/api/v1/search?query={name}&tags=comment | Comments, mentions |

### Podcast Transcripts ✅✅✅ (HIGHEST VOICE VALUE)
| Source | Method |
|--------|--------|
| Lex Fridman | web_extract on lexfridman.com/.../transcript |
| Tyler Cowen | web_extract on conversationswithtyler.com |
| TED Talks | web_extract on ted.com/.../transcript |
| Sequoia | web_extract on sequoiacap.com/podcast |
| Discovery: web_search "{name} podcast transcript interview" |

### News/Blogs ✅✅
| Source | Method |
|--------|--------|
| TechCrunch, Wired, Verge, Ars | web_extract — full articles |
| Personal blogs | web_extract — longform self-expression |
| Substacks | web_extract — essays and comments |
| Wayback Machine | Works for blog archives (not Twitter) |

## TIER 3 — Limited / Conditional

### TikTok ✅✅ (FULL ACCESS)
| Method | Returns |
|--------|---------|
| HTML profile scraping | Parse __UNIVERSAL_DATA_FOR_REHYDRATION__ JSON at path __DEFAULT_SCOPE__.webapp.user-detail.userInfo.statsV2 → username, bio, followerCount, followingCount, heartCount, videoCount. Use statsV2 not stats for large numbers. |
| oEmbed per video | curl tiktok.com/oembed?url={video_url} → caption, author, thumbnail. No auth. |
| tikwm.com API | tikwm.com/api/user/info?unique_id={user} → full user stats. tikwm.com/api/?url={video_url} → play count, likes, comments, shares, duration. |
| HTML video scraping | tiktok.com/@{user}/video/{id} → parse __UNIVERSAL_DATA → webapp.video-detail → full video data with description, hashtags, engagement. |
| SocialBlade | web_extract socialblade.com/tiktok/user/{user} → followers, likes, growth trends. |
| Video discovery | web_search("site:tiktok.com/@{user}/video") → recent video URLs → scrape each |
| Tested: khaby.lame (160.5M), charlidamelio (156.7M), mrbeast (124.7M) |

### Spotify ✅ (podcasters only)
| Method | Returns |
|--------|---------|
| web_extract on show page | Episode listings with guests, topics, durations |

### Stack Overflow ✅
| Method | Returns |
|--------|---------|
| web_extract on profile | Reputation, tags, top answers, bio |

### Crunchbase ✅ (executives/founders only)
| Method | Returns |
|--------|---------|
| web_extract on crunchbase.com/person/{slug} | Full career history, education, investments, board positions |

### LinkedIn ⚠️ (indirect only)
| Method | Returns |
|--------|---------|
| web_search site:linkedin.com/in | Name, headline, company, location from snippets |
| Crunchbase | Full career history (better than LinkedIn for execs) |
| Corporate press pages | Official professional bios |
| RocketReach/SignalHire snippets | Title confirmation from web_search |

## TIER 4 — Blocked / Dead

| Platform | Status |
|----------|--------|
| LinkedIn direct | BLOCKED (web_extract domain blocked) |
| Discord | WALLED (not publicly indexable) |
| Telegram t.me | BLOCKED in some environments |
| Threads Official API | AUTH REQUIRED (graph.threads.net needs OAuth) |
| Threads ActivityPub outbox | 404 for all tested users |
| Instagram direct | BLOCKED (use Private API instead) |
| Most Nitter instances | DEAD (only nitter.cz works, but UNRELIABLE — see note) |
| Google Cache of Twitter | EMPTY |
| Wayback for tweets | USELESS (JS rendering) |
| Twitter Syndication API | RATE LIMITED |
| Archive.today | 429 + CAPTCHA |
| imginn/picuki/dumpoir/gramhir | 403 |
| Facebook Graph API | AUTH REQUIRED |

## Quick Reference: Research Pipeline by Person Type

### Tech Founder/CEO
X API → Bluesky → GitHub README → Crunchbase → Podcast transcripts → Medium RSS → HN → Product Hunt → LinkedIn snippets → News profiles

### AI Researcher
X API → Bluesky → Google Scholar → Semantic Scholar → arXiv → GitHub → Podcast transcripts → Blog/Substack → Reddit → Mastodon (sigmoid.social)

### Public Figure / Politician
X API → Facebook OG → Instagram API → YouTube → Podcast transcripts → News profiles → Quora → Goodreads → Wikipedia

### Content Creator
X API → Instagram API → TikTok → YouTube → Twitch → Podcast → Medium → Reddit → Bluesky → Threads OG

### Academic
Google Scholar → Semantic Scholar → University page → Conference talks → Podcast transcripts → Mastodon → Blog → GitHub → Reddit → HN
