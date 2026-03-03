---
name: twitter-post
description: Post tweets to Twitter/X using the API. Supports text and image posts. Used to announce meme coin launches, share updates, and post memes. Uses Tweepy with OAuth 1.0a.
triggers:
  - post to twitter
  - send tweet
  - tweet about
  - announce on twitter
  - post meme coin launch
version: 1
---

# Twitter/X Posting Skill

Post tweets (text and images) to Twitter/X via the API using Tweepy and OAuth 1.0a.

## Prerequisites

- Python 3 with `tweepy` library (`pip3 install tweepy`)
- Twitter/X developer account (https://developer.x.com)
- API credentials stored in `~/.config/twitter-post/.env`

## Setup (One-Time)

### 1. Create X Developer App

1. Go to https://developer.x.com and sign in
2. Create a new Project and App
3. Set app permissions to **"Read and Write"**
4. Under "Keys and Tokens", generate:
   - API Key (Consumer Key)
   - API Key Secret (Consumer Secret)
   - Access Token
   - Access Token Secret

### 2. Store Credentials

Create the config directory and credentials file:

```bash
mkdir -p ~/.config/twitter-post
```

Then create `~/.config/twitter-post/.env` with your editor containing:

```
X_API_KEY=your-api-key
X_API_KEY_SECRET=your-api-key-secret
X_ACCESS_TOKEN=your-access-token
X_ACCESS_TOKEN_SECRET=your-access-token-secret
```

Lock down permissions:

```bash
chmod 600 ~/.config/twitter-post/.env
```

### 3. Verify Authentication

```python
import tweepy, os

def load_credentials():
    """Load X API credentials from ~/.config/twitter-post/.env"""
    creds = {}
    env_path = os.path.expanduser("~/.config/twitter-post/.env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                creds[key.strip()] = val.strip()
    return creds

creds = load_credentials()
client = tweepy.Client(
    consumer_key=creds["X_API_KEY"],
    consumer_secret=creds["X_API_KEY_SECRET"],
    access_token=creds["X_ACCESS_TOKEN"],
    access_token_secret=creds["X_ACCESS_TOKEN_SECRET"],
)
me = client.get_me()
print(f"Authenticated as: @{me.data.username}")
```

## Posting a Text-Only Tweet

```python
import tweepy, os

def load_credentials():
    creds = {}
    with open(os.path.expanduser("~/.config/twitter-post/.env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                creds[key.strip()] = val.strip()
    return creds

creds = load_credentials()
client = tweepy.Client(
    consumer_key=creds["X_API_KEY"],
    consumer_secret=creds["X_API_KEY_SECRET"],
    access_token=creds["X_ACCESS_TOKEN"],
    access_token_secret=creds["X_ACCESS_TOKEN_SECRET"],
)

response = client.create_tweet(text="Your tweet text here")
tweet_id = response.data["id"]
print(f"Tweet posted: https://x.com/i/status/{tweet_id}")
```

## Posting a Tweet with an Image

Image upload requires OAuth 1.0a via `tweepy.API` (v1.1 media upload), then the tweet
is created via `tweepy.Client` (v2).

```python
import tweepy, os

def load_credentials():
    creds = {}
    with open(os.path.expanduser("~/.config/twitter-post/.env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                creds[key.strip()] = val.strip()
    return creds

creds = load_credentials()

# v1.1 API for media upload
auth = tweepy.OAuth1UserHandler(
    creds["X_API_KEY"],
    creds["X_API_KEY_SECRET"],
    creds["X_ACCESS_TOKEN"],
    creds["X_ACCESS_TOKEN_SECRET"],
)
api = tweepy.API(auth)

# v2 Client for tweeting
client = tweepy.Client(
    consumer_key=creds["X_API_KEY"],
    consumer_secret=creds["X_API_KEY_SECRET"],
    access_token=creds["X_ACCESS_TOKEN"],
    access_token_secret=creds["X_ACCESS_TOKEN_SECRET"],
)

# Step 1: Upload image
media = api.media_upload("/path/to/image.png")

# Step 2: Tweet with image
response = client.create_tweet(
    text="Your tweet text here",
    media_ids=[media.media_id]
)
tweet_id = response.data["id"]
print(f"Tweet posted: https://x.com/i/status/{tweet_id}")
```

## Posting a Meme Coin Launch Announcement

Template for announcing a new pump.fun token launch:

```python
def post_launch_announcement(token_name, ticker, mint_address, pump_url, logo_path=None):
    """Post a meme coin launch announcement to Twitter/X."""
    import tweepy, os

    creds = {}
    with open(os.path.expanduser("~/.config/twitter-post/.env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                creds[key.strip()] = val.strip()

    client = tweepy.Client(
        consumer_key=creds["X_API_KEY"],
        consumer_secret=creds["X_API_KEY_SECRET"],
        access_token=creds["X_ACCESS_TOKEN"],
        access_token_secret=creds["X_ACCESS_TOKEN_SECRET"],
    )

    tweet_text = (
        f"${ticker} ({token_name}) just launched on pump.fun!\n\n"
        f"CA: {mint_address}\n\n"
        f"{pump_url}"
    )

    media_ids = None
    if logo_path and os.path.exists(logo_path):
        auth = tweepy.OAuth1UserHandler(
            creds["X_API_KEY"],
            creds["X_API_KEY_SECRET"],
            creds["X_ACCESS_TOKEN"],
            creds["X_ACCESS_TOKEN_SECRET"],
        )
        api = tweepy.API(auth)
        media = api.media_upload(logo_path)
        media_ids = [media.media_id]

    response = client.create_tweet(text=tweet_text, media_ids=media_ids)
    tweet_id = response.data["id"]
    tweet_url = f"https://x.com/i/status/{tweet_id}"
    print(f"Launch announced: {tweet_url}")
    return tweet_url
```

## Pipeline Integration

This skill receives inputs from **pump-deploy**:
- `token_name` — e.g. "TacoHeist"
- `ticker` — e.g. "TACO"
- `mint_address` — the token's contract address
- `pump_url` — e.g. "https://pump.fun/coin/MINT_ADDRESS"
- `logo_path` — path to the token logo image (optional)

Output:
- `tweet_url` — URL of the posted tweet

Pipeline flow:
```
meme-research → pump-deploy → twitter-post → content-gen (stage 3)
                    ↑
              solana-wallet
```

## Rate Limits & Quotas

- **Free tier**: 500 posts/month (~16/day), works for launch announcements
- **Pay-per-use**: credits deducted per API call (new Feb 2026 option)
- **Basic tier**: $200/month for 10,000 posts/month (overkill for this use case)
- Rate limit hit: HTTP 429 with `Retry-After` header

## Tweet Character Limit

- Max 280 characters per tweet
- URLs count as 23 characters regardless of actual length
- Images don't count toward character limit

## Pitfalls

- **Media upload uses v1.1 API** — requires OAuth 1.0a even when tweeting via v2. The tweepy.API(auth) object handles this.
- **App permissions must be "Read and Write"** — the default is "Read" only. Change in the X Developer Portal under your app settings. Regenerate tokens after changing permissions.
- **Free tier 280 char limit** — free accounts are hard-capped at 280 characters per tweet. Twitter returns a generic 403 error (not a helpful "too long" message) when you exceed this, making it easy to confuse with auth issues. Always check character count before posting. URLs count as 23 chars regardless of actual length.
- **Free tier** — if you hit 500/month, tweets silently fail or return 403. Monitor your usage.
- **Duplicate tweets** — Twitter rejects exact duplicate text within a short window. Vary your tweet text.
- **Sensitive content** — crypto/token promotion may trigger X's automated flagging. Keep tweets factual, avoid "guaranteed returns" language.
- **Image format** — PNG or JPG. Max 5MB for images. Recommended: 1200x675 for optimal display, or square 1024x1024 for logos.

## Cost Notes

- Twitter Free tier: $0/month
- Pay-per-use: varies, but posting is cheap (a few cents per tweet)
- For automated pipeline: Free tier is sufficient (even hourly research won't generate 500 launches/month)
