# YouTube Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `youtube`
**Base URL proxied:** `www.googleapis.com`

## API Path Pattern

```
/youtube/youtube/v3/{resource}
```

## Common Endpoints

### Search Videos
```bash
GET /youtube/youtube/v3/search?part=snippet&q=coding+tutorial&type=video&maxResults=10
```

Query parameters:
- `part` - Required: `snippet`
- `q` - Search query
- `type` - Filter: `video`, `channel`, `playlist`
- `maxResults` - Results per page (1-50)
- `order` - Sort: `date`, `rating`, `relevance`, `title`, `viewCount`
- `videoDuration` - `short` (<4min), `medium` (4-20min), `long` (>20min)

Example:

```bash
maton youtube search videos 'coding tutorial' --limit 10
```

### Get Video Details
```bash
GET /youtube/youtube/v3/videos?part=snippet,statistics,contentDetails&id={videoId}
```

Parts available: `snippet`, `statistics`, `contentDetails`, `status`, `player`

Example:

```bash
maton youtube video view {videoId}
```

### Get Trending Videos
```bash
GET /youtube/youtube/v3/videos?part=snippet,statistics&chart=mostPopular&regionCode=US&maxResults=10
```

Example:

```bash
maton youtube video list --region US --limit 10
```

### Rate Video
```bash
POST /youtube/youtube/v3/videos/rate?id={videoId}&rating=like
```

Rating values: `like`, `dislike`, `none`

Example:

```bash
maton youtube video rate {videoId} --rating like
```

### Get My Channel
```bash
GET /youtube/youtube/v3/channels?part=snippet,statistics,contentDetails&mine=true
```

Example:

```bash
maton youtube channel mine
```

### Get Channel Details
```bash
GET /youtube/youtube/v3/channels?part=snippet,statistics&id={channelId}
```

Example:

```bash
maton youtube channel view {channelId}
```

### List My Playlists
```bash
GET /youtube/youtube/v3/playlists?part=snippet,contentDetails&mine=true&maxResults=25
```

Example:

```bash
maton youtube playlist list --limit 25
```

### Create Playlist
```bash
POST /youtube/youtube/v3/playlists?part=snippet,status
Content-Type: application/json

{
  "snippet": {
    "title": "My New Playlist",
    "description": "A collection of videos"
  },
  "status": {
    "privacyStatus": "private"
  }
}
```

Privacy values: `public`, `private`, `unlisted`

Example:

```bash
maton youtube playlist create --title 'My New Playlist' --description 'A collection of videos' --privacy private
```

### Delete Playlist
```bash
DELETE /youtube/youtube/v3/playlists?id={playlistId}
```

Example:

```bash
maton youtube playlist delete {playlistId}
```

### List Playlist Items
```bash
GET /youtube/youtube/v3/playlistItems?part=snippet,contentDetails&playlistId={playlistId}&maxResults=50
```

Example:

```bash
maton youtube playlist items {playlistId} --limit 50
```

### Add Video to Playlist
```bash
POST /youtube/youtube/v3/playlistItems?part=snippet
Content-Type: application/json

{
  "snippet": {
    "playlistId": "PLxyz123",
    "resourceId": {
      "kind": "youtube#video",
      "videoId": "abc123xyz"
    },
    "position": 0
  }
}
```

Example:

```bash
maton youtube playlist add-video --playlist PLxyz123 --video abc123xyz --position 0
```

### List My Subscriptions
```bash
GET /youtube/youtube/v3/subscriptions?part=snippet&mine=true&maxResults=50
```

Example:

```bash
maton youtube subscription list --limit 50
```

### Subscribe to Channel
```bash
POST /youtube/youtube/v3/subscriptions?part=snippet
Content-Type: application/json

{
  "snippet": {
    "resourceId": {
      "kind": "youtube#channel",
      "channelId": "UCxyz123"
    }
  }
}
```

Example:

```bash
maton youtube subscription create --channel UCxyz123
```

### List Video Comments
```bash
GET /youtube/youtube/v3/commentThreads?part=snippet,replies&videoId={videoId}&maxResults=100
```

Example:

```bash
maton youtube comment list --video {videoId} --limit 100
```

### Add Comment to Video
```bash
POST /youtube/youtube/v3/commentThreads?part=snippet
Content-Type: application/json

{
  "snippet": {
    "videoId": "abc123xyz",
    "topLevelComment": {
      "snippet": {
        "textOriginal": "Great video!"
      }
    }
  }
}
```

Example:

```bash
maton youtube comment create --video abc123xyz --text 'Great video!'
```

## Pagination

YouTube uses cursor-based pagination via `pageToken`. The CLI handles this automatically with `--paginate`:

```bash
maton youtube playlist items {playlistId} --paginate
```

## Notes

- Video IDs are 11 characters (e.g., `dQw4w9WgXcQ`)
- Channel IDs start with `UC` (e.g., `UCxyz123`)
- Playlist IDs start with `PL` (user) or `UU` (uploads)
- Use `pageToken` for pagination through large result sets
- The `part` parameter is required and determines what data is returned
- Quota costs vary by endpoint - search is expensive (100 units), reads are cheap (1 unit)

## Resources

- [YouTube Data API Overview](https://developers.google.com/youtube/v3)
- [Search](https://developers.google.com/youtube/v3/docs/search/list)
- [Videos](https://developers.google.com/youtube/v3/docs/videos)
- [Channels](https://developers.google.com/youtube/v3/docs/channels)
- [Playlists](https://developers.google.com/youtube/v3/docs/playlists)
- [PlaylistItems](https://developers.google.com/youtube/v3/docs/playlistItems)
- [Subscriptions](https://developers.google.com/youtube/v3/docs/subscriptions)
- [Comments](https://developers.google.com/youtube/v3/docs/comments)
- [Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost)
- [Maton CLI Manual](https://cli.maton.ai/manual)
