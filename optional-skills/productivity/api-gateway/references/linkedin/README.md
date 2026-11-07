# LinkedIn Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `linkedin`
**Base URL proxied:** `api.linkedin.com`

## API Path Pattern

```
/linkedin/rest/{resource}
```

## Required Headers

```
LinkedIn-Version: 202506
```

## Common Endpoints

### Get Current User Profile
```bash
GET /linkedin/rest/me
LinkedIn-Version: 202506
```

### Create Text Post
```bash
POST /linkedin/rest/posts
Content-Type: application/json
LinkedIn-Version: 202506

{
  "author": "urn:li:person:{personId}",
  "lifecycleState": "PUBLISHED",
  "visibility": "PUBLIC",
  "commentary": "Hello LinkedIn!",
  "distribution": {
    "feedDistribution": "MAIN_FEED"
  }
}
```

### Create Article/URL Share
```bash
POST /linkedin/rest/posts
Content-Type: application/json
LinkedIn-Version: 202506

{
  "author": "urn:li:person:{personId}",
  "lifecycleState": "PUBLISHED",
  "visibility": "PUBLIC",
  "commentary": "Check this out!",
  "distribution": {
    "feedDistribution": "MAIN_FEED"
  },
  "content": {
    "article": {
      "source": "https://example.com",
      "title": "Title",
      "description": "Description"
    }
  }
}
```

### Initialize Image Upload
```bash
POST /linkedin/rest/images?action=initializeUpload
Content-Type: application/json
LinkedIn-Version: 202506

{
  "initializeUploadRequest": {
    "owner": "urn:li:person:{personId}"
  }
}
```

### Ad Library - Search Ads
```bash
GET /linkedin/rest/adLibrary?q=criteria&keyword=linkedin
LinkedIn-Version: 202506
```

### Job Library - Search Jobs
```bash
GET /linkedin/rest/jobLibrary?q=criteria&keyword=software
LinkedIn-Version: 202506
```

## Marketing API (Advertising)

Required headers for all Marketing API calls:
```
LinkedIn-Version: 202506
```

### List Ad Accounts
```bash
GET /linkedin/rest/adAccounts?q=search
```

### Get Ad Account
```bash
GET /linkedin/rest/adAccounts/{adAccountId}
```

### Create Ad Account
```bash
POST /linkedin/rest/adAccounts
Content-Type: application/json

{
  "name": "Ad Account Name",
  "currency": "USD",
  "reference": "urn:li:organization:{orgId}",
  "type": "BUSINESS"
}
```

### List Campaign Groups
```bash
GET /linkedin/rest/adAccounts/{adAccountId}/adCampaignGroups
```

### Create Campaign Group
```bash
POST /linkedin/rest/adAccounts/{adAccountId}/adCampaignGroups
Content-Type: application/json

{
  "name": "Campaign Group Name",
  "status": "DRAFT"
}
```

### Get Campaign Group
```bash
GET /linkedin/rest/adAccounts/{adAccountId}/adCampaignGroups/{campaignGroupId}
```

### List Campaigns
```bash
GET /linkedin/rest/adAccounts/{adAccountId}/adCampaigns
```

### Create Campaign
```bash
POST /linkedin/rest/adAccounts/{adAccountId}/adCampaigns
Content-Type: application/json

{
  "campaignGroup": "urn:li:sponsoredCampaignGroup:{groupId}",
  "name": "Campaign Name",
  "status": "DRAFT",
  "objectiveType": "BRAND_AWARENESS"
}
```

### Get Campaign
```bash
GET /linkedin/rest/adAccounts/{adAccountId}/adCampaigns/{campaignId}
```

### List Organization ACLs
```bash
GET /linkedin/rest/organizationAcls?q=roleAssignee
LinkedIn-Version: 202506
```

### Lookup Organization by Vanity Name
```bash
GET /linkedin/rest/organizations?q=vanityName&vanityName=microsoft
```

### Get Organization Share Statistics
```bash
GET /linkedin/rest/organizationalEntityShareStatistics?q=organizationalEntity&organizationalEntity=urn:li:organization:12345
```

### Get Organization Posts
```bash
GET /linkedin/rest/posts?q=author&author=urn:li:organization:12345
```

## Media Upload

> **CRITICAL — URL Encoding:** The upload URLs returned by initialize endpoints contain URL-encoded characters (e.g., `%253D`) that get corrupted when passed through shell variables or `curl`. You **MUST** use Python `urllib` for the entire upload flow — parse the JSON response with `json.load()` and use the URL directly in Python without ever passing it through the shell.

> **Upload URLs are pre-signed.** They point to `www.linkedin.com` (NOT `api.linkedin.com`), do NOT go through the gateway, and do NOT require an Authorization header.

### Initialize Image Upload
```bash
POST /linkedin/rest/images?action=initializeUpload
Content-Type: application/json
LinkedIn-Version: 202506

{"initializeUploadRequest": {"owner": "urn:li:person:{personId}"}}
```

### Video Upload (4-step process)

Video uploads require: initialize → upload binary → finalize → create post.

**Complete working example:**
```bash
python <<'EOF'
import urllib.request, os, json

GATEWAY = 'https://api.maton.ai'
HEADERS = {
    'Authorization': f'Bearer {os.environ["MATON_API_KEY"]}',
    'Content-Type': 'application/json',
    'LinkedIn-Version': '202506',
    'X-Restli-Protocol-Version': '2.0.0',
}

# Step 1: Initialize upload (via gateway)
file_path = '/path/to/video.mp4'
init_data = json.dumps({
    'initializeUploadRequest': {
        'owner': 'urn:li:person:{personId}',
        'fileSizeBytes': os.path.getsize(file_path),
        'uploadCaptions': False,
        'uploadThumbnail': False,
    }
}).encode()
req = urllib.request.Request(f'{GATEWAY}/linkedin/rest/videos?action=initializeUpload', data=init_data, method='POST')
for k, v in HEADERS.items(): req.add_header(k, v)
init_resp = json.load(urllib.request.urlopen(req))
upload_url = init_resp['value']['uploadInstructions'][0]['uploadUrl']
video_urn = init_resp['value']['video']

# Step 2: Upload binary DIRECTLY to pre-signed URL (NOT through gateway, NO auth header)
with open(file_path, 'rb') as f:
    video_data = f.read()
upload_req = urllib.request.Request(upload_url, data=video_data, method='PUT')
upload_req.add_header('Content-Type', 'application/octet-stream')
upload_resp = urllib.request.urlopen(upload_req)
etag = upload_resp.headers['etag']

# Step 3: Finalize upload (via gateway)
finalize_data = json.dumps({
    'finalizeUploadRequest': {
        'video': video_urn,
        'uploadToken': '',
        'uploadedPartIds': [etag],
    }
}).encode()
req = urllib.request.Request(f'{GATEWAY}/linkedin/rest/videos?action=finalizeUpload', data=finalize_data, method='POST')
for k, v in HEADERS.items(): req.add_header(k, v)
urllib.request.urlopen(req)

# Step 4: Create post with video (via gateway)
post_data = json.dumps({
    'author': 'urn:li:person:{personId}',
    'lifecycleState': 'PUBLISHED',
    'visibility': 'PUBLIC',
    'commentary': 'Check out this video!',
    'distribution': {'feedDistribution': 'MAIN_FEED'},
    'content': {'media': {'id': video_urn}},
}).encode()
req = urllib.request.Request(f'{GATEWAY}/linkedin/rest/posts', data=post_data, method='POST')
for k, v in HEADERS.items(): req.add_header(k, v)
resp = urllib.request.urlopen(req)
print(f'Video post created! {resp.headers.get("location")}')
EOF
```

**Video specs:** 3s–30min, 75KB–500MB, MP4 format. For videos >4MB, LinkedIn returns multiple `uploadInstructions` — upload each chunk and collect all etags.

### Initialize Document Upload
```bash
POST /linkedin/rest/documents?action=initializeUpload
Content-Type: application/json
LinkedIn-Version: 202506

{"initializeUploadRequest": {"owner": "urn:li:person:{personId}"}}
```

## Ad Targeting

### Get Targeting Facets
```bash
GET /linkedin/rest/adTargetingFacets
LinkedIn-Version: 202506
```

Returns 31 targeting facets (skills, industries, titles, locations, etc.)

## Little Text Format (Commentary Field)

The `commentary` field in posts uses LinkedIn's "Little Text Format". **Reserved characters must be escaped with a backslash or the post content will be truncated.**

### Reserved Characters (Must Escape)

| Character | Escape As |
|-----------|-----------|
| `\` | `\\` |
| `\|` | `\\|` |
| `{` | `\{` |
| `}` | `\}` |
| `@` | `\@` |
| `[` | `\[` |
| `]` | `\]` |
| `(` | `\(` |
| `)` | `\)` |
| `<` | `\<` |
| `>` | `\>` |
| `#` | `\#` |
| `*` | `\*` |
| `_` | `\_` |
| `~` | `\~` |

### Example

```json
{
  "commentary": "Hello\\! Check out these bullet points:\\n\\n\\* Point 1\\n\\* Point 2\\n\\* More info \\(details inside\\)"
}
```

### Mentions and Hashtags

- **Mention a person:** `@[Display Name](urn:li:person:123)`
- **Mention an organization:** `@[Company Name](urn:li:organization:456)`
- **Hashtag:** `{hashtag|\\#|MyTag}` or simply `#hashtag` for single words

## Notes

- Authentication is automatic - the router injects the OAuth token
- Include `LinkedIn-Version: 202506` header for all REST API calls
- Author URN format: `urn:li:person:{personId}`
- Get person ID from `/rest/me` endpoint
- **Commentary uses Little Text Format** — escape reserved characters (`|{}@[]()<>#\*_~`) with backslash or content will be truncated
- Image uploads are 3-step: initialize, upload binary, create post
- Video uploads are 4-step: initialize, upload binary, finalize, create post
- **Media upload URLs point to `www.linkedin.com` (not `api.linkedin.com`).** They are pre-signed — do NOT send through the gateway, do NOT add an Authorization header. MUST use Python `urllib` (not shell `curl`) due to URL encoding issues.
- Rate limits: 150 requests/day per member, 100K/day per app

## Visibility Options

- `PUBLIC` - Viewable by anyone
- `CONNECTIONS` - 1st-degree connections only

## Share Media Categories

- `NONE` - Text only
- `ARTICLE` - URL share
- `IMAGE` - Image post
- `VIDEO` - Video post

## Resources

- [LinkedIn API Overview](https://learn.microsoft.com/en-us/linkedin/)
- [Share on LinkedIn](https://learn.microsoft.com/en-us/linkedin/consumer/integrations/self-serve/share-on-linkedin)
- [Profile API](https://learn.microsoft.com/en-us/linkedin/shared/integrations/people/profile-api)
- [Marketing API](https://learn.microsoft.com/en-us/linkedin/marketing/)
- [Ad Accounts](https://learn.microsoft.com/en-us/linkedin/marketing/integrations/ads/account-structure/create-and-manage-accounts)
- [Campaigns](https://learn.microsoft.com/en-us/linkedin/marketing/integrations/ads/account-structure/create-and-manage-campaigns)
