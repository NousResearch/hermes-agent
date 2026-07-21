# YouTube Data API v3

Discovery source: `https://youtube.googleapis.com/$discovery/rest?version=v3`
Public API base URL: `https://api.mybrandmetrics.com`
Methods: 83
Schemas: 199

## Method Index

### abuseReports
- `youtube.abuseReports.insert` - `POST /youtube/v3/abuseReports`

### activities
- `youtube.activities.list` - `GET /youtube/v3/activities`

### captions
- `youtube.captions.delete` - `DELETE /youtube/v3/captions`
- `youtube.captions.download` - `GET /youtube/v3/captions/{id}`
- `youtube.captions.insert` - `POST /youtube/v3/captions`
- `youtube.captions.list` - `GET /youtube/v3/captions`
- `youtube.captions.update` - `PUT /youtube/v3/captions`

### channelBanners
- `youtube.channelBanners.insert` - `POST /youtube/v3/channelBanners/insert`

### channelSections
- `youtube.channelSections.delete` - `DELETE /youtube/v3/channelSections`
- `youtube.channelSections.insert` - `POST /youtube/v3/channelSections`
- `youtube.channelSections.list` - `GET /youtube/v3/channelSections`
- `youtube.channelSections.update` - `PUT /youtube/v3/channelSections`

### channels
- `youtube.channels.list` - `GET /youtube/v3/channels`
- `youtube.channels.update` - `PUT /youtube/v3/channels`

### commentThreads
- `youtube.commentThreads.insert` - `POST /youtube/v3/commentThreads`
- `youtube.commentThreads.list` - `GET /youtube/v3/commentThreads`

### comments
- `youtube.comments.delete` - `DELETE /youtube/v3/comments`
- `youtube.comments.insert` - `POST /youtube/v3/comments`
- `youtube.comments.list` - `GET /youtube/v3/comments`
- `youtube.comments.markAsSpam` - `POST /youtube/v3/comments/markAsSpam`
- `youtube.comments.setModerationStatus` - `POST /youtube/v3/comments/setModerationStatus`
- `youtube.comments.update` - `PUT /youtube/v3/comments`

### i18nLanguages
- `youtube.i18nLanguages.list` - `GET /youtube/v3/i18nLanguages`

### i18nRegions
- `youtube.i18nRegions.list` - `GET /youtube/v3/i18nRegions`

### liveBroadcasts
- `youtube.liveBroadcasts.bind` - `POST /youtube/v3/liveBroadcasts/bind`
- `youtube.liveBroadcasts.delete` - `DELETE /youtube/v3/liveBroadcasts`
- `youtube.liveBroadcasts.insert` - `POST /youtube/v3/liveBroadcasts`
- `youtube.liveBroadcasts.insertCuepoint` - `POST /youtube/v3/liveBroadcasts/cuepoint`
- `youtube.liveBroadcasts.list` - `GET /youtube/v3/liveBroadcasts`
- `youtube.liveBroadcasts.transition` - `POST /youtube/v3/liveBroadcasts/transition`
- `youtube.liveBroadcasts.update` - `PUT /youtube/v3/liveBroadcasts`

### liveChatBans
- `youtube.liveChatBans.delete` - `DELETE /youtube/v3/liveChat/bans`
- `youtube.liveChatBans.insert` - `POST /youtube/v3/liveChat/bans`

### liveChatMessages
- `youtube.liveChatMessages.delete` - `DELETE /youtube/v3/liveChat/messages`
- `youtube.liveChatMessages.insert` - `POST /youtube/v3/liveChat/messages`
- `youtube.liveChatMessages.list` - `GET /youtube/v3/liveChat/messages`
- `youtube.liveChatMessages.transition` - `POST /youtube/v3/liveChat/messages/transition`

### liveChatModerators
- `youtube.liveChatModerators.delete` - `DELETE /youtube/v3/liveChat/moderators`
- `youtube.liveChatModerators.insert` - `POST /youtube/v3/liveChat/moderators`
- `youtube.liveChatModerators.list` - `GET /youtube/v3/liveChat/moderators`

### liveStreams
- `youtube.liveStreams.delete` - `DELETE /youtube/v3/liveStreams`
- `youtube.liveStreams.insert` - `POST /youtube/v3/liveStreams`
- `youtube.liveStreams.list` - `GET /youtube/v3/liveStreams`
- `youtube.liveStreams.update` - `PUT /youtube/v3/liveStreams`

### members
- `youtube.members.list` - `GET /youtube/v3/members`

### membershipsLevels
- `youtube.membershipsLevels.list` - `GET /youtube/v3/membershipsLevels`

### playlistImages
- `youtube.playlistImages.delete` - `DELETE /youtube/v3/playlistImages`
- `youtube.playlistImages.insert` - `POST /youtube/v3/playlistImages`
- `youtube.playlistImages.list` - `GET /youtube/v3/playlistImages`
- `youtube.playlistImages.update` - `PUT /youtube/v3/playlistImages`

### playlistItems
- `youtube.playlistItems.delete` - `DELETE /youtube/v3/playlistItems`
- `youtube.playlistItems.insert` - `POST /youtube/v3/playlistItems`
- `youtube.playlistItems.list` - `GET /youtube/v3/playlistItems`
- `youtube.playlistItems.update` - `PUT /youtube/v3/playlistItems`

### playlists
- `youtube.playlists.delete` - `DELETE /youtube/v3/playlists`
- `youtube.playlists.insert` - `POST /youtube/v3/playlists`
- `youtube.playlists.list` - `GET /youtube/v3/playlists`
- `youtube.playlists.update` - `PUT /youtube/v3/playlists`

### search
- `youtube.search.list` - `GET /youtube/v3/search`

### subscriptions
- `youtube.subscriptions.delete` - `DELETE /youtube/v3/subscriptions`
- `youtube.subscriptions.insert` - `POST /youtube/v3/subscriptions`
- `youtube.subscriptions.list` - `GET /youtube/v3/subscriptions`

### superChatEvents
- `youtube.superChatEvents.list` - `GET /youtube/v3/superChatEvents`

### tests
- `youtube.tests.insert` - `POST /youtube/v3/tests`

### thirdPartyLinks
- `youtube.thirdPartyLinks.delete` - `DELETE /youtube/v3/thirdPartyLinks`
- `youtube.thirdPartyLinks.insert` - `POST /youtube/v3/thirdPartyLinks`
- `youtube.thirdPartyLinks.list` - `GET /youtube/v3/thirdPartyLinks`
- `youtube.thirdPartyLinks.update` - `PUT /youtube/v3/thirdPartyLinks`

### thumbnails
- `youtube.thumbnails.set` - `POST /youtube/v3/thumbnails/set`

### videoAbuseReportReasons
- `youtube.videoAbuseReportReasons.list` - `GET /youtube/v3/videoAbuseReportReasons`

### videoCategories
- `youtube.videoCategories.list` - `GET /youtube/v3/videoCategories`

### videoTrainability
- `youtube.videoTrainability.get` - `GET /youtube/v3/videoTrainability`

### videos
- `youtube.videos.delete` - `DELETE /youtube/v3/videos`
- `youtube.videos.getRating` - `GET /youtube/v3/videos/getRating`
- `youtube.videos.insert` - `POST /youtube/v3/videos`
- `youtube.videos.list` - `GET /youtube/v3/videos`
- `youtube.videos.rate` - `POST /youtube/v3/videos/rate`
- `youtube.videos.reportAbuse` - `POST /youtube/v3/videos/reportAbuse`
- `youtube.videos.update` - `PUT /youtube/v3/videos`

### watermarks
- `youtube.watermarks.set` - `POST /youtube/v3/watermarks/set`
- `youtube.watermarks.unset` - `POST /youtube/v3/watermarks/unset`

### youtube.v3
- `youtube.youtube.v3.updateCommentThreads` - `PUT /youtube/v3/commentThreads`

### youtube.v3.liveChat.messages
- `youtube.youtube.v3.liveChat.messages.stream` - `GET /youtube/v3/liveChat/messages/stream`

## Method Reference

### `youtube.abuseReports.insert`

- Resource: `abuseReports`
- HTTP: `POST /youtube/v3/abuseReports`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `AbuseReport`
- Response schema: `AbuseReport`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/abuseReports" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.activities.list`

- Resource: `activities`
- HTTP: `GET /youtube/v3/activities`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `ActivityListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more activity resource properties that the API response will include. If the parameter identifies a property that contains child properties, the child pr... |
| `channelId` | `query` | no | `string` |  |
| `home` | `query` | no | `boolean` |  |
| `mine` | `query` | no | `boolean` |  |
| `publishedAfter` | `query` | no | `string` |  |
| `publishedBefore` | `query` | no | `string` |  |
| `regionCode` | `query` | no | `string` |  |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/activities" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.captions.delete`

- Resource: `captions`
- HTTP: `DELETE /youtube/v3/captions`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOf` | `query` | no | `string` | ID of the Google+ Page for the channel that the request is be on behalf of |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.captions.download`

- Resource: `captions`
- HTTP: `GET /youtube/v3/captions/{id}`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`
- Supports media download: yes

Downloads a caption track.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `path` | yes | `string` | The ID of the caption track to download, required for One Platform. |
| `tlang` | `query` | no | `string` | tlang is the language code; machine translate the captions into this language. |
| `tfmt` | `query` | no | `string` | Convert the captions into this format. Supported options are sbv, srt, and vtt. |
| `onBehalfOf` | `query` | no | `string` | ID of the Google+ Page for the channel that the request is be on behalf of |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions/${ID}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtube.captions.insert`

- Resource: `captions`
- HTTP: `POST /youtube/v3/captions`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Caption`
- Response schema: `Caption`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the caption resource parts that the API response will include. Set the parameter value to snippet. |
| `sync` | `query` | no | `boolean` | Extra parameter to allow automatically syncing the uploaded caption/transcript with the audio. |
| `onBehalfOf` | `query` | no | `string` | ID of the Google+ Page for the channel that the request is be on behalf of |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "part=snippet"
```

### `youtube.captions.list`

- Resource: `captions`
- HTTP: `GET /youtube/v3/captions`
- Required parameters: `videoId`, `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `CaptionListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more caption resource parts that the API response will include. The part names that you can include in the parameter value are id and snippet. |
| `videoId` | `query` | yes | `string` | Returns the captions for the specified video. |
| `id` | `query` | no | `string` | Returns the captions with the given IDs for Stubby or Apiary. |
| `onBehalfOf` | `query` | no | `string` | ID of the Google+ Page for the channel that the request is on behalf of. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet" \
  --url-query "videoId=${VIDEO_ID}"
```

### `youtube.captions.update`

- Resource: `captions`
- HTTP: `PUT /youtube/v3/captions`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Caption`
- Response schema: `Caption`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more caption resource parts that the API response will include. The part names that you can include in the parameter value are id and snippet. |
| `sync` | `query` | no | `boolean` | Extra parameter to allow automatically syncing the uploaded caption/transcript with the audio. |
| `onBehalfOf` | `query` | no | `string` | ID of the Google+ Page for the channel that the request is on behalf of. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/captions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "part=snippet"
```

### `youtube.channelBanners.insert`

- Resource: `channelBanners`
- HTTP: `POST /youtube/v3/channelBanners/insert`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.upload`
- Request schema: `ChannelBannerResource`
- Response schema: `ChannelBannerResource`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `channelId` | `query` | no | `string` | Unused, channel_id is currently derived from the security context of the requestor. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channelBanners/insert" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channelBanners/insert" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}"
```

### `youtube.channelSections.delete`

- Resource: `channelSections`
- HTTP: `DELETE /youtube/v3/channelSections`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channelSections" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.channelSections.insert`

- Resource: `channelSections`
- HTTP: `POST /youtube/v3/channelSections`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `ChannelSection`
- Response schema: `ChannelSection`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The part names that you can incl... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channelSections" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.channelSections.list`

- Resource: `channelSections`
- HTTP: `GET /youtube/v3/channelSections`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `ChannelSectionListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more channelSection resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, a... |
| `id` | `query` | no | `string` | Return the ChannelSections with the given IDs for Stubby or Apiary. |
| `mine` | `query` | no | `boolean` | Return the ChannelSections owned by the authenticated user. |
| `channelId` | `query` | no | `string` | Return the ChannelSections owned by the specified channel ID. |
| `hl` | `query` | no | `string` | Return content in specified language |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channelSections" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.channelSections.update`

- Resource: `channelSections`
- HTTP: `PUT /youtube/v3/channelSections`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `ChannelSection`
- Response schema: `ChannelSection`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The part names that you can incl... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channelSections" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.channels.list`

- Resource: `channels`
- HTTP: `GET /youtube/v3/channels`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/youtubepartner-channel-audit`
- Request schema: `none`
- Response schema: `ChannelListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more channel resource properties that the API response will include. If the parameter identifies a property that contains child properties, the child pro... |
| `mine` | `query` | no | `boolean` | Return the ids of channels owned by the authenticated user. |
| `id` | `query` | no | `string` | Return the channels with the specified IDs. |
| `mySubscribers` | `query` | no | `boolean` | Return the channels subscribed to the authenticated user |
| `categoryId` | `query` | no | `string` | Return the channels within the specified guide category ID. |
| `managedByMe` | `query` | no | `boolean` | Return the channels managed by the authenticated user. |
| `forUsername` | `query` | no | `string` | Return the channel associated with a YouTube username. |
| `forHandle` | `query` | no | `string` | Return the channel associated with a YouTube handle. |
| `hl` | `query` | no | `string` | Stands for "host language". Specifies the localization language of the metadata to be filled into snippet.localized. The field is filled with the default metadata if there is no localization in the specified language.... |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.channels.update`

- Resource: `channels`
- HTTP: `PUT /youtube/v3/channels`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Channel`
- Response schema: `Channel`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The API currently only allows th... |
| `onBehalfOfContentOwner` | `query` | no | `string` | The *onBehalfOfContentOwner* parameter indicates that the authenticated user is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/channels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.commentThreads.insert`

- Resource: `commentThreads`
- HTTP: `POST /youtube/v3/commentThreads`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `CommentThread`
- Response schema: `CommentThread`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter identifies the properties that the API response will include. Set the parameter value to snippet. The snippet part has a quota cost of 2 units. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/commentThreads" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.commentThreads.list`

- Resource: `commentThreads`
- HTTP: `GET /youtube/v3/commentThreads`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `CommentThreadListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more commentThread resource properties that the API response will include. |
| `id` | `query` | no | `string` | Returns the comment threads with the given IDs for Stubby or Apiary. |
| `videoId` | `query` | no | `string` | Returns the comment threads of the specified video. |
| `postId` | `query` | no | `string` | Returns the comment threads of the specified post. |
| `channelId` | `query` | no | `string` | Returns the comment threads for all the channel comments (ie does not include comments left on videos). |
| `allThreadsRelatedToChannelId` | `query` | no | `string` | Returns the comment threads of all videos of the channel and the channel comments as well. |
| `moderationStatus` | `query` | no | `string` | Limits the returned comment threads to those with the specified moderation status. Not compatible with the 'id' filter. Valid values: published, heldForReview, likelySpam. Allowed: published, heldForReview, likelySpam... |
| `searchTerms` | `query` | no | `string` | Limits the returned comment threads to those matching the specified key words. Not compatible with the 'id' filter. |
| `textFormat` | `query` | no | `string` | The requested text format for the returned comments. Allowed: textFormatUnspecified, html, plainText. |
| `order` | `query` | no | `string` | Allowed: orderUnspecified, time, relevance. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/commentThreads" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.comments.delete`

- Resource: `comments`
- HTTP: `DELETE /youtube/v3/comments`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/comments" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.comments.insert`

- Resource: `comments`
- HTTP: `POST /youtube/v3/comments`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `Comment`
- Response schema: `Comment`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter identifies the properties that the API response will include. Set the parameter value to snippet. The snippet part has a quota cost of 2 units. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/comments" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.comments.list`

- Resource: `comments`
- HTTP: `GET /youtube/v3/comments`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `CommentListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more comment resource properties that the API response will include. |
| `id` | `query` | no | `string` | Returns the comments with the given IDs for One Platform. |
| `parentId` | `query` | no | `string` | Returns replies to the specified comment. Note, currently YouTube features only one level of replies (ie replies to top level comments). However replies to replies may be supported in the future. |
| `textFormat` | `query` | no | `string` | The requested text format for the returned comments. Allowed: textFormatUnspecified, html, plainText. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/comments" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.comments.markAsSpam`

- Resource: `comments`
- HTTP: `POST /youtube/v3/comments/markAsSpam`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Expresses the caller's opinion that one or more comments should be flagged as spam.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` | Flags the comments with the given IDs as spam in the caller's opinion. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/comments/markAsSpam" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.comments.setModerationStatus`

- Resource: `comments`
- HTTP: `POST /youtube/v3/comments/setModerationStatus`
- Required parameters: `id`, `moderationStatus`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Sets the moderation status of one or more comments.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` | Modifies the moderation status of the comments with the given IDs |
| `moderationStatus` | `query` | yes | `string` | Specifies the requested moderation status. Note, comments can be in statuses, which are not available through this call. For example, this call does not allow to mark a comment as 'likely spam'. Valid values: 'heldFor... |
| `banAuthor` | `query` | no | `boolean` | If set to true the author of the comment gets added to the ban list. This means all future comments of the author will autmomatically be rejected. Only valid in combination with STATUS_REJECTED. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/comments/setModerationStatus" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}" \
  --url-query "moderationStatus=published"
```

### `youtube.comments.update`

- Resource: `comments`
- HTTP: `PUT /youtube/v3/comments`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `Comment`
- Response schema: `Comment`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter identifies the properties that the API response will include. You must at least include the snippet part in the parameter value since that part contains all of the properties that the API request... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/comments" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.i18nLanguages.list`

- Resource: `i18nLanguages`
- HTTP: `GET /youtube/v3/i18nLanguages`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `I18nLanguageListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the i18nLanguage resource properties that the API response will include. Set the parameter value to snippet. |
| `hl` | `query` | no | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/i18nLanguages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.i18nRegions.list`

- Resource: `i18nRegions`
- HTTP: `GET /youtube/v3/i18nRegions`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `I18nRegionListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the i18nRegion resource properties that the API response will include. Set the parameter value to snippet. |
| `hl` | `query` | no | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/i18nRegions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.liveBroadcasts.bind`

- Resource: `liveBroadcasts`
- HTTP: `POST /youtube/v3/liveBroadcasts/bind`
- Required parameters: `id`, `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `LiveBroadcast`

Bind a broadcast to a stream.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` | Broadcast to bind to the stream |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, co... |
| `streamId` | `query` | no | `string` | Stream to bind, if not set unbind the current one. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts/bind" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}" \
  --url-query "part=snippet"
```

### `youtube.liveBroadcasts.delete`

- Resource: `liveBroadcasts`
- HTTP: `DELETE /youtube/v3/liveBroadcasts`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Delete a given broadcast.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` | Broadcast to delete. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.liveBroadcasts.insert`

- Resource: `liveBroadcasts`
- HTTP: `POST /youtube/v3/liveBroadcasts`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveBroadcast`
- Response schema: `LiveBroadcast`

Inserts a new stream for the authenticated user.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The part properties that you can... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveBroadcasts.insertCuepoint`

- Resource: `liveBroadcasts`
- HTTP: `POST /youtube/v3/liveBroadcasts/cuepoint`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Cuepoint`
- Response schema: `Cuepoint`

Insert cuepoints in a broadcast

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | no | `string` | Broadcast to insert ads to, or equivalently `external_video_id` for internal use. |
| `part` | `query` | no | `string` | The *part* parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, co... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts/cuepoint" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveBroadcasts.list`

- Resource: `liveBroadcasts`
- HTTP: `GET /youtube/v3/liveBroadcasts`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `LiveBroadcastListResponse`

Retrieve the list of broadcasts associated with the given channel.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, co... |
| `broadcastStatus` | `query` | no | `string` | Return broadcasts with a certain status, e.g. active broadcasts. Allowed: broadcastStatusFilterUnspecified, all, active, upcoming, completed. |
| `id` | `query` | no | `string` | Return broadcasts with the given ids from Stubby or Apiary. |
| `mine` | `query` | no | `boolean` |  |
| `broadcastType` | `query` | no | `string` | Return only broadcasts with the selected type. Allowed: broadcastTypeFilterUnspecified, all, event, persistent. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.liveBroadcasts.transition`

- Resource: `liveBroadcasts`
- HTTP: `POST /youtube/v3/liveBroadcasts/transition`
- Required parameters: `id`, `broadcastStatus`, `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `LiveBroadcast`

Transition a broadcast to a given status.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `broadcastStatus` | `query` | yes | `string` | The status to which the broadcast is going to transition. Allowed: statusUnspecified, testing, live, complete. |
| `id` | `query` | yes | `string` | Broadcast to transition. |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, co... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts/transition" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "broadcastStatus=statusUnspecified" \
  --url-query "id=${ID}" \
  --url-query "part=snippet"
```

### `youtube.liveBroadcasts.update`

- Resource: `liveBroadcasts`
- HTTP: `PUT /youtube/v3/liveBroadcasts`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveBroadcast`
- Response schema: `LiveBroadcast`

Updates an existing broadcast for the authenticated user.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The part properties that you can... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveBroadcasts" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveChatBans.delete`

- Resource: `liveChatBans`
- HTTP: `DELETE /youtube/v3/liveChat/bans`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Deletes a chat ban.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/bans" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.liveChatBans.insert`

- Resource: `liveChatBans`
- HTTP: `POST /youtube/v3/liveChat/bans`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveChatBan`
- Response schema: `LiveChatBan`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response returns. Set the parameter value to snippet. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/bans" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveChatMessages.delete`

- Resource: `liveChatMessages`
- HTTP: `DELETE /youtube/v3/liveChat/messages`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Deletes a chat message.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/messages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.liveChatMessages.insert`

- Resource: `liveChatMessages`
- HTTP: `POST /youtube/v3/liveChat/messages`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveChatMessage`
- Response schema: `LiveChatMessage`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes. It identifies the properties that the write operation will set as well as the properties that the API response will include. Set the parameter value to snippet. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/messages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveChatMessages.list`

- Resource: `liveChatMessages`
- HTTP: `GET /youtube/v3/liveChat/messages`
- Required parameters: `liveChatId`, `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `LiveChatMessageListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `liveChatId` | `query` | yes | `string` | The id of the live chat for which comments should be returned. |
| `part` | `query` | yes | `string` | The *part* parameter specifies the liveChatComment resource parts that the API response will include. Supported values are id, snippet, and authorDetails. |
| `hl` | `query` | no | `string` | Specifies the localization language in which the system messages should be returned. |
| `profileImageSize` | `query` | no | `integer` | Specifies the size of the profile image that should be returned for each user. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. Not used in the streaming RPC. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken property identify other pages that could be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/messages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "liveChatId=${LIVE_CHAT_ID}" \
  --url-query "part=snippet"
```

### `youtube.liveChatMessages.transition`

- Resource: `liveChatMessages`
- HTTP: `POST /youtube/v3/liveChat/messages/transition`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `LiveChatMessage`

Transition a durable chat event.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | no | `string` | The ID that uniquely identify the chat message event to transition. |
| `status` | `query` | no | `string` | The status to which the chat event is going to transition. Allowed: statusUnspecified, closed. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/messages/transition" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtube.liveChatModerators.delete`

- Resource: `liveChatModerators`
- HTTP: `DELETE /youtube/v3/liveChat/moderators`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Deletes a chat moderator.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/moderators" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.liveChatModerators.insert`

- Resource: `liveChatModerators`
- HTTP: `POST /youtube/v3/liveChat/moderators`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveChatModerator`
- Response schema: `LiveChatModerator`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response returns. Set the parameter value to snippet. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/moderators" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveChatModerators.list`

- Resource: `liveChatModerators`
- HTTP: `GET /youtube/v3/liveChat/moderators`
- Required parameters: `liveChatId`, `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `LiveChatModeratorListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `liveChatId` | `query` | yes | `string` | The id of the live chat for which moderators should be returned. |
| `part` | `query` | yes | `string` | The *part* parameter specifies the liveChatModerator resource parts that the API response will include. Supported values are id and snippet. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/moderators" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "liveChatId=${LIVE_CHAT_ID}" \
  --url-query "part=snippet"
```

### `youtube.liveStreams.delete`

- Resource: `liveStreams`
- HTTP: `DELETE /youtube/v3/liveStreams`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `none`
- Response schema: `none`

Deletes an existing stream for the authenticated user.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveStreams" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.liveStreams.insert`

- Resource: `liveStreams`
- HTTP: `POST /youtube/v3/liveStreams`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveStream`
- Response schema: `LiveStream`

Inserts a new stream for the authenticated user.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The part properties that you can... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveStreams" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.liveStreams.list`

- Resource: `liveStreams`
- HTTP: `GET /youtube/v3/liveStreams`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `LiveStreamListResponse`

Retrieve the list of streams associated with the given channel. --

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more liveStream resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, cdn,... |
| `id` | `query` | no | `string` | Return LiveStreams with the given ids from Stubby or Apiary. |
| `mine` | `query` | no | `boolean` |  |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveStreams" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.liveStreams.update`

- Resource: `liveStreams`
- HTTP: `PUT /youtube/v3/liveStreams`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`
- Request schema: `LiveStream`
- Response schema: `LiveStream`

Updates an existing stream for the authenticated user.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. The part properties that you can... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveStreams" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.members.list`

- Resource: `members`
- HTTP: `GET /youtube/v3/members`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.channel-memberships.creator`
- Request schema: `none`
- Response schema: `MemberListResponse`

Retrieves a list of members that match the request criteria for a channel.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the member resource parts that the API response will include. Set the parameter value to snippet. |
| `mode` | `query` | no | `string` | Parameter that specifies which channel members to return. Allowed: listMembersModeUnknown, updates, all_current. |
| `hasAccessToLevel` | `query` | no | `string` | Filter members in the results set to the ones that have access to a level. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `filterByMemberChannelId` | `query` | no | `string` | Comma separated list of channel IDs. Only data about members that are part of this list will be included in the response. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/members" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.membershipsLevels.list`

- Resource: `membershipsLevels`
- HTTP: `GET /youtube/v3/membershipsLevels`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.channel-memberships.creator`
- Request schema: `none`
- Response schema: `MembershipsLevelListResponse`

Retrieves a list of all pricing levels offered by a creator to the fans.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the membershipsLevel resource parts that the API response will include. Supported values are id and snippet. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/membershipsLevels" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.playlistImages.delete`

- Resource: `playlistImages`
- HTTP: `DELETE /youtube/v3/playlistImages`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | no | `string` | Id to identify this image. This is returned from by the List method. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistImages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtube.playlistImages.insert`

- Resource: `playlistImages`
- HTTP: `POST /youtube/v3/playlistImages`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `PlaylistImage`
- Response schema: `PlaylistImage`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | no | `string` | The *part* parameter specifies the properties that the API response will include. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistImages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistImages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "part=snippet"
```

### `youtube.playlistImages.list`

- Resource: `playlistImages`
- HTTP: `GET /youtube/v3/playlistImages`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `PlaylistImageListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `parent` | `query` | no | `string` | Return PlaylistImages for this playlist id. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `part` | `query` | no | `string` | The *part* parameter specifies a comma-separated list of one or more playlistImage resource properties that the API response will include. If the parameter identifies a property that contains child properties, the chi... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistImages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.playlistImages.update`

- Resource: `playlistImages`
- HTTP: `PUT /youtube/v3/playlistImages`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `PlaylistImage`
- Response schema: `PlaylistImage`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | no | `string` | The *part* parameter specifies the properties that the API response will include. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistImages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistImages" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "part=snippet"
```

### `youtube.playlistItems.delete`

- Resource: `playlistItems`
- HTTP: `DELETE /youtube/v3/playlistItems`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.playlistItems.insert`

- Resource: `playlistItems`
- HTTP: `POST /youtube/v3/playlistItems`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `PlaylistItem`
- Response schema: `PlaylistItem`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.playlistItems.list`

- Resource: `playlistItems`
- HTTP: `GET /youtube/v3/playlistItems`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `PlaylistItemListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more playlistItem resource properties that the API response will include. If the parameter identifies a property that contains child properties, the chil... |
| `id` | `query` | no | `string` |  |
| `playlistId` | `query` | no | `string` | Return the playlist items within the given playlist. |
| `videoId` | `query` | no | `string` | Return the playlist items associated with the given video ID. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.playlistItems.update`

- Resource: `playlistItems`
- HTTP: `PUT /youtube/v3/playlistItems`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `PlaylistItem`
- Response schema: `PlaylistItem`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. Note that this method will overr... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlistItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.playlists.delete`

- Resource: `playlists`
- HTTP: `DELETE /youtube/v3/playlists`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlists" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.playlists.insert`

- Resource: `playlists`
- HTTP: `POST /youtube/v3/playlists`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Playlist`
- Response schema: `Playlist`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlists" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.playlists.list`

- Resource: `playlists`
- HTTP: `GET /youtube/v3/playlists`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `PlaylistListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more playlist resource properties that the API response will include. If the parameter identifies a property that contains child properties, the child pr... |
| `id` | `query` | no | `string` | Return the playlists with the given IDs for Stubby or Apiary. |
| `mine` | `query` | no | `boolean` | Return the playlists owned by the authenticated user. |
| `channelId` | `query` | no | `string` | Return the playlists owned by the specified channel ID. |
| `hl` | `query` | no | `string` | Return content in specified language |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlists" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.playlists.update`

- Resource: `playlists`
- HTTP: `PUT /youtube/v3/playlists`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Playlist`
- Response schema: `Playlist`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. Note that this method will overr... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/playlists" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.search.list`

- Resource: `search`
- HTTP: `GET /youtube/v3/search`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `SearchListResponse`

Retrieves a list of search resources

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more search resource properties that the API response will include. Set the parameter value to snippet. |
| `q` | `query` | no | `string` | Textual search terms to match. |
| `type` | `query` | no | `string` | Restrict results to a particular set of resource types from One Platform. |
| `order` | `query` | no | `string` | Sort order of the results. Allowed: searchSortUnspecified, date, rating, viewCount, relevance, title, videoCount. |
| `relevanceLanguage` | `query` | no | `string` | Return results relevant to this language. |
| `videoDimension` | `query` | no | `string` | Filter on 3d videos. Allowed: any, 2d, 3d. |
| `videoDefinition` | `query` | no | `string` | Filter on the definition of the videos. Allowed: any, standard, high. |
| `videoLicense` | `query` | no | `string` | Filter on the license of the videos. Allowed: any, youtube, creativeCommon. |
| `videoDuration` | `query` | no | `string` | Filter on the duration of the videos. Allowed: videoDurationUnspecified, any, short, medium, long. |
| `videoCaption` | `query` | no | `string` | Filter on the presence of captions on the videos. Allowed: videoCaptionUnspecified, any, closedCaption, none. |
| `videoEmbeddable` | `query` | no | `string` | Filter on embeddable videos. Allowed: videoEmbeddableUnspecified, any, true. |
| `videoSyndicated` | `query` | no | `string` | Filter on syndicated videos. Allowed: videoSyndicatedUnspecified, any, true. |
| `videoCategoryId` | `query` | no | `string` | Filter on videos in a specific category. |
| `videoType` | `query` | no | `string` | Filter on videos of a specific type. Allowed: videoTypeUnspecified, any, movie, episode. |
| `eventType` | `query` | no | `string` | Filter on the livestream status of the videos. Allowed: none, upcoming, live, completed. |
| `location` | `query` | no | `string` | Filter on location of the video |
| `locationRadius` | `query` | no | `string` | Filter on distance from the location (specified above). |
| `channelId` | `query` | no | `string` | Filter on resources belonging to this channelId. |
| `publishedAfter` | `query` | no | `string` | Filter on resources published after this date. |
| `publishedBefore` | `query` | no | `string` | Filter on resources published before this date. |
| `topicId` | `query` | no | `string` | Restrict results to a particular topic. |
| `videoPaidProductPlacement` | `query` | no | `string` | Allowed: videoPaidProductPlacementUnspecified, any, true. |
| `forContentOwner` | `query` | no | `boolean` | Search owned by a content owner. |
| `regionCode` | `query` | no | `string` | Display the content as seen by viewers in this country. |
| `channelType` | `query` | no | `string` | Add a filter on the channel search. Allowed: channelTypeUnspecified, any, show. |
| `safeSearch` | `query` | no | `string` | Indicates whether the search results should include restricted content as well as standard content. Allowed: safeSearchSettingUnspecified, none, moderate, strict. |
| `forMine` | `query` | no | `boolean` | Search for the private videos of the authenticated user. |
| `forDeveloper` | `query` | no | `boolean` | Restrict the search to only retrieve videos uploaded using the project id of the authenticated user. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/search" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.subscriptions.delete`

- Resource: `subscriptions`
- HTTP: `DELETE /youtube/v3/subscriptions`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/subscriptions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.subscriptions.insert`

- Resource: `subscriptions`
- HTTP: `POST /youtube/v3/subscriptions`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Subscription`
- Response schema: `Subscription`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/subscriptions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.subscriptions.list`

- Resource: `subscriptions`
- HTTP: `GET /youtube/v3/subscriptions`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `SubscriptionListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more subscription resource properties that the API response will include. If the parameter identifies a property that contains child properties, the chil... |
| `id` | `query` | no | `string` | Return the subscriptions with the given IDs for Stubby or Apiary. |
| `mine` | `query` | no | `boolean` | Flag for returning the subscriptions of the authenticated user. |
| `channelId` | `query` | no | `string` | Return the subscriptions of the given channel owner. |
| `forChannelId` | `query` | no | `string` | Return the subscriptions to the subset of these channels that the authenticated user is subscribed to. |
| `order` | `query` | no | `string` | The order of the returned subscriptions Allowed: subscriptionOrderUnspecified, relevance, unread, alphabetical. |
| `mySubscribers` | `query` | no | `boolean` | Return the subscribers of the given channel owner. |
| `myRecentSubscribers` | `query` | no | `boolean` |  |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/subscriptions" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.superChatEvents.list`

- Resource: `superChatEvents`
- HTTP: `GET /youtube/v3/superChatEvents`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `SuperChatEventListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the superChatEvent resource parts that the API response will include. This parameter is currently not supported. |
| `hl` | `query` | no | `string` | Return rendered funding amounts in specified language. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/superChatEvents" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.tests.insert`

- Resource: `tests`
- HTTP: `POST /youtube/v3/tests`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `TestItem`
- Response schema: `TestItem`

POST method.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` |  |
| `externalChannelId` | `query` | no | `string` |  |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/tests" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.thirdPartyLinks.delete`

- Resource: `thirdPartyLinks`
- HTTP: `DELETE /youtube/v3/thirdPartyLinks`
- Required parameters: `linkingToken`, `type`
- OAuth scopes: not declared
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `linkingToken` | `query` | yes | `string` | Delete the partner links with the given linking token. |
| `type` | `query` | yes | `string` | Type of the link to be deleted. Allowed: linkUnspecified, channelToStoreLink. |
| `externalChannelId` | `query` | no | `string` | Channel ID to which changes should be applied, for delegation. |
| `part` | `query` | no | `string` | Do not use. Required for compatibility. |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/thirdPartyLinks" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "linkingToken=${LINKING_TOKEN}" \
  --url-query "type=linkUnspecified"
```

### `youtube.thirdPartyLinks.insert`

- Resource: `thirdPartyLinks`
- HTTP: `POST /youtube/v3/thirdPartyLinks`
- Required parameters: `part`
- OAuth scopes: not declared
- Request schema: `ThirdPartyLink`
- Response schema: `ThirdPartyLink`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the thirdPartyLink resource parts that the API request and response will include. Supported values are linkingToken, status, and snippet. |
| `externalChannelId` | `query` | no | `string` | Channel ID to which changes should be applied, for delegation. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/thirdPartyLinks" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.thirdPartyLinks.list`

- Resource: `thirdPartyLinks`
- HTTP: `GET /youtube/v3/thirdPartyLinks`
- Required parameters: `part`
- OAuth scopes: not declared
- Request schema: `none`
- Response schema: `ThirdPartyLinkListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the thirdPartyLink resource parts that the API response will include. Supported values are linkingToken, status, and snippet. |
| `linkingToken` | `query` | no | `string` | Get a third party link with the given linking token. |
| `type` | `query` | no | `string` | Get a third party link of the given type. Allowed: linkUnspecified, channelToStoreLink. |
| `externalChannelId` | `query` | no | `string` | Channel ID to which changes should be applied, for delegation. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/thirdPartyLinks" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.thirdPartyLinks.update`

- Resource: `thirdPartyLinks`
- HTTP: `PUT /youtube/v3/thirdPartyLinks`
- Required parameters: `part`
- OAuth scopes: not declared
- Request schema: `ThirdPartyLink`
- Response schema: `ThirdPartyLink`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the thirdPartyLink resource parts that the API request and response will include. Supported values are linkingToken, status, and snippet. |
| `externalChannelId` | `query` | no | `string` | Channel ID to which changes should be applied, for delegation. |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/thirdPartyLinks" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.thumbnails.set`

- Resource: `thumbnails`
- HTTP: `POST /youtube/v3/thumbnails/set`
- Required parameters: `videoId`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.upload`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `ThumbnailSetResponse`

As this is not an insert in a strict sense (it supports uploading/setting of a thumbnail for multiple videos, which doesn't result in creation of a single resource), I use a custom verb here.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `videoId` | `query` | yes | `string` | Returns the Thumbnail with the given video IDs for Stubby or Apiary. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/thumbnails/set" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "videoId=${VIDEO_ID}"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/thumbnails/set" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "videoId=${VIDEO_ID}"
```

### `youtube.videoAbuseReportReasons.list`

- Resource: `videoAbuseReportReasons`
- HTTP: `GET /youtube/v3/videoAbuseReportReasons`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `VideoAbuseReportReasonListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the videoCategory resource parts that the API response will include. Supported values are id and snippet. |
| `hl` | `query` | no | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videoAbuseReportReasons" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.videoCategories.list`

- Resource: `videoCategories`
- HTTP: `GET /youtube/v3/videoCategories`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `VideoCategoryListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies the videoCategory resource properties that the API response will include. Set the parameter value to snippet. |
| `id` | `query` | no | `string` | Returns the video categories with the given IDs for Stubby or Apiary. |
| `regionCode` | `query` | no | `string` |  |
| `hl` | `query` | no | `string` |  |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videoCategories" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.videoTrainability.get`

- Resource: `videoTrainability`
- HTTP: `GET /youtube/v3/videoTrainability`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `VideoTrainability`

Returns the trainability status of a video.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | no | `string` | The ID of the video to retrieve. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videoTrainability" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtube.videos.delete`

- Resource: `videos`
- HTTP: `DELETE /youtube/v3/videos`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Deletes a resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.videos.getRating`

- Resource: `videos`
- HTTP: `GET /youtube/v3/videos/getRating`
- Required parameters: `id`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `VideoGetRatingResponse`

Retrieves the ratings that the authorized user gave to a list of specified videos.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos/getRating" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}"
```

### `youtube.videos.insert`

- Resource: `videos`
- HTTP: `POST /youtube/v3/videos`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.upload`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Video`
- Response schema: `Video`

Inserts a new resource into this collection.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. Note that not all parts contain... |
| `autoLevels` | `query` | no | `boolean` | Should auto-levels be applied to the upload. |
| `stabilize` | `query` | no | `boolean` | Should stabilize be applied to the upload. |
| `notifySubscribers` | `query` | no | `boolean` | Notify the channel subscribers about the new video. As default, the notification is enabled. |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |
| `onBehalfOfContentOwnerChannel` | `query` | no | `string` | This parameter can only be used in a properly authorized request. *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwnerChannel* parameter specifies the YouTube channe... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "part=snippet"
```

### `youtube.videos.list`

- Resource: `videos`
- HTTP: `GET /youtube/v3/videos`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `VideoListResponse`

Retrieves a list of resources, possibly filtered.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter specifies a comma-separated list of one or more video resource properties that the API response will include. If the parameter identifies a property that contains child properties, the child prope... |
| `id` | `query` | no | `string` | Return videos with the given ids. |
| `myRating` | `query` | no | `string` | Return videos liked/disliked by the authenticated user. Does not support RateType.RATED_TYPE_NONE. Allowed: none, like, dislike. |
| `chart` | `query` | no | `string` | Return the videos that are in the specified chart. Allowed: chartUnspecified, mostPopular. |
| `videoCategoryId` | `query` | no | `string` | Use chart that is specific to the specified video category |
| `regionCode` | `query` | no | `string` | Use a chart that is specific to the specified region |
| `maxWidth` | `query` | no | `integer` | Return the player with maximum height specified in |
| `maxHeight` | `query` | no | `integer` |  |
| `hl` | `query` | no | `string` | Stands for "host language". Specifies the localization language of the metadata to be filled into snippet.localized. The field is filled with the default metadata if there is no localization in the specified language.... |
| `locale` | `query` | no | `string` |  |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. *Note:* This parameter is supported for use in conjunction with the myRating and chart parameters, but it is... |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved. *Note:*... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.videos.rate`

- Resource: `videos`
- HTTP: `POST /youtube/v3/videos/rate`
- Required parameters: `id`, `rating`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Adds a like or dislike rating to a video or removes a rating from a video.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `id` | `query` | yes | `string` |  |
| `rating` | `query` | yes | `string` | Allowed: none, like, dislike. |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos/rate" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "id=${ID}" \
  --url-query "rating=none"
```

### `youtube.videos.reportAbuse`

- Resource: `videos`
- HTTP: `POST /youtube/v3/videos/reportAbuse`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `VideoAbuseReport`
- Response schema: `none`

Report abuse for a video.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos/reportAbuse" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json"
```

### `youtube.videos.update`

- Resource: `videos`
- HTTP: `PUT /youtube/v3/videos`
- Required parameters: `part`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `Video`
- Response schema: `Video`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | yes | `string` | The *part* parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include. Note that this method will overr... |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/videos" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

### `youtube.watermarks.set`

- Resource: `watermarks`
- HTTP: `POST /youtube/v3/watermarks/set`
- Required parameters: `channelId`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.upload`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `InvideoBranding`
- Response schema: `none`

Allows upload of watermark image and setting it for a channel.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `channelId` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/watermarks/set" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "channelId=${CHANNEL_ID}"
```

**MyBrandMetrics media upload curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/watermarks/set" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  -H "Content-Type: ${MIME_TYPE:-application/octet-stream}" \
  --data-binary "@${MEDIA_FILE}" \
  --url-query "channelId=${CHANNEL_ID}"
```

### `youtube.watermarks.unset`

- Resource: `watermarks`
- HTTP: `POST /youtube/v3/watermarks/unset`
- Required parameters: `channelId`
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtubepartner`
- Request schema: `none`
- Response schema: `none`

Allows removal of channel watermark.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `channelId` | `query` | yes | `string` |  |
| `onBehalfOfContentOwner` | `query` | no | `string` | *Note:* This parameter is intended exclusively for YouTube content partners. The *onBehalfOfContentOwner* parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on b... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/watermarks/unset" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "channelId=${CHANNEL_ID}"
```

### `youtube.youtube.v3.liveChat.messages.stream`

- Resource: `youtube.v3.liveChat.messages`
- HTTP: `GET /youtube/v3/liveChat/messages/stream`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.force-ssl`, `https://www.googleapis.com/auth/youtube.readonly`
- Request schema: `none`
- Response schema: `LiveChatMessageListResponse`

Allows a user to load live chat through a server-streamed RPC.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `liveChatId` | `query` | no | `string` | The id of the live chat for which comments should be returned. |
| `hl` | `query` | no | `string` | Specifies the localization language in which the system messages should be returned. |
| `profileImageSize` | `query` | no | `integer` | Specifies the size of the profile image that should be returned for each user. |
| `maxResults` | `query` | no | `integer` | The *maxResults* parameter specifies the maximum number of items that should be returned in the result set. Not used in the streaming RPC. |
| `pageToken` | `query` | no | `string` | The *pageToken* parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken property identify other pages that could be retrieved. |
| `part` | `query` | no | `string` | The *part* parameter specifies the liveChatComment resource parts that the API response will include. Supported values are id, snippet, and authorDetails. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/liveChat/messages/stream" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "part=snippet"
```

### `youtube.youtube.v3.updateCommentThreads`

- Resource: `youtube.v3`
- HTTP: `PUT /youtube/v3/commentThreads`
- Required parameters: none
- OAuth scopes: not declared
- Request schema: `CommentThread`
- Response schema: `CommentThread`

Updates an existing resource.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `part` | `query` | no | `string` | The *part* parameter specifies a comma-separated list of commentThread resource properties that the API response will include. You must at least include the snippet part in the parameter value since that part contains... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtube/commentThreads" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json" \
  --url-query "part=snippet"
```

## Schemas

| Schema | Type | Description |
| --- | --- | --- |
| `AbuseReport` | `object` |  |
| `AbuseType` | `object` |  |
| `AccessPolicy` | `object` | Rights management policy for YouTube resources. |
| `Activity` | `object` | An *activity* resource contains information about an action that a particular channel, or user, has taken on YouTube.The actions reported in activity feeds include rating a vide... |
| `ActivityContentDetails` | `object` | Details about the content of an activity: the video that was shared, the channel that was subscribed to, etc. |
| `ActivityContentDetailsBulletin` | `object` | Details about a channel bulletin post. |
| `ActivityContentDetailsChannelItem` | `object` | Details about a resource which was added to a channel. |
| `ActivityContentDetailsComment` | `object` | Information about a resource that received a comment. |
| `ActivityContentDetailsFavorite` | `object` | Information about a video that was marked as a favorite video. |
| `ActivityContentDetailsLike` | `object` | Information about a resource that received a positive (like) rating. |
| `ActivityContentDetailsPlaylistItem` | `object` | Information about a new playlist item. |
| `ActivityContentDetailsPromotedItem` | `object` | Details about a resource which is being promoted. |
| `ActivityContentDetailsRecommendation` | `object` | Information that identifies the recommended resource. |
| `ActivityContentDetailsSocial` | `object` | Details about a social network post. |
| `ActivityContentDetailsSubscription` | `object` | Information about a channel that a user subscribed to. |
| `ActivityContentDetailsUpload` | `object` | Information about the uploaded video. |
| `ActivityListResponse` | `object` |  |
| `ActivitySnippet` | `object` | Basic details about an activity, including title, description, thumbnails, activity type and group. Next ID: 12 |
| `Caption` | `object` | A *caption* resource represents a YouTube caption track. A caption track is associated with exactly one YouTube video. |
| `CaptionListResponse` | `object` |  |
| `CaptionSnippet` | `object` | Basic details about a caption track, such as its language and name. |
| `CdnSettings` | `object` | Brief description of the live stream cdn settings. |
| `Channel` | `object` | A *channel* resource contains information about a YouTube channel. |
| `ChannelAuditDetails` | `object` | The auditDetails object encapsulates channel data that is relevant for YouTube Partners during the audit process. |
| `ChannelBannerResource` | `object` | A channel banner returned as the response to a channel_banner.insert call. |
| `ChannelBrandingSettings` | `object` | Branding properties of a YouTube channel. |
| `ChannelContentDetails` | `object` | Details about the content of a channel. |
| `ChannelContentOwnerDetails` | `object` | The contentOwnerDetails object encapsulates channel data that is relevant for YouTube Partners linked with the channel. |
| `ChannelConversionPing` | `object` | Pings that the app shall fire (authenticated by biscotti cookie). Each ping has a context, in which the app must fire the ping, and a url identifying the ping. |
| `ChannelConversionPings` | `object` | The conversionPings object encapsulates information about conversion pings that need to be respected by the channel. |
| `ChannelListResponse` | `object` |  |
| `ChannelLocalization` | `object` | Channel localization setting |
| `ChannelProfileDetails` | `object` |  |
| `ChannelSection` | `object` |  |
| `ChannelSectionContentDetails` | `object` | Details about a channelsection, including playlists and channels. |
| `ChannelSectionListResponse` | `object` |  |
| `ChannelSectionLocalization` | `object` | ChannelSection localization setting |
| `ChannelSectionSnippet` | `object` | Basic details about a channel section, including title, style and position. |
| `ChannelSectionTargeting` | `object` | ChannelSection targeting setting. |
| `ChannelSettings` | `object` | Branding properties for the channel view. |
| `ChannelSnippet` | `object` | Basic details about a channel, including title, description and thumbnails. |
| `ChannelStatistics` | `object` | Statistics about a channel: number of subscribers, number of videos in the channel, etc. |
| `ChannelStatus` | `object` | JSON template for the status part of a channel. |
| `ChannelToStoreLinkDetails` | `object` | Information specific to a store on a merchandising platform linked to a YouTube channel. |
| `ChannelToStoreLinkDetailsBillingDetails` | `object` | Information specific to billing. |
| `ChannelToStoreLinkDetailsMerchantAffiliateProgramDetails` | `object` | Information specific to merchant affiliate program. |
| `ChannelTopicDetails` | `object` | Freebase topic information related to the channel. |
| `Comment` | `object` | A *comment* represents a single YouTube comment. |
| `CommentListResponse` | `object` |  |
| `CommentSnippet` | `object` | Basic details about a comment, such as its author and text. |
| `CommentSnippetAuthorChannelId` | `object` | Contains the id of the author's YouTube channel, if any. |
| `CommentThread` | `object` | A *comment thread* represents information that applies to a top level comment and all its replies. It can also include the top level comment itself and some of the replies. |
| `CommentThreadListResponse` | `object` |  |
| `CommentThreadReplies` | `object` | Comments written in (direct or indirect) reply to the top level comment. |
| `CommentThreadSnippet` | `object` | Basic details about a comment thread. |
| `ContentRating` | `object` | Ratings schemes. The country-specific ratings are mostly for movies and shows. LINT.IfChange |
| `Cuepoint` | `object` | Note that there may be a 5-second end-point resolution issue. For instance, if a cuepoint comes in for 22:03:27, we may stuff the cuepoint into 22:03:25 or 22:03:30, depending.... |
| `CuepointSchedule` | `object` | Schedule to insert cuepoints into a broadcast by ads automator. |
| `Entity` | `object` |  |
| `GeoPoint` | `object` | Geographical coordinates of a point, in WGS84. |
| `I18nLanguage` | `object` | An *i18nLanguage* resource identifies a UI language currently supported by YouTube. |
| `I18nLanguageListResponse` | `object` |  |
| `I18nLanguageSnippet` | `object` | Basic details about an i18n language, such as language code and human-readable name. |
| `I18nRegion` | `object` | A *i18nRegion* resource identifies a region where YouTube is available. |
| `I18nRegionListResponse` | `object` |  |
| `I18nRegionSnippet` | `object` | Basic details about an i18n region, such as region code and human-readable name. |
| `ImageSettings` | `object` | Branding properties for images associated with the channel. |
| `IngestionInfo` | `object` | Describes information necessary for ingesting an RTMP, HTTP, or SRT stream. |
| `InvideoBranding` | `object` | Describes an invideo branding. |
| `InvideoPosition` | `object` | Describes the spatial position of a visual widget inside a video. It is a union of various position types, out of which only will be set one. |
| `InvideoTiming` | `object` | Describes a temporal position of a visual widget inside a video. |
| `LanguageTag` | `object` |  |
| `LevelDetails` | `object` |  |
| `LiveBroadcast` | `object` | A *liveBroadcast* resource represents an event that will be streamed, via live video, on YouTube. |
| `LiveBroadcastContentDetails` | `object` | Detailed settings of a broadcast. |
| `LiveBroadcastListResponse` | `object` |  |
| `LiveBroadcastMonetizationDetails` | `object` | Monetization settings of a broadcast. |
| `LiveBroadcastSnippet` | `object` | Basic broadcast information. |
| `LiveBroadcastStatistics` | `object` | Statistics about the live broadcast. These represent a snapshot of the values at the time of the request. Statistics are only returned for live broadcasts. |
| `LiveBroadcastStatus` | `object` | Live broadcast state. |
| `LiveChatBan` | `object` | A `__liveChatBan__` resource represents a ban for a YouTube live chat. |
| `LiveChatBanSnippet` | `object` |  |
| `LiveChatFanFundingEventDetails` | `object` |  |
| `LiveChatGiftDetails` | `object` | Details about the gift event, this is only set if the type is 'giftEvent'. |
| `LiveChatGiftMembershipReceivedDetails` | `object` |  |
| `LiveChatMemberMilestoneChatDetails` | `object` |  |
| `LiveChatMembershipGiftingDetails` | `object` |  |
| `LiveChatMessage` | `object` | A *liveChatMessage* resource represents a chat message in a YouTube Live Chat. |
| `LiveChatMessageAuthorDetails` | `object` |  |
| `LiveChatMessageDeletedDetails` | `object` |  |
| `LiveChatMessageListResponse` | `object` |  |
| `LiveChatMessageRetractedDetails` | `object` |  |
| `LiveChatMessageSnippet` | `object` | Next ID: 35 |
| `LiveChatModerator` | `object` | A *liveChatModerator* resource represents a moderator for a YouTube live chat. A chat moderator has the ability to ban/unban users from a chat, remove message, etc. |
| `LiveChatModeratorListResponse` | `object` |  |
| `LiveChatModeratorSnippet` | `object` |  |
| `LiveChatNewSponsorDetails` | `object` |  |
| `LiveChatPollDetails` | `object` |  |
| `LiveChatPollDetailsPollMetadata` | `object` |  |
| `LiveChatPollDetailsPollMetadataPollOption` | `object` |  |
| `LiveChatSuperChatDetails` | `object` |  |
| `LiveChatSuperStickerDetails` | `object` |  |
| `LiveChatTextMessageDetails` | `object` |  |
| `LiveChatUserBannedMessageDetails` | `object` |  |
| `LiveStream` | `object` | A live stream describes a live ingestion point. |
| `LiveStreamConfigurationIssue` | `object` |  |
| `LiveStreamContentDetails` | `object` | Detailed settings of a stream. |
| `LiveStreamHealthStatus` | `object` |  |
| `LiveStreamListResponse` | `object` |  |
| `LiveStreamSnippet` | `object` |  |
| `LiveStreamStatus` | `object` | Brief description of the live stream status. |
| `LocalizedProperty` | `object` |  |
| `LocalizedString` | `object` |  |
| `Member` | `object` | A *member* resource represents a member for a YouTube channel. A member provides recurring monetary support to a creator and receives special benefits. |
| `MemberListResponse` | `object` |  |
| `MemberSnippet` | `object` |  |
| `MembershipsDetails` | `object` |  |
| `MembershipsDuration` | `object` |  |
| `MembershipsDurationAtLevel` | `object` |  |
| `MembershipsLevel` | `object` | A *membershipsLevel* resource represents an offer made by YouTube creators for their fans. Users can become members of the channel by joining one of the available levels. They w... |
| `MembershipsLevelListResponse` | `object` |  |
| `MembershipsLevelSnippet` | `object` |  |
| `MonitorStreamInfo` | `object` | Settings and Info of the monitor stream |
| `PageInfo` | `object` | Paging details for lists of resources, including total number of items available and number of resources returned in a single page. |
| `Playlist` | `object` | A *playlist* resource represents a YouTube playlist. A playlist is a collection of videos that can be viewed sequentially and shared with other users. A playlist can contain up... |
| `PlaylistContentDetails` | `object` |  |
| `PlaylistImage` | `object` |  |
| `PlaylistImageListResponse` | `object` |  |
| `PlaylistImageSnippet` | `object` | A *playlistImage* resource identifies another resource, such as a image, that is associated with a playlist. In addition, the playlistImage resource contains details about the i... |
| `PlaylistItem` | `object` | A *playlistItem* resource identifies another resource, such as a video, that is included in a playlist. In addition, the playlistItem resource contains details about the include... |
| `PlaylistItemContentDetails` | `object` |  |
| `PlaylistItemListResponse` | `object` |  |
| `PlaylistItemSnippet` | `object` | Basic details about a playlist, including title, description and thumbnails. Basic details of a YouTube Playlist item provided by the author. Next ID: 15 |
| `PlaylistItemStatus` | `object` | Information about the playlist item's privacy status. |
| `PlaylistListResponse` | `object` |  |
| `PlaylistLocalization` | `object` | Playlist localization setting |
| `PlaylistPlayer` | `object` |  |
| `PlaylistSnippet` | `object` | Basic details about a playlist, including title, description and thumbnails. |
| `PlaylistStatus` | `object` |  |
| `PropertyValue` | `object` | A pair Property / Value. |
| `RelatedEntity` | `object` |  |
| `ResourceId` | `object` | A resource id is a generic reference that points to another YouTube resource. |
| `SearchListResponse` | `object` |  |
| `SearchResult` | `object` | A search result contains information about a YouTube video, channel, or playlist that matches the search parameters specified in an API request. While a search result points to... |
| `SearchResultSnippet` | `object` | Basic details about a search result, including title, description and thumbnails of the item referenced by the search result. |
| `Subscription` | `object` | A *subscription* resource contains information about a YouTube user subscription. A subscription notifies a user when new videos are added to a channel or when another user take... |
| `SubscriptionContentDetails` | `object` | Details about the content to witch a subscription refers. |
| `SubscriptionListResponse` | `object` |  |
| `SubscriptionSnippet` | `object` | Basic details about a subscription, including title, description and thumbnails of the subscribed item. |
| `SubscriptionSubscriberSnippet` | `object` | Basic details about a subscription's subscriber including title, description, channel ID and thumbnails. |
| `SuperChatEvent` | `object` | A `__superChatEvent__` resource represents a Super Chat purchase on a YouTube channel. |
| `SuperChatEventListResponse` | `object` |  |
| `SuperChatEventSnippet` | `object` |  |
| `SuperStickerMetadata` | `object` |  |
| `TestItem` | `object` |  |
| `TestItemTestItemSnippet` | `object` |  |
| `ThirdPartyLink` | `object` | A *third party account link* resource represents a link between a YouTube account or a channel and an account on a third-party service. |
| `ThirdPartyLinkListResponse` | `object` |  |
| `ThirdPartyLinkSnippet` | `object` | Basic information about a third party account link, including its type and type-specific information. |
| `ThirdPartyLinkStatus` | `object` | The third-party link status object contains information about the status of the link. |
| `Thumbnail` | `object` | A thumbnail is an image representing a YouTube resource. |
| `ThumbnailDetails` | `object` | Internal representation of thumbnails for a YouTube resource. |
| `ThumbnailSetResponse` | `object` |  |
| `TokenPagination` | `object` | Stub token pagination template to suppress results. |
| `Video` | `object` | A *video* resource represents a YouTube video. |
| `VideoAbuseReport` | `object` |  |
| `VideoAbuseReportReason` | `object` | A `__videoAbuseReportReason__` resource identifies a reason that a video could be reported as abusive. Video abuse report reasons are used with `video.ReportAbuse`. |
| `VideoAbuseReportReasonListResponse` | `object` |  |
| `VideoAbuseReportReasonSnippet` | `object` | Basic details about a video category, such as its localized title. |
| `VideoAbuseReportSecondaryReason` | `object` |  |
| `VideoAgeGating` | `object` |  |
| `VideoCategory` | `object` | A *videoCategory* resource identifies a category that has been or could be associated with uploaded videos. |
| `VideoCategoryListResponse` | `object` |  |
| `VideoCategorySnippet` | `object` | Basic details about a video category, such as its localized title. |
| `VideoContentDetails` | `object` | Details about the content of a YouTube Video. |
| `VideoContentDetailsRegionRestriction` | `object` | DEPRECATED Region restriction of the video. |
| `VideoFileDetails` | `object` | Describes original video file properties, including technical details about audio and video streams, but also metadata information like content length, digitization time, or geo... |
| `VideoFileDetailsAudioStream` | `object` | Information about an audio stream. |
| `VideoFileDetailsVideoStream` | `object` | Information about a video stream. |
| `VideoGetRatingResponse` | `object` |  |
| `VideoListResponse` | `object` |  |
| `VideoLiveStreamingDetails` | `object` | Details about the live streaming metadata. |
| `VideoLocalization` | `object` | Localized versions of certain video properties (e.g. title). |
| `VideoMonetizationDetails` | `object` | Details about monetization of a YouTube Video. |
| `VideoPaidProductPlacementDetails` | `object` | Details about paid content, such as paid product placement, sponsorships or endorsement, contained in a YouTube video and a method to inform viewers of paid promotion. This data... |
| `VideoPlayer` | `object` | Player to be used for a video playback. |
| `VideoProcessingDetails` | `object` | Describes processing status and progress and availability of some other Video resource parts. |
| `VideoProcessingDetailsProcessingProgress` | `object` | Video processing progress and completion time estimate. |
| `VideoProjectDetails` | `object` | DEPRECATED. b/157517979: This part was never populated after it was added. However, it sees non-zero traffic because there is generated client code in the wild that refers to it... |
| `VideoRating` | `object` | Basic details about rating of a video. |
| `VideoRecordingDetails` | `object` | Recording information associated with the video. |
| `VideoSnippet` | `object` | Basic details about a video, including title, description, uploader, thumbnails and category. |
| `VideoStatistics` | `object` | Statistics about the video, such as the number of times the video was viewed or liked. |
| `VideoStatus` | `object` | Basic details about a video category, such as its localized title. Next Id: 19 |
| `VideoSuggestions` | `object` | Specifies suggestions on how to improve video content, including encoding hints, tag suggestions, and editor suggestions. |
| `VideoSuggestionsTagSuggestion` | `object` | A single tag suggestion with its relevance information. |
| `VideoTopicDetails` | `object` | Freebase topic information related to the video. |
| `VideoTrainability` | `object` | Specifies who is allowed to train on the video. |
| `WatchSettings` | `object` | Branding properties for the watch. All deprecated. |
