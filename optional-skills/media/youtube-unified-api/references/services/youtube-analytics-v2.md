# YouTube Analytics API v2

Discovery source: `https://youtubeanalytics.googleapis.com/$discovery/rest?version=v2`
Public API base URL: `https://api.mybrandmetrics.com`
Methods: 8
Schemas: 12

## Method Index

### groupItems
- `youtubeAnalytics.groupItems.delete` - `DELETE /v2/groupItems`
- `youtubeAnalytics.groupItems.insert` - `POST /v2/groupItems`
- `youtubeAnalytics.groupItems.list` - `GET /v2/groupItems`

### groups
- `youtubeAnalytics.groups.delete` - `DELETE /v2/groups`
- `youtubeAnalytics.groups.insert` - `POST /v2/groups`
- `youtubeAnalytics.groups.list` - `GET /v2/groups`
- `youtubeAnalytics.groups.update` - `PUT /v2/groups`

### reports
- `youtubeAnalytics.reports.query` - `GET /v2/reports`

## Method Reference

### `youtubeAnalytics.groupItems.delete`

- Resource: `groupItems`
- HTTP: `DELETE /v2/groupItems`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `EmptyResponse`

Removes an item from a group.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |
| `id` | `query` | no | `string` | The `id` parameter specifies the YouTube group item ID of the group item that is being deleted. |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groupItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubeAnalytics.groupItems.insert`

- Resource: `groupItems`
- HTTP: `POST /v2/groupItems`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `GroupItem`
- Response schema: `GroupItem`

Creates a group item.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groupItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json"
```

### `youtubeAnalytics.groupItems.list`

- Resource: `groupItems`
- HTTP: `GET /v2/groupItems`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `ListGroupItemsResponse`

Returns a collection of group items that match the API request parameters.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |
| `groupId` | `query` | no | `string` | The `groupId` parameter specifies the unique ID of the group for which you want to retrieve group items. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groupItems" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubeAnalytics.groups.delete`

- Resource: `groups`
- HTTP: `DELETE /v2/groups`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `EmptyResponse`

Deletes a group.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |
| `id` | `query` | no | `string` | The `id` parameter specifies the YouTube group ID of the group that is being deleted. |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groups" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubeAnalytics.groups.insert`

- Resource: `groups`
- HTTP: `POST /v2/groups`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `Group`
- Response schema: `Group`

Creates a group.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groups" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json"
```

### `youtubeAnalytics.groups.list`

- Resource: `groups`
- HTTP: `GET /v2/groups`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `ListGroupsResponse`

Returns a collection of groups that match the API request parameters. For example, you can retrieve all groups that the authenticated user owns, or you can retrieve one or more groups by their unique IDs.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |
| `id` | `query` | no | `string` | The `id` parameter specifies a comma-separated list of the YouTube group ID(s) for the resource(s) that are being retrieved. Each group must be owned by the authenticated user. In a `group` resource, the `id` property... |
| `mine` | `query` | no | `boolean` | This parameter can only be used in a properly authorized request. Set this parameter's value to true to retrieve all groups owned by the authenticated user. |
| `pageToken` | `query` | no | `string` | The `pageToken` parameter identifies a specific page in the result set that should be returned. In an API response, the `nextPageToken` property identifies the next page that can be retrieved. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groups" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubeAnalytics.groups.update`

- Resource: `groups`
- HTTP: `PUT /v2/groups`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `Group`
- Response schema: `Group`

Modifies a group. For example, you could change a group's title.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | This parameter can only be used in a properly authorized request. **Note:** This parameter is intended exclusively for YouTube content partners that own and manage many different YouTube channels. The `onBehalfOfConte... |

**MyBrandMetrics curl**

```bash
curl -sS -X PUT "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/groups" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json"
```

### `youtubeAnalytics.reports.query`

- Resource: `reports`
- HTTP: `GET /v2/reports`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/youtube`, `https://www.googleapis.com/auth/youtube.readonly`, `https://www.googleapis.com/auth/youtubepartner`, `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `QueryResponse`

Retrieve your YouTube Analytics reports.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `ids` | `query` | no | `string` | Identifies the YouTube channel or content owner for which you are retrieving YouTube Analytics data. - To request data for a YouTube user, set the `ids` parameter value to `channel==CHANNEL_ID`, where `CHANNEL_ID` spe... |
| `startDate` | `query` | no | `string` | The start date for fetching YouTube Analytics data. The value should be in `YYYY-MM-DD` format. required: true, pattern: "[0-9]{4}-[0-9]{2}-[0-9]{2} |
| `endDate` | `query` | no | `string` | The end date for fetching YouTube Analytics data. The value should be in `YYYY-MM-DD` format. required: true, pattern: [0-9]{4}-[0-9]{2}-[0-9]{2} |
| `metrics` | `query` | no | `string` | A comma-separated list of YouTube Analytics metrics, such as `views` or `likes,dislikes`. See the [Available Reports](/youtube/analytics/v2/available_reports) document for a list of the reports that you can retrieve a... |
| `dimensions` | `query` | no | `string` | A comma-separated list of YouTube Analytics dimensions, such as `views` or `ageGroup,gender`. See the [Available Reports](/youtube/analytics/v2/available_reports) document for a list of the reports that you can retrie... |
| `currency` | `query` | no | `string` | The currency to which financial metrics should be converted. The default is US Dollar (USD). If the result contains no financial metrics, this flag will be ignored. Responds with an error if the specified currency is... |
| `filters` | `query` | no | `string` | A list of filters that should be applied when retrieving YouTube Analytics data. The [Available Reports](/youtube/analytics/v2/available_reports) document identifies the dimensions that can be used to filter each repo... |
| `includeHistoricalChannelData` | `query` | no | `boolean` | If set to true historical data (i.e. channel data from before the linking of the channel to the content owner) will be retrieved.", |
| `maxResults` | `query` | no | `integer` | The maximum number of rows to include in the response.", minValue: 1 |
| `sort` | `query` | no | `string` | A comma-separated list of dimensions or metrics that determine the sort order for YouTube Analytics data. By default the sort order is ascending. The '`-`' prefix causes descending sort order.", pattern: [-0-9a-zA-Z,]+ |
| `startIndex` | `query` | no | `integer` | An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter (one-based, inclusive).", minValue: 1 |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeAnalytics/reports" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --url-query "ids=channel==MINE" \
  --url-query "startDate=2026-01-01" \
  --url-query "endDate=2026-01-31" \
  --url-query "metrics=views,estimatedMinutesWatched"
```

## Schemas

| Schema | Type | Description |
| --- | --- | --- |
| `EmptyResponse` | `object` | Empty response. |
| `ErrorProto` | `object` | Describes one specific error. |
| `Errors` | `object` | Request Error information. The presence of an error field signals that the operation has failed. |
| `Group` | `object` | A group. |
| `GroupContentDetails` | `object` | A group's content details. |
| `GroupItem` | `object` | A group item. |
| `GroupItemResource` | `object` |  |
| `GroupSnippet` | `object` | A group snippet. |
| `ListGroupItemsResponse` | `object` | Response message for GroupsService.ListGroupItems. |
| `ListGroupsResponse` | `object` | Response message for GroupsService.ListGroups. |
| `QueryResponse` | `object` | Response message for TargetedQueriesService.Query. |
| `ResultTableColumnHeader` | `object` | The description of a column of the result table. |
