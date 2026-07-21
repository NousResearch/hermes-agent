# YouTube Reporting API v1

Discovery source: `https://youtubereporting.googleapis.com/$discovery/rest?version=v1`
Public API base URL: `https://api.mybrandmetrics.com`
Methods: 8
Schemas: 18

## Method Index

### jobs
- `youtubereporting.jobs.create` - `POST /v1/jobs`
- `youtubereporting.jobs.delete` - `DELETE /v1/jobs/{jobId}`
- `youtubereporting.jobs.get` - `GET /v1/jobs/{jobId}`
- `youtubereporting.jobs.list` - `GET /v1/jobs`

### jobs.reports
- `youtubereporting.jobs.reports.get` - `GET /v1/jobs/{jobId}/reports/{reportId}`
- `youtubereporting.jobs.reports.list` - `GET /v1/jobs/{jobId}/reports`

### media
- `youtubereporting.media.download` - `GET /v1/media/{+resourceName}`

### reportTypes
- `youtubereporting.reportTypes.list` - `GET /v1/reportTypes`

## Method Reference

### `youtubereporting.jobs.create`

- Resource: `jobs`
- HTTP: `POST /v1/jobs`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `Job`
- Response schema: `Job`

Creates a job and returns it.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |

**MyBrandMetrics curl**

```bash
curl -sS -X POST "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/jobs" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json" \
  --json "@body.json"
```

### `youtubereporting.jobs.delete`

- Resource: `jobs`
- HTTP: `DELETE /v1/jobs/{jobId}`
- Required parameters: `jobId`
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `Empty`

Deletes a job.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `jobId` | `path` | yes | `string` | The ID of the job to delete. |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |

**MyBrandMetrics curl**

```bash
curl -sS -X DELETE "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/jobs/${JOB_ID}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubereporting.jobs.get`

- Resource: `jobs`
- HTTP: `GET /v1/jobs/{jobId}`
- Required parameters: `jobId`
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `Job`

Gets a job.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `jobId` | `path` | yes | `string` | The ID of the job to retrieve. |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/jobs/${JOB_ID}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubereporting.jobs.list`

- Resource: `jobs`
- HTTP: `GET /v1/jobs`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `ListJobsResponse`

Lists jobs.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |
| `pageSize` | `query` | no | `integer` | Requested page size. Server may return fewer jobs than requested. If unspecified, server will pick an appropriate default. |
| `pageToken` | `query` | no | `string` | A token identifying a page of results the server should return. Typically, this is the value of ListReportTypesResponse.next_page_token returned in response to the previous call to the `ListJobs` method. |
| `includeSystemManaged` | `query` | no | `boolean` | If set to true, also system-managed jobs will be returned; otherwise only user-created jobs will be returned. System-managed jobs can neither be modified nor deleted. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/jobs" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubereporting.jobs.reports.get`

- Resource: `jobs.reports`
- HTTP: `GET /v1/jobs/{jobId}/reports/{reportId}`
- Required parameters: `jobId`, `reportId`
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `Report`

Gets the metadata of a specific report.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `jobId` | `path` | yes | `string` | The ID of the job. |
| `reportId` | `path` | yes | `string` | The ID of the report to retrieve. |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/jobs/${JOB_ID}/reports/${REPORT_ID}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubereporting.jobs.reports.list`

- Resource: `jobs.reports`
- HTTP: `GET /v1/jobs/{jobId}/reports`
- Required parameters: `jobId`
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `ListReportsResponse`

Lists reports created by a specific job. Returns NOT_FOUND if the job does not exist.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `jobId` | `path` | yes | `string` | The ID of the job. |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |
| `pageSize` | `query` | no | `integer` | Requested page size. Server may return fewer report types than requested. If unspecified, server will pick an appropriate default. |
| `pageToken` | `query` | no | `string` | A token identifying a page of results the server should return. Typically, this is the value of ListReportsResponse.next_page_token returned in response to the previous call to the `ListReports` method. |
| `createdAfter` | `query` | no | `string` | If set, only reports created after the specified date/time are returned. |
| `startTimeAtOrAfter` | `query` | no | `string` | If set, only reports whose start time is greater than or equal the specified date/time are returned. |
| `startTimeBefore` | `query` | no | `string` | If set, only reports whose start time is smaller than the specified date/time are returned. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/jobs/${JOB_ID}/reports" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubereporting.media.download`

- Resource: `media`
- HTTP: `GET /v1/media/{+resourceName}`
- Required parameters: `resourceName`
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `GdataMedia`
- Supports media download: yes

Method for media download. Download is supported on the URI `/v1/media/{+name}?alt=media`.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `resourceName` | `path` | yes | `string` | Name of the media that is being downloaded. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/media/${RESOURCE_NAME}" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

### `youtubereporting.reportTypes.list`

- Resource: `reportTypes`
- HTTP: `GET /v1/reportTypes`
- Required parameters: none
- OAuth scopes: `https://www.googleapis.com/auth/yt-analytics-monetary.readonly`, `https://www.googleapis.com/auth/yt-analytics.readonly`
- Request schema: `none`
- Response schema: `ListReportTypesResponse`

Lists report types.

**Parameters**

| Name | Location | Required | Type | Notes |
| --- | --- | --- | --- | --- |
| `onBehalfOfContentOwner` | `query` | no | `string` | The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel). |
| `pageSize` | `query` | no | `integer` | Requested page size. Server may return fewer report types than requested. If unspecified, server will pick an appropriate default. |
| `pageToken` | `query` | no | `string` | A token identifying a page of results the server should return. Typically, this is the value of ListReportTypesResponse.next_page_token returned in response to the previous call to the `ListReportTypes` method. |
| `includeSystemManaged` | `query` | no | `boolean` | If set to true, also system-managed report types will be returned; otherwise only the report types that can be used to create new reporting jobs will be returned. |

**MyBrandMetrics curl**

```bash
curl -sS -X GET "${MYBRANDMETRICS_API_BASE_URL:-https://api.mybrandmetrics.com}/discovery/youtubeReporting/reportTypes" \
  -H "X-API-Key: $MYBRANDMETRICS_API_KEY" \
  -H "Accept: application/json"
```

## Schemas

| Schema | Type | Description |
| --- | --- | --- |
| `Empty` | `object` | A generic empty message that you can re-use to avoid defining duplicated empty messages in your APIs. A typical example is to use it as the request or the response type of an AP... |
| `GdataBlobstore2Info` | `object` | gdata |
| `GdataCompositeMedia` | `object` | gdata |
| `GdataContentTypeInfo` | `object` | gdata |
| `GdataDiffChecksumsResponse` | `object` | gdata |
| `GdataDiffDownloadResponse` | `object` | gdata |
| `GdataDiffUploadRequest` | `object` | gdata |
| `GdataDiffUploadResponse` | `object` | gdata |
| `GdataDiffVersionResponse` | `object` | gdata |
| `GdataDownloadParameters` | `object` | gdata |
| `GdataMedia` | `object` | gdata |
| `GdataObjectId` | `object` | gdata |
| `Job` | `object` | A job creating reports of a specific type. |
| `ListJobsResponse` | `object` | Response message for ReportingService.ListJobs. |
| `ListReportTypesResponse` | `object` | Response message for ReportingService.ListReportTypes. |
| `ListReportsResponse` | `object` | Response message for ReportingService.ListReports. |
| `Report` | `object` | A report's metadata including the URL from which the report itself can be downloaded. |
| `ReportType` | `object` | A report type. |
