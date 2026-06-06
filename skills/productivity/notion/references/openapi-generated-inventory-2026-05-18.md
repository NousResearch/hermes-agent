# Generated Notion OpenAPI Inventory — 2026-05-18

This file is a generated static inventory from the official public Notion OpenAPI spec. It complements the prose references; do not hand-edit endpoint counts without regenerating from the spec.

## Retrieval receipt

- Retrieved: `2026-05-18T04:46:46Z`
- URL: `https://developers.notion.com/openapi.json`
- SHA-256: `c781691e6316b679648c83ff8f18a9dd70943fa24dbe35be69d19ff1cb274174`
- Bytes: `786086`
- OpenAPI version: `3.1.0`
- API title: `Notion API`
- API info version: `1.0.0`
- Servers: `https://api.notion.com`
- Paths: `32`
- Operations: `47`
- Component schemas: `505`
- Notion-Version enum in spec: `2026-03-11`

Regenerate/check with `skills/productivity/notion/scripts/notion_api_surface_snapshot.py`.

## Operation inventory

### Blocks

- `DELETE /v1/blocks/{block_id}` — operationId `delete-a-block`; auth: Bearer; Delete a block
- `GET /v1/blocks/{block_id}` — operationId `retrieve-a-block`; auth: Bearer; Retrieve a block
- `PATCH /v1/blocks/{block_id}` — operationId `update-a-block`; body: application/json; auth: Bearer; Update a block
- `GET /v1/blocks/{block_id}/children` — operationId `get-block-children`; pagination/cursor surface; auth: Bearer; Retrieve block children
- `PATCH /v1/blocks/{block_id}/children` — operationId `patch-block-children`; body: application/json; auth: Bearer; Append block children

### Comments

- `GET /v1/comments` — operationId `list-comments`; pagination/cursor surface; auth: Bearer; List comments
- `POST /v1/comments` — operationId `create-a-comment`; body: application/json; auth: Bearer; Create a comment
- `DELETE /v1/comments/{comment_id}` — operationId `delete-a-comment`; auth: Bearer; Delete a comment
- `GET /v1/comments/{comment_id}` — operationId `retrieve-comment`; auth: Bearer; Retrieve a comment
- `PATCH /v1/comments/{comment_id}` — operationId `update-a-comment`; body: application/json; auth: Bearer; Update a comment

### Custom emojis

- `GET /v1/custom_emojis` — operationId `list-custom-emojis`; pagination/cursor surface; auth: Bearer; List custom emojis

### Data sources

- `POST /v1/data_sources` — operationId `create-a-database`; body: application/json; auth: Bearer; Create a data source
- `GET /v1/data_sources/{data_source_id}` — operationId `retrieve-a-data-source`; auth: Bearer; Retrieve a data source
- `PATCH /v1/data_sources/{data_source_id}` — operationId `update-a-data-source`; body: application/json; auth: Bearer; Update a data source
- `POST /v1/data_sources/{data_source_id}/query` — operationId `post-database-query`; pagination/cursor surface; body: application/json; auth: Bearer; Query a data source
- `GET /v1/data_sources/{data_source_id}/templates` — operationId `list-data-source-templates`; pagination/cursor surface; auth: Bearer; List templates in a data source

### Databases

- `POST /v1/databases` — operationId `create-database`; body: application/json; auth: Bearer; Create a database
- `GET /v1/databases/{database_id}` — operationId `retrieve-database`; auth: Bearer; Retrieve a database
- `PATCH /v1/databases/{database_id}` — operationId `update-database`; body: application/json; auth: Bearer; Update a database

### File uploads

- `GET /v1/file_uploads` — operationId `list-file-uploads`; pagination/cursor surface; auth: Bearer; List file uploads
- `POST /v1/file_uploads` — operationId `create-file`; body: application/json; auth: Bearer; Create a file upload
- `GET /v1/file_uploads/{file_upload_id}` — operationId `retrieve-file-upload`; auth: Bearer; Retrieve a file upload
- `POST /v1/file_uploads/{file_upload_id}/complete` — operationId `complete-file-upload`; auth: Bearer; Complete a multi-part file upload
- `POST /v1/file_uploads/{file_upload_id}/send` — operationId `upload-file`; body: multipart/form-data; auth: Bearer; Upload a file

### Meeting notes

- `POST /v1/blocks/meeting_notes/query` — operationId `query-meeting-notes`; body: application/json; auth: Bearer; Query meeting notes

### OAuth

- `POST /v1/oauth/introspect` — operationId `introspect-token`; body: application/json; auth: Basic/client credentials; Introspect a token
- `POST /v1/oauth/revoke` — operationId `revoke-token`; body: application/json; auth: Basic/client credentials; Revoke a token
- `POST /v1/oauth/token` — operationId `create-a-token`; body: application/json; auth: Basic/client credentials; Exchange an authorization code for an access and refresh token

### Pages

- `POST /v1/pages` — operationId `post-page`; body: application/json; auth: Bearer; Create a page
- `GET /v1/pages/{page_id}` — operationId `retrieve-a-page`; auth: Bearer; Retrieve a page
- `PATCH /v1/pages/{page_id}` — operationId `patch-page`; body: application/json; auth: Bearer; Update page
- `GET /v1/pages/{page_id}/markdown` — operationId `retrieve-page-markdown`; auth: Bearer; Retrieve a page as markdown
- `PATCH /v1/pages/{page_id}/markdown` — operationId `update-page-markdown`; body: application/json; auth: Bearer; Update a page's content as markdown
- `POST /v1/pages/{page_id}/move` — operationId `move-page`; body: application/json; auth: Bearer; Move a page
- `GET /v1/pages/{page_id}/properties/{property_id}` — operationId `retrieve-a-page-property`; pagination/cursor surface; auth: Bearer; Retrieve a page property item

### Search

- `POST /v1/search` — operationId `post-search`; pagination/cursor surface; body: application/json; auth: Bearer; Search by title

### Users

- `GET /v1/users` — operationId `get-users`; pagination/cursor surface; auth: Bearer; List all users
- `GET /v1/users/me` — operationId `get-self`; auth: Bearer; Retrieve your token's bot user
- `GET /v1/users/{user_id}` — operationId `get-user`; auth: Bearer; Retrieve a user

### Views

- `GET /v1/views` — operationId `list-views`; pagination/cursor surface; auth: Bearer; List views
- `POST /v1/views` — operationId `create-view`; body: application/json; auth: Bearer; Create a view
- `DELETE /v1/views/{view_id}` — operationId `delete-view`; auth: Bearer; Delete a view
- `GET /v1/views/{view_id}` — operationId `retrieve-a-view`; auth: Bearer; Retrieve a view
- `PATCH /v1/views/{view_id}` — operationId `update-a-view`; body: application/json; auth: Bearer; Update a view
- `POST /v1/views/{view_id}/queries` — operationId `create-view-query`; body: application/json; auth: Bearer; Create a view query
- `DELETE /v1/views/{view_id}/queries/{query_id}` — operationId `delete-view-query`; auth: Bearer; Delete a view query
- `GET /v1/views/{view_id}/queries/{query_id}` — operationId `get-view-query-results`; pagination/cursor surface; auth: Bearer; Get view query results

## First-class operation sets to remember

### Views

- `GET /v1/views` — `list-views`
- `POST /v1/views` — `create-view`
- `DELETE /v1/views/{view_id}` — `delete-view`
- `GET /v1/views/{view_id}` — `retrieve-a-view`
- `PATCH /v1/views/{view_id}` — `update-a-view`
- `POST /v1/views/{view_id}/queries` — `create-view-query`
- `DELETE /v1/views/{view_id}/queries/{query_id}` — `delete-view-query`
- `GET /v1/views/{view_id}/queries/{query_id}` — `get-view-query-results`

### Markdown pages

- `GET /v1/pages/{page_id}/markdown` — `retrieve-page-markdown`
- `PATCH /v1/pages/{page_id}/markdown` — `update-page-markdown`

### File uploads

- `GET /v1/file_uploads` — `list-file-uploads`
- `POST /v1/file_uploads` — `create-file`
- `GET /v1/file_uploads/{file_upload_id}` — `retrieve-file-upload`
- `POST /v1/file_uploads/{file_upload_id}/complete` — `complete-file-upload`
- `POST /v1/file_uploads/{file_upload_id}/send` — `upload-file`

### OAuth token maintenance

- `POST /v1/oauth/introspect` — `introspect-token`
- `POST /v1/oauth/revoke` — `revoke-token`
- `POST /v1/oauth/token` — `create-a-token`

## Webhook event keys in OpenAPI

- `commentCreated` — `POST` `webhook-comment-created`; Comment created
- `commentDeleted` — `POST` `webhook-comment-deleted`; Comment deleted
- `commentUpdated` — `POST` `webhook-comment-updated`; Comment updated
- `dataSourceContentUpdated` — `POST` `webhook-data-source-content-updated`; Data source content updated
- `dataSourceCreated` — `POST` `webhook-data-source-created`; Data source created
- `dataSourceDeleted` — `POST` `webhook-data-source-deleted`; Data source deleted
- `dataSourceMoved` — `POST` `webhook-data-source-moved`; Data source moved
- `dataSourceSchemaUpdated` — `POST` `webhook-data-source-schema-updated`; Data source schema updated
- `dataSourceUndeleted` — `POST` `webhook-data-source-undeleted`; Data source undeleted
- `databaseContentUpdated` — `POST` `webhook-database-content-updated`; Database content updated
- `databaseCreated` — `POST` `webhook-database-created`; Database created
- `databaseDeleted` — `POST` `webhook-database-deleted`; Database deleted
- `databaseMoved` — `POST` `webhook-database-moved`; Database moved
- `databaseSchemaUpdated` — `POST` `webhook-database-schema-updated`; Database schema updated
- `databaseUndeleted` — `POST` `webhook-database-undeleted`; Database undeleted
- `fileUploadCompleted` — `POST` `webhook-file-upload-completed`; File upload completed
- `fileUploadCreated` — `POST` `webhook-file-upload-created`; File upload created
- `fileUploadExpired` — `POST` `webhook-file-upload-expired`; File upload expired
- `fileUploadUploadFailed` — `POST` `webhook-file-upload-upload-failed`; File upload failed
- `pageContentUpdated` — `POST` `webhook-page-content-updated`; Page content updated
- `pageCreated` — `POST` `webhook-page-created`; Page created
- `pageDeleted` — `POST` `webhook-page-deleted`; Page deleted
- `pageLocked` — `POST` `webhook-page-locked`; Page locked
- `pageMoved` — `POST` `webhook-page-moved`; Page moved
- `pagePropertiesUpdated` — `POST` `webhook-page-properties-updated`; Page properties updated
- `pageTranscriptionBlockTranscriptDeleted` — `POST` `webhook-page-transcription-block-transcript-deleted`; Page transcript deleted
- `pageUndeleted` — `POST` `webhook-page-undeleted`; Page undeleted
- `pageUnlocked` — `POST` `webhook-page-unlocked`; Page unlocked
- `viewCreated` — `POST` `webhook-view-created`; View created
- `viewDeleted` — `POST` `webhook-view-deleted`; View deleted
- `viewUpdated` — `POST` `webhook-view-updated`; View updated

## Schema enum excerpts worth handling non-exhaustively

- `baseWebhookPayload.properties.api_version`: `2022-06-28`, `2025-09-03`, `2026-03-11`
- `dataSourceViewObjectResponse.properties.type`: `table`, `board`, `list`, `calendar`, `timeline`, `gallery`, `form`, `chart`, `map`, `dashboard`
- `fileUploadObjectResponse.properties.status`: `pending`, `uploaded`, `expired`, `failed`
- `partialDataSourceViewObjectResponse.properties.type`: `table`, `board`, `list`, `calendar`, `timeline`, `gallery`, `form`, `chart`, `map`, `dashboard`
- `verificationPropertyResponse.properties.state`: `verified`, `expired`
- `verificationPropertyStatusFilter.properties.status`: `verified`, `expired`, `none`
- `viewTypeRequest`: `table`, `board`, `list`, `calendar`, `timeline`, `gallery`, `form`, `chart`, `map`, `dashboard`
- `webhookDatabaseEventEntity.properties.type`: `block`, `database`, `data_source`
- `webhookExternalBlock.properties.type`: `page`, `database`, `block`
- `webhookParentBlock.properties.type`: `space`, `block`, `page`, `database`, `team`, `agent`
- `webhookUpdatedBlock.properties.type`: `page`, `database`, `block`

## Generated drift flags

Stale operation IDs detected in official OpenAPI:

- `POST /v1/data_sources` has operationId `create-a-database`; prefer method+path/canonical docs in generated clients.
- `POST /v1/data_sources/{data_source_id}/query` has operationId `post-database-query`; prefer method+path/canonical docs in generated clients.

Other standing drift flags:

- Treat additive fields/types as expected; Notion versioning allows response additions without version bumps.
- Use `code`, not human `message`, for error handling.
- Do not flatten deep `oneOf`/`anyOf`/`allOf` unions into closed enums; keep discriminator + unknown fallback.
