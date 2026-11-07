# Dyad SQLite Database Schema

Extracted from Dyad v1.6.2 (Aug 2026). The DB lives at
`~/Library/Application Support/dyad/sqlite.db` and runs in **WAL mode**
(concurrent reads are safe while Dyad runs; writes should be done with caution).

## Tables

### `apps` — Dyad projects

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `name` | text | | Project name |
| `path` | text | | Relative path (name only, resolves under `~/dyad-apps/`) |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |
| `github_org` | text | | GitHub org for repo sync |
| `github_repo` | text | | GitHub repo name |
| `supabase_project_id` | text | | Supabase project reference |
| `chat_context` | text | | Additional context for chats |
| `github_branch` | text | | Active GitHub branch |
| `vercel_project_id` | text | | Vercel deployment project |
| `vercel_project_name` | text | | Vercel project name |
| `vercel_team_id` | text | | Vercel team |
| `vercel_deployment_url` | text | | Live deployment URL |
| `neon_project_id` | text | | Neon DB project |
| `neon_development_branch_id` | text | | Neon dev branch |
| `neon_preview_branch_id` | text | | Neon preview branch |
| `install_command` | text | | Custom install command |
| `start_command` | text | | Custom start command |
| `is_favorite` | integer | 0 | Favorite flag |
| `supabase_parent_project_id` | text | | Supabase parent (branching) |
| `supabase_organization_slug` | text | | Supabase org slug |
| `theme_id` | text | | Applied theme |
| `neon_active_branch_id` | text | | Active Neon branch |
| `needs_app_blueprint` | integer | 0 | Blueprint pending flag |
| `neon_production_auth_cookie_secret` | text | | Neon prod auth |
| `neon_development_auth_cookie_secret` | text | | Neon dev auth |
| `collection_id` | integer | | FK → `app_collections(id)` |
| `selected_database_branch_type` | text | | DB branch type |

### `chats` — Chat sessions per project

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `app_id` | integer | | FK → `apps(id)` ON DELETE CASCADE |
| `title` | text | | Chat title |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `initial_commit_hash` | text | | Git commit at chat start |
| `compacted_at` | integer | | When compaction happened |
| `compaction_backup_path` | text | | Path to compaction backup |
| `pending_compaction` | integer | | Compaction pending flag |
| `chat_mode` | text | | Chat mode (e.g. "build", "agent") |

### `messages` — Individual chat messages

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `chat_id` | integer | | FK → `chats(id)` ON DELETE CASCADE |
| `role` | text | | "user", "assistant", etc. |
| `content` | text | | Message content (may contain markdown, code) |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `approval_state` | text | | Approval status for code proposals |
| `commit_hash` | text | | Git commit associated with message |
| `request_id` | text | | LLM request ID |
| `source_commit_hash` | text | | Source commit for this message |
| `max_tokens_used` | integer | | Token usage |
| `ai_messages_json` | text | | Full AI message JSON |
| `model` | text | | Model used (e.g. "claude-3.5-sonnet") |
| `using_free_agent_mode_quota` | integer | | Free tier flag |
| `is_compaction_summary` | integer | | Compaction summary flag |

### `mcp_servers` — MCP server registrations

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `name` | text | | Display name |
| `transport` | text | | "stdio" or "sse" |
| `command` | text | | Command to run (stdio transport) |
| `args` | text | | Arguments (space-separated or JSON) |
| `env_json` | text | | JSON object of env vars |
| `url` | text | | URL (sse transport) |
| `enabled` | integer | 0 | Whether server is active |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |
| `headers_json` | text | | JSON headers (sse transport) |
| `oauth_enabled` | integer | 0 | OAuth flag |
| `oauth_state` | text | | OAuth state token |
| `oauth_client_id` | text | | OAuth client ID |
| `oauth_client_secret` | text | | OAuth client secret |
| `oauth_scope` | text | | OAuth scopes |
| `oauth_callback_port` | integer | | OAuth callback port |

### `mcp_tool_consents` — Per-tool approval settings

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `server_id` | integer | | FK → `mcp_servers(id)` ON DELETE CASCADE |
| `tool_name` | text | | MCP tool name |
| `consent` | text | `'ask'` | "ask", "allow", or "deny" |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |

**Unique index:** `uniq_mcp_consent` on `(server_id, tool_name)`

### `prompts` — Reusable prompt templates

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `title` | text | | Prompt title |
| `description` | text | | Description |
| `content` | text | | Prompt content |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |
| `slug` | text | | URL slug |

**Unique index:** `prompts_slug_unique` on `(slug)`

### `versions` — Git commit versions per project

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `app_id` | integer | | FK → `apps(id)` ON DELETE CASCADE |
| `commit_hash` | text | | Git commit hash |
| `neon_db_timestamp` | text | | Neon DB timestamp |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |
| `is_favorite` | integer | 0 | Favorite flag |
| `note` | text | | Version note |

**Unique index:** `versions_app_commit_unique` on `(app_id, commit_hash)`

### `app_collections` — Project collections/folders

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `name` | text | | Collection name |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |

**Unique index:** `app_collections_name_unique` on `(name)`

### `language_model_providers` — Custom LLM providers

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | text PK | | Provider ID (string) |
| `name` | text | | Display name |
| `api_base_url` | text | | API endpoint |
| `env_var_name` | text | | Env var for API key |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |

### `language_models` — Custom model definitions

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `display_name` | text | | Model display name |
| `api_name` | text | | API model name |
| `builtin_provider_id` | text | | Reference to builtin provider |
| `custom_provider_id` | text | | FK → `language_model_providers(id)` ON DELETE CASCADE |
| `description` | text | | Model description |
| `max_output_tokens` | integer | | Max output tokens |
| `context_window` | integer | | Context window size |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |

### `custom_themes` — User-created UI themes

| Column | Type | Default | Notes |
|--------|------|---------|-------|
| `id` | integer PK | AUTOINCREMENT | Primary key |
| `name` | text | | Theme name |
| `description` | text | | Theme description |
| `prompt` | text | | Theme generation prompt |
| `created_at` | integer | `unixepoch()` | Unix timestamp |
| `updated_at` | integer | `unixepoch()` | Unix timestamp |

### `__drizzle_migrations` — Schema migration tracking

| Column | Type | Notes |
|--------|------|-------|
| `id` | serial PK | Migration ID |
| `hash` | text | Migration hash |
| `created_at` | numeric | When applied |

## Useful Queries

### List all projects with chat counts

```sql
SELECT a.id, a.name, a.path, a.github_repo,
       COUNT(DISTINCT c.id) AS chat_count,
       COUNT(DISTINCT m.id) AS message_count
FROM apps a
LEFT JOIN chats c ON c.app_id = a.id
LEFT JOIN messages m ON m.chat_id = c.id
GROUP BY a.id
ORDER BY a.created_at DESC;
```

### Recent messages across all projects

```sql
SELECT m.id, m.role, m.model, m.created_at,
       c.title AS chat_title, a.name AS project_name
FROM messages m
JOIN chats c ON m.chat_id = c.id
JOIN apps a ON c.app_id = a.id
ORDER BY m.created_at DESC
LIMIT 20;
```

### All MCP servers with consent counts

```sql
SELECT s.id, s.name, s.transport, s.enabled,
       COUNT(tc.id) AS tool_count,
       SUM(CASE WHEN tc.consent = 'allow' THEN 1 ELSE 0 END) AS auto_approved
FROM mcp_servers s
LEFT JOIN mcp_tool_consents tc ON tc.server_id = s.id
GROUP BY s.id;
```

### Project versions (git history)

```sql
SELECT v.commit_hash, v.note, v.is_favorite, v.created_at, a.name
FROM versions v
JOIN apps a ON v.app_id = a.id
ORDER BY v.created_at DESC
LIMIT 30;
```