# write_file_hook — Provenance Tracking v3.0

## What it does

## 📋 审阅指南
- **重点段落**：[待标注]
- **可跳过**：[待标注]
- **数据来源**：[待标注]
- **假设列表**：[待标注]
- **建议审阅方式**：[待标注]
Records every write operation (output provenance) and read operation (source provenance) automatically via `post_tool_call` hook.

- **write_file_hook.log** — `timestamp | session_id | absolute_path`
- **read_provenance.log** — `timestamp | session_id | tool_name | trust | source`

## TL;DR
> ⚠️ 自动生成，请审阅

主题：write_file_hook — Provenance Tracking v3.0 > What it does > 📋 审阅指南。关键点：重点段落；可跳过；数据来源。含 8 处数据点。


## Trust levels

| Level | Meaning | Tools |
|-------|---------|-------|
| EXTERNAL | Untrusted — always cross-verify | web_search, web_extract, browser_navigate |
| INTERNAL | Local/trusted | read_file, search_files, session_search, vision_analyze (local) |
| UNKNOWN | Subagent or opaque | delegate_task, send_message |
| LEGACY | Old-format log line (no trust field) | Backward compatibility |

## Covered tools

**Writes (5):** `write_file`, `patch`, `mcp_workspace_rw_write_file`, `mcp_hermes_backup_rw_write_file`, `mcp_workspace_rw_edit_file`

**Reads (9):** `web_extract`, `web_search`, `read_file`, `session_search`, `browser_navigate`, `search_files`, `vision_analyze`, `delegate_task`, `send_message`

## Integration

- `goal_check.sh trace_provenance()` — cross-references hook logs with done entries
- `done_marker_validate.sh` — validates session_id format (6-8 hex suffix)
- `logrotate.conf` — 10MB rotation, 5 old copies

## Rollback

```bash
# Disable plugin: remove "write_file_hook" from config.yaml plugins.enabled
# Delete logs: rm ~/.hermes/logs/write_file_hook.log ~/.hermes/logs/read_provenance.log
# Logs are append-only, no data loss from disabling
```

## Success metrics

- `goal_check.sh` false-positive rate for missing output files → zero
- Every done entry with `→ file:` has a matching write_file_hook.log entry
- Every done entry without `→ file:` gets discovered from hook log

## Live test

```bash
echo "test write $(date)" > /tmp/provenance_live_test.txt
# Then check: tail -1 ~/.hermes/logs/write_file_hook.log
# Should show: timestamp | session_id | /tmp/provenance_live_test.txt
```

🟡 假设：此产出由常青自动生成，未经杨旸手动标注假设。
