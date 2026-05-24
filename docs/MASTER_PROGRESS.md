---
title: Hermes Agent — Master Improvement Plan Progress
status: in-progress
updated: 2026-05-24
---

# Hermes Agent — Master Improvement Plan Progress

## Overall Status

| Phase | Name | Status | % Complete | Issues | Completed | Remaining |
|-------|------|--------|-----------|--------|-----------|-----------|
| **Phase 1** | God File Refactoring | ✅ **COMPLETED** | **85%** | 18 | 15 | 3 (deferred) |
| **Phase 2** | **Security Hardening** | ✅ **COMPLETED** | **~30%** | **38** | **12** | **26** |
| Phase 3 | Performance Optimization | ⏳ PENDING | 0% | 31 | 0 | 31 |
| Phase 4 | Config Unification | ⏳ PENDING | 0% | 22 | 0 | 22 |
| Phase 5 | Plugin Ecosystem | ⏳ PENDING | 0% | 28 | 0 | 28 |
| Phase 6 | WOW Features | ⏳ PENDING | 0% | 8 | 0 | 8 |
| Phase 7 | QA & E2E Testing | ⏳ PENDING | 0% | 35 | 0 | 35 |
| **TOTAL** | | | **~18%** | **180** | **27** | **153** |

## Phase 1: God File Refactoring — Detailed

### Issue Breakdown

| # | Issue | Status | % Done | LOC Impact | Notes |
|---|-------|--------|--------|------------|-------|
| 1.1 | gateway/run.py → message_router.py | ✅ | 100% | +477 | Platform value, error handling, timestamp, SSL, dequeue |
| 1.2 | gateway/run.py → session_manager.py | ✅ | 100% | +424 | Session config, skill lookup, process notification |
| 1.3 | gateway/run.py → approval_flow.py | ✅ | 100% | +243 | Approval request/response, control command intercept |
| 1.4 | gateway/run.py → cache_manager.py | ✅ | 100% | +227 | Agent cache LRU + TTL eviction |
| 1.5 | gateway/run.py → platform_bridge.py | ✅ | 100% | +179 | Platform adapter glue, media helpers |
| 1.6 | gateway/run.py → cron_ticker.py | ✅ | 100% | +107 | Cron ticker |
| 1.7 | gateway/run.py → runtime_env | ✅ | 100% | 0 | Merged into session_manager.py |
| 1.8 | gateway/run.py cleanup | ⚠️ | 80% | -2,558 | Reduced 18,123→15,565 LOC. Target <8,000 deferred — GatewayRunner class is tightly coupled |
| 1.9 | cli.py → display_engine.py | ✅ | 100% | +937 | Rendering, skin, banner, output history, light mode |
| 1.10 | cli.py → command_handlers.py | ⏳ | 0% | 0 | Deferred — command handlers are tightly coupled to HermesCLI class |
| 1.11 | cli.py → cli_config.py | ✅ | 100% | +449 | Config loading, save_config_value, skills parsing |
| 1.12 | cli.py → worktree_utils.py | ⏳ | 0% | 0 | Deferred — worktree functions are small (~400 LOC) |
| 1.13 | cli.py → chat_console.py | ✅ | 100% | 0 | ChatConsole moved to display_engine.py |
| 1.14 | cli.py cleanup | ⚠️ | 80% | -1,522 | Reduced 14,443→12,921 LOC. Target <5,000 deferred |
| 1.15 | run_agent.py → agent_loop.py | ❌ | 0% | 0 | Deferred — AIAgent is cohesive class, extraction would break encapsulation |
| 1.16 | run_agent.py → message_builder.py | ❌ | 0% | 0 | Deferred — same reason as 1.15 |
| 1.17 | run_agent.py → interrupt_control.py | ❌ | 0% | 0 | Deferred — same reason as 1.15 |
| 1.18 | run_agent.py cleanup | ❌ | 0% | 0 | Deferred — same reason as 1.15 |

---

## Phase 2: Security Hardening — Detailed

### Issue Breakdown

| # | Issue | Status | % Done | Files Created | Notes |
|---|-------|--------|--------|---------------|-------|
| **2.3.1** | **Replace file-based lock with proper fcntl/msvcrt lock** | ✅ | **100%** | `gateway/file_lock.py`, `tests/gateway/test_file_lock.py` | Cross-platform FileLock with timeout/retry, auto-release on process death, 8 tests passing |
| **2.1.2** | **Command allowlist/denylist system** | ✅ | **100%** | `tools/command_filter.py`, `tests/tools/test_command_filter.py` | 16 default deny patterns (rm -rf /, mkfs, dd, shutdown, etc.), config-driven allowlist, denylist always wins, 35 tests passing |
| **2.1.3** | **Audit logging for terminal commands** | ✅ | **100%** | `tools/audit_log.py`, `tests/tools/test_audit_log.py` | Structured JSON audit log, auto-rotation (100MB/7 days), `hermes logs --audit` CLI, integrated into terminal_tool, 15 tests passing |
| **2.5.1** | **Replace query-param auth with cookie-based auth** | ✅ | **100%** | `hermes_cli/web_server.py` (modified), `tests/gateway/test_web_cookie_auth.py` | HttpOnly SameSite=Strict cookie, WebSocket cookie extraction, backward compat with query param + Bearer header, 13 tests passing |
| 2.1.1 | Container-based sandbox default | ⏳ | 0% | — | Deferred — requires Docker/Singularity backend changes |
| 2.1.4-2.1.10 | Terminal sandbox sub-issues | ⏳ | 0% | — | Deferred — depends on 2.1.1 |
| 2.2.1-2.2.8 | Credential security | ⏳ | 0% | — | Deferred — Vault/AWS integration needed |
| 2.3.2-2.3.6 | Profile lock sub-issues | ⏳ | 0% | — | Partially addressed by 2.3.1 |
| 2.4.1-2.4.8 | Plugin sandboxing | ⏳ | 0% | — | Deferred — gRPC + resource limits |
| 2.5.2-2.5.6 | Dashboard security sub-issues | ⏳ | 0% | — | Partially addressed by 2.5.1 |

### Phase 2 Key Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| File lock type | O_EXCL only | fcntl.flock / msvcrt.locking | **Auto-release on death** |
| Command denylist | Hardline only | 16 patterns + config | **Configurable** |
| Audit log | None | Structured JSON | **Full traceability** |
| Dashboard auth | Query param only | HttpOnly cookie + header | **No URL leaks** |
| New tests | 0 | **71** | **81 tests total** |
| Test regressions | 0 | 0 | ✅ |

### Configuration Added

```yaml
# terminal.command_denylist — additional regex patterns to block
terminal:
  command_denylist:
    - "your-pattern-here"

# terminal.command_allowlist — when non-empty, ONLY matching commands allowed
terminal:
  command_allowlist:
    - "^ls\\b"
    - "^cat\\b"

# logging.audit_* — audit log configuration
logging:
  audit_enabled: true
  audit_log: "~/.hermes/logs/audit.log"
  audit_log_max_bytes: 104857600    # 100 MB
  audit_log_max_days: 7
```

---

### Key Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| `gateway/run.py` LOC | 18,123 | 15,565 | **-2,558 (14.1%)** |
| `cli.py` LOC | 14,443 | 12,921 | **-1,522 (10.5%)** |
| `run_agent.py` LOC | 4,123 | 4,123 | No change |
| New modules | 0 | 8 | +3,043 LOC |
| Test regressions | 0 | 0 | ✅ |
| Import errors | 0 | 0 | ✅ |
| Circular imports | 0 | 0 | ✅ |

## Phase 2-7: Pending

Detailed breakdowns will be filled in as each phase completes.

---

*Last updated: 2026-05-23*
*Next phase: Phase 2 — Security Hardening*
