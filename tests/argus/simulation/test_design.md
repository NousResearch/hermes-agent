# Argus Test Design Document

Stress Tests and Edge Cases Design

---

## 1. STRESS TEST FRAMEWORK

### 1.1 Volume Tests

| Test | Description | Target Metric |
|------|-------------|---------------|
| `sessions_100` | 100 concurrent sessions | Detection latency < 100ms per session |
| `sessions_1000` | 1000 sessions, 10% active | Memory usage < 500MB |
| `tool_calls_10k` | 10,000 tool calls in single session | Query time < 1s |
| `mixed_load` | 500 sessions × 50 tool calls each | Throughput > 1000 detections/sec |

### 1.2 Concurrency Patterns

```python
# Race condition simulation
scenario_race_condition():
    - Session A: repeat_tool_calls pattern
    - Session B: same pattern, offset by 1 call
    - Session C: interleaved tool names
    - Detect: No cross-session pollution

# Cascading entropy
scenario_cascade_farm():
    - 50 sessions all hitting error_cascade simultaneously
    - Database lock contention test
    - WAL mode performance under write pressure
```

### 1.3 Performance Benchmarks

| Operation | Baseline | Stress Target | Measurement |
|-----------|----------|---------------|-------------|
| `detect_repeat_tool_calls` | 10ms | < 50ms @ 10k rows | SQLite query time |
| `detect_stuck_loops` | 15ms | < 100ms @ 10k rows | Pattern matching |
| `detect_error_cascade` | 20ms | < 75ms @ 10k rows | Sequential scan |
| Full poll cycle | 100ms | < 500ms @ 100 sessions | End-to-end |

---

## 2. EDGE CASE TEST SUITE

### 2.1 Threshold Boundaries (Near-Miss Tests)

| Pattern | Below Threshold | At Threshold | Above Threshold |
|---------|-----------------|--------------|-----------------|
| `repeat_tool_calls` | 2 calls | 3 calls | 4 calls |
| | NO detection | WARNING | WARNING |
| | | | 5 calls = CRITICAL |
| `repeat_commands` | 2 commands | 3 commands | 4 commands |
| | NO detection | WARNING | WARNING |
| `error_cascade` | 2 errors | 3 errors | 4 errors |
| | NO detection | WARNING | WARNING |
| | | | 5 errors = CRITICAL |
| `stuck_loop` | 5 tools (no repeat) | 6 tools (A,B,C,A,B,C) | 8 tools (2 cycles) |
| | NO detection | CRITICAL | CRITICAL |

### 2.2 Time Window Boundaries

```python
# Window: 10 minutes for repeat patterns
test_time_boundaries():
    - Calls at: [0min, 4min, 9min]  -> NO detection (within window)
    - Calls at: [0min, 5min, 11min] -> NO detection (3rd outside window)
    - Calls at: [0min, 4min, 9min, 12min] -> detection on first 3
    - Verify: SQLite datetime('now', '-10 minutes') behavior
```

### 2.3 Malformed Data Handling

| Input Type | Malformed Variant | Expected Behavior |
|------------|-------------------|-------------------|
| `tool_args` | Invalid JSON | Graceful skip, log warning |
| `tool_args` | 10MB JSON blob | Truncate, process remainder |
| `timestamp` | NULL | Use CURRENT_TIMESTAMP |
| `timestamp` | Future date (tomorrow) | Accept, may miss detection |
| `timestamp` | Past date (1 year ago) | Accept, outside all windows |
| `session_id` | Empty string | Reject insert, log error |
| `session_id` | 1000 char string | Accept, hash if needed |
| `tool_name` | NULL | Treat as "unknown_tool" |
| `success` | Non-boolean ("maybe") | Cast to boolean, "maybe"=True |
| `file_changed` | NULL | Infer from success status |

### 2.4 Boundary Data Sizes

| Field | Empty | Normal | Large | Extreme |
|-------|-------|--------|-------|---------|
| `tool_args` | `{}` | 1KB JSON | 100KB | 1MB (test limit) |
| `error_message` | NULL | 100 chars | 10KB stack trace | 100KB (truncate) |
| `command` (terminal) | `""` | 200 chars | 10KB | 100KB (shell limit) |
| Rows per session | 0 | 100 | 10,000 | 100,000 (pagination test) |

### 2.5 Special Characters & Injection

```python
# SQL injection attempts in fields
test_sql_injection():
    - tool_args: "'; DROP TABLE sessions; --"
    - session_id: "1' OR '1'='1"
    - error_message: "\"; DELETE FROM tool_calls; --"
    - Verify: All parameterized queries, no raw string interpolation

# Unicode edge cases
    - tool_args with emoji: {"path": "/tmp/🗑️"}
    - session_id with CJK: "测试会话_001"
    - error_message with Arabic: "خطأ في الملف"
    - Verify: UTF-8 handling throughout
```

### 2.6 Empty/Null Conditions

| Scenario | Input | Expected |
|----------|-------|----------|
| Empty session | Session with 0 tool calls | No errors, empty detection list |
| All null timestamps | Tools calls with NULL ts | Default to now, may detect |
| Mixed null/filled | Some tools with ts, some without | Process what we have |
| Single tool call | Only 1 call in session | Never detect repeats |
| No terminal commands | Session with tools only | No repeat_commands detection |
| No file writes | Session with reads only | No no_file_changes detection |

---

## 3. PARAMETERIZED TEST GENERATOR

Systematic coverage via matrix generation:

```python
# Configuration matrix
TEST_MATRIX = {
    "repeat_tool_calls": {
        "count": [1, 2, 3, 4, 5, 10, 100],
        "time_spread_minutes": [0, 5, 9, 10, 11, 60],
        "tool_args_variation": ["identical", "slightly_different", "completely_different"],
    },
    "error_cascade": {
        "consecutive_errors": [0, 1, 2, 3, 4, 5, 10],
        "interleaved_successes": [0, 1, 2],
        "error_types": ["all_same", "varied", "random"],
    },
    "stuck_loop": {
        "pattern_length": [2, 3, 4, 5],
        "iterations": [1, 2, 3, 10],
        "noise_ratio": [0, 0.1, 0.5],  # Random tools between patterns
    },
}

# Generates: 7 × 6 × 3 = 126 tests for repeat_tool_calls alone
```

---

## 4. IMPLEMENTATION PRIORITY

### Phase 1: Critical Edge Cases (This Session)
- Threshold boundaries (2 vs 3 vs 5)
- Time window boundaries (9min vs 11min)
- Malformed JSON handling
- Empty/null field handling

### Phase 2: Volume Stress (Next Session)
- 100 concurrent sessions
- 10k tool calls single session
- Performance benchmarks

### Phase 3: Adversarial/Fuzzing (Future)
- SQL injection attempts
- Unicode torture tests
- Random pattern fuzzing
- Resource exhaustion tests

---

## 5. METRICS TO COLLECT

| Metric | Unit | Target | Alert If |
|--------|------|--------|----------|
| Detection latency | ms/query | < 50 | > 100 |
| Memory usage | MB | < 256 | > 512 |
| DB query time | ms | < 10 | > 50 |
| False positive rate | % | < 1 | > 5 |
| False negative rate | % | 0 | > 0 |

---

What should we implement first?

**A)** Threshold boundary tests (systematic 1-5 call ranges)
**B)** Time window boundary tests (minute-precision edge cases)
**C)** Malformed data injection suite
**D)** Volume stress test (100 sessions benchmark)
**E)** Parameterized matrix generator (full systematic coverage)

Pick one or propose a mix.