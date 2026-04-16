# Trace System V2 - Enhanced Call Chain Tracing

## Overview

The Trace System V2 is an enhanced call chain tracing system for Hermes Agent that addresses the limitations of the original implementation. It provides three-level indexing, intelligent compression, real-time streaming with async storage, and session boundary detection.

## Key Features

### 1. Three-Level Indexing
- **Level 1**: `session_id` - Groups related traces together
- **Level 2**: `trace_id` - Identifies individual requests within a session
- **Level 3**: `tool_call_id` - Tracks individual tool executions

This hierarchical structure enables efficient querying at different granularities.

### 2. Intelligent Compression
The system intelligently compresses tool results while preserving important content:

- **Important patterns preserved**: Error messages, warnings, success indicators, URLs, file paths, IP addresses, hashes
- **Noise filtered**: Empty lines, separators, debug messages, trace messages
- **Configurable limits**: Maximum length for tool results and arguments

### 3. Real-Time Streaming + Async Storage
- **Real-time streaming**: Events are immediately available through `TraceStreamContext`
- **Async storage**: Events are batched and stored asynchronously to avoid blocking execution
- **Configurable batching**: Adjustable batch size and flush interval

### 4. Session Boundary Detection
The system automatically detects session boundaries based on:

- **Compression events**: When context compression occurs
- **Time gaps**: Significant time gaps (> 30 minutes)
- **Message count**: Large changes in message count (> 50 messages)

## Architecture

### Core Components

1. **TraceEvent**: Lightweight dataclass with minimal fields
2. **TraceSession**: Session metadata and statistics
3. **IntelligentCompressor**: Compresses tool results and arguments
4. **SessionBoundaryDetector**: Detects session boundaries
5. **TraceStreamContext**: Real-time event streaming
6. **AsyncStorageManager**: Asynchronous batch storage

### Database Schema

The system uses SQLite with optimized indexes:

- **trace_sessions**: Session metadata
- **trace_traces**: Trace metadata within sessions
- **trace_tool_calls**: Individual tool call records
- **trace_events**: Detailed event log
- **trace_error_patterns**: Error pattern analysis
- **trace_feedbacks**: User feedback
- **trace_tool_stats**: Aggregated tool statistics
- **trace_session_boundaries**: Session boundary markers

## Usage

### Basic Usage

```python
from core.trace_system_v2 import TraceSystemV2, create_trace_callbacks

# Get trace system instance
system = TraceSystemV2()

# Create a session
session_id = system.create_session(metadata={'user': 'test'})

# Create callbacks for tool execution
start_cb, complete_cb = create_trace_callbacks(session_id, 'trace-123')

# Use callbacks during tool execution
start_cb('tool-001', 'read_file', {'path': '/test.txt'})
# ... tool execution ...
complete_cb('tool-001', 'read_file', {'path': '/test.txt'}, 'File contents')

# End session
system.end_session(session_id)
```

### Advanced Usage

```python
from core.trace_system_v2 import TraceSystemV2, TraceEvent, EventType

system = TraceSystemV2()

# Create session with metadata
session_id = system.create_session(metadata={'version': 'v2'})

# Record LLM events
system.record_llm_request(session_id, 'trace-123', 'gpt-4', 10)
system.record_llm_response(session_id, 'trace-123', 'gpt-4', 'Response', 1234.5)

# Record compression event
system.record_compression(session_id, 'trace-123', 10000, 5000, 100.0)

# Analyze session
analysis = system.analyze_session(session_id)
print(f"Tool calls: {len(analysis['tool_analysis'])}")
print(f"Errors: {len(analysis['errors'])}")
```

## Migration from V1

### Automatic Migration

The migration script can automatically migrate data from the original trace system:

```python
from db.migrations import upgrade, check_migration_status

# Check current status
status = check_migration_status()
print(f"V1 events: {status['v1_event_count']}")
print(f"V2 sessions: {status['v2_session_count']}")

# Perform migration
result = upgrade(dry_run=False)
print(f"Migrated {result['stats']['events_migrated']} events")
```

### Backward Compatibility

The system maintains backward compatibility with the original trace system:

```python
from core.trace_system_v2 import record_event

# Old-style function still works
record_event(
    session_id='session-123',
    trace_id='trace-456',
    event_type='tool_start',
    tool_name='read_file',
    tool_args={'path': '/test.txt'}
)
```

## Configuration

Trace configuration is loaded from `~/.hermes/config.yaml`:

```yaml
trace:
  enabled: true
  sample_rate: 1.0
  v2_enabled: true
  v2_db_path: "~/.hermes/trace/trace_v2.db"
```

## Performance Considerations

1. **Batch Operations**: Events are batched before storage to reduce I/O
2. **WAL Mode**: SQLite uses Write-Ahead Logging for better concurrency
3. **Indexes**: Optimized indexes for common query patterns
4. **Compression**: Tool results are compressed to save storage
5. **Sampling**: Configurable sampling rate for high-volume environments

## Testing

Run the test suite:

```bash
python -m unittest tests.test_trace_system_v2 -v
```

## File Structure

```
hermes-agent/
├── core/
│   ├── __init__.py
│   └── trace_system_v2.py          # Main trace system implementation
├── db/
│   ├── __init__.py
│   ├── trace_schema_v2.sql         # Database schema
│   ├── trace_manager_v2.py         # Database operations
│   └── migrations/
│       ├── __init__.py
│       └── upgrade_trace_system.py # Migration script
└── tests/
    └── test_trace_system_v2.py     # Test suite
```

## Future Enhancements

1. **Distributed Tracing**: Support for distributed traces across multiple agents
2. **Real-time Dashboard**: Web-based dashboard for trace visualization
3. **Advanced Analytics**: Machine learning-based anomaly detection
4. **Export Formats**: Support for OpenTelemetry, Jaeger, and Zipkin formats
5. **Retention Policies**: Automatic cleanup of old trace data

## Troubleshooting

### Database Locked Errors

If you encounter database locked errors:

1. Check for long-running transactions
2. Increase the timeout in `TraceManagerV2._get_connection()`
3. Ensure proper connection cleanup

### Memory Usage

For high-volume tracing:

1. Reduce batch size in `AsyncStorageManager`
2. Increase flush frequency
3. Enable sampling with `sample_rate < 1.0`

### Migration Issues

If migration fails:

1. Check the backup file exists
2. Verify sufficient disk space
3. Check file permissions on `~/.hermes/trace/`

## License

Part of the Hermes Agent project.