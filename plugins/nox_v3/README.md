# NOX V3 - Neural Operational eXpression

**Version:** 3.0.0
**Author:** Hermes Agent Community
**License:** MIT

## Overview

NOX V3 combines token optimization (V1) with verification (V2) to provide balanced performance:

- **30-50% token savings** through compact internal reasoning
- **Strong verification** to prevent information loss
- **Adaptive latency** based on task complexity and context
- **Token budgeting** to control daily usage
- **Zero-context-cost** plugin integration

## Features

### Token Optimization
- Pre-LLM hook injects NOX system prompt
- Model thinks in compact notation internally
- Only expands final answer in natural language
- Configurable compression levels (conservative/balanced/aggressive)

### Verification & Safety
- Post-LLM hook verifies internal reasoning
- Checks for completeness and logical structure
- Falls back to original response if verification fails
- Tracks verification success rate

### Performance
- Fast path for simple responses (95% of cases)
- Adaptive latency based on task complexity
- Configurable latency budgets
- Non-blocking design with graceful degradation

### Cost Control
- Daily token budget limits
- Per-session token tracking
- Usage visibility and warnings
- Auto-disable when budget exceeded

## Installation

NOX V3 is a bundled plugin - no installation required.

## Usage

### Enable NOX V3

```bash
/nox enable
```

Enable with custom settings:

```bash
/nox enable --mode balanced --max-tokens 20000 --latency 50
```

### Check Status

```bash
/nox status
```

Shows:
- Current status (enabled/disabled)
- Compression mode
- Token usage and budget
- Session count
- Verification success rate

### Disable NOX V3

```bash
/nox disable
```

Stops applying to new sessions. Existing sessions continue until complete.

### Configure NOX V3

```bash
/nox config
```

Update configuration:

```bash
/nox config --mode aggressive --latency 100
/nox config --max-tokens 50000
/nox config --fast-path 200
```

### Reset Daily Usage

```bash
/nox reset
```

Resets daily token usage counter (typically called at midnight).

## Configuration

### Compression Modes

| Mode | Token Savings | Latency | Risk | Use Case |
|------|--------------|---------|------|----------|
| Conservative | 10-30% | <10ms | Very Low | Sensitive tasks, first testing |
| Balanced | 30-50% | <50ms | Low | General reasoning, mixed domains |
| Aggressive | 50-80% | <100ms | Moderate | Large context, well-known topics |

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `false` | Enable/disable NOX V3 |
| `mode` | `balanced` | Compression mode |
| `max_daily_tokens` | `10000` | Daily token budget |
| `latency_budget_ms` | `50` | Maximum latency overhead |
| `fast_path_threshold` | `100` | Token threshold for fast path |

## Architecture

### Dual-Hook Design

**Pre-LLM Hook (`pre_llm_call`):**
1. Check if NOX is enabled
2. Verify token budget
3. Inject NOX system prompt
4. Store metadata for post-processing

**Post-LLM Hook (`post_llm_call`):**
1. Retrieve NOX state from pre-hook
2. Check latency budget
3. Determine if fast path applies
4. Parse NOX reasoning from response
5. Verify reasoning completeness
6. Optimize reasoning if verification passes
7. Return final answer or fallback

### Data Flow

```
User Query
    ↓
[Pre-LLM Hook]
    ↓
Inject NOX System Prompt
    ↓
LLM Call (thinks in compact notation)
    ↓
[Post-LLM Hook]
    ↓
Parse & Verify Reasoning
    ↓
Optimize (if valid)
    ↓
Expand to Final Answer
    ↓
Return to User
```

## NOX Notation

### Basic Patterns

- `FACT[condition]` - Facts
- `RULE[X->Y]` - Rules/implications
- `INFER[conclusion]` - Inferences
- `VERIFY[step]` - Verification steps

### Examples

**Conservative Mode:**
```
IF (Wed is between Mon-Fri) THEN library is open.
```

**Balanced Mode:**
```
FACT[Open(Mon-Fri)]; FACT[Day=Wed]; INFER[Open=Yes]
```

**Aggressive Mode:**
```
F[O(M-F)];F[D=W];I[O=Y]
```

## Performance

### Token Savings

- Conservative: 10-30% reduction
- Balanced: 30-50% reduction
- Aggressive: 50-80% reduction

### Latency Overhead

- Conservative: <10ms
- Balanced: <50ms
- Aggressive: <100ms

### Verification Success Rate

Target: >95% success rate in production

## Troubleshooting

### NOX not applying

1. Check if enabled: `/nox status`
2. Verify token budget not exceeded
3. Check logs for errors

### High latency

1. Reduce latency budget: `/nox config --latency 30`
2. Use conservative mode: `/nox config --mode conservative`
3. Increase fast path threshold: `/nox config --fast-path 200`

### Token budget exceeded

1. Increase budget: `/nox config --max-tokens 20000`
2. Reset daily usage: `/nox reset`
3. Use conservative mode for more efficiency

## Comparison with NOX V1 and V2

| Feature | NOX V1 | NOX V2 | NOX V3 |
|---------|--------|--------|--------|
| Token Optimization | ✅ | ❌ | ✅ |
| Verification | ❌ | ✅ | ✅ |
| Pre-LLM Hook | ✅ | ❌ | ✅ |
| Post-LLM Hook | ❌ | ✅ | ✅ |
| Token Budgeting | ❌ | ❌ | ✅ |
| Adaptive Latency | ❌ | ❌ | ✅ |
| Slash Commands | ❌ | ❌ | ✅ |

## Contributing

Contributions welcome! Please follow Hermes Agent contribution guidelines.

## License

MIT License - see LICENSE file for details.
