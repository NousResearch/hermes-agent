### hermes logs

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Allows viewing, tailing, and filtering Hermes log files. Users can specify which log to view, the number of lines, follow in real-time, filter by log level, session ID, time, or component.

```APIDOC
## hermes logs

### Description
View, tail, and filter Hermes log files. All logs are stored in `~/.hermes/logs/` (or `<profile>/logs/` for non-default profiles).

### Command
`hermes logs [log_name] [options]`

### Arguments
- **log_name** (string) - Optional - Which log to view: `agent` (default), `errors`, `gateway`, or `list` to show available files with sizes.

### Options
- **-n, --lines <N>** (integer) - Optional - Number of lines to show (default: 50).
- **-f, --follow** (boolean) - Optional - Follow the log in real time, like `tail -f`. Press Ctrl+C to stop.
- **--level <LEVEL>** (string) - Optional - Minimum log level to show: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- **--session <ID>** (string) - Optional - Filter lines containing a session ID substring.
- **--since <TIME>** (string) - Optional - Show lines from a relative time ago: `30m`, `1h`, `2d`, etc. Supports `s` (seconds), `m` (minutes), `h` (hours), `d` (days).
- **--component <NAME>** (string) - Optional - Filter by component: `gateway`, `agent`, `tools`, `cli`, `cron`.

### CLI Examples
```bash
# View the last 50 lines of agent.log (default)
hermes logs

# Follow agent.log in real time
hermes logs -f

# View the last 100 lines of gateway.log
hermes logs gateway -n 100

# Show only warnings and errors from the last hour
hermes logs --level WARNING --since 1h

# Filter by a specific session
hermes logs --session abc123

# Follow errors.log, starting from 30 minutes ago
hermes logs errors --since 30m -f

# List all log files with their sizes
hermes logs list
```
```

--------------------------------

### Hermes Logs Command Examples

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Demonstrates various uses of the `hermes logs` command, including viewing default logs, following logs in real-time, filtering by log name, level, session, and time, and listing all log files.

```bash
hermes logs
```

```bash
hermes logs -f
```

```bash
hermes logs gateway -n 100
```

```bash
hermes logs --level WARNING --since 1h
```

```bash
hermes logs --session abc123
```

```bash
hermes logs errors --since 30m -f
```

```bash
hermes logs list
```

--------------------------------

### Common Hermes Cron Diagnostic Commands (Bash)

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/guides/cron-troubleshooting.md

Provides a set of commands for listing cron jobs, testing job execution, editing job configurations, viewing logs, and verifying installed skills. Use these to troubleshoot cron job issues.

```bash
hermes cron list                    # Show all jobs, states, next_run times
```

```bash
hermes cron run <job_id>            # Schedule for next tick (for testing)
```

```bash
hermes cron edit <job_id>           # Fix configuration issues
```

```bash
hermes logs                         # View recent Hermes logs
```

```bash
hermes skills list                  # Verify installed skills
```

--------------------------------

### Troubleshooting Hermes Agent Docker Commands

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/docker.md

These commands provide various ways to check the health, logs, version, and resource usage of your Hermes Agent Docker container.

```sh
docker logs --tail 50 hermes
```

```sh
docker run -it --rm nousresearch/hermes-agent:latest version
```

```sh
docker stats hermes
```

--------------------------------

### CLI Command: hermes logs

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

View, tail, and filter agent/gateway/error log files.

```APIDOC
## CLI Command: hermes logs

### Description
View, tail, and filter agent/gateway/error log files.

### Method
CLI

### Command Syntax
`hermes logs`

### Example Usage
```bash
hermes logs tail
```
```