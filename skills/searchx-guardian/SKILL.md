# SearchX Guardian

Automated health monitoring and recovery for SearXNG Docker container.

## 1. Existence
- **Goal**: Ensure SearXNG is always available by monitoring for 502 Bad Gateway errors.
- **Target**: `http://localhost:8888`
- **Action**: Restart the `searxng` Docker container on failure.

## 2. Classify
- **Type**: System Guardian / Health Monitor
- **Trigger**: Periodic check (Cron or Loop)
- **Dependency**: Docker

## 3. Scope
- **In Scope**:
    - HTTP health check on port 8888.
    - Container restart command.
    - Simple logging of recovery actions.
- **Out of Scope**:
    - Complex log analysis.
    - Updating the Docker image.
    - Managing other containers.

## 4. Architecture
- **Implementation**: Bash script.
- **Mechanism**: `curl` for status check, `docker restart` for recovery.
- **Deployment**: Systemd service or Cron job.

## 5. Build
- **Script**: `guardian.sh`
- **Service**: `searchx-guardian.service`

## 6. Validate
- **Test Case 1**: Service starts and monitors.
- **Test Case 2**: Simulate 502 (stop container) -> Verify restart.
- **Test Case 3**: Normal operation (200 OK) -> No action taken.
