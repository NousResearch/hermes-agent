# Dockerfile for atropos-agent sandbox server
# Runs inside Nomad containers to handle tool execution
# Includes bubblewrap for namespace-based slot isolation

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Bubblewrap for namespace isolation
    bubblewrap \
    # `script` for PTY allocation (used for stable tmux+asciinema startup)
    util-linux \
    # Git for SWE-style tasks (cloning repos)
    git \
    # tmux for stateful terminal sessions (Phase 4.7+)
    tmux \
    # Common tools agents might need
    curl \
    wget \
    jq \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (sandbox server + optional terminal recording)
RUN pip install --no-cache-dir aiohttp asciinema

# Copy the sandbox server
COPY sandbox_server.py /app/sandbox_server.py

WORKDIR /app

# Create data directory for slot workspaces
RUN mkdir -p /data

# Verify bubblewrap is installed and working
RUN bwrap --version

EXPOSE 8080

# Default command - can be overridden by Nomad job spec
CMD ["python", "sandbox_server.py", "--port", "8080", "--slots", "10", "--data-dir", "/data"]
