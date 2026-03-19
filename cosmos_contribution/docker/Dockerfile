# cosmos - Self-Evolving Companion AI
# Multi-stage Docker build for optimized image size

# ============ Stage 1: Builder ============
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ============ Stage 2: Runtime ============
FROM python:3.11-slim as runtime

# Labels
LABEL org.opencontainers.image.title="cosmos"
LABEL org.opencontainers.image.description="Self-Evolving Companion AI for Claude Code"
LABEL org.opencontainers.image.version="2.8.0"
LABEL org.opencontainers.image.source="https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer-ver.4.2"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    cosmos_DATA_DIR=/data \
    cosmos_CONFIG_DIR=/config \
    cosmos_LOG_LEVEL=INFO

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY cosmos/ /app/cosmos/
COPY configs/ /app/configs/
COPY main.py /app/
COPY pyproject.toml /app/

# Create data and config directories
RUN mkdir -p /data /config /models

# Create non-root user
RUN useradd -m -u 1000 cosmos && \
    chown -R cosmos:cosmos /app /data /config /models

USER cosmos

# 8000 - MCP Server
# 8501 - Streamlit UI
# 8888 - UDP Discovery
# 9999 - TCP Swarm Fabric
EXPOSE 8000 8501 8888 9999

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cosmos; print('healthy')" || exit 1

# Default command: Run MCP server
CMD ["python", "-m", "cosmos.mcp_server"]


# ============ Stage 3: GPU Runtime (Optional) ============
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu-runtime

# Labels
LABEL org.opencontainers.image.title="cosmos GPU"
LABEL org.opencontainers.image.description="Self-Evolving Companion AI (GPU-enabled)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    cosmos_DATA_DIR=/data \
    cosmos_CONFIG_DIR=/config \
    NVIDIA_VISIBLE_DEVICES=all

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgomp1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY cosmos/ /app/cosmos/
COPY configs/ /app/configs/
COPY main.py /app/
COPY pyproject.toml /app/

# Create directories
RUN mkdir -p /data /config /models

# Create user
RUN useradd -m -u 1000 cosmos && \
    chown -R cosmos:cosmos /app /data /config /models

USER cosmos

EXPOSE 8000 8501 8888 9999

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.11 -c "import cosmos; print('healthy')" || exit 1

CMD ["python3.11", "-m", "cosmos.mcp_server"]
