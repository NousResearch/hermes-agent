# Nomad Development Configuration (Hermes-Agent)
# Run with: nomad agent -dev -config=nomad-dev.hcl
#
# This is intended for local development only.

client {
  enabled = true

  options {
    # Enable Docker volume mounts for persistent slot workspaces
    "docker.volumes.enabled" = "true"
  }
}

# Docker driver plugin configuration
plugin "docker" {
  config {
    # CRITICAL: Enable volume mounts
    volumes {
      enabled = true
    }

    # Allow privileged containers if needed
    allow_privileged = false

    # Garbage collection settings
    gc {
      image       = true
      # NOTE: For local dev we often rely on locally built images like `atropos-sandbox:local`.
      # A short image GC delay can delete these between runs, causing confusing "Failed to pull"
      # crash loops. Keep this comfortably long; tighten it for CI/production if needed.
      image_delay = "24h"
      container   = true
    }
  }
}

