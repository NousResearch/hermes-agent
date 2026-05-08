"""
Mem0 SDK 2.0.2 self-hosted compatibility patch.

Issue: MemoryClient.__init__ unconditionally creates Project(),
which validates org_id/project_id — self-hosted doesn't have these.

Fix: Monkey-patch MemoryClient.__init__ to skip Project init
when host points to a private/self-hosted endpoint.

Location: plugins/memory/mem0/selfhost_patch.py (survives pip upgrades)
"""

import re
import logging

logger = logging.getLogger(__name__)

_SELF_HOSTED_RE = re.compile(
    r'^(localhost|127\.[\d.]+|0\.0\.0\.0|'
    r'10\.[\d.]+|'
    r'172\.(1[6-9]|2[0-9]|3[01])\.[\d.]+|'
    r'192\.168\.[\d.]+|'
    r'100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.[\d.]+)'
)


def _is_self_hosted(host: str) -> bool:
    """Check if host points to a private/self-hosted endpoint."""
    clean = re.sub(r'^https?://', '', host).rstrip('/')
    return bool(_SELF_HOSTED_RE.match(clean))


def patch():
    """Apply self-hosted compatibility patch to mem0ai SDK 2.0.2+.
    
    Idempotent: safe to call multiple times.
    Graceful: if SDK < 2.0.2 (no Project class), no-op.
    """
    try:
        from mem0.client.main import MemoryClient
    except ImportError:
        logger.debug("mem0ai not installed, skipping selfhost patch")
        return False

    if getattr(MemoryClient, '_selfhost_patched', False):
        return True

    original_init = MemoryClient.__init__

    def patched_init(self, api_key=None, host=None, client=None):
        resolved_host = host or "https://api.mem0.ai"

        if _is_self_hosted(resolved_host):
            # Self-hosted: catch Project validation error
            try:
                original_init(self, api_key=api_key, host=host, client=client)
            except ValueError as e:
                if "org_id and project_id" in str(e):
                    self.project = None
                    logger.info("mem0ai SDK: skipped Project init (self-hosted, no org_id/project_id)")
                else:
                    raise
        else:
            original_init(self, api_key=api_key, host=host, client=client)

    MemoryClient.__init__ = patched_init
    MemoryClient._selfhost_patched = True
    logger.info("mem0ai SDK self-hosted patch applied")
    return True
