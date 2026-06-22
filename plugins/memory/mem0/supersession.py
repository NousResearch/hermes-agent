"""Supersession chain tracking for mem0 memories.

Instead of overwriting memories, maintain a version chain:
  old_memory (v1, superseded) → new_memory (v2, current)

This allows:
- Audit trail: see how facts evolved over time
- Rollback: recover old versions if needed
- Contradiction resolution: clearly mark which version is authoritative

Design: agentmemory-inspired supersession chains, adapted for mem0 self-hosted.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CHAIN_DIR = Path.home() / ".hermes" / "state" / "mem0_supersessions"
_CHAIN_INDEX = _CHAIN_DIR / "index.json"


def _ensure_dir():
    _CHAIN_DIR.mkdir(parents=True, exist_ok=True)


def load_chains() -> Dict[str, List[Dict[str, Any]]]:
    """Load all supersession chains.

    Returns: {chain_id: [{version, mem0_id, text, timestamp, supersedes}, ...]}
    """
    if not _CHAIN_INDEX.exists():
        return {}
    try:
        return json.loads(_CHAIN_INDEX.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_chains(chains: Dict[str, List[Dict[str, Any]]]):
    """Persist supersession chains to disk."""
    _ensure_dir()
    try:
        _CHAIN_INDEX.write_text(
            json.dumps(chains, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    except Exception as e:
        logger.debug("Failed to save supersession chains: %s", e)


def find_chain_for_memory(mem0_id: str) -> Optional[Tuple[str, int]]:
    """Find which chain a memory belongs to.

    Returns: (chain_id, version) or None if not in any chain.
    """
    chains = load_chains()
    for chain_id, versions in chains.items():
        for v in versions:
            if v.get('mem0_id') == mem0_id:
                return (chain_id, v.get('version', 0))
    return None


def create_supersession(
    old_mem0_id: str,
    old_text: str,
    new_mem0_id: str,
    new_text: str,
    reason: str = "SUPERSEDE",
) -> str:
    """Create a supersession chain linking old → new memory.

    Args:
        old_mem0_id: Mem0 ID of the superseded memory
        old_text: Text of the old memory
        new_mem0_id: Mem0 ID of the new memory
        new_text: Text of the new memory
        reason: Why superseded (SUPERSEDE/EXTEND)

    Returns: chain_id
    """
    import hashlib
    chains = load_chains()

    # Check if old memory is already in a chain
    existing = find_chain_for_memory(old_mem0_id)
    if existing:
        chain_id = existing[0]
        chain = chains[chain_id]
    else:
        # Create new chain
        chain_id = hashlib.sha256(
            f"{old_mem0_id}:{old_text[:50]}".encode()
        ).hexdigest()[:12]
        chain = [{
            'version': 1,
            'mem0_id': old_mem0_id,
            'text': old_text[:200],
            'timestamp': time.time(),
            'supersedes': None,
            'is_latest': False,
        }]
        chains[chain_id] = chain

    # Add new version
    next_version = max(v['version'] for v in chain) + 1
    chain.append({
        'version': next_version,
        'mem0_id': new_mem0_id,
        'text': new_text[:200],
        'timestamp': time.time(),
        'supersedes': old_mem0_id,
        'is_latest': True,
        'reason': reason,
    })

    # Mark all previous versions as not latest
    for v in chain[:-1]:
        v['is_latest'] = False

    save_chains(chains)
    logger.info("Supersession chain created: %s v%d → v%d (%s)",
                chain_id, next_version - 1, next_version, reason)
    return chain_id


def get_chain_history(chain_id: str) -> List[Dict[str, Any]]:
    """Get the full version history of a supersession chain."""
    chains = load_chains()
    return chains.get(chain_id, [])


def get_all_superseded() -> List[Dict[str, Any]]:
    """Get all memories that have been superseded (not latest)."""
    chains = load_chains()
    superseded = []
    for chain_id, versions in chains.items():
        for v in versions:
            if not v.get('is_latest', True):
                superseded.append({
                    'chain_id': chain_id,
                    'mem0_id': v['mem0_id'],
                    'text': v['text'],
                    'superseded_by': v.get('supersedes'),
                    'version': v['version'],
                })
    return superseded


def annotate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add supersession info to a search result if it's in a chain."""
    mem0_id = result.get('id', '')
    if not mem0_id:
        return result
    chain_info = find_chain_for_memory(mem0_id)
    if chain_info:
        chain_id, version = chain_info
        chains = load_chains()
        chain = chains.get(chain_id, [])
        latest = next((v for v in chain if v.get('is_latest')), None)
        result['supersession'] = {
            'chain_id': chain_id,
            'version': version,
            'total_versions': len(chain),
            'is_latest': latest and latest['mem0_id'] == mem0_id,
        }
        if latest and latest['mem0_id'] != mem0_id:
            result['supersession']['latest_version'] = latest['text'][:100]
    return result


def get_stats() -> Dict[str, Any]:
    """Get supersession chain statistics."""
    chains = load_chains()
    total_chains = len(chains)
    total_versions = sum(len(v) for v in chains.values())
    superseded_count = sum(
        1 for versions in chains.values()
        for v in versions if not v.get('is_latest', True)
    )
    return {
        'total_chains': total_chains,
        'total_versions': total_versions,
        'superseded_memories': superseded_count,
        'active_chains': sum(1 for v in chains.values() if len(v) > 1),
    }
