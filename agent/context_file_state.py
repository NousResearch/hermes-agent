"""Startup context identities paired with the exact cached prompt, never re-derived on restore."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class ContextFileSnapshot:
    identities: set[tuple] = field(default_factory=set)
    digests: set[str] = field(default_factory=set)

    def include(self, record: Any) -> None:
        if record.identity is not None:
            self.identities.add(record.identity)
        if record.content:
            self.digests.add(content_digest(record.content))

    def serialize(self) -> str:
        entries = [
            {"kind": "inode", "device": item[1], "inode": item[2]} if item[0] == "inode"
            else {"kind": "path", "path": str(item[1])}
            for item in sorted(self.identities, key=repr)
        ]
        return json.dumps({"version": 1, "identities": entries, "digests": sorted(self.digests)},
                          separators=(",", ":"), sort_keys=True)

    @classmethod
    def parse(cls, manifest: str) -> "ContextFileSnapshot":
        data = json.loads(manifest)
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("invalid context identity manifest")
        if not isinstance(data.get("identities"), list):
            raise ValueError("invalid context identity manifest")
        identities = {_parse_identity(entry) for entry in data["identities"]}
        # July's v1 manifest stored identities alone; accept those without inventing content digests.
        digests = data.get("digests", [])
        if not isinstance(digests, list) or any(not isinstance(d, str) or len(d) != 64
                                               or any(c not in "0123456789abcdef" for c in d) for d in digests):
            raise ValueError("invalid context content digests")
        return cls(identities, set(digests))


def _parse_identity(entry: Any) -> tuple:
    if not isinstance(entry, dict):
        raise ValueError("invalid context identity entry")
    if entry.get("kind") == "inode":
        device, inode = entry.get("device"), entry.get("inode")
        if type(device) is int and type(inode) is int and device >= 0 and inode > 0:
            return ("inode", device, inode)
    if entry.get("kind") == "path":
        path = entry.get("path")
        if isinstance(path, str) and Path(path).is_absolute():
            return ("path", path)
    raise ValueError("invalid context identity entry")


def restore_context_manifest(agent: Any, manifest: Any, prompt: str) -> None:
    """Bind only persisted startup state. Legacy rows keep their prompt until its next real boundary."""
    snapshot = ContextFileSnapshot()
    if isinstance(manifest, str) and manifest and prompt:
        try:
            snapshot = ContextFileSnapshot.parse(manifest)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid context identity manifest for session %s; keeping cached prompt",
                           getattr(agent, "session_id", None), exc_info=True)
            manifest = None
    else:
        manifest = None
    agent._context_file_identity_manifest = manifest
    agent._context_file_identity_prompt_hash = content_digest(prompt)
    tracker = getattr(agent, "_subdirectory_hints", None)
    register = getattr(tracker, "register_startup_context", None)
    if callable(register):
        register(snapshot)


def bind_context_snapshot(agent: Any, snapshot: ContextFileSnapshot, prompt: str) -> None:
    restore_context_manifest(agent, snapshot.serialize(), prompt)


def context_manifest_kwargs(agent: Any, prompt: str) -> dict:
    """Omit the additive DB argument for old embedders/fakes without a captured snapshot."""
    manifest = getattr(agent, "_context_file_identity_manifest", None)
    if (isinstance(manifest, str) and isinstance(prompt, str)
            and getattr(agent, "_context_file_identity_prompt_hash", None) == content_digest(prompt)):
        return {"context_file_identities": manifest}
    return {}


def persist_system_prompt(agent: Any, prompt: str, *, db: Any = None, session_id: str | None = None) -> None:
    store = db if db is not None else agent._session_db
    store.update_system_prompt(session_id or agent.session_id, prompt, **context_manifest_kwargs(agent, prompt))
