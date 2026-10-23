"""In-memory stand-in for a server speaking the SEP-2640 skills extension.

Deliberately not built on the real SDK: the point of these tests is to control the
wire precisely — page boundaries, repeated cursors, digests that stop matching,
advertised-but-unimplemented methods — none of which a real server will do on
demand. The fake answers through :class:`FakeSkillsSender`, whose two entry points
match the production sender's, so the manager's own bridging choice is exercised
rather than bypassed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

SKILLS_EXTENSION_ID = "io.modelcontextprotocol/skills"


class JsonRpcError(Exception):
    """JSON-RPC error shape the manager's translator duck-types on (``code``/``message``)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def digest_of(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class FakeSkill:
    """One skill the fake server publishes, with its files."""

    uri: str
    name: str
    description: str = "a fake skill"
    files: dict[str, str] = field(default_factory=dict)
    frontmatter_extra: dict[str, Any] = field(default_factory=dict)
    dynamic: bool = False

    def main_text(self) -> str:
        return self.files.get("SKILL.md", f"---\nname: {self.name}\ndescription: {self.description}\n---\nbody\n")

    def entry(self, *, digest_overrides: Optional[dict[str, str]] = None) -> dict[str, Any]:
        root = self.uri.rsplit("/", 1)[0]
        files = dict(self.files) or {"SKILL.md": self.main_text()}
        resources = [
            {"uri": self.uri if rel == "SKILL.md" else f"{root}/{rel}",
             "size": len(text.encode("utf-8")),
             "digest": (digest_overrides or {}).get(rel) or digest_of(text)}
            for rel, text in files.items()
        ]
        frontmatter: dict[str, Any] = {"name": self.name, "description": self.description}
        frontmatter.update(self.frontmatter_extra)
        return {"uri": self.uri, "frontmatter": frontmatter,
                "resources": "dynamic" if self.dynamic else resources}


class FakeSkillsServer:
    """A controllable server: catalog, capabilities, failures, and call counters."""

    def __init__(self, skills: list[FakeSkill], *, page_size: int = 25,
                 directory_read: bool = True, advertise: bool = True,
                 advertise_resources: bool = True, ttl_ms: Optional[int] = 0) -> None:
        self.skills = skills
        self.page_size = page_size
        self.directory_read = directory_read
        self.advertise = advertise
        self.advertise_resources = advertise_resources
        self.ttl_ms = ttl_ms
        self.calls: list[str] = []
        # uri -> {relative path: forged digest}. Per skill, because the point of a
        # forged digest is that ONE entry stops matching.
        self.digest_overrides: dict[str, dict[str, str]] = {}
        # Skills the server answers skills/get for but omits from skills/list: the spec
        # allows a partial listing, and a host must not read absence from it as deletion.
        self.unlisted: set[str] = set()
        self.repeat_cursor: Optional[str] = None
        self.fail_next_list = False
        self.fail_next_read = False
        self.unimplemented: set[str] = set()

    # ── capabilities ────────────────────────────────────────────────────────

    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {"tools": {}}
        if self.advertise_resources:
            caps["resources"] = {"listChanged": True}
        if self.advertise:
            caps["extensions"] = {SKILLS_EXTENSION_ID: {"directoryRead": self.directory_read}}
        return caps

    def initialize_result(self) -> Any:
        return _Namespace(capabilities=self.capabilities())

    # ── request handling ────────────────────────────────────────────────────

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        self.calls.append(method)
        if method in self.unimplemented:
            raise JsonRpcError(-32601, f"Method not found: {method}")
        if method == "skills/list":
            if self.fail_next_list:
                self.fail_next_list = False
                raise JsonRpcError(-32603, "internal error while listing")
            return self._list(params.get("cursor"))
        if method == "skills/get":
            return self._get(params.get("uri"))
        if method == "resources/read":
            return self._read(params.get("uri"))
        raise JsonRpcError(-32601, f"Method not found: {method}")

    def _list(self, cursor: Optional[str]) -> dict[str, Any]:
        start = 0
        if cursor:
            if self.repeat_cursor is not None and cursor == self.repeat_cursor:
                return {"skills": [], "nextCursor": self.repeat_cursor, **self._cache_fields()}
            uris = [s.uri for s in self.skills]
            if cursor not in uris:
                raise JsonRpcError(-32602, f"unknown cursor {cursor!r}")
            start = uris.index(cursor) + 1
        listed = [s for s in self.skills if s.uri not in self.unlisted]
        window = listed[start:start + self.page_size]
        nxt = None
        if start + self.page_size < len(listed):
            nxt = listed[start + self.page_size - 1].uri
        return {"skills": [s.entry(digest_overrides=self.digest_overrides.get(s.uri)) for s in window],
                "nextCursor": nxt, **self._cache_fields()}

    def _cache_fields(self) -> dict[str, Any]:
        return {"resultType": "complete", "ttlMs": self.ttl_ms, "cacheScope": "private"}

    def _get(self, uri: Optional[str]) -> dict[str, Any]:
        skill = next((s for s in self.skills if s.uri == uri), None)
        if skill is None:
            raise JsonRpcError(-32602, f"no Skill is published at {uri!r}")
        return {"skill": skill.entry(digest_overrides=self.digest_overrides.get(uri)), **self._cache_fields()}

    def _read(self, uri: Optional[str]) -> dict[str, Any]:
        if self.fail_next_read:
            self.fail_next_read = False
            raise JsonRpcError(-32603, "internal error while reading")
        for skill in self.skills:
            root = skill.uri.rsplit("/", 1)[0]
            for rel, text in (skill.files or {"SKILL.md": skill.main_text()}).items():
                full = skill.uri if rel == "SKILL.md" else f"{root}/{rel}"
                if full == uri:
                    return {"contents": [{"uri": uri, "text": text, "blob": None, "mimeType": "text/markdown"}]}
        raise JsonRpcError(-32602, f"Resource not found: {uri!r}")


class _Namespace:
    """Attribute view over a plain dict, matching how the manager reads results."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def __repr__(self) -> str:
        return f"_Namespace({json.dumps({k: v for k, v in self.__dict__.items() if k != 'skills'}, default=str)})"


class FakeSkillsSender:
    """The sender contract the manager calls, backed by a :class:`FakeSkillsServer`.

    ``send`` and ``send_sync`` both dispatch inline and differ only in being a
    coroutine: the production difference between them is *which loop* the request
    rides, and that belongs to the real sender, not to the manager under test.
    """

    def __init__(self, server: FakeSkillsServer, *, connected: bool = True) -> None:
        self.server = server
        self.connected = connected

    def _dispatch(self, server_name: str, op: str, request: Any, result_type: Any) -> Any:
        if not self.connected:
            from agent.skills_contracts import SkillError, SkillErrorCode
            raise SkillError(SkillErrorCode.UNAVAILABLE, f"MCP server '{server_name}' is not connected")
        payload = self.server.handle(op, _params_of(request))
        return result_type.model_validate(payload, by_name=False)

    async def send(self, server_name: str, op: str, request: Any, result_type: Any, timeout: float) -> Any:
        return self._dispatch(server_name, op, request, result_type)

    def send_sync(self, server_name: str, op: str, request: Any, result_type: Any, timeout: float) -> Any:
        return self._dispatch(server_name, op, request, result_type)


def _params_of(request: Any) -> dict[str, Any]:
    params = getattr(request, "params", None)
    return params.model_dump(exclude_none=True) if params is not None else {}
