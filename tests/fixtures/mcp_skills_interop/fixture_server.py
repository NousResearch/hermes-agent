#!/usr/bin/env python3
"""Portable synthetic SEP-2640 MCP server built on the Python MCP SDK."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from typing import Any

import anyio
from pydantic import BaseModel, ConfigDict
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

EXTENSION = "io.modelcontextprotocol/skills"

PRIMARY_MD = b"""---
name: portable-demo
description: Demonstrate portable Skills Over MCP with supporting files
license: CC0-1.0
metadata:
  fixture: public-interop
---

# Portable demo

Read `references/GUIDE.md` for a benign phrase. The binary sample at
`assets/pixel.png` is data and should only be materialized when requested.
"""
COLLISION_MD = b"""---
name: portable-demo
description: A second same-name skill proving URI-scoped identity
license: CC0-1.0
metadata:
  fixture: public-interop-collision
---

# Collision demo

This distinct skill intentionally shares a display name.
"""
ALTERNATE_MD = b"""---
name: portable-alt
description: A skill served under a non-skill URI scheme
license: CC0-1.0
metadata:
  fixture: public-interop-alternate-scheme
---

# Alternate scheme demo

Skill identity comes from discovery and the server origin, not the URI scheme.
"""
UNLISTED_MD = b"""---
name: uri-only
description: A valid static skill intentionally omitted from skills/list
---

# URI-only skill
"""
PARENT_MD = b"---\nname: parent\ndescription: Canonical enclosing skill\n---\n\n# Parent\n"
CHILD_MD = b"---\nname: child\ndescription: Canonical nested skill\n---\n\n# Child\n"
BAD_PARENT_MD = b"---\nname: bad-parent\ndescription: Inconsistent enclosing skill\n---\n"
BAD_CHILD_MD = b"---\nname: bad-child\ndescription: Inconsistent nested skill\n---\n"
GUIDE = b"Portable interop phrase: cobalt-lantern-27.\n"
BINARY = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63606060f80f00010401005fe5c34b0000000049454e44ae426082")

FILES: dict[str, tuple[bytes, str]] = {
    "skill://portable-demo/SKILL.md": (PRIMARY_MD, "text/markdown"),
    "skill://portable-demo/references/GUIDE.md": (GUIDE, "text/markdown"),
    "skill://portable-demo/assets/pixel.png": (BINARY, "image/png"),
    "skill://catalog-b/portable-demo/SKILL.md": (COLLISION_MD, "text/markdown"),
    "fixture-skill://catalog-c/portable-alt/SKILL.md": (ALTERNATE_MD, "text/markdown"),
    "skill://uri-only/SKILL.md": (UNLISTED_MD, "text/markdown"),
    "skill://nested/parent/SKILL.md": (PARENT_MD, "text/markdown"),
    "skill://nested/parent/child/SKILL.md": (CHILD_MD, "text/markdown"),
    "skill://bad/bad-parent/SKILL.md": (BAD_PARENT_MD, "text/markdown"),
    "skill://bad/bad-parent/bad-child/SKILL.md": (BAD_CHILD_MD, "text/markdown"),
}
FRONTMATTER = {
    "skill://portable-demo/SKILL.md": {
        "name": "portable-demo",
        "description": "Demonstrate portable Skills Over MCP with supporting files",
        "license": "CC0-1.0",
        "metadata": {"fixture": "public-interop"},
    },
    "skill://catalog-b/portable-demo/SKILL.md": {
        "name": "portable-demo",
        "description": "A second same-name skill proving URI-scoped identity",
        "license": "CC0-1.0",
        "metadata": {"fixture": "public-interop-collision"},
    },
    "fixture-skill://catalog-c/portable-alt/SKILL.md": {
        "name": "portable-alt",
        "description": "A skill served under a non-skill URI scheme",
        "license": "CC0-1.0",
        "metadata": {"fixture": "public-interop-alternate-scheme"},
    },
    "skill://uri-only/SKILL.md": {
        "name": "uri-only", "description": "A valid static skill intentionally omitted from skills/list",
    },
    "skill://nested/parent/SKILL.md": {
        "name": "parent", "description": "Canonical enclosing skill",
    },
    "skill://nested/parent/child/SKILL.md": {
        "name": "child", "description": "Canonical nested skill",
    },
    "skill://bad/bad-parent/SKILL.md": {
        "name": "bad-parent", "description": "Inconsistent enclosing skill",
    },
    "skill://bad/bad-parent/bad-child/SKILL.md": {
        "name": "bad-child", "description": "Inconsistent nested skill",
    },
}


class SkillsListParams(types.RequestParams):
    cursor: str | None = None


class SkillsGetParams(types.RequestParams):
    uri: str


class DirectoryReadParams(types.RequestParams):
    uri: str
    cursor: str | None = None


class ExtensionResult(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    resultType: str = "complete"


def descriptor(uri: str) -> dict[str, Any]:
    raw, _mime = FILES[uri]
    return {"uri": uri, "digest": "sha256:" + hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def skill_entry(uri: str) -> dict[str, Any]:
    root = uri.removesuffix("SKILL.md")
    resources = [descriptor(item) for item in FILES if item.startswith(root)]
    resources.sort(key=lambda item: item["uri"])
    return {"uri": uri, "frontmatter": FRONTMATTER[uri], "resources": resources}


class Fixture:
    def __init__(self, events_path: str | None) -> None:
        self.events_path = events_path
        self.server = Server(
            "public-skills-interop-fixture",
            version="1.0.0",
            description="Synthetic public SEP-2640 interoperability fixture",
            instructions="Synthetic skills are available through the declared skills extension.",
            on_list_resources=self.list_resources,
            on_read_resource=self.read_resource,
        )
        self.server.extensions = {EXTENSION: {"directoryRead": True}}
        self.server.add_request_handler("skills/list", SkillsListParams, self.list_skills)
        self.server.add_request_handler("skills/get", SkillsGetParams, self.get_skill)
        self.server.add_request_handler("resources/directory/read", DirectoryReadParams, self.read_directory)

    def event(self, method: str, **details: Any) -> None:
        if not self.events_path:
            return
        with open(self.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"method": method, **details}, sort_keys=True) + "\n")

    async def list_skills(self, _ctx: Any, params: SkillsListParams) -> ExtensionResult:
        self.event("skills/list", cursor=params.cursor)
        if params.cursor is None:
            return ExtensionResult(skills=[skill_entry("skill://portable-demo/SKILL.md")], nextCursor="page-2")
        if params.cursor == "page-2":
            return ExtensionResult(
                skills=[skill_entry("skill://catalog-b/portable-demo/SKILL.md")], nextCursor="page-3")
        if params.cursor == "page-3":
            bad_child = skill_entry("skill://bad/bad-parent/bad-child/SKILL.md")
            bad_child["resources"][0]["digest"] = "sha256:" + "0" * 64
            return ExtensionResult.model_validate({"skills": [
                    skill_entry("fixture-skill://catalog-c/portable-alt/SKILL.md"),
                    skill_entry("skill://nested/parent/SKILL.md"),
                    skill_entry("skill://nested/parent/child/SKILL.md"),
                    skill_entry("skill://bad/bad-parent/SKILL.md"),
                    bad_child,
                ]})
        raise MCPError(code=-32602, message="Invalid params")

    async def get_skill(self, _ctx: Any, params: SkillsGetParams) -> ExtensionResult:
        self.event("skills/get", uri=params.uri)
        if params.uri not in FRONTMATTER:
            raise MCPError(code=-32602, message="Invalid params")
        return ExtensionResult(skill=skill_entry(params.uri))

    async def list_resources(self, _ctx: Any, _params: Any) -> types.ListResourcesResult:
        self.event("resources/list")
        return types.ListResourcesResult(resources=[])

    async def read_resource(self, _ctx: Any, params: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        uri = str(params.uri)
        self.event("resources/read", uri=uri)
        item = FILES.get(uri)
        if item is None:
            raise MCPError(code=-32602, message="Invalid params")
        raw, mime = item
        if mime.startswith("text/"):
            content: types.TextResourceContents | types.BlobResourceContents = types.TextResourceContents(
                uri=uri, mimeType=mime, text=raw.decode("utf-8"))
        else:
            content = types.BlobResourceContents(uri=uri, mimeType=mime, blob=base64.b64encode(raw).decode("ascii"))
        return types.ReadResourceResult(contents=[content])

    async def read_directory(self, _ctx: Any, params: DirectoryReadParams) -> ExtensionResult:
        uri = params.uri.rstrip("/")
        self.event("resources/directory/read", uri=uri, cursor=params.cursor)
        prefix = uri + "/"
        children: dict[str, dict[str, Any]] = {}
        for item, (raw, mime) in FILES.items():
            if not item.startswith(prefix):
                continue
            remainder = item[len(prefix):]
            first = remainder.split("/", 1)[0]
            child_uri = prefix + first
            is_dir = "/" in remainder
            children[child_uri] = {
                "uri": child_uri,
                "name": first,
                "mimeType": "inode/directory" if is_dir else mime,
                **({} if is_dir else {"size": len(raw)}),
            }
        if not children:
            raise MCPError(code=-32602, message="Invalid params")
        return ExtensionResult(resources=[children[key] for key in sorted(children)])

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="append JSONL request observations to this runtime path")
    args = parser.parse_args()
    anyio.run(Fixture(args.events).run)


if __name__ == "__main__":
    main()
