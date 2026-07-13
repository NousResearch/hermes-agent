# Note Capture Runtime Event Schema

This document defines the corrected runtime event shape for the canonical
note-capture projection flow.

Use it when aligning Hermes runtime capture code, runtime memories, and
vault-routing instructions with the repo-side projection contract.

## Design Rules

Keep these meanings separate:

- `target_id`: stable canonical identity
- `resolved_target_path`: resolved live target-relative path
- `staged_relative_path`: staged artifact location under trusted runtime storage
- `target_status`: target lifecycle state
- `projection.*.state`: per-store projection/materialization state

Do not:

- use a folder path as `target_id`
- use a staging file path as `resolved_target_path`
- use `staged` as `target_status`
- collapse target lifecycle and projection state into one field

## Corrected Example Event

```json
{
  "event_id": "ce_20260713_rosmcp_001",
  "capture_model": "canonical_event_log",
  "event_type": "note_capture",
  "captured_at": "2026-07-13T12:00:00Z",
  "capture_source": "hermes-agent:slack",
  "content_type": "knowledge_reference",
  "content_class": "report",
  "title": "Return on Security MCP - Cyber VC Integration Report",
  "source": {
    "origin": "slack",
    "author": "NeilRobinson",
    "original_filename": "2026-07-13 ROS MCP Cyber VC Integration Report.md"
  },
  "routing": {
    "target": "3.Areas/cyber futures frontier",
    "target_id": "vault.second_brain.areas.cyber_futures_frontier",
    "target_class": "obsidian_vault",
    "logical_path": "areas/cyber_futures_frontier",
    "display_path": "3.Areas/cyber futures frontier",
    "resolved_target_path": "3.Areas/cyber futures frontier",
    "target_status": "active",
    "trust_boundary": "icloud_obsidian",
    "confidence": 0.93,
    "reason": "Cyber VC workflow report belongs with the existing cyber futures frontier operating area."
  },
  "routing_review": {
    "uncertainty": "low",
    "alternative_targets": [
      "4.Resources/knowledge",
      "4.Resources/Content Library",
      "1.Inbox"
    ],
    "rejected_alternatives": {
      "4.Resources/knowledge": "Too general; this is specific to Cyber Future Frontier operating context.",
      "4.Resources/Content Library": "Not consumable external content.",
      "1.Inbox": "Routing confidence is high enough not to defer."
    }
  },
  "canonical_content": {
    "format": "markdown",
    "filename": "2026-07-13 ROS MCP Cyber VC Integration Report.md"
  },
  "projection": {
    "stores": {
      "vault": {
        "state": "pending",
        "target_relative_path": "3.Areas/cyber futures frontier/2026-07-13 ROS MCP Cyber VC Integration Report.md",
        "staged_relative_path": "staging/vault/ce_20260713_rosmcp_001/3.Areas/cyber futures frontier/2026-07-13 ROS MCP Cyber VC Integration Report.md"
      },
      "second_brain": {
        "state": "pending",
        "target_relative_path": "runtime-source/3.Areas/cyber futures frontier/2026-07-13 ROS MCP Cyber VC Integration Report.md",
        "staged_relative_path": "staging/second_brain/ce_20260713_rosmcp_001/runtime-source/3.Areas/cyber futures frontier/2026-07-13 ROS MCP Cyber VC Integration Report.md"
      }
    }
  },
  "sync_contract": {
    "capture_model": "canonical_event_log",
    "write_policy": "trusted_staging_only",
    "projection_status_source": "structured_status",
    "stores": [
      "vault",
      "second_brain"
    ]
  }
}
```

## JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CanonicalNoteCaptureEvent",
  "type": "object",
  "required": [
    "event_id",
    "capture_model",
    "event_type",
    "captured_at",
    "title",
    "routing",
    "projection",
    "sync_contract"
  ],
  "properties": {
    "event_id": { "type": "string", "minLength": 1 },
    "capture_model": { "const": "canonical_event_log" },
    "event_type": { "const": "note_capture" },
    "captured_at": { "type": "string", "format": "date-time" },
    "capture_source": { "type": "string" },
    "content_type": { "type": "string" },
    "content_class": { "type": "string" },
    "title": { "type": "string", "minLength": 1 },
    "source": {
      "type": "object",
      "properties": {
        "origin": { "type": "string" },
        "author": { "type": "string" },
        "original_filename": { "type": "string" }
      },
      "additionalProperties": true
    },
    "routing": {
      "type": "object",
      "required": [
        "target",
        "target_id",
        "target_class",
        "logical_path",
        "display_path",
        "resolved_target_path",
        "target_status",
        "trust_boundary"
      ],
      "properties": {
        "target": { "type": "string", "minLength": 1 },
        "target_id": { "type": "string", "minLength": 1 },
        "target_class": {
          "type": "string",
          "enum": ["obsidian_vault", "filesystem"]
        },
        "logical_path": { "type": "string", "minLength": 1 },
        "display_path": { "type": "string", "minLength": 1 },
        "resolved_target_path": { "type": "string", "minLength": 1 },
        "target_status": {
          "type": "string",
          "enum": ["active", "deferred", "pending_migration", "unavailable"]
        },
        "trust_boundary": { "type": "string", "minLength": 1 },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "reason": { "type": "string" }
      },
      "additionalProperties": true
    },
    "routing_review": {
      "type": "object",
      "properties": {
        "uncertainty": { "type": "string" },
        "alternative_targets": {
          "type": "array",
          "items": { "type": "string" }
        },
        "rejected_alternatives": {
          "type": "object",
          "additionalProperties": { "type": "string" }
        }
      },
      "additionalProperties": true
    },
    "canonical_content": {
      "type": "object",
      "properties": {
        "format": { "type": "string" },
        "filename": { "type": "string" }
      },
      "additionalProperties": true
    },
    "projection": {
      "type": "object",
      "required": ["stores"],
      "properties": {
        "stores": {
          "type": "object",
          "minProperties": 1,
          "additionalProperties": {
            "type": "object",
            "required": ["state", "target_relative_path", "staged_relative_path"],
            "properties": {
              "state": {
                "type": "string",
                "enum": ["pending", "projected", "failed", "superseded"]
              },
              "target_relative_path": { "type": "string", "minLength": 1 },
              "staged_relative_path": { "type": "string", "minLength": 1 }
            },
            "additionalProperties": true
          }
        }
      },
      "additionalProperties": true
    },
    "sync_contract": {
      "type": "object",
      "required": ["capture_model", "write_policy", "projection_status_source", "stores"],
      "properties": {
        "capture_model": { "const": "canonical_event_log" },
        "write_policy": { "const": "trusted_staging_only" },
        "projection_status_source": { "type": "string" },
        "stores": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        }
      },
      "additionalProperties": true
    }
  },
  "additionalProperties": true
}
```

## Pydantic Model

```python
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TargetClass = Literal["obsidian_vault", "filesystem"]
TargetStatus = Literal["active", "deferred", "pending_migration", "unavailable"]
ProjectionState = Literal["pending", "projected", "failed", "superseded"]


class SourceRef(BaseModel):
    origin: Optional[str] = None
    author: Optional[str] = None
    original_filename: Optional[str] = None


class RoutingMetadata(BaseModel):
    target: str
    target_id: str
    target_class: TargetClass
    logical_path: str
    display_path: str
    resolved_target_path: str
    target_status: TargetStatus
    trust_boundary: str
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    reason: Optional[str] = None


class RoutingReview(BaseModel):
    uncertainty: Optional[str] = None
    alternative_targets: List[str] = Field(default_factory=list)
    rejected_alternatives: Dict[str, str] = Field(default_factory=dict)


class CanonicalContent(BaseModel):
    format: Optional[str] = None
    filename: Optional[str] = None


class ProjectionStoreEntry(BaseModel):
    state: ProjectionState
    target_relative_path: str
    staged_relative_path: str


class ProjectionStateBlock(BaseModel):
    stores: Dict[str, ProjectionStoreEntry]


class SyncContract(BaseModel):
    capture_model: Literal["canonical_event_log"]
    write_policy: Literal["trusted_staging_only"]
    projection_status_source: str
    stores: List[str]


class CanonicalNoteCaptureEvent(BaseModel):
    event_id: str
    capture_model: Literal["canonical_event_log"]
    event_type: Literal["note_capture"]
    captured_at: str
    capture_source: Optional[str] = None
    content_type: Optional[str] = None
    content_class: Optional[str] = None
    title: str
    source: Optional[SourceRef] = None
    routing: RoutingMetadata
    routing_review: Optional[RoutingReview] = None
    canonical_content: Optional[CanonicalContent] = None
    projection: ProjectionStateBlock
    sync_contract: SyncContract
```

## Runtime Prompt Addendum

Use this short rule block in runtime prompts or memories when the full schema
is too heavy:

```md
For generic note capture, preserve four separate concepts:
- canonical identity (`target_id`)
- resolved live target path (`resolved_target_path`)
- staged artifact path (`staged_relative_path`)
- projection state (`pending` / `projected` / `failed`)

Do not use a folder path as `target_id`.
Do not use a staging path as `resolved_target_path`.
Do not use `staged` as `target_status`.
Use `target_status` only for target lifecycle:
`active`, `deferred`, `pending_migration`, `unavailable`.
```
