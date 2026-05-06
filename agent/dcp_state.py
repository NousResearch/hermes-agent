"""State model for the DCP context engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class CompressionBlock:
    block_id: int
    run_id: int
    mode: Literal["range", "message"]
    topic: str
    summary: str
    active: bool = True
    start_ref: str | None = None
    end_ref: str | None = None
    message_refs: list[str] = field(default_factory=list)
    included_block_ids: list[int] = field(default_factory=list)
    consumed_block_ids: list[int] = field(default_factory=list)
    created_at: float = 0.0
    deactivated_at: float | None = None
    deactivated_by_block_id: int | None = None

    @property
    def ref(self) -> str:
        return f"b{self.block_id}"


@dataclass(slots=True)
class DCPSessionState:
    session_id: str | None = None
    next_message_ref: int = 1
    next_block_id: int = 1
    next_run_id: int = 1
    ref_by_message_key: dict[str, str] = field(default_factory=dict)
    message_key_by_ref: dict[str, str] = field(default_factory=dict)
    index_by_ref: dict[str, int] = field(default_factory=dict)
    blocks_by_id: dict[int, CompressionBlock] = field(default_factory=dict)
    active_block_ids: set[int] = field(default_factory=set)
    last_prompt_tokens: int = 0
    last_user_turn_index: int = 0
    turns_since_last_compress: int = 0
    messages_since_last_user: int = 0
    manual_mode: bool | Literal["compress-pending"] = False
    pending_manual_focus: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def new_message_ref(self) -> str:
        ref = f"m{self.next_message_ref:04d}"
        self.next_message_ref += 1
        return ref

    def new_block_id(self) -> int:
        block_id = self.next_block_id
        self.next_block_id += 1
        return block_id

    def new_run_id(self) -> int:
        run_id = self.next_run_id
        self.next_run_id += 1
        return run_id

    def active_blocks(self) -> list[CompressionBlock]:
        return [
            self.blocks_by_id[block_id]
            for block_id in sorted(self.active_block_ids)
            if block_id in self.blocks_by_id and self.blocks_by_id[block_id].active
        ]
