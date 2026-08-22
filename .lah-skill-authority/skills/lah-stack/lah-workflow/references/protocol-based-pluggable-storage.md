# Protocol-based Pluggable Storage Pattern

## Overview

Add a durable backend to an existing in-memory storage class that satisfies
a `typing.Protocol`. Callers stay unchanged because they depend only on the
protocol, not the concrete class.

## Step 1: Define the Protocol (in-storage already)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EvidenceStorage(Protocol):
    def store(self, record: EvidenceRecord) -> bool: ...
    def exists(self, canonical_url: str, content_hash: str) -> bool: ...
    def all(self) -> list[EvidenceRecord]: ...
    def count(self) -> int: ...
    def find(self, canonical_url: str) -> list[EvidenceRecord]: ...
```

All implementations must satisfy this protocol.

## Step 2: Keep InMemoryEvidenceStorage unchanged

## Step 3: Add JSONLEvidenceStorage

Key implementation decisions:

1. **Append-only JSONL.** Each `store()` appends one JSON line. The file is
   never rewritten. Dedup checks happen via an in-memory index rebuilt on
   load.

2. **In-memory index on init.** `_load()` reads the entire JSONL file into
   a `dict[str, list[EvidenceRecord]]` keyed by `canonical_url`. This gives
   O(1) dedup checks.

3. **Fail-closed on corruption.** If any line has invalid JSON or missing
   required fields, the constructor raises `ValueError`. Blank lines are
   silently skipped.

4. **Serialization round-trip.** The dataclass gets `to_dict()` and
   `from_dict()` methods. `from_dict()` validates required fields
   (`canonical_url`, `content_hash`, `content`) and raises `ValueError` on
   incomplete records.

### Implementation template

```python
class JSONLEvidenceStorage:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._records: dict[str, list[EvidenceRecord]] = {}
        self._load()

    def store(self, record: EvidenceRecord) -> bool:
        if self.exists(record.canonical_url, record.content_hash):
            return False
        self._append_line(record)
        self._records.setdefault(record.canonical_url, []).append(record)
        return True

    def exists(self, canonical_url: str, content_hash: str) -> bool:
        records_for_url = self._records.get(canonical_url, [])
        return any(r.content_hash == content_hash for r in records_for_url)

    def all(self) -> list[EvidenceRecord]:
        return [r for records in self._records.values() for r in records]

    def count(self) -> int:
        return sum(len(rs) for rs in self._records.values())

    def find(self, canonical_url: str) -> list[EvidenceRecord]:
        return list(reversed(self._records.get(canonical_url, [])))

    def _load(self) -> None:
        self._records = {}
        if not self._path.exists():
            return
        raw = self._path.read_text("utf-8")
        for i, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL at {self._path} line {i}: {exc}") from exc
            record = EvidenceRecord.from_dict(data)
            self._records.setdefault(record.canonical_url, []).append(record)

    def _append_line(self, record: EvidenceRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
```

### Serialization on the dataclass

```python
@dataclass(frozen=True)
class EvidenceRecord:
    SCHEMA_VERSION: ClassVar[str] = "evidence_record.v1"

    canonical_url: str
    content_hash: str
    content: str
    title: str
    metadata: dict = field(default_factory=dict)
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    collector: str = ""

    def to_dict(self) -> dict:
        return {
            "__schema_version__": self.SCHEMA_VERSION,
            "canonical_url": self.canonical_url,
            "content_hash": self.content_hash,
            "content": self.content,
            "title": self.title,
            "metadata": self.metadata,
            "collected_at": self.collected_at,
            "collector": self.collector,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceRecord:
        required = {"canonical_url", "content_hash", "content"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"EvidenceRecord.from_dict missing required fields: {missing}"
            )
        return cls(
            canonical_url=str(data["canonical_url"]),
            content_hash=str(data["content_hash"]),
            content=str(data["content"]),
            title=str(data.get("title", "")),
            metadata=dict(data.get("metadata", {})),
            collected_at=str(data.get("collected_at", "")),
            collector=str(data.get("collector", "")),
        )
```

## Tests

Key test categories for the durable backend:

| Category | Example |
|----------|---------|
| Protocol compliance | `isinstance(storage, EvidenceStorage)` |
| CRUD | store, exists, all, find, count |
| Persistence across reload | Write records, create fresh instance on same file, verify count |
| Dedup across reload | Duplicate stored before reload is still duplicate after |
| Append semantics | Second instance appends, doesn't overwrite |
| Corrupted input | `ValueError` on corrupt JSON line |
| Invalid shape | `ValueError` on missing required fields |
| Blank line tolerance | Empty lines are skipped |
| New file is empty | Non-existent file = empty, not error |
| Tempdir cleanup | `tempfile.NamedTemporaryFile` auto-cleanup |

### Dedup key semantics

Dedup uses the **combination of `canonical_url` and `content_hash`**:
- Same URL + same hash → DUPLICATE (skipped)
- Same URL + different hash → NEW (content updated)
- Different URL + same hash → NEW (same content elsewhere)

## Decision: JSONL vs SQLite for this pattern

| Factor | JSONL | stdlib SQLite |
|--------|-------|---------------|
| Dependencies | `json` only (stdlib) | `sqlite3` only (stdlib) |
| Human-readable | Yes — `cat`, `grep`, `wc -l` | No — requires SQL or tool |
| Dict fields | Native JSON serialization | Needs JSON column or separate table |
| Schema migrations | Never needed | Required on every schema change |
| Append/overwrite semantics | Append-only by design | INSERT (needs cleanup) |
| Data integrity on partial write | Last line may be partial (on crash) | WAL protects against partial writes |
| Concurrent access | Unsafe | Safer (but not full ACID in concurrent mode) |

**Prefer JSONL** for: local-only, single-process, append-heavy workloads
where human readability matters.

**Prefer SQLite** for: concurrent access, complex queries, transactional
guarantees, or data exceeding tens of MB.
