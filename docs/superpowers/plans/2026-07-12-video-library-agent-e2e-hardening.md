# Video Library Agent E2E Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current named video library safe to rescan, fully operable through Agent-facing tools, and verifiably able to hand three matched local shots to MoneyPrinterTurbo for a playable MP4.

**Architecture:** Keep `capabilities/video_library` responsible for authorized discovery, clip analysis, indexing, search, lazy materialization, and renderer-neutral timelines. Extend the existing MoneyPrinter MCP edge instead of Hermes Agent Core: the Agent searches a named library, creates a provenance-preserving timeline, caches materialized shots into MoneyPrinter's local whitelist, and submits the existing render task. A deterministic acceptance runner exercises the same contracts with local test audio, so no paid TTS call is needed.

**Tech Stack:** Python 3.11+, SQLite, FFmpeg/ffprobe, pytest, FastAPI capability routes, Hermes MCP tool wrappers, MoneyPrinterTurbo FastAPI sidecar, MoviePy/FFmpeg.

---

## Scope and file map

This is the first independently testable delivery from the broader controllable one-click-production design. It does not add structured director intent, vector retrieval, or sentence-level audio timing.

**Files to modify**

- `capabilities/video_library/config.py` — define and test the single rule for generated library directories.
- `capabilities/video_library/batch.py` — exclude generated outputs during discovery; expose truthful status and safe derived-record pruning.
- `capabilities/video_library/store.py` — delete one asset record transactionally without deleting any source file.
- `capabilities/video_library/cli.py` — expose dry-run-first `prune-derived` maintenance.
- `capabilities/moneyprinter/mcp/tools.py` — make named-library import/analyze/timeline operations complete and allow Agent local-material rendering.
- `apps/desktop/src/app/video-studio/moneyprinter-client.ts` — add `failed_assets` to the status contract.
- `apps/desktop/src/app/video-studio/unified-material-library-panel.tsx` — distinguish failed assets from failed semantic clips.
- `skills/local-video-asset-indexer/SKILL.md` — teach the Hermes Agent to audit derived pollution before scanning.

**Files to create**

- `scripts/video_library_acceptance.py` — deterministic read/plan/cache/render acceptance runner.
- `tests/capabilities/test_video_library_acceptance.py` — contract test for the runner without starting a real sidecar.

**Existing tests to extend**

- `tests/capabilities/test_video_library_config.py`
- `tests/capabilities/test_video_library_batch.py`
- `tests/capabilities/test_video_library_store.py`
- `tests/capabilities/test_video_library_cli.py`
- `tests/capabilities/test_moneyprinter_mcp_tools.py`
- `apps/desktop/src/app/video-studio/unified-material-library-panel.test.tsx`

## Task 1: Prevent generated-output recursion

**Files:**

- Modify: `capabilities/video_library/config.py:11-43`
- Modify: `capabilities/video_library/batch.py:61-75`
- Test: `tests/capabilities/test_video_library_config.py`
- Test: `tests/capabilities/test_video_library_batch.py`

- [ ] **Step 1: Write the failing generated-path classifier test**

Add to `tests/capabilities/test_video_library_config.py`:

```python
from capabilities.video_library.config import is_generated_library_path


def test_generated_library_path_only_matches_managed_output_directories(tmp_path):
    root = tmp_path / "牛肉面资产库"
    library = VideoLibraryConfig(
        id="beef-noodle",
        mode="linked",
        name="牛肉面资产库",
        root=root,
        source_roots=(root,),
        taxonomy="beef-noodle-v1",
    )

    assert is_generated_library_path(library, root / "02_精选镜头" / "clip.mp4") is True
    assert is_generated_library_path(library, root / "03_关键帧" / "frame.jpg") is True
    assert is_generated_library_path(library, root / "04_素材分析" / "report.md") is True
    assert is_generated_library_path(library, root / "timelines" / "timeline.json") is True
    assert is_generated_library_path(library, root / ".hermes-assets" / "managed-assets" / "copy.mp4") is True
    assert is_generated_library_path(library, root / "01_原始素材" / "raw.mp4") is False
    assert is_generated_library_path(library, tmp_path / "outside.mp4") is False
```

- [ ] **Step 2: Run the classifier test and verify it fails**

Run:

```bash
.venv/bin/pytest tests/capabilities/test_video_library_config.py::test_generated_library_path_only_matches_managed_output_directories -q
```

Expected: collection fails because `is_generated_library_path` does not exist.

- [ ] **Step 3: Implement the generated-path classifier**

Add to `capabilities/video_library/config.py`:

```python
GENERATED_LIBRARY_DIRECTORIES = frozenset(
    {"02_精选镜头", "03_关键帧", "04_素材分析", "timelines", ".hermes-assets"}
)


def is_generated_library_path(library: VideoLibraryConfig, path: Path | str) -> bool:
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(library.root.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] in GENERATED_LIBRARY_DIRECTORIES
```

Export both names from `__all__`.

- [ ] **Step 4: Write the failing recursive-discovery test**

Add to `tests/capabilities/test_video_library_batch.py`:

```python
def test_scan_excludes_generated_video_directories_when_library_root_is_a_source(tmp_path):
    root = tmp_path / "牛肉面资产库"
    (root / "01_原始素材").mkdir(parents=True)
    (root / "02_精选镜头" / "asset-1").mkdir(parents=True)
    (root / ".hermes-assets" / "managed-assets").mkdir(parents=True)
    (root / "01_原始素材" / "raw.mp4").write_bytes(b"raw")
    (root / "02_精选镜头" / "asset-1" / "clip.mp4").write_bytes(b"derived")
    (root / ".hermes-assets" / "managed-assets" / "copy.mp4").write_bytes(b"derived")
    library = VideoLibraryConfig(
        id="beef-noodle",
        mode="linked",
        name="牛肉面资产库",
        root=root,
        source_roots=(root,),
        taxonomy="beef-noodle-v1",
    )

    result = VideoLibraryBatchRunner(library).scan(dry_run=True)

    assert result.total == 1
```

- [ ] **Step 5: Filter generated paths before authorization**

Import `is_generated_library_path` in `batch.py` and change `_discover()`:

```python
for candidate in sorted(source_root.rglob("*"), key=lambda path: str(path).casefold()):
    if is_generated_library_path(self.library, candidate):
        continue
    if candidate.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        continue
    try:
        files.append(resolve_source_path(self.library, candidate))
    except (FileNotFoundError, ValueError) as exc:
        errors.append({"file": str(candidate), "message": str(exc), "stage": "authorization"})
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
.venv/bin/pytest tests/capabilities/test_video_library_config.py tests/capabilities/test_video_library_batch.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add capabilities/video_library/config.py capabilities/video_library/batch.py tests/capabilities/test_video_library_config.py tests/capabilities/test_video_library_batch.py
git commit -m "fix(video-library): exclude generated outputs from scans"
```

## Task 2: Audit and safely prune polluted database records

**Files:**

- Modify: `capabilities/video_library/store.py:260-274`
- Modify: `capabilities/video_library/batch.py:179-219`
- Modify: `capabilities/video_library/cli.py:13-49`
- Modify: `apps/desktop/src/app/video-studio/moneyprinter-client.ts:65-85`
- Modify: `apps/desktop/src/app/video-studio/unified-material-library-panel.tsx:210-225`
- Test: `tests/capabilities/test_video_library_store.py`
- Test: `tests/capabilities/test_video_library_batch.py`
- Test: `tests/capabilities/test_video_library_cli.py`
- Test: `apps/desktop/src/app/video-studio/unified-material-library-panel.test.tsx`

- [ ] **Step 1: Write the failing metadata-only deletion test**

Add to `tests/capabilities/test_video_library_store.py`:

```python
def test_delete_asset_record_cascades_metadata_without_deleting_source(tmp_path):
    root = tmp_path / "library"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    store = VideoLibraryStore(root=root)
    asset = store.import_asset(source, source_mode="linked", library_id="beef-noodle")

    deleted = store.delete_asset_record(asset["id"])

    assert deleted is True
    assert store.get_asset(asset["id"]) is None
    assert source.is_file()
```

- [ ] **Step 2: Implement metadata-only deletion**

Add to `VideoLibraryStore`:

```python
def delete_asset_record(self, asset_id: str) -> bool:
    with self.connect() as conn:
        cursor = conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        return cursor.rowcount == 1
```

Do not unlink `source_path`, `managed_path`, keyframes, or materialized clips in this method.

- [ ] **Step 3: Write pruning and truthful-status tests**

Add to `tests/capabilities/test_video_library_batch.py` using a monkeypatched `resolve_library_config` and a real temporary store:

```python
def test_prune_derived_assets_is_dry_run_by_default(tmp_path, monkeypatch):
    library = _library(tmp_path, [])
    derived = library.root / "02_精选镜头" / "old.mp4"
    derived.parent.mkdir(parents=True)
    derived.write_bytes(b"derived")
    service = build_library_service(library)
    asset = service.store.import_asset(derived, source_mode="linked", library_id=library.id)
    monkeypatch.setattr(batch, "resolve_library_config", lambda _library_id: library)

    preview = batch.prune_derived_assets(library.id)

    assert preview["matched"] == 1
    assert preview["deleted"] == 0
    assert service.store.get_asset(asset["id"]) is not None

    executed = batch.prune_derived_assets(library.id, execute=True)
    assert executed["deleted"] == 1
    assert service.store.get_asset(asset["id"]) is None
    assert derived.is_file()
```

Also assert `library_status()` exposes both `failed_assets` and `semantic_failed` rather than hiding failed assets behind the clip count.

- [ ] **Step 4: Implement derived-record audit and status fields**

Add to `batch.py`:

```python
def prune_derived_assets(library_id: str, *, execute: bool = False) -> dict[str, Any]:
    library = resolve_library_config(library_id)
    service = build_library_service(library)
    matched = [
        asset
        for asset in service.store.list_assets()
        if is_generated_library_path(library, str(asset.get("source_path") or ""))
    ]
    deleted = 0
    if execute:
        for asset in matched:
            deleted += int(service.store.delete_asset_record(str(asset["id"])))
    return {
        "deleted": deleted,
        "execute": execute,
        "library_id": library.id,
        "matched": len(matched),
        "records": [
            {"asset_id": asset["id"], "source_path": asset["source_path"], "status": asset["status"]}
            for asset in matched
        ],
    }
```

In `library_status()`, compute:

```python
assets = service.store.list_assets()
semantic_failed = sum(1 for clip in clips if clip.get("status") == "semantic_failed")
return {
    "assets": len(assets),
    "clips": len(clips),
    "failed": semantic_failed,
    "failed_assets": sum(1 for asset in assets if asset.get("status") == "failed"),
    "semantic_failed": semantic_failed,
    # retain the existing fields
}
```

- [ ] **Step 5: Add dry-run-first CLI pruning**

Extend `_parser()`:

```python
prune = commands.add_parser("prune-derived")
prune.add_argument("--library", required=True)
prune.add_argument("--execute", action="store_true")
```

Dispatch it with:

```python
elif args.command == "prune-derived":
    payload = prune_derived_assets(args.library, execute=bool(args.execute))
```

Add a CLI test proving that omission of `--execute` forwards `execute=False`.

- [ ] **Step 6: Display honest status in Desktop**

Add `failed_assets: number` and optional `semantic_failed?: number` to `VideoLibraryStatus`. Change the status line to:

```tsx
素材 {status?.assets ?? '--'} · 镜头 {status?.clips ?? '--'} · 失败素材 {status?.failed_assets ?? '--'} ·
语义失败 {status?.semantic_failed ?? status?.failed ?? '--'} · 低置信度 {status?.low_confidence ?? '--'}
```

Update the component fixture and assertion accordingly.

- [ ] **Step 7: Run Task 2 tests**

```bash
.venv/bin/pytest tests/capabilities/test_video_library_store.py tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_cli.py -q
npm exec vitest run apps/desktop/src/app/video-studio/unified-material-library-panel.test.tsx
```

Expected: Python and Vitest suites pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add capabilities/video_library/store.py capabilities/video_library/batch.py capabilities/video_library/cli.py tests/capabilities/test_video_library_store.py tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_cli.py apps/desktop/src/app/video-studio/moneyprinter-client.ts apps/desktop/src/app/video-studio/unified-material-library-panel.tsx apps/desktop/src/app/video-studio/unified-material-library-panel.test.tsx
git commit -m "feat(video-library): audit and prune derived records"
```

## Task 3: Complete the named-library Agent tool contract

**Files:**

- Modify: `capabilities/moneyprinter/mcp/tools.py:375-438`
- Test: `tests/capabilities/test_moneyprinter_mcp_tools.py:127-168`

- [ ] **Step 1: Write failing named import, analyze, status, and timeline tests**

Add tests that monkeypatch video-library adapter functions and assert exact bodies:

```python
def test_mcp_video_library_named_operations_forward_library_id(monkeypatch):
    from capabilities.video_library import adapter as video_library_adapter

    calls = []
    monkeypatch.setattr(
        video_library_adapter,
        "import_asset_data",
        lambda body: (calls.append(("import", body)) or (200, {"ok": True, "data": {}, "error": None})),
    )
    monkeypatch.setattr(
        video_library_adapter,
        "analyze_asset_data",
        lambda asset_id, body: (
            calls.append(("analyze", asset_id, body)) or (200, {"ok": True, "data": {}, "error": None})
        ),
    )
    monkeypatch.setattr(
        video_library_adapter,
        "create_timeline_data",
        lambda body: (calls.append(("timeline", body)) or (200, {"ok": True, "data": {}, "error": None})),
    )
    monkeypatch.setattr(
        video_library_adapter,
        "library_status_data",
        lambda library_id: (200, {"ok": True, "data": {"library_id": library_id}, "error": None}),
    )

    mp_tools.video_library_import_asset("/tmp/source.mp4", library_id="beef-noodle")
    mp_tools.video_library_analyze_asset("asset-1", library_id="beef-noodle")
    mp_tools.video_library_create_timeline(
        ["clip-1"],
        library_id="beef-noodle",
        script=[{"id": "segment-1", "text": "顾客吃面"}],
    )
    status = json.loads(mp_tools.video_library_get_status("beef-noodle"))

    assert calls[0] == ("import", {"sourcePath": "/tmp/source.mp4", "libraryId": "beef-noodle"})
    assert calls[1][2]["libraryId"] == "beef-noodle"
    assert calls[2][1]["libraryId"] == "beef-noodle"
    assert calls[2][1]["script"][0]["id"] == "segment-1"
    assert status["data"]["library_id"] == "beef-noodle"
```

- [ ] **Step 2: Run the focused MCP test and verify it fails**

```bash
.venv/bin/pytest tests/capabilities/test_moneyprinter_mcp_tools.py::test_mcp_video_library_named_operations_forward_library_id -q
```

Expected: failure because the signatures/status tool do not exist.

- [ ] **Step 3: Extend the tool signatures without breaking legacy calls**

Use these signatures and bodies:

```python
def video_library_import_asset(source_path: str, library_id: str = "") -> str:
    body = {"sourcePath": source_path}
    if library_id:
        body["libraryId"] = library_id
    status, payload = adapter.import_asset_data(body)


def video_library_analyze_asset(
    asset_id: str,
    threshold: float = 0.32,
    min_clip_seconds: float = 1.0,
    fallback_clip_seconds: float = 5.0,
    library_id: str = "",
) -> str:
    body = {
        "fallbackClipSeconds": fallback_clip_seconds,
        "minClipSeconds": min_clip_seconds,
        "threshold": threshold,
    }
    if library_id:
        body["libraryId"] = library_id
    status, payload = adapter.analyze_asset_data(asset_id, body)


def video_library_get_status(library_id: str) -> str:
    status, payload = adapter.library_status_data(library_id)
    return _json_result(payload, status=status)


def video_library_create_timeline(
    clip_ids: list[str],
    aspect: str = "9:16",
    library_id: str = "",
    script: Optional[list[dict[str, Any]]] = None,
) -> str:
    body: dict[str, Any] = {"aspect": aspect, "clipIds": clip_ids, "script": script or []}
    if library_id:
        body["libraryId"] = library_id
    status, payload = adapter.create_timeline_data(body)
```

Register `video_library_get_status` in `TOOL_SPECS`. Update tool descriptions to state that `library_id` is required for named Obsidian libraries.

- [ ] **Step 4: Run MCP regression tests**

```bash
.venv/bin/pytest tests/capabilities/test_moneyprinter_mcp_tools.py -q
```

Expected: all tests pass, including the legacy import test that omits `library_id`.

- [ ] **Step 5: Commit Task 3**

```bash
git add capabilities/moneyprinter/mcp/tools.py tests/capabilities/test_moneyprinter_mcp_tools.py
git commit -m "feat(video-library): complete named Agent tools"
```

## Task 4: Let the Agent cache selected shots and submit a local render

**Files:**

- Modify: `capabilities/moneyprinter/mcp/tools.py:77-130`
- Test: `tests/capabilities/test_moneyprinter_mcp_tools.py`

- [ ] **Step 1: Write the failing cache-tool test**

```python
def test_mcp_cache_local_material_uses_existing_adapter(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "upload_local_material_data",
        lambda body: (
            200,
            {"ok": True, "data": {"material": {"file": body["filename"]}}, "error": None},
        ),
    )

    payload = json.loads(
        mp_tools.moneyprinter_cache_local_material(
            "/vault/02_精选镜头/clip.mp4",
            "beef-noodle-asset-clip.mp4",
        )
    )

    assert payload["data"]["material"]["file"] == "beef-noodle-asset-clip.mp4"
```

- [ ] **Step 2: Implement and register the cache tool**

```python
def moneyprinter_cache_local_material(source_path: str, filename: str) -> str:
    """Copy one selected materialized shot into MoneyPrinter's local whitelist."""
    from capabilities.moneyprinter import adapter

    status, payload = adapter.upload_local_material_data(
        {"filename": filename, "sourcePath": source_path}
    )
    return _json_result(payload, status=status)
```

Register it next to `moneyprinter_generate_video`.

- [ ] **Step 3: Write the failing local-render payload test**

```python
def test_mcp_generate_video_accepts_cached_local_materials(monkeypatch):
    seen = {}

    async def fake_create(body):
        seen.update(body)
        return 200, {"ok": True, "data": {"task": {"id": "task-local"}}, "error": None}

    monkeypatch.setattr(adapter, "create_video_data", fake_create)
    monkeypatch.setattr(mp_tools, "_ensure_service_if_needed", lambda: None)

    payload = json.loads(
        mp_tools.moneyprinter_generate_video(
            video_subject="牛肉面门店",
            video_script="顾客吃面。员工端碗。成品面特写。",
            video_source="local",
            local_materials=["one.mp4", "two.mp4", "three.mp4"],
            match_materials_to_script=True,
            auto_start=False,
            bgm_type="none",
        )
    )

    assert payload["data"]["task_id"] == "task-local"
    assert seen["video_source"] == "local"
    assert [item["url"] for item in seen["video_materials"]] == ["one.mp4", "two.mp4", "three.mp4"]
    assert seen["match_materials_to_script"] is True
```

- [ ] **Step 4: Extend `moneyprinter_generate_video` minimally**

Add parameters:

```python
local_materials: Optional[list[str]] = None,
custom_audio_file: str = "",
match_materials_to_script: bool = False,
```

Add to the request body:

```python
"custom_audio_file": custom_audio_file,
"match_materials_to_script": match_materials_to_script,
```

When `video_source == "local"`, add:

```python
body["video_materials"] = [
    {"duration": 0, "provider": "local", "url": name}
    for name in (local_materials or [])
]
```

The adapter remains responsible for filename sanitization and whitelist existence checks.

- [ ] **Step 5: Run MCP tests**

```bash
.venv/bin/pytest tests/capabilities/test_moneyprinter_mcp_tools.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add capabilities/moneyprinter/mcp/tools.py tests/capabilities/test_moneyprinter_mcp_tools.py
git commit -m "feat(moneyprinter): render Agent-selected local shots"
```

## Task 5: Add a deterministic acceptance runner

**Files:**

- Create: `scripts/video_library_acceptance.py`
- Create: `tests/capabilities/test_video_library_acceptance.py`

- [ ] **Step 1: Write the failing planning-mode acceptance test**

The test must inject a fake client so it runs without real model calls or sidecar state:

```python
from scripts.video_library_acceptance import AcceptanceClient, build_acceptance_plan


class FakeClient(AcceptanceClient):
    def search(self, library_id: str, query: str, limit: int = 5):
        suffix = query.replace(" ", "-")
        return [{
            "id": f"clip-{suffix}",
            "asset_id": f"asset-{suffix}",
            "description": query,
            "quality_score": 0.9,
            "confidence": 0.9,
            "score": 1.0,
            "source_file_path": f"/source/{suffix}.mp4",
        }]


def test_acceptance_plan_selects_one_distinct_shot_per_required_intent():
    plan = build_acceptance_plan(FakeClient(), "beef-noodle")

    assert [row["query"] for row in plan["selections"]] == [
        "前厅顾客吃面",
        "员工端碗上餐",
        "成品牛肉面特写",
    ]
    assert len({row["asset_id"] for row in plan["selections"]}) == 3
```

- [ ] **Step 2: Implement the acceptance plan and CLI**

`scripts/video_library_acceptance.py` must define:

```python
REQUIRED_INTENTS = ("前厅顾客吃面", "员工端碗上餐", "成品牛肉面特写")


class AcceptanceClient:
    def search(self, library_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        status, payload = video_library_adapter.list_clips_data(
            library_id=library_id, query=query, limit=limit
        )
        if status != 200 or not payload.get("ok"):
            raise RuntimeError((payload.get("error") or {}).get("message") or query)
        return list((payload.get("data") or {}).get("clips") or [])


def build_acceptance_plan(client: AcceptanceClient, library_id: str) -> dict[str, Any]:
    used_assets: set[str] = set()
    selections = []
    for query in REQUIRED_INTENTS:
        candidates = client.search(library_id, query, limit=5)
        selected = next((row for row in candidates if row["asset_id"] not in used_assets), None)
        if selected is None:
            raise RuntimeError(f"no distinct candidate for {query}")
        used_assets.add(str(selected["asset_id"]))
        selections.append({"query": query, **selected})
    return {"library_id": library_id, "selections": selections}
```

CLI flags:

- `--library` required;
- `--output` defaults to `<library-root>/04_素材分析/验收/agent-e2e-plan.json`;
- `--render` defaults false;
- `--audio` optional custom audio source;
- `--timeout` defaults 300 seconds.

Without `--render`, write only the plan JSON and do not materialize clips, cache files, start services, or create a task.

- [ ] **Step 3: Write the render-orchestration contract test**

Add this test and make `execute_render()` depend only on the client methods shown here:

```python
from pathlib import Path

from scripts.video_library_acceptance import execute_render


class FakeRenderClient(FakeClient):
    def __init__(self, final_path: Path):
        self.calls = []
        self.final_path = final_path

    def create_timeline(self, library_id, clip_ids, script):
        self.calls.append(("timeline", library_id, clip_ids, script))
        video = [
            {"clipId": clip_id, "file": f"/materialized/{clip_id}.mp4"}
            for clip_id in clip_ids
        ]
        shots = [
            {
                "assetId": f"asset-{index}",
                "clipId": clip_id,
                "libraryId": library_id,
                "sourcePath": f"/source/{clip_id}.mp4",
                "sourceSha256": f"sha-{index}",
            }
            for index, clip_id in enumerate(clip_ids)
        ]
        return {
            "path": "/library/timelines/test.json",
            "timeline": {"tracks": {"video": video}, "shotPlan": shots},
        }

    def cache_material(self, source_path, filename):
        self.calls.append(("cache", source_path, filename))
        return filename

    def cache_audio(self, source_path):
        self.calls.append(("audio", source_path))
        return "acceptance.mp3"

    def create_video(self, body):
        self.calls.append(("render", body))
        return "task-local"

    def get_task(self, task_id):
        self.calls.append(("poll", task_id))
        return {"id": task_id, "status": "complete", "final_video_path": str(self.final_path)}


def test_execute_render_preserves_named_library_order_and_provenance(tmp_path):
    final_path = tmp_path / "final-1.mp4"
    final_path.write_bytes(b"mp4")
    client = FakeRenderClient(final_path)
    plan = build_acceptance_plan(client, "beef-noodle")

    result = execute_render(
        client,
        plan,
        audio_path=tmp_path / "acceptance.mp3",
        timeout_seconds=5,
    )

    timeline_call = next(call for call in client.calls if call[0] == "timeline")
    render_call = next(call for call in client.calls if call[0] == "render")
    assert timeline_call[1] == "beef-noodle"
    assert render_call[1]["video_source"] == "local"
    assert render_call[1]["match_materials_to_script"] is True
    assert [item["url"] for item in render_call[1]["video_materials"]] == [
        call[2] for call in client.calls if call[0] == "cache"
    ]
    assert result["task_id"] == "task-local"
    assert result["timeline_path"] == "/library/timelines/test.json"
    assert result["final_video_path"] == str(final_path)
    assert len(result["selected_sources"]) == 3
```

- [ ] **Step 4: Implement render mode using existing adapters**

Render mode must:

1. call `video_library_adapter.create_timeline_data()` with the selected clip IDs, named library, and three script rows;
2. extract `timeline.tracks.video[].file` and `timeline.shotPlan[]`;
3. cache each file with `moneyprinter_adapter.upload_local_material_data()` using a provenance-derived filename;
4. if `--audio` is supplied, cache it with `upload_custom_audio_data()`;
5. call `moneyprinter_adapter.create_video_data()` with local materials, sequential matching, subtitles off for synthetic audio, BGM none, and one output;
6. poll `moneyprinter_adapter.get_task_data()` until complete/failed/timeout;
7. require that `final_video_path` exists and is non-empty;
8. write the final acceptance JSON atomically.

Do not call MiniMax, Pexels, Pixabay, or Coverr.

- [ ] **Step 5: Run acceptance-runner tests**

```bash
.venv/bin/pytest tests/capabilities/test_video_library_acceptance.py -q
```

Expected: planning and render-orchestration contract tests pass without network access.

- [ ] **Step 6: Commit Task 5**

```bash
git add scripts/video_library_acceptance.py tests/capabilities/test_video_library_acceptance.py
git commit -m "test(video-library): add Agent render acceptance runner"
```

## Task 6: Teach the Hermes indexing Agent the maintenance gate

**Files:**

- Modify: `skills/local-video-asset-indexer/SKILL.md:29-76`
- Test: `tests/capabilities/test_video_library_cli.py`

- [ ] **Step 1: Add the audit command to the skill before scanning**

Insert after status:

````markdown
### 3. 检查派生素材污染

```bash
python -m capabilities.video_library.cli prune-derived --library beef-noodle
```

该命令默认只预览。如果 `matched` 大于 0，先报告记录，确认它们的来源全部位于资产库派生目录，再执行：

```bash
python -m capabilities.video_library.cli prune-derived --library beef-noodle --execute
```

该操作只删除 SQLite 中错误收录的资产记录，不删除任何原片或派生视频文件。
````

Renumber the later steps. Keep dry-run before real scan.

- [ ] **Step 2: Assert the CLI help exposes the safe default**

Add a CLI parser test that verifies `prune-derived --library beef-noodle` dispatches `execute=False` and the `--execute` form dispatches `True`.

- [ ] **Step 3: Run skill-adjacent tests**

```bash
.venv/bin/pytest tests/capabilities/test_video_library_cli.py tests/capabilities/test_video_library_batch.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit Task 6**

```bash
git add skills/local-video-asset-indexer/SKILL.md tests/capabilities/test_video_library_cli.py
git commit -m "docs(video-library): teach Agent derived-record audit"
```

## Task 7: Run the real beef-noodle acceptance test

**Files and state:**

- Read: `/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home/config.yaml`
- Backup/write: `/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/.hermes-assets/index.sqlite`
- Write: `/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/04_素材分析/验收/`
- Write: `external/MoneyPrinterTurbo/storage/local_videos/`
- Write: `external/MoneyPrinterTurbo/storage/custom_audio/`
- Write: `external/MoneyPrinterTurbo/storage/tasks/<task-id>/`

- [ ] **Step 1: Verify live runtime ownership**

```bash
ps -axo pid,ppid,command | rg "Hermes Dev|apps/desktop.*electron|hermes serve"
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

Expected: current repository owns the development desktop process and MoneyPrinter sidecar. Do not kill unrelated listeners.

- [ ] **Step 2: Run focused automated regression**

```bash
.venv/bin/pytest tests/capabilities/test_video_library_config.py tests/capabilities/test_video_library_store.py tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_cli.py tests/capabilities/test_video_library_e2e.py tests/capabilities/test_video_library_acceptance.py tests/capabilities/test_moneyprinter_mcp_tools.py -q
npm exec vitest run apps/desktop/src/app/video-studio/unified-material-library-panel.test.tsx apps/desktop/src/app/video-studio/use-named-video-library.test.tsx apps/desktop/src/app/video-studio/named-library-matching.test.ts
```

Expected: all selected Python and Desktop suites pass.

- [ ] **Step 3: Back up and preview the polluted records**

```bash
DB='/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/.hermes-assets/index.sqlite'
BACKUP="$DB.before-derived-prune-$(date +%Y%m%d-%H%M%S).bak"
cp -p "$DB" "$BACKUP"
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python -m capabilities.video_library.cli prune-derived --library beef-noodle
```

Expected: backup exists; preview reports the known derived records and `deleted: 0`.

- [ ] **Step 4: Execute pruning and verify the second scan is clean**

```bash
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python -m capabilities.video_library.cli prune-derived --library beef-noodle --execute
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python -m capabilities.video_library.cli scan --library beef-noodle --dry-run
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python -m capabilities.video_library.cli status --library beef-noodle
```

Expected: derived records are removed; dry-run does not rediscover `02_精选镜头`; status exposes `failed_assets` separately.

- [ ] **Step 5: Create local synthetic acceptance audio**

```bash
mkdir -p external/MoneyPrinterTurbo/storage/custom_audio
ffmpeg -hide_banner -loglevel error -f lavfi -i anullsrc=r=44100:cl=stereo -t 9 \
  -c:a libmp3lame -y external/MoneyPrinterTurbo/storage/custom_audio/beef-noodle-agent-e2e.mp3
```

Expected: a non-empty 9-second MP3 exists; no provider/API call occurs.

- [ ] **Step 6: Run planning mode and inspect the selected sources**

```bash
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python scripts/video_library_acceptance.py --library beef-noodle
```

Expected: three distinct selections for customer eating, employee serving, and finished-noodle close-up; each selected source is from an authorized raw source, not an asset-library generated directory.

- [ ] **Step 7: Run the real local render**

```bash
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python scripts/video_library_acceptance.py \
  --library beef-noodle \
  --audio external/MoneyPrinterTurbo/storage/custom_audio/beef-noodle-agent-e2e.mp3 \
  --render \
  --timeout 300
```

Expected: result JSON reports `complete`, exactly three selected local shots, one task ID, and a non-empty `final-1.mp4`.

- [ ] **Step 8: Verify the final media and provenance**

```bash
RESULT='/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/04_素材分析/验收/agent-e2e-result.json'
jq '{library_id,task_id,selected_sources,final_video_path}' "$RESULT"
FINAL=$(jq -r '.final_video_path' "$RESULT")
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height \
  -of json "$FINAL"
```

Expected:

- video stream is H.264-compatible and portrait output is 9:16;
- audio stream exists;
- duration is approximately 9 seconds;
- selected sources match the three planned intents;
- no online material-provider request appears in the MoneyPrinter task log.

- [ ] **Step 9: Run development Desktop smoke test**

Start or reuse only the current development build:

```bash
hermes-dev-desktop status
```

In Video Studio, choose `牛肉面资产库`, use the same three-sentence script, choose local materials, and click the automatic generation action. Verify the created task preview opens the new MP4 and the task message names `beef-noodle`.

- [ ] **Step 10: Commit acceptance evidence**

Do not commit MP4, SQLite backups, cached videos, custom audio, API keys, or Obsidian runtime outputs. Record the task ID, result JSON path, final MP4 path, duration, codecs, and selected source paths in the final handoff only.

## Final verification

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only deliberate implementation files are tracked. Runtime artifacts remain outside git.

The delivery is complete only when:

1. generated directories cannot be rediscovered as raw inputs;
2. the database reports and safely prunes derived records without deleting source files;
3. Agent tools preserve `library_id` through import, analyze, status, search, and timeline creation;
4. Agent-selected materialized clips can be cached and submitted as ordered local materials;
5. the three beef-noodle intents resolve to authorized raw sources;
6. the local, no-paid-TTS acceptance run produces a playable MP4;
7. the current Hermes development Desktop can preview the generated task.
