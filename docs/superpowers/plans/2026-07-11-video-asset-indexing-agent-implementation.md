# Video Asset Indexing Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Hermes “拉片专家” profile that can scan an explicitly configured Obsidian video directory and automatically create a resumable, searchable, shot-level semantic asset index without modifying original videos.

**Architecture:** Extend the existing `capabilities/video_library` Phase 1 instead of creating a parallel library. Add configured library roots, linked-source imports, lazy clip materialization, schema-validated vision analysis, batch scanning, Obsidian projections, and a CLI/Skill orchestration surface; keep MoneyPrinterTurbo as the downstream renderer.

**Tech Stack:** Python 3.11+, SQLite, FFmpeg/FFprobe, Hermes auxiliary vision client, PyYAML/config.yaml, Markdown, pytest, Hermes Skills and Profiles.

---

## Scope and follow-up boundary

This plan ends when the agent can scan a real mixed directory and semantic queries return correct shots. It deliberately does not add Video Studio script-to-shot ranking or MoneyPrinter timeline rendering. Those become a second plan after this ingestion/indexing milestone passes, because they are independently testable consumers of the index.

Existing dirty-worktree changes under `capabilities/video_library/`, Desktop Video Studio, MoneyPrinter, and their tests belong to the current user work. Preserve them and commit only files explicitly listed per task.

## File map

| File | Responsibility |
| --- | --- |
| `capabilities/video_library/config.py` | Parse and validate configured libraries and allowed source roots |
| `capabilities/video_library/store.py` | Migrations, linked assets, semantic clip fields, job state |
| `capabilities/video_library/service.py` | Technical analysis, lazy clip materialization, semantic stage orchestration |
| `capabilities/video_library/taxonomy.py` | Controlled 牛肉面 taxonomy and normalization |
| `capabilities/video_library/semantic.py` | Vision request construction, strict JSON parsing and scoring |
| `capabilities/video_library/batch.py` | Directory scan, fingerprint dedupe, per-file isolation and resume |
| `capabilities/video_library/obsidian.py` | Deterministic Markdown projections and library statistics |
| `capabilities/video_library/cli.py` | Rung-2 CLI used by the Skill |
| `capabilities/video_library/adapter.py` | Stable API envelopes for libraries, scan and semantic search |
| `capabilities/moneyprinter/mcp/tools.py` | Existing MCP extension for agent callers |
| `skills/local-video-asset-indexer/` | Batch-ingestion Skill and profile SOUL template |
| `skills/director-lapian/` | Reviewed director-analysis Skill supplied by the user |
| `tests/capabilities/test_video_library_config.py` | Config parsing and path authorization |
| `tests/capabilities/test_video_library_store.py` | Schema migration, linked imports and persistence |
| `tests/capabilities/test_video_library_service.py` | Analysis, lazy materialization and timeline behavior |
| `tests/capabilities/test_video_library_semantic.py` | Taxonomy and vision-output contracts |
| `tests/capabilities/test_video_library_batch.py` | Batch isolation, resume and idempotence |
| `tests/capabilities/test_video_library_obsidian.py` | Markdown projection behavior |
| `tests/capabilities/test_video_library_cli.py` | CLI JSON and dry-run contracts |
| `tests/capabilities/test_video_library_web_routes.py` | Authenticated API contracts |

### Task 1: Import and repair the supplied director-lapian Skill

**Files:**
- Create: `skills/director-lapian/SKILL.md`
- Create: `skills/director-lapian/agents/openai.yaml`
- Create: `skills/director-lapian/references/craft_pattern_library.md`
- Create: `skills/director-lapian/references/delivery_audio_workflow.md`
- Create: `skills/director-lapian/references/professional_director_analysis.md`
- Create: `skills/director-lapian/references/report_structure.md`
- Create: `skills/director-lapian/references/style_and_quality.md`
- Create: `skills/director-lapian/references/style_transfer_bible.md`
- Create: `skills/director-lapian/scripts/detect_shot_cuts.py`
- Create: `skills/director-lapian/scripts/lapian_delivery_status.py`
- Create: `skills/director-lapian/scripts/lapian_finalize_manifest.py`
- Create: `skills/director-lapian/scripts/md_image_report_to_docx.py`
- Create: `skills/director-lapian/scripts/md_lapian_preview_pdf.py`
- Create: `skills/director-lapian/scripts/prepare_lapian_evidence.py`
- Create: `skills/director-lapian/scripts/qa_lapian_delivery.py`
- Create: `skills/director-lapian/scripts/select_lapian_report_frames.py`
- Create: `skills/director-lapian/scripts/test_lapian_workflow_tools.py`
- Create: `skills/director-lapian/scripts/video_frame_sampler.py`
- Modify: `skills/director-lapian/scripts/lapian_delivery_status.py:13-19`
- Modify: `skills/director-lapian/scripts/test_lapian_workflow_tools.py:152-188`

- [ ] **Step 1: Copy only source files from the reviewed archive**

Extract the supplied archive to a temporary directory, copy `SKILL.md`, `agents/`, `references/`, and `scripts/*.py`, and exclude every `__pycache__` and `.pyc` file.

Run:

```bash
find skills/director-lapian -type d -name __pycache__ -o -type f -name '*.pyc'
```

Expected: no output.

- [ ] **Step 2: Preserve the failing regression test and verify it fails**

The archive already contains this assertion:

```python
qa_json = paths["qa_dir"] / "lapian_delivery_qa.json"
qa_json.write_text("{}", encoding="utf-8")
(paths["qa_dir"] / "测试片_交付清单.json").write_text("{}", encoding="utf-8")
scan = status.scan_project(paths["project"])
assert_true(scan["paths"]["qa_json"] == str(qa_json), "status should prefer QA audit JSON over delivery manifest")
```

Run:

```bash
python3 skills/director-lapian/scripts/test_lapian_workflow_tools.py
```

Expected: FAIL at `status should prefer QA audit JSON over delivery manifest`.

- [ ] **Step 3: Make stable QA output the first choice**

Replace `find_latest_qa()` with:

```python
def find_latest_qa(project_dir: Path) -> Path | None:
    qa_dir = qa.find_child_by_prefix(project_dir, "06_") or project_dir / "06_QA审计"
    stable = qa_dir / "lapian_delivery_qa.json"
    if stable.is_file():
        return stable
    preferred = qa.find_latest_file(qa_dir, ["*qa*.json", "*QA*.json", "*审计*.json", "*audit*.json"])
    if preferred:
        return preferred
    return qa.find_latest_file(qa_dir, ["*.json"])
```

- [ ] **Step 4: Run compile and workflow tests**

Run:

```bash
python3 -m py_compile skills/director-lapian/scripts/*.py
python3 skills/director-lapian/scripts/test_lapian_workflow_tools.py
```

Expected: both commands exit 0; workflow script prints its success summary.

- [ ] **Step 5: Commit the imported, repaired Skill**

```bash
git add skills/director-lapian
git commit -m "feat(skills): add director lapian workflow"
```

### Task 2: Add configured video-library roots and path authorization

**Files:**
- Create: `capabilities/video_library/config.py`
- Create: `tests/capabilities/test_video_library_config.py`
- Modify: `hermes_cli/config.py` in the default configuration map near other capability settings

- [ ] **Step 1: Write failing configuration and allowlist tests**

Add tests covering one configured Obsidian library, unknown IDs, source-root escape, and symlink escape:

```python
from pathlib import Path

import pytest

from capabilities.video_library.config import load_library_configs, resolve_source_path


def test_loads_linked_obsidian_library(tmp_path: Path):
    root = tmp_path / "牛肉面资产库"
    source = root / "01_原始素材"
    source.mkdir(parents=True)
    libraries = load_library_configs({
        "video_libraries": [{
            "id": "beef-noodle",
            "name": "牛肉面资产库",
            "root": str(root),
            "source_roots": [str(source)],
            "mode": "linked",
            "taxonomy": "beef-noodle-v1",
        }]
    })
    assert libraries["beef-noodle"].database_path == root / ".hermes-assets" / "index.sqlite"


def test_rejects_source_outside_allowlist(tmp_path: Path):
    source = tmp_path / "allowed"
    source.mkdir()
    config = load_library_configs({
        "video_libraries": [{"id": "beef-noodle", "root": str(tmp_path / "vault"), "source_roots": [str(source)]}]
    })["beef-noodle"]
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"video")
    with pytest.raises(ValueError, match="outside configured source roots"):
        resolve_source_path(config, outside)
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

```bash
pytest tests/capabilities/test_video_library_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: capabilities.video_library.config`.

- [ ] **Step 3: Implement immutable library configuration**

Create `config.py` with this public contract:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoLibraryConfig:
    id: str
    name: str
    root: Path
    source_roots: tuple[Path, ...]
    mode: str
    taxonomy: str

    @property
    def database_path(self) -> Path:
        return self.root / ".hermes-assets" / "index.sqlite"

    @property
    def selected_clips_dir(self) -> Path:
        return self.root / "02_精选镜头"

    @property
    def keyframes_dir(self) -> Path:
        return self.root / "03_关键帧"


def load_library_configs(config: dict[str, Any]) -> dict[str, VideoLibraryConfig]:
    result: dict[str, VideoLibraryConfig] = {}
    for raw in config.get("video_libraries") or []:
        library_id = str(raw.get("id") or "").strip()
        if not library_id or library_id in result:
            raise ValueError("video library ids must be non-empty and unique")
        root = Path(str(raw.get("root") or "")).expanduser().resolve()
        sources = tuple(Path(str(item)).expanduser().resolve() for item in raw.get("source_roots") or [])
        if not sources:
            raise ValueError(f"video library {library_id} requires source_roots")
        result[library_id] = VideoLibraryConfig(
            id=library_id,
            name=str(raw.get("name") or library_id),
            root=root,
            source_roots=sources,
            mode=str(raw.get("mode") or "linked"),
            taxonomy=str(raw.get("taxonomy") or "beef-noodle-v1"),
        )
    return result


def resolve_source_path(library: VideoLibraryConfig, path: Path | str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=True)
    if not any(candidate.is_relative_to(root) for root in library.source_roots):
        raise ValueError("video source is outside configured source roots")
    return candidate
```

Add `"video_libraries": []` to the default YAML configuration. Do not add an environment variable.

- [ ] **Step 4: Run config tests**

```bash
pytest tests/capabilities/test_video_library_config.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add capabilities/video_library/config.py tests/capabilities/test_video_library_config.py hermes_cli/config.py
git commit -m "feat(video-library): add configured library roots"
```

### Task 3: Migrate the store for linked assets and semantic records

**Files:**
- Modify: `capabilities/video_library/store.py:44-175,277-404`
- Modify: `tests/capabilities/test_video_library_store.py`

- [ ] **Step 1: Write failing migration and linked-import tests**

```python
def test_linked_import_does_not_copy_source(tmp_path):
    source_root = tmp_path / "vault" / "01_原始素材"
    source_root.mkdir(parents=True)
    source = source_root / "拉面.mp4"
    source.write_bytes(b"same-video")
    store = VideoLibraryStore(root=tmp_path / "vault", db_path=tmp_path / "vault/.hermes-assets/index.sqlite")

    asset = store.import_asset(source, source_mode="linked")

    assert asset["managed_path"] == str(source.resolve())
    assert asset["source_mode"] == "linked"
    assert not (tmp_path / "vault/assets").exists()


def test_semantic_clip_fields_round_trip(tmp_path):
    store, asset = make_store_with_asset(tmp_path)
    job = store.create_analysis_job(asset["id"], analyzer_version="semantic-v1")
    result = store.commit_analysis(asset["id"], job["id"], {}, [{
        "start_seconds": 0,
        "end_seconds": 4,
        "source_file_path": asset["managed_path"],
        "file_path": "",
        "keyframe_path": str(tmp_path / "key.jpg"),
        "description": "厨师拉面",
        "semantic_json": {"content": {"actions": ["拉面"]}},
        "quality_score": 0.91,
        "confidence": 0.88,
        "tags": [{"name": "动作/拉面", "confidence": 0.95, "source": "semantic-controlled"}],
    }])
    assert result["clips"][0]["description"] == "厨师拉面"
    assert result["clips"][0]["semantic_json"]["content"]["actions"] == ["拉面"]
```

- [ ] **Step 2: Verify failures against the old schema**

```bash
pytest tests/capabilities/test_video_library_store.py -q
```

Expected: FAIL because `source_mode` and semantic columns do not exist.

- [ ] **Step 3: Add idempotent migrations**

Keep the existing `CREATE TABLE` statements, then add a helper that checks `PRAGMA table_info` before every `ALTER TABLE`:

```python
def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
```

Required additions:

```python
_ensure_column(conn, "assets", "library_id", "library_id TEXT NOT NULL DEFAULT 'default'")
_ensure_column(conn, "assets", "source_mode", "source_mode TEXT NOT NULL DEFAULT 'managed'")
_ensure_column(conn, "assets", "source_mtime_ns", "source_mtime_ns INTEGER NOT NULL DEFAULT 0")
_ensure_column(conn, "clips", "source_file_path", "source_file_path TEXT NOT NULL DEFAULT ''")
_ensure_column(conn, "clips", "semantic_json", "semantic_json TEXT NOT NULL DEFAULT '{}'")
_ensure_column(conn, "clips", "quality_score", "quality_score REAL NOT NULL DEFAULT 0")
_ensure_column(conn, "clips", "confidence", "confidence REAL NOT NULL DEFAULT 0")
_ensure_column(conn, "clips", "materialized", "materialized INTEGER NOT NULL DEFAULT 1")
_ensure_column(conn, "analysis_jobs", "stage", "stage TEXT NOT NULL DEFAULT 'discovered'")
_ensure_column(conn, "analysis_jobs", "attempts", "attempts INTEGER NOT NULL DEFAULT 0")
```

Decode `semantic_json` in row-return helpers and encode it on writes. Extend `import_asset()` with `source_mode` and `library_id`; copy only in managed mode.

- [ ] **Step 4: Run store tests and migration twice**

```bash
pytest tests/capabilities/test_video_library_store.py -q
pytest tests/capabilities/test_video_library_store.py -q
```

Expected: both runs pass; the second run proves migrations are idempotent.

- [ ] **Step 5: Commit**

```bash
git add capabilities/video_library/store.py tests/capabilities/test_video_library_store.py
git commit -m "feat(video-library): support linked semantic assets"
```

### Task 4: Add lazy clip materialization

**Files:**
- Modify: `capabilities/video_library/service.py:54-166,167-232`
- Modify: `capabilities/video_library/media.py:171-260`
- Modify: `tests/capabilities/test_video_library_service.py`

- [ ] **Step 1: Write failing linked-analysis and materialization tests**

```python
def test_linked_analysis_extracts_keyframes_without_clip_mp4(tmp_path, monkeypatch):
    service, asset = make_linked_service(tmp_path)
    monkeypatch.setattr(media, "probe_media", lambda _: {"duration_seconds": 10, "width": 1080, "height": 1920, "fps": 30})
    monkeypatch.setattr(media, "detect_scene_boundaries", lambda *a, **k: [(0, 5), (5, 10)])
    monkeypatch.setattr(media, "extract_keyframe", write_fake_keyframe)
    monkeypatch.setattr(media, "extract_clip", lambda *a, **k: pytest.fail("linked analysis must not materialize clips"))

    result = service.analyze_asset(asset["id"])

    assert all(not clip["materialized"] for clip in result["clips"])
    assert all(clip["file_path"] == "" for clip in result["clips"])


def test_materialize_clip_extracts_exact_source_range(tmp_path, monkeypatch):
    service, clip = make_service_with_unmaterialized_clip(tmp_path)
    captured = {}
    monkeypatch.setattr(media, "extract_clip", lambda source, output, **kwargs: captured.update(source=source, output=output, **kwargs))
    result = service.materialize_clip(clip["id"])
    assert captured["start_seconds"] == clip["start_seconds"]
    assert captured["end_seconds"] == clip["end_seconds"]
    assert result["materialized"] is True
```

- [ ] **Step 2: Run the service tests**

```bash
pytest tests/capabilities/test_video_library_service.py -q
```

Expected: FAIL because analysis always calls `extract_clip()` and no `materialize_clip()` exists.

- [ ] **Step 3: Branch analysis by source mode**

In `analyze_asset()`, always create keyframes. For `source_mode == "managed"`, retain current eager clip extraction. For `linked`, record:

```python
clip_inputs.append({
    "start_seconds": start,
    "end_seconds": end,
    "source_file_path": str(source),
    "file_path": "",
    "keyframe_path": str(asset_keyframe_dir / f"{filename}.jpg"),
    "materialized": False,
    "tags": _technical_tags(metadata, end - start),
})
```

Add `materialize_clip(clip_id)` that resolves the selected output under a concrete per-asset directory such as `02_精选镜头/asset_4fd21d07a8c2b10e6d0149a3/`, calls `media.extract_clip()` with the stored source range, then atomically updates `file_path` and `materialized`.

- [ ] **Step 4: Make timeline creation materialize selected clips**

Before path validation in `create_timeline()`:

```python
if not clip["materialized"]:
    clip = self.materialize_clip(clip["id"])
file_path = Path(clip["file_path"]).expanduser().resolve(strict=True)
```

- [ ] **Step 5: Run service and real-media tests**

```bash
pytest tests/capabilities/test_video_library_service.py tests/capabilities/test_video_library_media.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add capabilities/video_library/service.py capabilities/video_library/media.py tests/capabilities/test_video_library_service.py
git commit -m "feat(video-library): materialize linked clips on demand"
```

### Task 5: Add controlled taxonomy and schema-validated vision analysis

**Files:**
- Create: `capabilities/video_library/taxonomy.py`
- Create: `capabilities/video_library/semantic.py`
- Create: `tests/capabilities/test_video_library_semantic.py`
- Modify: `capabilities/video_library/service.py`

- [ ] **Step 1: Write failing taxonomy and analyzer tests**

```python
def test_normalizes_controlled_and_free_tags():
    result = normalize_semantic_result({
        "content": {"subjects": ["厨师"], "actions": ["抻面", "拉面"]},
        "creative": {"commercial_functions": ["品质证明"]},
        "quality": {"overall_score": 1.4},
        "analysis": {"confidence": -0.2},
    }, taxonomy="beef-noodle-v1")
    assert "动作/拉面" in result.controlled_tags
    assert result.quality_score == 1.0
    assert result.confidence == 0.0


def test_vision_analyzer_sends_keyframe_not_video(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    calls = []
    monkeypatch.setattr("capabilities.video_library.semantic.call_llm", lambda **kwargs: fake_json_response(calls, kwargs))
    result = analyze_keyframes([frame], taxonomy="beef-noodle-v1")
    content = calls[0]["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert not any(".mp4" in str(part) for part in content)
    assert result.summary == "厨师手工拉面"
```

- [ ] **Step 2: Verify failures**

```bash
pytest tests/capabilities/test_video_library_semantic.py -q
```

Expected: FAIL because taxonomy and semantic modules do not exist.

- [ ] **Step 3: Implement the 牛肉面 v1 controlled taxonomy**

Create constants for subjects, scenes, actions, production stages, shot sizes, angles, camera motions, visual features, moods, commercial functions, audio values, and usability. Include aliases such as `抻面 -> 拉面` and `下面 -> 下锅`.

Expose:

```python
def normalize_controlled(dimension: str, values: list[str], taxonomy: str) -> list[str]:
    """Return unique `dimension/value` tags from the configured vocabulary."""
```

- [ ] **Step 4: Implement strict semantic output**

Create `SemanticClipResult` and `normalize_semantic_result()`. Build a Chinese prompt that demands one JSON object and includes the allowed taxonomy. Use Hermes' existing provider router:

```python
from agent.auxiliary_client import call_llm

response = call_llm(
    messages=[{"role": "user", "content": content_parts}],
    task="vision",
    temperature=0.1,
    timeout=120,
)
raw = response.choices[0].message.content or ""
payload = json.loads(extract_json_object(raw))
return normalize_semantic_result(payload, taxonomy=taxonomy)
```

Limit frames per shot to three and resize before base64 encoding. The prompt must state that low-confidence fields should be empty rather than invented.

- [ ] **Step 5: Integrate semantic analysis as a second service stage**

Add `semantic_analyzer: Callable | None` injection to `VideoLibraryService`. After technical clip commit, analyze each clip keyframe, update description/tags/semantic fields, and advance job stages `extracting_evidence -> semantic_analysis -> indexing`.

Do not hold a SQLite transaction open during model calls.

- [ ] **Step 6: Run semantic and service tests**

```bash
pytest tests/capabilities/test_video_library_semantic.py tests/capabilities/test_video_library_service.py -q
```

Expected: all tests pass with a fake analyzer; no network calls occur in tests.

- [ ] **Step 7: Commit**

```bash
git add capabilities/video_library/taxonomy.py capabilities/video_library/semantic.py capabilities/video_library/service.py tests/capabilities/test_video_library_semantic.py tests/capabilities/test_video_library_service.py
git commit -m "feat(video-library): add semantic shot analysis"
```

### Task 6: Add batch scanning, resume, and Obsidian projections

**Files:**
- Create: `capabilities/video_library/batch.py`
- Create: `capabilities/video_library/obsidian.py`
- Create: `tests/capabilities/test_video_library_batch.py`
- Create: `tests/capabilities/test_video_library_obsidian.py`

- [ ] **Step 1: Write failing batch behavior tests**

```python
def test_scan_continues_after_one_file_fails(tmp_path, monkeypatch):
    library = make_library(tmp_path, names=["good.mp4", "bad.mp4", "good2.mov"])
    runner = make_batch_runner(library)
    monkeypatch.setattr(runner, "process_file", lambda path: (_ for _ in ()).throw(RuntimeError("broken")) if path.name == "bad.mp4" else {"status": "complete"})
    result = runner.scan()
    assert result.total == 3
    assert result.complete == 2
    assert result.failed == 1


def test_second_scan_skips_unchanged_content(tmp_path, monkeypatch):
    library = make_library(tmp_path, names=["one.mp4"])
    runner = make_batch_runner(library)
    first = runner.scan()
    second = runner.scan()
    assert first.complete == 1
    assert second.skipped == 1
```

- [ ] **Step 2: Write failing projection tests**

```python
def test_projection_writes_readable_asset_page(tmp_path):
    path = write_asset_page(tmp_path, asset_fixture(), clip_fixtures())
    text = path.read_text(encoding="utf-8")
    assert "# 厨师下午拉面01" in text
    assert "动作/拉面" in text
    assert "00:08.400-00:13.700" in text


def test_projection_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "04_素材分析/单条视频分析/asset.md"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr("os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        write_markdown_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
```

- [ ] **Step 3: Run tests and confirm failures**

```bash
pytest tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_obsidian.py -q
```

Expected: FAIL because both modules are missing.

- [ ] **Step 4: Implement batch scan isolation**

`VideoLibraryBatchRunner.scan()` must recursively enumerate only supported video suffixes under configured source roots, sort paths deterministically, and return:

```python
@dataclass
class BatchScanResult:
    library_id: str
    total: int = 0
    complete: int = 0
    skipped: int = 0
    failed: int = 0
    low_confidence: int = 0
    unusable: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
```

Catch exceptions per file, store the failed stage, and continue. Skip when the SHA-256 exists with semantic analyzer version `semantic-v1` complete.

- [ ] **Step 5: Implement deterministic Markdown projection**

Write one page per source asset under `04_素材分析/单条视频分析/` and regenerate `素材统计.md`. Include source path, fingerprint, analysis model/version, clip time ranges, keyframes, fixed tags, free tags, description, quality and status. Use temp-file + `os.replace()`.

- [ ] **Step 6: Run batch and projection tests**

```bash
pytest tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_obsidian.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add capabilities/video_library/batch.py capabilities/video_library/obsidian.py tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_obsidian.py
git commit -m "feat(video-library): scan and project Obsidian assets"
```

### Task 7: Expose the batch workflow through CLI, API, and existing MCP

**Files:**
- Create: `capabilities/video_library/cli.py`
- Create: `tests/capabilities/test_video_library_cli.py`
- Modify: `capabilities/video_library/adapter.py:23-112`
- Modify: `capabilities/moneyprinter/mcp/tools.py:375-416,497-514`
- Modify: `hermes_cli/web_server.py:441-511`
- Modify: `gateway/platforms/api_server.py:1659-1738,4916-4921`
- Modify: `tests/capabilities/test_video_library_web_routes.py`
- Modify: `tests/capabilities/test_moneyprinter_mcp_tools.py`

- [ ] **Step 1: Write failing adapter and CLI tests**

```python
def test_scan_library_adapter_uses_configured_id(monkeypatch):
    monkeypatch.setattr(adapter, "scan_library", lambda library_id: {"libraryId": library_id, "complete": 4})
    status, payload = adapter.scan_library_data("beef-noodle")
    assert status == 200
    assert payload["data"]["libraryId"] == "beef-noodle"


def test_cli_scan_prints_machine_readable_json(monkeypatch, capsys):
    monkeypatch.setattr(cli, "scan_library", lambda library_id: {"libraryId": library_id, "failed": 0})
    assert cli.main(["scan", "--library", "beef-noodle"]) == 0
    assert json.loads(capsys.readouterr().out)["libraryId"] == "beef-noodle"


def test_cli_dry_run_does_not_process_files(monkeypatch, capsys):
    monkeypatch.setattr(cli, "dry_run_library", lambda library_id: {
        "libraryId": library_id,
        "dryRun": True,
        "total": 3,
        "writesPlanned": [],
    })
    assert cli.main(["scan", "--library", "beef-noodle", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dryRun"] is True
    assert payload["writesPlanned"] == []
```

- [ ] **Step 2: Verify failures**

```bash
pytest tests/capabilities/test_video_library_cli.py tests/capabilities/test_video_library_web_routes.py tests/capabilities/test_moneyprinter_mcp_tools.py -q
```

Expected: FAIL because scan/list-library surfaces do not exist.

- [ ] **Step 3: Implement the CLI**

Support:

```bash
python -m capabilities.video_library.cli libraries
python -m capabilities.video_library.cli scan --library beef-noodle
python -m capabilities.video_library.cli scan --library beef-noodle --dry-run
python -m capabilities.video_library.cli status --library beef-noodle
python -m capabilities.video_library.cli search --library beef-noodle --query '厨师拉面'
```

Every command prints one UTF-8 JSON document and returns nonzero on failure. `--dry-run` may enumerate and hash files, but must not call FFmpeg, call a model, create a database, or write Markdown.

- [ ] **Step 4: Add stable adapters and routes**

Add:

```text
GET  /api/capabilities/video-library/libraries
POST /api/capabilities/video-library/libraries/{library_id}/scan
GET  /api/capabilities/video-library/libraries/{library_id}/status
GET  /api/capabilities/video-library/clips?library_id=&query=&tag=
```

All backends must call the same adapter functions and preserve `{"ok","data","error"}` envelopes.

- [ ] **Step 5: Extend MCP without adding a core tool**

Register `video_library_scan_library(library_id)` and extend `video_library_search_clips` with `library_id` and free-text `query`. Keep the MCP capability optional and do not edit `_HERMES_CORE_TOOLS`.

- [ ] **Step 6: Run focused route and MCP tests**

```bash
pytest tests/capabilities/test_video_library_cli.py tests/capabilities/test_video_library_web_routes.py tests/capabilities/test_moneyprinter_mcp_tools.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add capabilities/video_library/cli.py capabilities/video_library/adapter.py capabilities/moneyprinter/mcp/tools.py hermes_cli/web_server.py gateway/platforms/api_server.py tests/capabilities/test_video_library_cli.py tests/capabilities/test_video_library_web_routes.py tests/capabilities/test_moneyprinter_mcp_tools.py
git commit -m "feat(video-library): expose configured batch scans"
```

### Task 8: Create the asset-indexing Skill and 拉片专家 profile template

**Files:**
- Create: `skills/local-video-asset-indexer/SKILL.md`
- Create: `skills/local-video-asset-indexer/references/shot-schema.md`
- Create: `skills/local-video-asset-indexer/references/beef-noodle-taxonomy.md`
- Create: `skills/local-video-asset-indexer/assets/SOUL.md`
- Create: `tests/skills/test_local_video_asset_indexer.py`

- [ ] **Step 1: Write failing Skill package tests**

```python
def test_asset_indexer_skill_has_required_contract():
    root = Path("skills/local-video-asset-indexer")
    text = (root / "SKILL.md").read_text(encoding="utf-8")
    assert "name: local-video-asset-indexer" in text
    assert "python -m capabilities.video_library.cli scan" in text
    assert "不得移动、改名或删除原始视频" in text
    assert (root / "references/shot-schema.md").is_file()
    assert (root / "assets/SOUL.md").is_file()
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/skills/test_local_video_asset_indexer.py -q
```

Expected: FAIL because the Skill does not exist.

- [ ] **Step 3: Write the focused Skill**

The Skill must:

- Trigger on scanning, importing, classifying or updating local video asset libraries.
- Resolve a configured library ID before work.
- Run `libraries`, then `scan`, then `status`; never invent paths from prose.
- Treat the CLI JSON summary as execution truth.
- Report totals for complete/skipped/failed/low-confidence/unusable.
- Never generate a director-length report for self-shot ingestion.
- Route requests for爆款分析 to `director-lapian`.
- Read the schema and taxonomy references completely before interpreting output.

- [ ] **Step 4: Add the Profile SOUL template**

The template should identify the profile as “拉片专家”, route self-shot and reference-video requests to different Skills, preserve source files, and avoid scanning outside configured roots. Do not embed user-specific absolute paths.

- [ ] **Step 5: Run Skill tests**

```bash
pytest tests/skills/test_local_video_asset_indexer.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add skills/local-video-asset-indexer tests/skills/test_local_video_asset_indexer.py
git commit -m "feat(skills): add local video asset indexer"
```

### Task 9: Verify profile creation in an isolated Hermes home

**Files:**
- Test only; do not modify the user's production `~/.hermes`

- [ ] **Step 1: Create an isolated profile root**

```bash
export HERMES_HOME="$(mktemp -d)/hermes-home"
python -m hermes_cli.main profile create video-analysis-expert
```

Expected: profile directory exists under `$HERMES_HOME/profiles/video-analysis-expert/` with `SOUL.md` and `skills/`.

- [ ] **Step 2: Install the approved SOUL template and synchronize Skills**

```bash
cp skills/local-video-asset-indexer/assets/SOUL.md "$HERMES_HOME/profiles/video-analysis-expert/SOUL.md"
HERMES_HOME="$HERMES_HOME/profiles/video-analysis-expert" python -c 'from tools.skills_sync import sync_skills; print(sync_skills(quiet=True))'
```

Expected: both `local-video-asset-indexer` and `director-lapian` appear under the isolated profile's `skills/`.

- [ ] **Step 3: Verify profile isolation**

```bash
test -f "$HERMES_HOME/profiles/video-analysis-expert/SOUL.md"
test -d "$HERMES_HOME/profiles/video-analysis-expert/skills/local-video-asset-indexer"
test -d "$HERMES_HOME/profiles/video-analysis-expert/skills/director-lapian"
```

Expected: all commands exit 0.

No commit is needed because this task mutates only the temporary test home.

### Task 10: Run the full automated verification suite

**Files:**
- Modify only tests if a verified contract is missing; do not weaken assertions to make failures disappear

- [ ] **Step 1: Run all video-library tests**

```bash
pytest tests/capabilities/test_video_library_config.py tests/capabilities/test_video_library_store.py tests/capabilities/test_video_library_media.py tests/capabilities/test_video_library_service.py tests/capabilities/test_video_library_semantic.py tests/capabilities/test_video_library_batch.py tests/capabilities/test_video_library_obsidian.py tests/capabilities/test_video_library_cli.py tests/capabilities/test_video_library_web_routes.py -q
```

Expected: all pass.

- [ ] **Step 2: Run MCP, Skill and director workflow tests**

```bash
pytest tests/capabilities/test_moneyprinter_mcp_tools.py tests/skills/test_local_video_asset_indexer.py -q
python3 skills/director-lapian/scripts/test_lapian_workflow_tools.py
```

Expected: all commands exit 0.

- [ ] **Step 3: Run static checks**

```bash
python3 -m py_compile capabilities/video_library/*.py skills/director-lapian/scripts/*.py
git diff --check
```

Expected: no output from `git diff --check`; compile exits 0.

- [ ] **Step 4: Confirm verification left no uncommitted implementation changes**

```bash
git status --short
```

Expected: only pre-existing user changes outside this plan may remain. If a plan-owned file is modified, return to the task that owns it, fix the failure there, rerun that task's focused tests, and commit it with that task's stated commit message before continuing.

### Task 11: Dry-run a real 牛肉面 directory, then perform the first authorized scan

**Files:**
- Runtime configuration only: the active Hermes Dev profile `config.yaml`
- Runtime output only: the user-approved Obsidian `牛肉面资产库/`

- [ ] **Step 1: Add the real library configuration without exposing secrets**

Add `video_libraries` to the active Hermes Dev profile's `config.yaml`, with the exact user-approved Vault root and `01_原始素材` path. Keep API keys in `.env`; do not place credentials in YAML.

- [ ] **Step 2: Verify access before writing**

```bash
python -m capabilities.video_library.cli libraries
python -m capabilities.video_library.cli status --library beef-noodle
```

Expected: JSON identifies `beef-noodle`, its linked mode, and writable library directories. If the Vault is not writable, stop here and report the exact path.

- [ ] **Step 3: Run a deterministic dry scan**

Use the `--dry-run` contract implemented and tested in Task 7. It enumerates supported videos and hash status without FFmpeg extraction, model calls, DB writes or Markdown writes.

```bash
python -m capabilities.video_library.cli scan --library beef-noodle --dry-run
```

Expected: JSON totals match the actual source directory and `writesPlanned` lists only paths under the configured asset library root.

- [ ] **Step 4: Run the authorized real scan**

```bash
python -m capabilities.video_library.cli scan --library beef-noodle
```

Expected: command exits 0 even when individual assets fail; JSON reports complete/skipped/failed counts and exact failed stages.

- [ ] **Step 5: Verify generated artifacts and search quality**

```bash
python -m capabilities.video_library.cli search --library beef-noodle --query '厨师拉面'
python -m capabilities.video_library.cli search --library beef-noodle --query '牛肉切片'
python -m capabilities.video_library.cli search --library beef-noodle --query '热汤浇入碗中'
```

Expected: each query returns semantically relevant shots with source path, time range, keyframe, description, tags, score and usability. Inspect at least the top three keyframes for each query.

- [ ] **Step 6: Verify idempotence**

Run the real scan a second time:

```bash
python -m capabilities.video_library.cli scan --library beef-noodle
```

Expected: unchanged assets are reported as skipped; no duplicate assets or model calls are created.

- [ ] **Step 7: Record the verified runtime result**

Update `docs/capabilities/video-material-library-logic-map.md` with actual counts, analyzer model/version, known failures and the exact completed phase. Do not claim Video Studio render closure in this phase.

- [ ] **Step 8: Commit runtime documentation only**

```bash
git add docs/capabilities/video-material-library-logic-map.md
git commit -m "docs(video-library): record semantic indexing verification"
```

## Final acceptance gate

Do not start the follow-up Video Studio matching/render plan until all of these are true:

- The real library scan is idempotent.
- Original source videos are byte-for-byte unchanged.
- At least three representative Chinese semantic queries return correct top results.
- Low-confidence and unusable clips are marked and ranked appropriately.
- A failed asset does not stop the batch.
- The isolated “拉片专家” profile contains both Skills and selects the correct one by request type.
- No new Hermes Agent Core tool was added.
