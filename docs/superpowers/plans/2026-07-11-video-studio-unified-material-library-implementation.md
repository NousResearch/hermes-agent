# Video Studio Unified Material Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three competing material surfaces with one named-library workflow that ingests user footage, matches copy to tagged clips, requires human confirmation, and invisibly bridges selected clips into MoneyPrinterTurbo.

**Architecture:** Obsidian-backed named libraries become the only user-facing source of truth. The old unnamed Hermes store remains readable only for migration, while MoneyPrinter `storage/local_videos` becomes a hidden render cache with collision-safe names. A focused controller and unified panel own library state; `index.tsx` only connects confirmed clips to the existing generation form.

**Tech Stack:** Python 3, FastAPI, SQLite, PyYAML, React, TypeScript, Vitest, Testing Library, Electron path picker, MoneyPrinterTurbo, Obsidian Markdown.

---

## File Map

- Create `capabilities/video_library/management.py` and `tests/capabilities/test_video_library_management.py` for source-root changes and legacy migration.
- Modify `capabilities/video_library/adapter.py`, `hermes_cli/web_server.py`, and route tests for management APIs.
- Modify `moneyprinter-client.ts` and its tests for named import, source-root, and migration calls.
- Modify `use-named-video-library.ts` and tests for unified controller state.
- Create `material-cache.ts` and tests for collision-safe render-cache names.
- Create `unified-material-library-panel.tsx` and tests; delete the superseded named panel.
- Modify `index.tsx` and tests to remove old local/default-library surfaces.
- Update both video logic maps with regression and real-demo evidence.

## Task 1: Guarded Source-Root Management

**Files:**
- Create: `capabilities/video_library/management.py`
- Create: `tests/capabilities/test_video_library_management.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
import pytest
from capabilities.video_library.management import add_library_source_root


def test_add_source_root_persists_a_real_directory_once(tmp_path, monkeypatch):
    source = tmp_path / "merchant-footage"
    source.mkdir()
    raw = {"video_libraries": [{
        "id": "beef-noodle", "name": "牛肉面资产库",
        "root": str(tmp_path / "vault"), "source_roots": [str(source)],
        "mode": "linked", "taxonomy": "beef-noodle-v1",
    }]}
    writes = []
    monkeypatch.setattr("capabilities.video_library.management.read_raw_config", lambda: raw)
    monkeypatch.setattr("capabilities.video_library.management.save_config", lambda value, **_: writes.append(value))

    result = add_library_source_root("beef-noodle", source)

    assert result["source_roots"] == [str(source.resolve())]
    assert len(writes) == 1


def test_add_source_root_rejects_a_file(tmp_path):
    path = tmp_path / "not-a-directory.mp4"
    path.write_bytes(b"video")
    with pytest.raises(ValueError, match="directory"):
        add_library_source_root("beef-noodle", path)
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/capabilities/test_video_library_management.py`

Expected: import failure because `management.py` does not exist.

- [ ] **Step 3: Implement the minimal API**

```python
from pathlib import Path
from typing import Any
from hermes_cli.config import read_raw_config, save_config
from .config import load_library_configs


def add_library_source_root(library_id: str, source_root: str | Path) -> dict[str, Any]:
    candidate = Path(source_root).expanduser().resolve(strict=True)
    if not candidate.is_dir():
        raise ValueError("video library source root must be a directory")
    raw = read_raw_config()
    load_library_configs(raw)
    normalized = str(library_id).strip().lower()
    entries = raw.get("video_libraries") or []
    entry = next((item for item in entries if str(item.get("id", "")).lower() == normalized), None)
    if entry is None:
        raise KeyError(f"unknown video library: {normalized}")
    roots = [str(Path(value).expanduser().resolve()) for value in entry.get("source_roots") or []]
    if str(candidate) not in roots:
        roots.append(str(candidate))
        entry["source_roots"] = roots
    save_config(raw, strip_defaults=False, preserve_keys={("video_libraries",)})
    return {"library_id": normalized, "source_roots": roots}
```

Do not accept a client-supplied library root, mode, or taxonomy.

- [ ] **Step 4: Run GREEN and commit**

```bash
.venv/bin/pytest -q tests/capabilities/test_video_library_management.py tests/capabilities/test_video_library_config.py
git add capabilities/video_library/management.py tests/capabilities/test_video_library_management.py
git commit -m "feat(video-library): manage named source roots"
```

Expected: all tests pass and duplicate roots remain idempotent.

## Task 2: Auditable Legacy Migration

**Files:**
- Modify: `capabilities/video_library/management.py`
- Modify: `tests/capabilities/test_video_library_management.py`

- [ ] **Step 1: Write failing migration tests**

```python
def test_migrate_legacy_library_is_idempotent_and_preserves_legacy(tmp_path, monkeypatch):
    source = tmp_path / "legacy.mp4"
    source.write_bytes(b"same-content")
    legacy = VideoLibraryStore(root=tmp_path / "legacy")
    old = legacy.import_asset(source)
    target = VideoLibraryStore(root=tmp_path / "target")
    monkeypatch.setattr("capabilities.video_library.management.get_legacy_store", lambda: legacy)
    monkeypatch.setattr("capabilities.video_library.management.get_named_store", lambda _id: target)

    first = migrate_legacy_library("beef-noodle")
    second = migrate_legacy_library("beef-noodle")

    assert first["imported"] == 1
    assert second["skipped"] == 1
    assert legacy.get_asset(old["id"])["id"] == old["id"]
    assert len(target.list_assets()) == 1
```

Add a partial-failure test requiring `source_asset_id`, `target_asset_id`, `state`, and `error` per record.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/capabilities/test_video_library_management.py -k migrate`

Expected: `migrate_legacy_library` is missing.

- [ ] **Step 3: Implement migration**

```python
def migrate_legacy_library(library_id: str) -> dict[str, Any]:
    legacy, target = get_legacy_store(), get_named_store(library_id)
    existing = {asset["sha256"] for asset in target.list_assets()}
    records, imported, skipped, failed = [], 0, 0, 0
    for asset in legacy.list_assets():
        try:
            migrated = target.import_asset(Path(asset["managed_path"]), source_mode="managed", library_id=library_id)
            state = "skipped" if migrated["sha256"] in existing else "imported"
            imported += int(state == "imported")
            skipped += int(state == "skipped")
            existing.add(migrated["sha256"])
            records.append({"source_asset_id": asset["id"], "target_asset_id": migrated["id"], "state": state, "error": ""})
        except Exception as exc:
            failed += 1
            records.append({"source_asset_id": asset["id"], "target_asset_id": "", "state": "failed", "error": str(exc)})
    return {"library_id": library_id, "total": len(records), "imported": imported, "skipped": skipped, "failed": failed, "records": records}
```

The production store factories resolve both stores server-side. Never delete legacy assets.

- [ ] **Step 4: Run GREEN and commit**

```bash
.venv/bin/pytest -q tests/capabilities/test_video_library_management.py
git add capabilities/video_library/management.py tests/capabilities/test_video_library_management.py
git commit -m "feat(video-library): migrate legacy assets safely"
```

## Task 3: Management Routes and Desktop Client

**Files:**
- Modify: `capabilities/video_library/adapter.py`
- Modify: `hermes_cli/web_server.py`
- Modify: `tests/capabilities/test_video_library_web_routes.py`
- Modify: `apps/desktop/src/app/video-studio/moneyprinter-client.ts`
- Modify: `apps/desktop/src/app/video-studio/moneyprinter-client.test.ts`

- [ ] **Step 1: Write failing route tests**

```python
def test_source_root_route_forwards_named_library(monkeypatch, client):
    captured = {}
    monkeypatch.setattr(adapter, "add_library_source_root_data", lambda library_id, body: (captured.update(id=library_id, body=body) or (200, {"ok": True, "data": {}, "error": None})))
    response = client.post("/api/capabilities/video-library/libraries/beef-noodle/source-roots", json={"path": "/vault/material"})
    assert response.status_code == 200
    assert captured == {"id": "beef-noodle", "body": {"path": "/vault/material"}}
```

Add a migration-route test and an empty-path rejection test.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/pytest -q tests/capabilities/test_video_library_web_routes.py -k 'source_root or migrate'`

Expected: missing routes.

- [ ] **Step 3: Add adapter functions and FastAPI routes**

```python
def add_library_source_root_data(library_id: str, body: dict[str, Any]):
    try:
        return 200, _envelope(add_library_source_root(library_id, str(body.get("path") or "")))
    except Exception as exc:
        return _failure(exc)


def migrate_legacy_library_data(library_id: str):
    try:
        return 200, _envelope(migrate_legacy_library(library_id))
    except Exception as exc:
        return _failure(exc)
```

Register `POST /libraries/{library_id}/source-roots` and `POST /libraries/{library_id}/migrate-legacy` beside the existing named routes.

- [ ] **Step 4: Write failing client tests**

```typescript
await videoLibraryClient.addSourceRoot('beef-noodle', '/vault/material')
await videoLibraryClient.migrateLegacyLibrary('beef-noodle')
await videoLibraryClient.importAsset('beef-noodle', '/vault/material/source.mov')
expect(api).toHaveBeenNthCalledWith(1, expect.objectContaining({
  path: '/api/capabilities/video-library/libraries/beef-noodle/source-roots',
  method: 'POST', body: { path: '/vault/material' }
}))
expect(api).toHaveBeenNthCalledWith(3, expect.objectContaining({
  path: '/api/capabilities/video-library/assets',
  body: { libraryId: 'beef-noodle', sourcePath: '/vault/material/source.mov' }
}))
```

- [ ] **Step 5: Extend `videoLibraryClient`**

Add `VideoLibraryMigrationResult`, `addSourceRoot`, `migrateLegacyLibrary`, and named-only `importAsset(libraryId, sourcePath)`. Every renderer write must carry `libraryId`.

- [ ] **Step 6: Run GREEN and commit**

```bash
.venv/bin/pytest -q tests/capabilities/test_video_library_web_routes.py
npm --prefix apps/desktop run test:ui -- src/app/video-studio/moneyprinter-client.test.ts
npm --prefix apps/desktop run typecheck
git add capabilities/video_library/adapter.py hermes_cli/web_server.py tests/capabilities/test_video_library_web_routes.py apps/desktop/src/app/video-studio/moneyprinter-client.ts apps/desktop/src/app/video-studio/moneyprinter-client.test.ts
git commit -m "feat(video-library): expose named library management"
```

## Task 4: Unified Named-Library Controller

**Files:**
- Modify: `apps/desktop/src/app/video-studio/use-named-video-library.ts`
- Modify: `apps/desktop/src/app/video-studio/use-named-video-library.test.tsx`

- [ ] **Step 1: Write failing controller tests**

```typescript
it('imports and analyzes files only in the selected library', async () => {
  act(() => result.current.selectLibrary('beef-noodle'))
  await act(() => result.current.importFiles(['/vault/material/a.mov']))
  expect(client.importAsset).toHaveBeenCalledWith('beef-noodle', '/vault/material/a.mov')
  expect(client.analyzeAsset).toHaveBeenCalledWith('beef-noodle', 'asset-1')
})

it('returns a dry-run before a real directory scan', async () => {
  act(() => result.current.selectLibrary('beef-noodle'))
  await act(() => result.current.addSourceRoot('/vault/new-material'))
  expect(client.addSourceRoot).toHaveBeenCalledWith('beef-noodle', '/vault/new-material')
  expect(client.scanLibrary).toHaveBeenCalledWith('beef-noodle', true)
  expect(client.scanLibrary).not.toHaveBeenCalledWith('beef-noodle', false)
})
```

Also assert switching libraries clears candidates, confirmations, migration result, dry-run result, and selected render files.

- [ ] **Step 2: Run RED**

Run: `npm --prefix apps/desktop run test:ui -- src/app/video-studio/use-named-video-library.test.tsx`

- [ ] **Step 3: Implement controller actions**

Add `importFiles`, `addSourceRoot`, `confirmScan`, `migrateLegacyLibrary`, `selectedMaterialFiles`, and per-file errors. Named import must be followed by named analysis. `addSourceRoot` only runs dry-run; `confirmScan` is the only real scan action. `selectLibrary` clears all cross-library state.

- [ ] **Step 4: Run GREEN and commit**

```bash
npm --prefix apps/desktop run test:ui -- src/app/video-studio/use-named-video-library.test.tsx
git add apps/desktop/src/app/video-studio/use-named-video-library.ts apps/desktop/src/app/video-studio/use-named-video-library.test.tsx
git commit -m "feat(video-studio): control one selected material library"
```

## Task 5: Collision-Safe MoneyPrinter Cache Bridge

**Files:**
- Create: `apps/desktop/src/app/video-studio/material-cache.ts`
- Create: `apps/desktop/src/app/video-studio/material-cache.test.ts`
- Modify: `apps/desktop/src/app/video-studio/index.tsx`

- [ ] **Step 1: Write failing pure-helper tests**

```typescript
expect(cacheFilenameForClip('clip-a', '/a/clip-0001.mp4')).toBe('clip-a-clip-0001.mp4')
expect(cacheFilenameForClip('clip-b', '/b/clip-0001.mp4')).toBe('clip-b-clip-0001.mp4')
expect(timelineVideoSelections({ tracks: { video: [
  { clipId: 'clip-b', file: '/b.mp4' }, { clipId: 'clip-a', file: '/a.mp4' }
] } })).toEqual([
  { clipId: 'clip-b', file: '/b.mp4' }, { clipId: 'clip-a', file: '/a.mp4' }
])
```

- [ ] **Step 2: Run RED**

Run: `npm --prefix apps/desktop run test:ui -- src/app/video-studio/material-cache.test.ts`

- [ ] **Step 3: Implement helpers**

```typescript
export function cacheFilenameForClip(clipId: string, file: string): string {
  const basename = file.split(/[\\/]/).pop() || 'selected-clip.mp4'
  return `${clipId.replace(/[^a-zA-Z0-9_-]/g, '_')}-${basename}`
}
```

`timelineVideoSelections` validates `tracks.video`, preserves order, and omits rows without both `clipId` and `file`.

- [ ] **Step 4: Use stable names in timeline creation**

```typescript
for (const selection of timelineVideoSelections(result.timeline)) {
  await moneyprinterClient.uploadLocalMaterial({
    filename: cacheFilenameForClip(selection.clipId, selection.file),
    sourcePath: selection.file
  })
}
```

- [ ] **Step 5: Run GREEN and commit**

```bash
npm --prefix apps/desktop run test:ui -- src/app/video-studio/material-cache.test.ts src/app/video-studio/index.test.tsx
git add apps/desktop/src/app/video-studio/material-cache.ts apps/desktop/src/app/video-studio/material-cache.test.ts apps/desktop/src/app/video-studio/index.tsx
git commit -m "fix(video-studio): isolate the render material cache"
```

## Task 6: One Unified Material Library Panel

**Files:**
- Create: `apps/desktop/src/app/video-studio/unified-material-library-panel.tsx`
- Create: `apps/desktop/src/app/video-studio/unified-material-library-panel.test.tsx`
- Modify: `apps/desktop/src/app/video-studio/index.tsx`
- Modify: `apps/desktop/src/app/video-studio/index.test.tsx`
- Delete: `apps/desktop/src/app/video-studio/named-library-panel.tsx`
- Delete: `apps/desktop/src/app/video-studio/named-library-panel.test.tsx`

- [ ] **Step 1: Write failing UI tests**

```typescript
expect(screen.getAllByText('素材库')).toHaveLength(1)
expect(screen.queryByText('本地素材')).toBeNull()
expect(screen.queryByText('视频素材库')).toBeNull()
expect(screen.queryByText('Obsidian 具名资产库')).toBeNull()
expect(screen.getByRole('button', { name: '添加素材文件' })).toBeDisabled()
expect(screen.getByRole('button', { name: '选择素材目录' })).toBeDisabled()
```

Add tests proving only confirmed clips appear under `本次已选镜头` and migration stays disabled until a target library is selected.

- [ ] **Step 2: Run RED**

Run: `npm --prefix apps/desktop run test:ui -- src/app/video-studio/unified-material-library-panel.test.tsx src/app/video-studio/index.test.tsx`

- [ ] **Step 3: Implement `UnifiedMaterialLibraryPanel`**

Expose explicit callbacks for select library, add files, select directory, dry-run scan, confirm scan, migration, segment/all matching, clip confirmation, and timeline creation. Render one heading, one selector, library status, management actions, candidate cards, migration notice, and confirmed render basket.

- [ ] **Step 4: Integrate native pickers**

```typescript
const paths = await selectDesktopPaths({
  title: '添加素材到当前资产库', multiple: true,
  filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'avi', 'flv'] }]
})
await namedLibrary.importFiles(paths)

const [directory] = await selectDesktopPaths({
  title: '选择当前资产库的素材目录', directories: true, multiple: false
})
if (directory) await namedLibrary.addSourceRoot(directory)
```

- [ ] **Step 5: Remove competing state and UI**

Remove the cache listing/checklist, generic `videoLibraryAssets` and `videoLibraryClips`, generic refresh/analyze/add-to-mix handlers, dual-write upload handler, and old named panel. Keep `form.localMaterials` only as the hidden final `video_materials` payload populated from confirmed timeline clips.

- [ ] **Step 6: Add explicit migration confirmation**

Import `ConfirmDialog` from `@/components/ui/confirm-dialog`, not `window.confirm`. Hold a local `migrationOpen` boolean and render:

```tsx
<ConfirmDialog
  confirmLabel="确认迁移"
  description={`将旧版素材迁移到 ${selectedLibrary.name}（${selectedLibrary.root}）。旧文件会保留。`}
  onClose={() => setMigrationOpen(false)}
  onConfirm={onMigrateLegacy}
  open={migrationOpen}
  title={`迁移 ${legacyAssetCount} 个旧版素材`}
/>
```

After completion show result counts `新增 / 已存在 / 失败`; never offer deletion.

- [ ] **Step 7: Run GREEN and commit**

```bash
npm --prefix apps/desktop run test:ui -- src/app/video-studio
npm --prefix apps/desktop run typecheck
git add apps/desktop/src/app/video-studio
git commit -m "feat(video-studio): unify the material library entry"
```

## Task 7: Logic Maps and Automated Regression

**Files:**
- Modify: `docs/capabilities/video-studio-logic-map.md`
- Modify: `docs/capabilities/video-material-library-logic-map.md`

- [ ] **Step 1: Run full regression**

```bash
.venv/bin/pytest -q tests/capabilities/test_video_library*.py
npm --prefix apps/desktop run test:ui -- src/app/video-studio
npm --prefix apps/desktop run typecheck
git diff --check
```

Expected: zero failures and zero type/whitespace errors. Record fresh counts.

- [ ] **Step 2: Update Video Studio logic map**

Add stable IDs for the one library page, file/folder/scan/migration/match/confirm actions, and the flow `select -> ingest -> index -> match -> confirm -> cache -> render`. Replace PAGE-005 and RULE-012 statements that describe parallel surfaces.

- [ ] **Step 3: Update material-library logic map**

Document named Obsidian roots, linked source roots, semantic tags, legacy migration-only store, hidden MoneyPrinter cache, human confirmation, path boundaries, and known footage gaps.

- [ ] **Step 4: Commit**

```bash
git add docs/capabilities/video-studio-logic-map.md docs/capabilities/video-material-library-logic-map.md
git commit -m "docs(video-studio): map the unified material workflow"
```

## Task 8: Real Beef-Noodle Desktop Demonstration

**Files:**
- Verify runtime; append observed evidence to both logic maps only.

- [ ] **Step 1: Start the correct desktop**

```bash
hermes-dev-desktop stop
hermes-dev-desktop start
```

Expected: renderer `127.0.0.1:5174` with isolated Hermes Dev home.

- [ ] **Step 2: Verify initial UI with Computer Use**

Open Video Studio in the development Electron app. Confirm exactly one `素材库` heading; no competing old headings; selector value `请选择资产库`; add/directory/scan/match/timeline actions disabled.

- [ ] **Step 3: Select and incrementally scan `beef-noodle`**

Confirm 6 unique assets, 24 ready clips, 0 failed, 0 low-confidence. Dry-run first and verify planned writes stay under the Obsidian root. Confirm the real scan reports 11 skipped and 0 failed.

- [ ] **Step 4: Run the real matching case**

```text
热气腾腾的牛肉汤刚刚出锅，大块牛肉铺满整碗。

师傅在后厨备餐，汤锅热气升腾，饭点就来吃一碗。
```

Run single and match-all, inspect candidates, manually confirm two clips, and do not claim true `动作/拉面` footage without that tag.

- [ ] **Step 5: Create timeline and final preview**

Verify every timeline file is under the asset library `02_精选镜头`, stable cache names appear under MoneyPrinter `storage/local_videos`, and `video_materials` contains only confirmed clips. Run the existing audio/subtitle/video path to a playable final MP4.

- [ ] **Step 6: Verify clearing and record evidence**

Clear the selector and confirm candidates, confirmations, timeline, and render basket return to zero. Record exact automated counts, timeline path, cache filenames, task ID, MP4 path, and footage gaps in both logic maps.

- [ ] **Step 7: Re-run regression and commit evidence**

Run Task 7 commands again. If evidence changed the maps:

```bash
git add docs/capabilities/video-studio-logic-map.md docs/capabilities/video-material-library-logic-map.md
git commit -m "test(video-studio): record unified library acceptance"
```
