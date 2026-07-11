# Video Studio Named Library Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit named-library selection and script-to-shot matching to Hermes Desktop Video Studio while preserving the separate generic upload library.

**Architecture:** Extend the existing video-library client so every named-library request carries `libraryId`, fix backend tag writes to resolve the selected database, and put selection/matching state in a focused React hook and panel. Video Studio remains the composition root; no library is selected on entry, and changing libraries clears candidates and confirmed shots.

**Tech Stack:** React 19, TypeScript, Vitest, Testing Library, FastAPI, Python, pytest, existing `capabilities.video_library` service/store.

---

## File map

- Modify `capabilities/video_library/adapter.py`: route tag writes and limited searches through the named service.
- Modify `hermes_cli/web_server.py`: forward `library_id`, query, tag, and limit.
- Modify `tests/capabilities/test_video_library_web_routes.py`: prove named-library HTTP isolation.
- Modify `apps/desktop/src/app/video-studio/moneyprinter-client.ts`: named-library types and request methods.
- Modify `apps/desktop/src/app/video-studio/moneyprinter-client.test.ts`: request mapping tests.
- Create `apps/desktop/src/app/video-studio/named-library-matching.ts`: pure segment and match-state helpers.
- Create `apps/desktop/src/app/video-studio/named-library-matching.test.ts`: pure behavior tests.
- Create `apps/desktop/src/app/video-studio/use-named-video-library.ts`: library controller.
- Create `apps/desktop/src/app/video-studio/use-named-video-library.test.tsx`: hook tests.
- Create `apps/desktop/src/app/video-studio/named-library-panel.tsx`: selector and candidates.
- Create `apps/desktop/src/app/video-studio/named-library-panel.test.tsx`: component tests.
- Modify `apps/desktop/src/app/video-studio/index.tsx`: compose the feature.

### Task 1: Preserve named-library isolation in the backend

**Files:**
- Modify: `capabilities/video_library/adapter.py:96-135`
- Modify: `hermes_cli/web_server.py:506-536`
- Test: `tests/capabilities/test_video_library_web_routes.py`

- [ ] **Step 1: Write the failing route test**

```python
def test_named_library_clip_query_and_tag_write_are_isolated(clients, monkeypatch):
    _anonymous, authenticated, default_service = clients
    from capabilities.video_library import adapter

    named = VideoLibraryService(VideoLibraryStore(root=default_service.store.root.parent / "named"))
    calls = []
    monkeypatch.setattr(adapter, "get_named_service", lambda _library_id: named)
    monkeypatch.setattr(
        named.store,
        "search_clips",
        lambda query, *, tag=None, limit=50: calls.append((query, tag, limit)) or [{"id": "named-clip"}],
    )
    monkeypatch.setattr(
        named.store,
        "replace_clip_tags",
        lambda clip_id, tags: [{"id": "tag-1", "name": tags[0]["name"]}],
    )

    listed = authenticated.get(
        "/api/capabilities/video-library/clips",
        params={"library_id": "beef-noodle", "query": "热气牛肉", "tag": "场景/后厨", "limit": 5},
    )
    tagged = authenticated.post(
        "/api/capabilities/video-library/clips/named-clip/tags",
        json={"libraryId": "beef-noodle", "tags": ["人工确认"]},
    )

    assert listed.json()["data"]["clips"] == [{"id": "named-clip"}]
    assert calls == [("热气牛肉", "场景/后厨", 5)]
    assert tagged.json()["data"]["tags"][0]["name"] == "人工确认"
```

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/capabilities/test_video_library_web_routes.py::test_named_library_clip_query_and_tag_write_are_isolated -q
```

Expected: FAIL because `limit` is not forwarded and tag replacement uses the default service.

- [ ] **Step 3: Implement minimal forwarding**

In `replace_clip_tags_data`:

```python
library_id = str(body.get("libraryId") or body.get("library_id") or "").strip() or None
tags = _request_service(library_id).store.replace_clip_tags(clip_id, normalized)
```

Add `limit: int = 50` to `list_clips_data`, pass it to `store.search_clips`, and forward the FastAPI `limit` query parameter.

- [ ] **Step 4: Run GREEN**

```bash
.venv/bin/python -m pytest tests/capabilities/test_video_library_web_routes.py tests/capabilities/test_video_library_store.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add capabilities/video_library/adapter.py hermes_cli/web_server.py tests/capabilities/test_video_library_web_routes.py
git commit -m "fix(video-library): preserve named library isolation"
```

### Task 2: Complete the desktop named-library client

**Files:**
- Modify: `apps/desktop/src/app/video-studio/moneyprinter-client.ts:430-740`
- Test: `apps/desktop/src/app/video-studio/moneyprinter-client.test.ts`

- [ ] **Step 1: Write a failing request contract test**

```typescript
it('sends the selected library id on named-library requests', async () => {
  const api = vi.fn().mockResolvedValue({ data: {}, error: null, ok: true })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  await videoLibraryClient.listLibraries()
  await videoLibraryClient.listAssets('beef-noodle')
  await videoLibraryClient.listClips('beef-noodle', { limit: 5, query: '热气 牛肉' })
  await videoLibraryClient.replaceClipTags('beef-noodle', 'clip-1', ['人工确认'])
  await videoLibraryClient.createTimeline('beef-noodle', ['clip-1'], '9:16', [{ text: '后厨现煮' }])

  expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/capabilities/video-library/libraries' }))
  expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/capabilities/video-library/assets?library_id=beef-noodle' }))
  expect(api).toHaveBeenCalledWith(expect.objectContaining({
    body: { libraryId: 'beef-noodle', tags: ['人工确认'] }
  }))
  expect(api).toHaveBeenCalledWith(expect.objectContaining({
    body: { aspect: '9:16', clipIds: ['clip-1'], libraryId: 'beef-noodle', script: [{ text: '后厨现煮' }] }
  }))
})
```

- [ ] **Step 2: Run RED**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio/moneyprinter-client.test.ts
```

Expected: FAIL because the named methods/signatures do not exist.

- [ ] **Step 3: Implement the client contract**

Add `VideoLibraryDescriptor`, `VideoLibraryStatus`, and `VideoLibraryClipQuery`. Use `URLSearchParams`, then implement:

```typescript
listLibraries()
getLibraryStatus(libraryId)
scanLibrary(libraryId, dryRun)
listAssets(libraryId?)
listClips(libraryId?, query?)
analyzeAsset(libraryId, assetId)
replaceClipTags(libraryId, clipId, tags)
createTimeline(libraryId, clipIds, aspect, script)
```

Keep optional `libraryId` only for generic upload-library compatibility.

- [ ] **Step 4: Run GREEN**

Run the same Vitest command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/video-studio/moneyprinter-client.ts apps/desktop/src/app/video-studio/moneyprinter-client.test.ts
git commit -m "feat(video-studio): add named library client"
```

### Task 3: Model script-to-shot matching

**Files:**
- Create: `apps/desktop/src/app/video-studio/named-library-matching.ts`
- Create: `apps/desktop/src/app/video-studio/named-library-matching.test.ts`

- [ ] **Step 1: Write failing pure tests**

```typescript
it('splits editable script into stable non-empty segments', () => {
  expect(segmentVideoScript('后厨现煮。\n\n大块牛肉看得见！')).toEqual([
    { id: 'segment-1', text: '后厨现煮。' },
    { id: 'segment-2', text: '大块牛肉看得见！' }
  ])
})

it('clears candidates and confirmations when the library changes', () => {
  const dirty = { candidatesBySegment: { 'segment-1': [clip] }, confirmedBySegment: { 'segment-1': 'clip-1' }, errorsBySegment: {} }
  expect(clearLibraryMatches(dirty)).toEqual(emptyMatchState())
})

it('does not treat the first candidate as human confirmation', () => {
  const next = setSegmentCandidates(emptyMatchState(), 'segment-1', [clip])
  expect(next.confirmedBySegment).toEqual({})
})
```

- [ ] **Step 2: Run RED**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio/named-library-matching.test.ts
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement pure helpers**

Export `ScriptSegment`, `NamedLibraryMatchState`, `segmentVideoScript`, `emptyMatchState`, `clearLibraryMatches`, `setSegmentCandidates`, `confirmSegmentClip`, and `setSegmentError`. Split on blank lines first, sentence punctuation second, trim empty values, and assign IDs by final order.

- [ ] **Step 4: Run GREEN**

Run the same Vitest command. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/video-studio/named-library-matching.ts apps/desktop/src/app/video-studio/named-library-matching.test.ts
git commit -m "feat(video-studio): model script shot matching"
```

### Task 4: Implement the named-library controller

**Files:**
- Create: `apps/desktop/src/app/video-studio/use-named-video-library.ts`
- Create: `apps/desktop/src/app/video-studio/use-named-video-library.test.tsx`

- [ ] **Step 1: Write failing hook tests**

```typescript
it('loads libraries without automatically selecting one', async () => {
  const client = fakeClient({ libraries: [{ id: 'beef-noodle', name: '牛肉面资产库' }] })
  const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
  await waitFor(() => expect(result.current.libraries).toHaveLength(1))
  expect(result.current.selectedLibraryId).toBe('')
  expect(client.listAssets).not.toHaveBeenCalled()
})

it('clears matches when switching libraries', async () => {
  const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
  await act(() => result.current.selectLibrary('beef-noodle'))
  await act(() => result.current.matchSegment('segment-1'))
  await act(() => result.current.confirmClip('segment-1', 'clip-1'))
  await act(() => result.current.selectLibrary('second-library'))
  expect(result.current.matches.confirmedBySegment).toEqual({})
})

it('keeps successful segments when match-all partially fails', async () => {
  const { result } = renderHook(() => useNamedVideoLibrary({ client: partiallyFailingClient, script: '第一段。\n\n第二段。' }))
  await act(() => result.current.selectLibrary('beef-noodle'))
  await act(() => result.current.matchAll())
  expect(result.current.matches.candidatesBySegment['segment-1']).toHaveLength(1)
  expect(result.current.matches.errorsBySegment['segment-2']).toBeTruthy()
})
```

- [ ] **Step 2: Run RED**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio/use-named-video-library.test.tsx
```

Expected: FAIL because the hook does not exist.

- [ ] **Step 3: Implement the hook**

Expose libraries, selected ID, status, assets, clips, segments, match state, busy state, `selectLibrary`, `refreshSelectedLibrary`, `scanSelectedLibrary`, `matchSegment`, `matchAll`, `confirmClip`, and `createTimeline`.

Rules:

- Initial selection is always empty and never read from storage.
- Selecting/clearing/changing a library clears prior matches.
- No selection rejects matching.
- `matchAll` uses `Promise.allSettled`.
- Timeline clip IDs follow segment order and always include `selectedLibraryId`.

- [ ] **Step 4: Run GREEN**

Run the same hook test. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/video-studio/use-named-video-library.ts apps/desktop/src/app/video-studio/use-named-video-library.test.tsx
git commit -m "feat(video-studio): control named video libraries"
```

### Task 5: Build the manual selector and candidate panel

**Files:**
- Create: `apps/desktop/src/app/video-studio/named-library-panel.tsx`
- Create: `apps/desktop/src/app/video-studio/named-library-panel.test.tsx`

- [ ] **Step 1: Write failing component tests**

```typescript
it('requires manual selection before matching', () => {
  render(<NamedLibraryPanel {...baseProps} selectedLibraryId="" />)
  expect(screen.getByLabelText('视频资产库')).toHaveValue('')
  expect(screen.getByRole('button', { name: '自动匹配全部文案' })).toBeDisabled()
  expect(screen.getByText('请先选择资产库')).toBeTruthy()
})

it('shows candidates without silently confirming one', () => {
  render(<NamedLibraryPanel {...matchedProps} />)
  expect(screen.getByText('后厨煮面工位近景')).toBeTruthy()
  expect(screen.getByText('质量 0.80')).toBeTruthy()
  expect(screen.getByRole('button', { name: '选用这个镜头' })).toBeTruthy()
  expect(screen.queryByText('已确认')).toBeNull()
})
```

- [ ] **Step 2: Run RED**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio/named-library-panel.test.tsx
```

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement the panel**

Use an empty-first `<select aria-label="视频资产库">`, status counts, refresh/scan actions, single-segment and match-all actions, candidate cards with keyframe/description/time/quality/confidence/score/source, explicit confirmation, and a timeline button enabled only after confirmation.

- [ ] **Step 4: Run GREEN**

Run the same component test. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop/src/app/video-studio/named-library-panel.tsx apps/desktop/src/app/video-studio/named-library-panel.test.tsx
git commit -m "feat(video-studio): add named library matching panel"
```

### Task 6: Integrate with Video Studio

**Files:**
- Modify: `apps/desktop/src/app/video-studio/index.tsx:130-430`
- Modify: `apps/desktop/src/app/video-studio/index.tsx:1450-1605`
- Test: `apps/desktop/src/app/video-studio/named-library-panel.test.tsx`

- [ ] **Step 1: Add a failing integration test**

```typescript
it('does not restore a named library from the saved MoneyPrinter draft', () => {
  writeKey('hermes-video-studio-moneyprinter-draft-v1', JSON.stringify({ selectedLibraryId: 'beef-noodle' }))
  render(<VideoStudioView />)
  expect(screen.getByLabelText('视频资产库')).toHaveValue('')
})
```

- [ ] **Step 2: Run RED**

Run the focused Video Studio tests. Expected: FAIL because the panel is not composed.

- [ ] **Step 3: Compose hook and panel**

```typescript
const namedLibrary = useNamedVideoLibrary({
  client: videoLibraryClient,
  script: form.videoScript,
  terms: form.videoTerms
})
```

Keep two labeled sections: “上传素材库” for existing generic import/analyze behavior and “Obsidian 具名资产库” for the new panel. Named clips must be materialized through a named timeline before entering the existing local-material mix.

- [ ] **Step 4: Run Video Studio tests and typecheck**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio
npm --workspace apps/desktop run typecheck
```

Expected: all tests pass; TypeScript exits 0.

- [ ] **Step 5: Commit**

Stage only the named-library implementation hunks and files, then:

```bash
git commit -m "feat(video-studio): integrate named asset matching"
```

### Task 7: Regression and desktop acceptance

**Files:** Verify only; change implementation files only when evidence exposes a defect.

- [ ] **Step 1: Run backend regression**

```bash
.venv/bin/python -m pytest \
  tests/capabilities/test_video_library_config.py \
  tests/capabilities/test_video_library_store.py \
  tests/capabilities/test_video_library_service.py \
  tests/capabilities/test_video_library_semantic.py \
  tests/capabilities/test_video_library_batch.py \
  tests/capabilities/test_video_library_web_routes.py \
  tests/capabilities/test_moneyprinter_mcp_tools.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run desktop regression**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio
npm --workspace apps/desktop run typecheck
```

Expected: all tests pass and typecheck exits 0.

- [ ] **Step 3: Start the correct app**

```bash
hermes-dev-desktop start
```

Expected: `hermes-desktop-dev` is running on 127.0.0.1:5174 with isolated Hermes Dev paths.

- [ ] **Step 4: Perform click-level acceptance**

1. Open Video Studio.
2. Confirm the selector is empty.
3. Manually select `beef-noodle`.
4. Confirm 6 assets and 24 clips.
5. Enter two script paragraphs.
6. Test “匹配此段”.
7. Test “自动匹配全部文案”.
8. Explicitly select at least one candidate.
9. Create a timeline and verify its path is inside the beef-noodle root.
10. Clear the selector and verify candidates disappear.

- [ ] **Step 5: Inspect the final worktree**

```bash
git diff --check
git status --short
```

Preserve unrelated dirty-worktree changes. Do not create an empty commit.
