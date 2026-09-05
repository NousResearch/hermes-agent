---
title: "UPSTREAM-REVIEW-PLAN-005: canonical upstream integration state model"
status: "DESIGN_DRAFT_REVIEW_PENDING"
date: 2026-09-05
type: architecture
ticket: t_0c855a90
target_repo: hermes-agent
---

# UPSTREAM-REVIEW-PLAN-005: 上游更新整合模型（review-only / apply 分離）

## 1. 背景

目前的上游更新流程需要一個可重播、可回滾、可供人工批准的狀態模型，
在 review-only 與 apply/deploy 之間做到明確隔離。目標是避免合併候選尚未核准即上線，
並對每一次上游更新都保留可追蹤的候選身分（sha / release-id / 父 commit）。

## 2. 設計決議（Canonical Model）

### 2.1 選型

採用：**Rebase-based candidate + snapshot promotion**（固定比較模型）

### 2.2 非採用項目

不採用：`merge-based review + 直接推進`。

### 2.3 為何採用 rebase

1. 讓待審查候選只保留「本地提交在上游基礎上的重放結果」，
   減少 merge bubble 導致的模糊差異。
2. 候選可完整對照：
   - `upstream_before_sha`
   - `main_before_sha`
   - `candidate_sha`
   - `replayed_local_commit_count`
3. 更容易做到「一個候選只對應一個 release-id」，並能在 apply 前驗證
   `main` 未變動。

## 3. 核心術語

- `candidate`: review-only 產生的上游候選分支/提交（未套用到 `main`）
- `run_id`: 單次更新流程識別碼，固定為時間戳字串（例如 `20260905-104500`）
- `release_id`: 套用後 snapshot 的唯一識別（例如 `hermes-upstream-20260905-104500`）
- `source_sha`: 候選來源分支（通常是 `main` 的原始 head）
- `parent_sha`: 候選提交的直接父 commit

## 4. 狀態機（共用）

**狀態名稱：** `LOCKED / PENDING / APPROVED / FAILED / BLOCKED`

### 4.1 `LOCKED`

- 意義：流程有作業中的獨占鎖，其他流程不得寫入候選/快照/鎖檔。
- 產生條件：script 啟動、關鍵操作進行中。
- 終止條件：完成該步驟並轉移到下一狀態。
- 錯誤處理：若持鎖時間超過 TTL，需人工介入清鎖與重跑。

### 4.2 `PENDING`

- 意義：候選已可審查（review-only 通過）或 apply/deploy 前待確認。
- 產生條件：候選已建立且簽核資訊完整。
- 終止條件：
  - 被 `APPROVED` 接受
  - 被 `BLOCKED` 阻斷（需人工介入）
  - 被 `FAILED` 阻斷（需重跑/修正）

### 4.3 `APPROVED`

- 意義：候選通過人工核准，可進入 apply。
- 產生條件：`approval_token` 檢核成功 + 候選 metadata 一致性驗證通過。
- 終止條件：進入 apply/deploy 流程或失效後回退。

### 4.4 `FAILED`

- 意義：流程中發生可重試、可回修的失敗。
- 終止條件：記錄錯誤原因後結束。
- 允許行為：保留紀錄並可由操作者清理後重跑。

### 4.5 `BLOCKED`

- 意義：流程被外部條件卡住（非腳本可自動修復）。
- 產生條件：例如衝突、dirty 工作樹、權限問題、候選與目前 `main` 不一致。
- 允許行為：人工修復後可重跑，或由審核人員 `REVISE`。

## 5. review-only 狀態流

```
LOCKED -> PENDING (review candidate) -> APPROVED / BLOCKED / FAILED
```

### 5.1 review-only 產生步驟

1. `preflight`（非獨占、只讀）
   - 驗證 `main` 分支、工作樹乾淨、remote/credential 可用、`upstream/main` 可抓取。
2. `LOCKED` 獲取
   - 同步寫入 `.upstream_update.lock`（含 `owner`, `run_id`, `pid`, `expires_at`）。
3. 取得 `upstream_sha`、`source_sha`
4. `rebase` 候選建立
   - 建議分支：`refs/upstream/review/<run_id>`
   - 若失敗，寫入衝突資訊並轉為 `BLOCKED`
5. 過濾空白變更與 noop 更新
   - 若 `upstream` 內容未變，直接 `BLOCKED` 並寫入 `up_to_date = true`
6. 驗證與元資料寫入
   - 測試、語法檢查、摘要（可選）
   - 寫入 metadata，狀態 `PENDING`
7. 通知（非套用）
   - 僅發通知，不重啟、不推送、不部署

### 5.2 review-only 失敗邏輯

- rebase 衝突：`BLOCKED`（需人工處理）
- 驗證失敗：`FAILED`（可重跑或修正）
- lock 遺失或過期：`FAILED`（防呆）

## 6. apply/deploy 狀態流

```
LOCKED (apply lock) -> PENDING (candidate recheck) -> APPROVED (verify)
    -> LOCKED (apply) -> FAILED/BLOCKED/DONE
```

### 6.1 apply/deploy 前置

1. 只接受 `APPROVED` 且 `run_id` 未過期、元資料一致的候選。
2. 檢查 `main` 仍未偏移；若已變動，轉 `BLOCKED`。
3. 同名 review branch 存在且未污染（與 metadata 比對）。

### 6.2 apply/deploy 實作步驟

1. fast-forward 套用候選到 `main`（必要且必需）。
2. 寫入 `applied_main_sha` 與 `release_id`。
3. 產生 snapshot / release artifact（必要時建立暫存目錄與 checksum）。
4. service 健康檢查與回滾策略檢核。
5. `post-apply` 驗證失敗則回滾到 `source_sha`，標記 `FAILED`。
6. 成功則標記 `DONE`（此文件使用 `DONE` 當 apply 階段終點，
   非主狀態枚舉）。

### 6.3 apply/deploy 失敗邏輯

- precheck 失配：`BLOCKED`
- 套用提交失敗：`FAILED`
- snapshot/build 失敗：`FAILED`
- restart/health 失敗：`FAILED`（自動回滾為必要）
- 推送失敗：`BLOCKED`，保留已套用主庫與可重試標記

## 7. 候選元資料（必要欄位）

每個候選必須寫入一份 JSON，至少包含：

```json
{
  "schema_version": "1.0",
  "run_id": "20260905-104500",
  "mode": "review-only",
  "status": "PENDING",
  "created_at_utc": "2026-09-05T02:45:00Z",
  "upstream_sha": "<sha>",
  "source_sha": "<main_head_before_rebase>",
  "parent_sha": "<candidate_parent>",
  "candidate_sha": "<sha_after_rebase>",
  "local_base_sha": "<git merge-base upstream/main main>",
  "replayed_local_commit_count": 3,
  "local_commit_ids": ["<old_sha_1>", "<old_sha_2>", "<old_sha_3>"],
  "rewritten_local_commit_ids": ["<new_sha_1>", "<new_sha_2>", "<new_sha_3>"],
  "release_id": "hermes-upstream-20260905-104500",
  "review_branch": "upstream/review/20260905-104500",
  "lock_id": "<lock_token>",
  "checks": {
    "preflight_ok": true,
    "syntax_ok": true,
    "tests_ok": false,
    "noop": false
  },
  "error_code": null,
  "retry_count": 0,
  "approval": {
    "approved_by": null,
    "approved_at_utc": null,
    "approval_token": null
  }
}
```

### 7.1 儲存路徑

建議放在：
`$HERMES_HOME/hermes-upstream-state/candidates/<run_id>.json`

### 7.2 不可遺漏欄位

- `candidate_sha`、`release_id`、`source_sha`、`parent_sha`、`status`、`mode`、`run_id`

## 8. 鎖檔與 marker 規則

1. 鎖檔：`$HERMES_HOME/hermes-upstream-state/update.lock`
2. review branch：`refs/upstream/review/<run_id>`
3. snapshot 目錄：`$HERMES_UPDATES_DIR/releases/<release_id>/`
4. 過期清理：
   - 鎖檔 TTL 過期
   - 與元資料不符的 review branch
   - 無法解析的 JSON 元資料
5. 一次只允許一個流程持有 lock；任何時候若發現二次持有即視為 `BLOCKED`。

## 9. 安全與回滾契約

- 不可直接在 live 系統上 patch；
  一定從候選 snapshot 驗證再切換。
- 所有錯誤訊息不得外洩 token/secret/訊息本體/憑證。
- 回滾順序：
  1) 還原 `main` 到 `source_sha`
  2) 清理 review branch 與 candidate markers
  3) 將 snapshot 標記為 `failed`
  4) 將 run state 設為 `FAILED`

## 10. 失效矩陣（範例）

- `NOOP_UPDATE`：`upstream_sha == source_sha 的祖先 + 無檔案差異` → `PENDING` with `noop=true`，可關閉提案。
- `DIRTY_WORKTREE`：`BLOCKED`
- `UPSTREAM_FETCH_FAIL`：`FAILED`（可重試）
- `REBASE_CONFLICT`：`BLOCKED`
- `LOCK_STALE`：`FAILED` 並提示人工清鎖
- `APPLY_PRECHECK_STALE_MAIN`：`BLOCKED`
- `POSTCHECK_FAIL`：`FAILED` 並回滾
- `HEALTH_FAIL`：`FAILED` 並回滾

## 11. reviewer checklist（實作前必填）

每位 reviewer 需回覆以下欄位：

- `model_decision`: 需確認是 rebase-based + review/apply 分離
- `state_machine`: 能否重建五狀態行為（LOCKED/PENDING/APPROVED/FAILED/BLOCKED）
- `metadata_schema`: 5 個關鍵欄位是否齊全
- `rollback_plan`: apply/deploy 失敗時是否有明確回滾點
- `operator_invariant`: 是否需要人工批准才能 apply/deploy

## 12. reviewer consensus record（執行前必需）

### 12.1 要求格式

```
decision: PASS | REVISE
reviewer: <AGY|Claude|other>
run_id: t_0c855a90 或對應 update run id
summary: <一句話結論>
findings:
  - id: F-001
    severity: P0|P1|P2|P3
    detail: <缺口/風險描述>
    required_action: <需要修正的事項>
correction_set: []  # 若 decision == PASS -> 空陣列；若 decision == REVISE -> 非空
  - id: C-001
    target: <文件段落>
    change_required: <應更改項目>
notes:
  - <補充說明>
```

### 12.2 實施 gate

- 任何 `decision: REVISE` 不得繼續實作變更；需先落地 correction_set 並回填。
- 所有 required reviewer 的 consensus 必須存在且一致（至少一個 `PASS`、且不存在 `BLOCKED`）。

## 13. 與既有機制的對接

- 既有 `hermes_cli/release_markers.py` 和 `scripts/release_snapshot.py` 可保留既定介面；
  本設計只要求在兩階段中補上 candidate metadata、狀態紀錄、人工批准 gate。
- `hermes_upstream_update_guard.sh` 應先以本文件的 `review-only` contract 為準，
  成功後才能驅動 `hermes_upstream_apply.sh`。