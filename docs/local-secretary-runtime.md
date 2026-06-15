# Local Secretary Runtime (RTX 3060 + Ryzen 5 4600)

Hermes を **コーディングエージェントではなくローカル秘書** として動かすための RTX 3060 向けランタイム手順。主推論は **llama.cpp server の OpenAI 互換 API**、Ollama は試験用 profile として残す。

## 推奨モデル

| 役割 | HuggingFace repo id | alias | ポート |
|------|---------------------|-------|--------|
| 主モデル | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M` | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M` | 8080 |
| フォールバック | `NousResearch/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M` | `hermes3-8b-fallback` | 8081 |
| 軽量補助 | `unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M` | `phi4-mini-aux` | 8082 |

**必須:** llama.cpp 起動時は常に `--jinja` を付ける。tool calling smoke test でプレーンテキスト返却が出たら `--jinja` 不足。

**Context:** Hermes 側 `context_length` 目標は **65536**。64000 未満は起動前チェックで警告/失敗。

## クイックスタート (Windows)

```powershell
# 1) llama.cpp 主モデル
powershell -ExecutionPolicy Bypass -File scripts/windows/start-llama-secretary.ps1

# 2) 契約 smoke test (JSON 出力)
powershell -ExecutionPolicy Bypass -File scripts/windows/check-local-llm.ps1

# 3) Hermes config 例を ~/.hermes/config.yaml にマージ
#    config/local-secretary.example.yaml を参照

# 4) Irodori-TTS (任意)
powershell -ExecutionPolicy Bypass -File scripts/windows/start-irodori-tts-server.ps1
powershell -ExecutionPolicy Bypass -File scripts/windows/test-irodori-tts.ps1
```

## GPU VRAM 別 tuning (RTX 3060)

### 12GB

- `HERMES_LLAMA_CTX=65536`
- `HERMES_LLAMA_GPU_LAYERS=99`
- `HERMES_LLAMA_CACHE_TYPE_K=q4_0` / `HERMES_LLAMA_CACHE_TYPE_V=q4_0`
- `HERMES_LLAMA_SPEC_TYPE=ngram-mod`

### 8GB

- 主モデル Q4_K_XL のまま `HERMES_LLAMA_GPU_LAYERS=32` から開始
- KV cache は `q4_0` 維持、OOM 時は `16` まで段階降下
- それでも無理なら `start-llama-secretary-fallback.ps1` (Hermes-3 8B)

### 6GB

- 主モデルより **Hermes-3 8B fallback を primary** にする運用を推奨
- `HERMES_LLAMA_CTX=65536` は維持しつつ quant / GPU layers を下げる
- Phi-4 mini は `:8082` の補助タスク専用

## llama.cpp profile 環境変数

| 変数 | 既定値 |
|------|--------|
| `HERMES_LLAMA_SERVER_EXE` | `%LOCALAPPDATA%\Programs\llama-turboquant\bin\llama-server.exe` |
| `HERMES_LLAMA_MODEL` | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M` |
| `HERMES_LLAMA_GGUF_PATH` | 空なら `--hf-repo`、ローカル運用では `~/.hermes/.env` で実GGUFへ設定 |
| `HERMES_LLAMA_ALIAS` | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M` |
| `HERMES_LLAMA_HOST` | `127.0.0.1` |
| `HERMES_LLAMA_PORT` | `8080` |
| `HERMES_LLAMA_CTX` | `65536` |
| `HERMES_LLAMA_GPU_LAYERS` | `99` |
| `HERMES_LLAMA_CACHE_TYPE_K/V` | `q4_0` |
| `HERMES_LLAMA_SPEC_TYPE` | `ngram-mod` |
| `HERMES_LLAMA_SPEC_DRAFT_N_MAX` | `64` |
| `HERMES_LLAMA_PROFILE` | `rtx3060` |

起動 script は `--help` を見て存在しない option は自動で外す:

- `--cache-type-k/v` 非対応 → f16 cache に警告付きフォールバック
- `--spec-type ngram-mod` 非対応 → speculative flags なしで再起動
- CUDA OOM → GPU layers を `99→32→24→16→8` と再試行
- 最終失敗 → `start-llama-secretary-fallback.ps1`

## Ollama trial profile

試験のみ。本番 primary は llama.cpp のまま。

```powershell
# config/local-secretary.ollama.example.yaml を参考に profile を分ける
hermes -p ollama-trial gateway
```

`num_ctx: 65536` を Ollama `extra_body.options` に必ず設定。

## TurboQuant experimental profile

`HERMES_LLAMA_SERVER_EXE` を TurboQuant ビルドの `llama-server.exe` に差し替え可能。KV `turbo4` / `f16` 組み合わせは VRAM と速度のトレードオフ。OOM 時は secretary launcher と同じ GPU layer 降下ロジックを使う。

## Speculative decoding profile

- `HERMES_LLAMA_SPEC_TYPE=ngram-mod` (対応ビルド)
- `HERMES_LLAMA_SPEC_DRAFT_N_MAX=64` → `--spec-ngram-mod-n-max`
- draft/MTP 系は別ビルド向け。非対応なら script が自動で外す

## Hermes config

`config/local-secretary.example.yaml` を `~/.hermes/config.yaml` にマージ:

- `model.provider: custom`
- `model.base_url: http://127.0.0.1:8080/v1`
- `model.context_length: 65536`
- `fallback_providers` で 8081/8082 へフェイルオーバー
- `local_secretary.*` で read/write 方針と緊急報道モード

## Google Workspace

既存 skill: `skills/productivity/google-workspace`

1. `~/.hermes/google_client_secret.json` を配置
2. `python skills/productivity/google-workspace/scripts/setup.py --auth-url`
3. token は `~/.hermes/google_token.json`

**Read-only 標準:** Gmail search / Calendar list
**Confirmation 必須:** Gmail send, Calendar create/update/delete

wrapper: `agent/local_secretary/google_workspace_actions.py`（JSON 統一 + gate）

Gmail search syntax 例: `from:boss newer_than:7d has:attachment`
Calendar range: `today` / `tomorrow` / `this_week`（`Asia/Tokyo` 固定）

## 緊急報道モード

`local_secretary.emergency_news_mode`:

- 単一ソースで断定しない (`require_multiple_sources: true`)
- 各項目に **取得時刻 JST** を付ける
- 優先ソース: 官公庁 → 通信社 → 放送局 → 交通機関 → 自治体
- 不明は「不明」と明示
- X 投稿・外部通知は **confirmation なしでは実行しない**

## 時刻表 / 交通 (拡張ポイント)

最初は Web / browser search ベース。`local_secretary.timetable` に adapter 名を置き、後から GTFS / 鉄道 API を差し替え可能。

必須フィールド:

- 出発地 / 目的地
- 出発 or 到着時刻
- 運休・遅延
- 取得時刻 (JST)

## X 投稿フロー

Skill: `skills/productivity/x-poster`

```powershell
py -3 skills/productivity/x-poster/scripts/x_post.py draft --text "下書き"
py -3 skills/productivity/x-poster/scripts/x_post.py publish --text "本文" --dry-run
py -3 skills/productivity/x-poster/scripts/x_post.py publish --text "本文" --confirmed
```

- `draft` → 自動 OK
- `publish` → **必ず `--confirmed`**（または dry-run）
- 確認画面項目: 文字数, URL, 添付, アカウント, 投稿文
- 認証情報は `~/.hermes/.env` のみ (`X_*`)

## Irodori-TTS

- Endpoint: `http://127.0.0.1:8088/v1/audio/speech`
- 起動: `scripts/windows/start-irodori-tts-server.ps1`
- 環境変数: `IRODORI_TTS_DIR`, `IRODORI_TTS_HOST`, `IRODORI_TTS_PORT`, `IRODORI_TTS_BACKEND`, `IRODORI_TTS_DEFAULT_VOICE`, `IRODORI_TTS_OUTPUT_DIR`
- CLI: `skills/audio/irodori-tts/scripts/irodori_tts.py`
- 長文は sentence chunking、metadata JSON を sidecar 出力
- 既定 **autoplay なし**

## Write-action confirmation 方針

| 自動 OK (read / local) | Confirmation 必須 |
|------------------------|-------------------|
| Web 検索 | X publish |
| Gmail 読取 / search | Gmail send |
| Calendar 読取 / list | Calendar create/update/delete |
| ニュース収集 (read) | 外部書込み全般 |
| TTS 生成 (local) | shell 実行 |

実装: `agent/local_secretary/write_action_gate.py`

## Troubleshooting

| 症状 | 対処 |
|------|------|
| tool call が文章で返る | `llama-server` に `--jinja` を付けて再起動 |
| context 不足 | `-c 65536` / Ollama `num_ctx` / Hermes `context_length` を確認 |
| CUDA OOM | quant 下げる、KV cache `q4_0`、GPU layers 降下、fallback 8B |
| 速度が遅い | ngram-mod、cache type、モデルサイズを比較 |
| Google OAuth 失敗 | token / scopes / test user / client secret パス |
| TTS が遅い | `IRODORI_TTS_BACKEND=cuda`、CPU fallback 時は chunk 数を減らす |

## テスト

```powershell
scripts/run_tests.sh tests/local/ -q
```

Live llama server がある場合のみ `@pytest.mark.integration` の live テストが走る。

## Secrets

**`.env` のみ。** git commit 禁止。例:

```env
HERMES_LLAMA_SERVER_EXE=C:/path/to/llama-server.exe
X_API_KEY=...
GOOGLE_CLIENT_SECRET=...  # ファイル運用推奨: google_client_secret.json
IRODORI_API_KEY=optional
X_DRY_RUN=1
```
