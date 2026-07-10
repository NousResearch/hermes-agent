# MEMORY.md (Hermes Brain)

## Purpose

長期的に有効な運用知見を保持し、同じ失敗を繰り返さないためのメモリ。

## Keep

- 再現性のある不具合原因と修正手順
- 起動・依存・環境差分（Windows/macOS/Linux）に関する注意点
- よく使う検証コマンドと期待される出力

## Avoid

- 秘密情報（鍵、トークン、個人情報）
- 一時的・再現不能なノイズログ

## Format

- 事実（What happened）
- 原因（Why）
- 対応（Fix）
- 再発防止（Prevention）

## Video routing (2026-07)

- **What:** 動画作成の第一選択は `hyperframes`（heygen-com/hyperframes, HTML→MP4）。`plugins/hyperframes` + `~/.hermes/skills/hyperframes`。
- **Why:** モーショングラフィックス・字幕・サイトキャプチャ・プロモが主用途。`manim-video` は純数学/幾何のみ。
- **Fix:** `hermes hyperframes install`（UACで Node/FFmpeg/npm CLI）。ツールセット `hyperframes` + `terminal` を有効化。
- **OAuth:** HyperFrames 本体に Google OAuth は不要。任意で `GEMINI_API_KEY` はレンダ後 `video_analyze` QA 用のみ（`.env` に API キー、OAuth ではない）。
