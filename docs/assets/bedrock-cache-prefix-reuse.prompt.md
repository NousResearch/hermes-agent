# Bedrock Cache Prefix Reuse Infographic

Use case: PR explanation image for non-engineers.

Goal: Explain that Hermes now reduces Bedrock cache write cost by keeping stable prompt parts reusable and moving volatile session-specific parts behind the cache checkpoint.

Displayed Japanese copy:

- Title: `Bedrock キャッシュ改善のしくみ`
- Subtitle: `毎回「書き込む」量を減らし、同じ内容は「読み込んで再利用」する変更`
- Before: `毎回、同じ大きな荷物を送り直す`
- After: `変わらない部分だけを固定して再利用`
- Key point: `ツール定義 + 安定したシステム指示` are reused, while `メモリ・日時・セッション情報` are passed at the tail.
- Metrics: `書き込み 35,478 -> 4,685`, `読み込み 0 -> 30,793`, `概算コスト $0.2219 -> $0.0449`

Style:

- 16:9 corporate infographic.
- White and light gray background.
- Red before panel, green/blue after panel, yellow volatile tail.
- Large Japanese sans-serif text.
- No logos, no people, no decorative background art.

Generation method:

- Deterministic Pillow rendering was used instead of AI image generation to avoid Japanese text corruption.
