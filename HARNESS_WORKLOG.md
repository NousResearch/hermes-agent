# Harness stabilization

目的: sandbox を維持したまま、実行上限・承認境界・検証証拠を実コードと回帰テストで改善する。

基準: 開始時 HEAD / ローカル origin/main は `a55c972e09`。作業ツリーは clean。
リモート最新状態は未確認。実利用の入口・失敗例はユーザーに確認中。

実装済み:
- 子エージェントのバッチ全体が残り上限に収まらなければ起動しない。
- 停止判定後の未開始ツールを共通ディスパッチで拒否する。既に実行中の並列処理を強制停止する変更ではない。
- SSH config の承認判定に実際の書き込みと同じ task cwd / symlink 解決を使う。
- 部分テストを targeted と記録し、collect-only/help/version を検証証拠にしない。
- TypeScript自動lintは offline + no-install。未導入時は未検証としてスキップ。
- `scripts/sandbox-coding/` に既存設定で停止・時間予算・検証を有効化するCLI向け設定と起動手順。

検証:
- 起動上限、SSH承認、検証証拠の14ケースが修正前に失敗することを確認。
- 同一バッチの停止後実行、自動lintの不要な通信も修正前に再現。
- 最初の関連17ファイル: 276 passed / 2 skipped。
- 追加9ファイル: 174 passed / 2 failed / 4 skipped。2失敗はmacOSの `/tmp` → `/private/tmp`
  正規化を無視した既存テストで、修正前の本体コードでも同じ失敗を確認。期待値を実ホストでの解決済みパスに修正。
- 修正した既存テストファイルの再実行: 47 passed / 2 skipped。
  最終結果は重複を除いて26ファイル、452 passed / 6 skipped / 0 failed。
  変更したPythonファイルの Ruff、`git diff --check` も成功。
- 自動lintの停滞を含んだ verification_evidence テストは31.9秒→1.8秒程度。
  全タスクの高速化率ではなく、依存未導入環境における待ち時間の除去。
- 実設定ローダーでCLIツール集合、Docker backend / network=false / mount_cwd=false、停止・検証有効化を確認。
- `.venv` は lockfile 通りに作成。既存ユーザー設定・稼働環境は変更していない。

未解決: 実利用タスクの品質・速度の比較、モデルAPI/実コンテナとの E2E。
この端末にDocker CLIがなく、実コンテナ起動は未実施。sandbox解除可とは判定していない。
次の一手: 代表タスクと使用モデル・入口を確定し、受け入れ条件付きで同一条件比較する。
