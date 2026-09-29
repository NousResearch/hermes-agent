# `hermes channel-skills`

チャンネル単位で自動生成されるスキルを確認・検証・再生成・削除するための管理CLI。

Phase 4では、Phase 3の自動生成管理に加えて、静的バインディングを統合表示し、実際に適用されるスキルを確認できる。

- 自動生成スキルのレジストリ確認
- `SKILL.md`とレジストリの整合性検証
- スキルの再生成
- レジストリからの関連付け解除
- 破損・欠損状態の修復
- 静的なチャンネル・スキル関連付け
- 自動生成ポリシーの有効化・無効化

`hermes skills`はスキルそのものの管理、`hermes channel-skills`はチャンネルとスキルの関連付け・自動生成状態の管理に使い分ける。

## 前提

コマンドは通常、HermesのCLI入口から実行する。

```bash
hermes channel-skills <command>
```

開発ツリーから直接確認する場合は、次の形式でも実行できる。

```bash
python -m hermes_cli.main channel-skills <command>
```

レジストリは次に保存される。

```text
$HERMES_HOME/channel-skills.yaml
```

自動生成されたスキル本体は、通常次の配下に保存される。

```text
$HERMES_HOME/skills/social-media/discord-channel-<channel-id>/SKILL.md
```

`HERMES_HOME`を明示しない場合は、現在のHermesプロファイルのホームが使われる。設定変更を伴うコマンドは、対象プロファイルを確認してから実行する。

## コマンド一覧

### `list`

登録済みチャンネルスキルを一覧表示する。

```bash
hermes channel-skills list
hermes channel-skills list --status ready
hermes channel-skills list --status missing
hermes channel-skills list --status corrupt
hermes channel-skills list --status manual_override
hermes channel-skills list --status generating
hermes channel-skills list --json
```

`--status`で状態を絞り込める。主な状態は次の通り。

- `ready`: レジストリとスキル本体が整合している
- `missing`: スキル本体が存在しない
- `corrupt`: frontmatter、チャンネルID、ハッシュなどの整合性に問題がある
- `manual_override`: 手動編集が検知された
- `generating`: 自動生成処理中として残っている

`--json`はスクリプトや監視から利用するためのJSON出力。

#### 静的バインディングを含める

通常の`list`は自動生成レジストリを表示する。静的バインディングも含める場合は`--all`を付ける。

```bash
hermes channel-skills list --all
hermes channel-skills list --all --source static
hermes channel-skills list --all --effective --json
```

各項目には次の情報が含まれる。

- `source`: `generated`または`static`
- `binding_type`: `generated`、`exact`、`profile`
- `effective_skills`: 継承解決後に実際へ適用されるスキル

### `show <channel_id>`

チャンネルスキルのメタデータとファイル検証結果を表示する。

```bash
hermes channel-skills show 1477833752460923011
hermes channel-skills show 1477833752460923011 --json
```

確認できる主な項目:

- チャンネルID
- プラットフォーム
- レジストリ状態
- プロファイル
- スキルファイルのパス
- ファイルの存在
- frontmatterの有無
- スキル名
- `metadata.hermes.channel_id`
- 解決済みスキル一覧

`show`はメタデータ確認向け。本文を含む詳細確認には`inspect`を使う。

### `effective <channel_id>`

指定チャンネルで、静的設定・プロファイル継承・生成レコードを解決した結果を表示する。

```bash
hermes channel-skills effective 1477833752460923011
hermes channel-skills effective 1477833752460923011 --json
hermes channel-skills effective <thread_id> --parent-id <parent_channel_id>
```

### `validate`

静的バインディングと自動生成レコードを検証する。

```bash
hermes channel-skills validate
hermes channel-skills validate --json
```

検査対象は、存在しないスキル、存在しないプロファイル、破損・欠損した自動生成スキル。

### `inspect <channel_id>`

メタデータに加えて、対象の`SKILL.md`本文を含む詳細情報をJSONで表示する。

```bash
hermes channel-skills inspect 1477833752460923011
hermes channel-skills inspect 1477833752460923011 --json
```

生成内容を確認したい場合や、手動編集の有無を調べたい場合に使う。

### `regenerate <channel_id>`

指定チャンネルの自動生成スキルを再生成する。

```bash
hermes channel-skills regenerate 1477833752460923011
hermes channel-skills regenerate 1477833752460923011 --dry-run
hermes channel-skills regenerate 1477833752460923011 --force
hermes channel-skills regenerate 1477833752460923011 --backup
hermes channel-skills regenerate 1477833752460923011 --json
```

#### `--dry-run`

生成結果を確認するだけで、スキルファイルとレジストリを変更しない。まずこれを実行してから本適用する。

#### `--force`

手動編集が検知されたスキルを上書きする。手動編集内容は失われるため、`inspect`やバックアップで内容を確認してから使う。

通常の再生成では、手動編集されたスキルを保護するため、`--force`なしの上書きを拒否する。

### `remove <channel_id>`

レジストリからチャンネルと生成スキルの関連付けを解除する。

```bash
hermes channel-skills remove 1477833752460923011
hermes channel-skills remove 1477833752460923011 --json
```

デフォルトではレジストリの関連付けだけを解除し、`SKILL.md`は削除しない。

スキル本体も削除する場合は、明示的に`--delete-skill --yes`を指定する。

```bash
hermes channel-skills remove 1477833752460923011 --delete-skill --yes
```

安全のため、次の対象は削除されない。

- `--yes`が付いていない場合
- 手動管理のスキル
- 他のチャンネルから参照されているスキル
- HermesのSkillsディレクトリ外にあるファイル
- `SKILL.md`がシンボリックリンクになっている場合

### `repair`

レジストリとスキル本体の不整合を検出する。

```bash
hermes channel-skills repair
hermes channel-skills repair --json
```

検出結果を確認してから、安全な修復を適用する。

```bash
hermes channel-skills repair --apply
```

`--apply`を付けない場合は検出のみ。破損したスキル本文を推測で上書きするコマンドではなく、主にレジストリ状態の修復に使う。

### `bind <channel_id>`

静的なチャンネル・スキル関連付けを設定する。プロファイルかスキル一覧のどちらか一方を指定する。

```bash
hermes channel-skills bind 1477833752460923011 \
  --profile discord-vocabulary

hermes channel-skills bind 1477833752460923011 \
  --skill discord-vocabulary-channel

hermes channel-skills bind 1477833752460923011 \
  --skill discord-vocabulary-channel \
  --skill another-skill
```

この設定は`config.yaml`の静的設定に保存される。自動生成レジストリの`channel-skills.yaml`とは別の管理対象。

### `unbind <channel_id>`

静的なチャンネル・スキル関連付けを解除する。

```bash
hermes channel-skills unbind 1477833752460923011
```

自動生成レジストリのレコードを削除するコマンドではない。自動生成済みスキルを解除する場合は`remove`を使う。

### `policy`

チャンネルスキルの初回自動生成ポリシーを管理する。

```bash
hermes channel-skills policy show
hermes channel-skills policy show --json
hermes channel-skills policy enable
hermes channel-skills policy disable
```

`enable`と`disable`は対象プロファイルの`config.yaml`を書き換える。許可チャンネル・許可ユーザーの設定は、必要に応じて`config.yaml`で別途指定する。

## 推奨運用フロー

### 既存状態を確認する

```bash
hermes channel-skills list --json
hermes channel-skills show <channel_id>
hermes channel-skills inspect <channel_id>
```

### 再生成する

```bash
hermes channel-skills regenerate <channel_id> --dry-run
hermes channel-skills regenerate <channel_id>
```

手動編集が検知された場合は、内容を保存したうえで必要に応じて次を使う。

```bash
hermes channel-skills regenerate <channel_id> --force
```

### 不整合を修復する

```bash
hermes channel-skills repair --json
hermes channel-skills repair --apply
hermes channel-skills list --json
```

### 関連付けだけ解除する

```bash
hermes channel-skills remove <channel_id>
```

### 生成物も削除する

```bash
hermes channel-skills remove <channel_id> --delete-skill --yes
```

## 安全性と実装上の注意

- レジストリ更新はロック下で行い、ファイルはatomic writeする。
- 同時実行時にレジストリの更新を失わないよう、更新前に最新状態を読み直す。
- 手動編集された`SKILL.md`は、明示的な`--force`なしでは上書きしない。
- 削除対象はHermesのSkillsディレクトリ配下に限定する。
- 静的設定と自動生成レジストリは別管理で、`bind`/`unbind`と`remove`は役割が異なる。
- `--json`は自動処理に使えるが、出力形式はPhase 3の実装に依存するため、利用側で未知のフィールドを許容する。

## トラブルシューティング

### `No channel skills found.`と表示される

現在のプロファイルに自動生成レジストリがない。`HERMES_HOME`またはプロファイル指定が意図したものか確認する。

```bash
echo "$HERMES_HOME"
hermes channel-skills list --json
```

### `missing`になっている

レジストリは存在するが、`SKILL.md`が見つからない。まず状態を確認する。

```bash
hermes channel-skills show <channel_id>
hermes channel-skills repair --dry-run
```

必要なら対象を再生成する。

```bash
hermes channel-skills regenerate <channel_id>
```

### 再生成が拒否される

手動編集が検知されている可能性がある。`inspect`で内容を確認し、上書きしてよい場合だけ`--force`を指定する。

### `--delete-skill`が拒否される

`--delete-skill`には`--yes`が必要。

```bash
hermes channel-skills remove <channel_id> --delete-skill --yes
```

それでも削除されない場合は、手動管理・参照中・Skillsディレクトリ外などの保護条件に該当している。

## Phase 4へ進む前の確認項目

- [ ] `list`、`show`、`inspect`の実CLI出力を確認した
- [ ] `regenerate --dry-run`と通常再生成を確認した
- [ ] 手動編集後の再生成拒否と`--force`を確認した
- [ ] `remove`がデフォルトでファイルを残すことを確認した
- [ ] `remove --delete-skill --yes`の削除保護を確認した
- [ ] `repair`と`repair --apply`を確認した
- [ ] `bind`/`unbind`が静的設定だけを変更することを確認した
- [ ] `policy enable/disable`が意図したプロファイルだけを変更することを確認した
- [ ] 初回投稿で自動生成されたスキルが同じ問い合わせに適用されることを確認した
- [ ] 2回目以降の投稿で既存スキルが再利用されることを確認した
- [ ] Discordのスレッド・チャンネル固有ルールに影響がないことを確認した
