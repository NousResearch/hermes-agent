#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:/Users/corek/AppData/Local/hermes/skills/personal-knowledge/episodic-memory/tools")
from episodic_memory import memory_store

memories = [
    ('Novel2Hermes_jpスキル: 日本語小説執筆支援パイプライン。fingerprint（文体ベクトル）抽出→plot生成(noise=0.35で制御されたセレンディピティ)→draft(char voice一貫性)→revise(keeper選別)。タグ: work/series/chapter/scene/beat/pov/type/theme/status/quality。月単位で文体フィンガープリント浮上、没アイデア再浮上、クロスシリーズリンク発見。episodic-memoryのCreative DNA応用。', ['novel', 'episodic-memory', 'creative', 'novel2hermes']),
    
    ('Music制作episodic-memory応用: 歌詞/コード/モチーフ/構成/サウンドデザイン/設定/リファレンスをベクトル化。タグ: project/track/status/key/bpm/genre/part/element/quality。noise検索「sad drop」noise=0.3→コード進行+料理メタファー+小説伏線回収="drop like broth"創発。クリエイティブ領域での異分野クロスオーバー強み。', ['music', 'episodic-memory', 'creative', 'audio']),
    
    ('Hermes profile-council: プロファイル間でSOUL/スキル/メモリ/プラグイン/設定を共有・同期する協議機構。各プロファイル独立だが、council経由で共通知識ベース参照可能。default/soudan等プロファイル固有の人格・設定を維持しつつ、横断的知見活用。プロファイル間の「会議」で意思決定支援。', ['hermes', 'profile-council', 'architecture', 'multi-profile']),
    
    ('Hermes skills/auto_tag実装詳細: TAG_PATTERNS辞書でキーワード→タグマッピング。novel(小説/章/登場人物/伏線/文体), php(php/laravel/composer/namespace/trait/interface), coding(python/function/class/decorator/async/await), infra(docker/k8s/terraform/ansible/systemd), config(yaml/toml/json/env/setting), cooking(出汁/醤油/味噌/レシピ/食材), music(bpm/コード/モチーフ/ミックス/マスタリング), sales(商談/見積/提案/クロージング/顧客), creative(プロット/キャラクター/世界観/テーマ)。新ドメインは辞書追加で拡張可能。auto_tag=True時のみ適用、手動tags優先。', ['coding', 'episodic-memory', 'auto-tag', 'feature', 'tagging']),
    
    ('Episodic Memoryスキル配布方針: 汎用ツール+NOVEL_WRITING_GUIDE.mdをREADME同梱。小説執筆=キラーアプリ(創作DNA+ノイズセレンディピティ対応ツール他になし)。musicセクション追加予定。GitHub: skillディレクトリ(SKILL.md, tools/, tests/)をパッケージ化し`hermes skill install`または手動コピーで配布。', ['coding', 'episodic-memory', 'distribution', 'github', 'skill']),
]

for text, tags in memories:
    r = memory_store(text, tags=tags)
    print(f'ID:{r["id"]} tags:{r["tags"]}')

print("Done")