#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"C:/Users/corek/AppData/Local/hermes/skills/personal-knowledge/episodic-memory/tools")
from episodic_memory import memory_store

memories = [
    ('Episodic Memory設計の核心合意: 「圧縮インデックスで広く引く → 必要なら元データで検証」の二段階アーキテクチャ。1bit量子化でもコサイン類似度の順序はそこそこ保たれる(SimHash/Binary Embeddings理論的裏付け)。fuzzy検索時のノイズ混入(noise_ratio)はバグでなく機能 - 人間のデフォルトモードネットワーク的セレンディピティを模倣。創造分野ではハルシネーションを後でチェックすれば良く、思考の種としてノイズを混ぜ込みたい。用途分岐: クリエイティブ→そのまま生成、事実・推論→元データ照合・再ランキング→確信度付き回答/不確実ならanirṇaya明示。', ['coding', 'episodic-memory', 'architecture', 'design', 'creative']),
    
    ('1bit量子化の実用的限界と対策: 384dim float32 → 384bit(1bit/dim)では情報落ち大。実用は4-8bit量子化+Product Quantizationが現実的だが、1bitでも「意味の近さ」の順序は保たれるためfuzzy recall(種出し)には十分。重要度スコアによる適応的量子化(重要な記憶ほど高精度)が脳の仕組みに近い。検証ステップのコスト対策: フル精度埋め込みキャッシュ階層設計、または生テキストをLLMでverify。クリエイティブ用ノイズ制御: top_k/similarity_threshold/温度パラメータで調整可能。', ['coding', 'episodic-memory', 'quantization', 'design', 'optimization']),
    
    ('Episodic Memoryユースケース実例: コーディング支援「あの関数前どこで書いたっけ」(verifiedで正確なコード取得、fuzzyでキーワード忘れてる時探す)、研究・論文メモ「この手法前読んだ論文で見た」(fuzzyで関連論文引き出し→verifiedで詳細確認)、創作・小説「この展開前に書いたアイデアと似てる」(noise込みで偶発的連想)、営業・会議メモ「先月のあの客の予算感」(タグ+時系列で絞り込み)。人間in-the-loop前提ならfuzzy→人間確認→verifiedの単一フローが最も自然でUIシンプル。', ['coding', 'episodic-memory', 'usecase', 'workflow']),
    
    ('二段階検索(粗いANN→正確な再ランキング)の産業標準実装例: Google/Bing(逆インデックス+量子化ベクトル→BERTクロスエンコーダー再スコア)、Meta推薦(Two-tower数億→数千→Heavy ranker数十)、RAG標準(Bi-encoder粗い→Cross-encoder正確)、ベクトルDB標準機能(Pinecone/Weaviate/Milvus/QdrantのHNSW/IVF-PQ→rerank API)、ColBERT(トークンレベル遅延相互作用)。Hermes Episodic Memoryはこれのローカル版・人間in-the-loop版・創造性ノイズ付き版。', ['coding', 'episodic-memory', 'architecture', 'industry-pattern', 'rag']),
]

for text, tags in memories:
    r = memory_store(text, tags=tags)
    print(f'ID:{r["id"]} tags:{r["tags"]}')

print("Done")