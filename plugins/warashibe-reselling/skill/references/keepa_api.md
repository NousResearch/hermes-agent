# Keepa API リファレンス

## Product Finder
- `GET /productfinder?key={API_KEY}&domainId=1&type=0&...`
- domainId: 1=Amazon.co.jp
- type=0: カテゴリ検索, type=1: ASINリスト
- 例: "Switch ゲームソフト"でFBA価格→メルカリ相場逆算

## Token消費
- 1リクエスト=1トークン + 推移量
- 無料枠: 月100トークン
- Pro: $20/月 (3000トークン)

## 主要パラメータ
- brand: メーカー名
- title: 商品タイトル
- salesRank: 売上ランキング
- current: 現在価格
- csv: 価格履歴 (1=Amazon, 2=New, 3=Used, 4=SalesRank)

## せどりフロー
1. Keepa Product Finderでaudio含む商品リスト取得
2. FBA価格 / Amazon価格取得
3. メルカリ/ヤフオク相場と比較
4. 利益率30%以上 & 回転14日以内 ならGO
