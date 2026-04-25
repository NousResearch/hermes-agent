#!/usr/bin/env python3
# ============================================================================
# Technical Analysis - 使用 FinMind 日 K 資料
# ============================================================================

import urllib.request
import json
import os
from datetime import datetime, timedelta

CACHE_DIR = "/home/ubuntu/.openclaw/cache/technical"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0yNyAxNTozNzo0NyIsInVzZXJfaWQiOiJwdXBweTA4MDgiLCJlbWFpbCI6InB1cHB5MDgwOEBnbWFpbC5jb20iLCJpcCI6IjExNC4xMzcuMTI5LjIxIn0.xD9y66Ggd4FY5HnEmutFoC-7L_qizPZb_mPn6TAAQj0"

WATCHLIST = {
    "2330": {"name": "台積電"},
    "2317": {"name": "鴻海"},
    "2454": {"name": "聯發科"},
    "2303": {"name": "聯電"},
    "3008": {"name": "大立光"},
    "0050": {"name": "台灣50"},
}

def fetch_daily_prices(code, days=60):
    """使用 FinMind 取得日 K 資料"""
    url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={code}&start_date=2026-02-01&end_date=2026-04-18&token={FINMIND_TOKEN}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        if data.get('status') != 200:
            return []
        
        return data.get('data', [])
    except Exception as e:
        print(f"抓取 {code} 失敗: {e}")
        return []

def calc_ma(data, period):
    if len(data) < period:
        return None
    closes = [d['close'] for d in data[-period:]]
    return sum(closes) / period

def calc_rsi(data, period=14):
    if len(data) < period + 1:
        return None
    
    changes = []
    for i in range(1, len(data)):
        diff = data[i]['close'] - data[i-1]['close']
        changes.append(diff if diff > 0 else 0)
    
    if len(changes) < period:
        return None
    
    avg_gain = sum(changes[-period:]) / period
    avg_loss = sum([abs(c) for c in changes[-period:]]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def calc_macd(data, fast=12, slow=26):
    if len(data) < slow:
        return None, None
    
    def calc_ema(period, data):
        prices = [d['close'] for d in data]
        ema = prices[0]
        k = 2 / (period + 1)
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema
    
    ema_fast = calc_ema(fast, data)
    ema_slow = calc_ema(slow, data)
    macd = ema_fast - ema_slow
    signal = macd * 0.9
    
    return round(macd, 2), round(signal, 2)

def analyze_stock(code, name):
    data = fetch_daily_prices(code, 60)
    
    if len(data) < 30:
        return {"code": code, "name": name, "error": f"資料不足({len(data)})"}
    
    # 排序（依日期）
    data.sort(key=lambda x: x['date'])
    
    ma5 = calc_ma(data, 5)
    ma10 = calc_ma(data, 10)
    ma20 = calc_ma(data, 20)
    ma60 = calc_ma(data, 60) if len(data) >= 60 else None
    
    rsi = calc_rsi(data, 14)
    macd, signal = calc_macd(data)
    
    current = data[-1]
    
    # 亞當策略判斷
    trend = "整理"
    if all([ma5, ma10, ma20]):
        if ma5 > ma10 > ma20:
            trend = "多頭"
        elif ma5 < ma10 < ma20:
            trend = "空頭"
    
    return {
        "code": code,
        "name": name,
        "price": current['close'],
        "date": current['date'],
        "open": current['open'],
        "high": current['max'],
        "low": current['min'],
        "ma5": round(ma5, 2) if ma5 else None,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": signal,
        "trend": trend,
    }

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("📈 **技術分析報告 (日 K)**")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    results = []
    for code, info in WATCHLIST.items():
        print(f"\n分析 {code} {info['name']}...")
        result = analyze_stock(code, info['name'])
        results.append(result)
    
    # 輸出報告
    print("\n" + "=" * 60)
    print("【均線摘要】")
    print(f"{'代碼':<6} {'名稱':<8} {'價格':>10} {'MA5':>10} {'MA10':>10} {'MA20':>10} {'RSI':>6} {'趨勢':<6}")
    print("-" * 70)
    
    for r in results:
        if "error" in r:
            continue
        print(f"{r['code']:<6} {r['name']:<8} {r['price']:>10.2f} {r['ma5']:>10.2f} {r['ma10']:>10.2f} {r['ma20']:>10.2f} {r['rsi']:>6.1f} {r['trend']:<6}")
    
    # 亞當策略訊號
    print("\n" + "=" * 60)
    print("【亞當策略訊號】")
    
    for r in results:
        if "error" in r:
            continue
        
        print(f"\n{r['code']} {r['name']}:")
        
        # 均線判斷
        if r['trend'] == "多頭":
            print(f"  🟢 趨勢: 多頭排列 (MA5 > MA10 > MA20)")
        elif r['trend'] == "空頭":
            print(f"  🔴 趨勢: 空頭排列 (MA5 < MA10 < MA20)")
        else:
            print(f"  🟡 趨勢: 整理中")
        
        # 價格與 MA5
        if r['price'] > r['ma5']:
            print(f"  ✅ 價格站上 MA5 ({r['price']} > {r['ma5']})")
        else:
            print(f"  ❌ 價格跌破 MA5 ({r['price']} < {r['ma5']})")
        
        # RSI
        if r['rsi'] > 70:
            print(f"  ⚠️ RSI 超買: {r['rsi']}")
        elif r['rsi'] < 30:
            print(f"  💡 RSI 超賣: {r['rsi']}")
        else:
            print(f"  RSI: {r['rsi']} (中性)")
        
        # MACD
        if r['macd'] and r['macd_signal']:
            if r['macd'] > r['macd_signal']:
                print(f"  ✅ MACD 金叉 ({r['macd']} > {r['macd_signal']})")
            else:
                print(f"  ❌ MACD 死叉 ({r['macd']} < {r['macd_signal']})")
    
    # 保存
    with open(f"{CACHE_DIR}/technical_report.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "stocks": results}, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ 報告已保存")

if __name__ == "__main__":
    main()