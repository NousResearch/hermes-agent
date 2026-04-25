#!/usr/bin/env python3
# ============================================================================
# Real-time Technical Analysis - 即時技術分析 v2
# 使用 Shioaji kbars 資料
# ============================================================================

import shioaji as sj
import os
import pandas as pd
import json
from datetime import datetime, timedelta

CACHE_DIR = "/home/ubuntu/.openclaw/cache/technical"
LOG_FILE = "/tmp/pipeline-technical.log"

WATCHLIST = {
    "2330": {"name": "台積電"},
    "2317": {"name": "鴻海"},
    "2454": {"name": "聯發科"},
    "2303": {"name": "聯電"},
    "3008": {"name": "大立光"},
    "0050": {"name": "台灣50"},
}

def log(msg):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def get_shioaji_api():
    api_key = os.environ.get("SINOTRADE_API_KEY")
    secret_key = os.environ.get("SINOTRADE_SECRET_KEY")
    api = sj.Shioaji()
    api.login(api_key=api_key, secret_key=secret_key)
    return api

def get_historical_prices(api, code, days=60):
    """取得歷史價格"""
    try:
        kbars = api.kbars(
            contract=api.Contracts.Stocks[code],
            start=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d")
        )
        
        if kbars is None:
            return []
        
        # kbars 是 Kbars 物件，要轉成 DataFrame
        df = pd.DataFrame(kbars)
        
        # 結構：row 0=Time, 1=Open, 2=High, 3=Low, 4=Close, 5=Volume, 6=Amount
        # 每個 row 的值是 list
        data = []
        for i in range(len(df)):
            row_name = df.iloc[i, 0]
            values = df.iloc[i, 1]
            if isinstance(values, list) and len(values) > 0:
                data.append({"field": row_name, "values": values})
        
        if not data:
            return []
        
        # 找出總筆數
        num_records = len(data[0]["values"]) if data else 0
        
        # 轉成 records 格式
        records = []
        for idx in range(num_records):
            record = {}
            for item in data:
                record[item["field"]] = item["values"][idx] if idx < len(item["values"]) else None
            records.append(record)
        
        return records
        
    except Exception as e:
        log(f"{code} 歷史資料失敗: {e}")
        return []

def calc_ma(data, period):
    if len(data) < period:
        return None
    closes = [d.get("Close", 0) for d in data[-period:]]
    return sum(closes) / period

def calc_rsi(data, period=14):
    if len(data) < period + 1:
        return None
    
    changes = []
    for i in range(1, len(data)):
        diff = data[i].get("Close", 0) - data[i-1].get("Close", 0)
        changes.append(diff if diff > 0 else 0)
    
    if len(changes) < period:
        return None
    
    avg_gain = sum(changes[-period:]) / period
    avg_loss = sum([abs(c) for c in changes[-period:]]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def analyze_stock(api, code, name):
    log(f"分析 {code} {name}...")
    data = get_historical_prices(api, code, 60)
    
    if len(data) < 30:
        return {"code": code, "name": name, "error": "資料不足"}
    
    # 計算均線
    ma5 = calc_ma(data, 5)
    ma10 = calc_ma(data, 10)
    ma20 = calc_ma(data, 20)
    ma60 = calc_ma(data, 60) if len(data) >= 60 else None
    
    # RSI
    rsi = calc_rsi(data, 14)
    
    # 當前價格
    current_price = data[-1].get("Close") if data else None
    
    return {
        "code": code,
        "name": name,
        "price": current_price,
        "ma5": round(ma5, 2) if ma5 else None,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "rsi": rsi,
    }

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    log("=== 開始技術分析 ===")
    
    try:
        api = get_shioaji_api()
        log("✅ Shioaji 登入成功")
        
        results = []
        for code, info in WATCHLIST.items():
            result = analyze_stock(api, code, info["name"])
            results.append(result)
        
        # 產生報告
        print("📈 **技術分析報告**")
        print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("━━━━━━━━━━━━━━━━━━━━")
        
        print("\n```")
        print(f"{'代碼':<6} {'名稱':<8} {'價格':>10} {'MA5':>10} {'MA10':>10} {'MA20':>10} {'RSI':>6}")
        print(f"{'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        
        for r in results:
            if "error" in r:
                continue
            price = r.get("price", 0)
            ma5 = r.get("ma5", 0) or 0
            ma10 = r.get("ma10", 0) or 0
            ma20 = r.get("ma20", 0) or 0
            rsi = r.get("rsi", 0) or 0
            print(f"{r['code']:<6} {r['name']:<8} {price:>10.2f} {ma5:>10.2f} {ma10:>10.2f} {ma20:>10.2f} {rsi:>6.1f}")
        print("```")
        
        # 保存
        with open(f"{CACHE_DIR}/technical_report.json", "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "stocks": results}, f, ensure_ascii=False, indent=2)
        
        log("=== 完成 ===")
        api.logout()
        
    except Exception as e:
        log(f"錯誤: {e}")
        print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    main()