#!/usr/bin/env python3
# ============================================================================
# Fundamental Pipeline - Layer 1: 抓取基本面資料
# 每天 06:30 執行 (晨報前)
# ============================================================================

import urllib.request
import json
import os
from datetime import datetime, timedelta

CACHE_DIR = "/home/ubuntu/.openclaw/cache/fundamental"
LOG_FILE = "/tmp/pipeline-fundamental-fetch.log"
STOCKS = ["2303", "3008", "2330", "2317", "2454"]  # 主要股票

def log(msg):
    """寫入 log"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def get_trading_date():
    """取得最近交易日"""
    for days_ago in range(5):
        d = datetime.now() - timedelta(days=days_ago)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={date_str}&stockNo=0050&response=json"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "很抱歉" not in data.get("stat", ""):
                return date_str
        except:
            pass
    return "20260417"

def fetch_all_stocks(trade_date):
    """一次抓取所有股票的基本面資料"""
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={trade_date}&response=json"
    
    log(f"抓取全部股票資料: {url}")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        results = {}
        for row in data.get("data", []):
            if len(row) >= 7:
                code = row[0]
                results[code] = {
                    "code": code,
                    "name": row[1],
                    "price": row[2],
                    "yield": row[3],
                    "year": row[4],
                    "PE": row[5] if row[5] != '-' else None,
                    "PB": row[6] if row[6] != '-' else None,
                    "season": row[7] if len(row) > 7 else None,
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
        
    except Exception as e:
        log(f"抓取失敗: {e}")
        return {}

def save_cache(code, data):
    """保存到快取"""
    cache_file = f"{CACHE_DIR}/{code}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    log("=== 開始基本面收集 ===")
    
    # 取得最近交易日
    trade_date = get_trading_date()
    log(f"最近交易日: {trade_date}")
    
    # 一次抓取所有股票
    all_data = fetch_all_stocks(trade_date)
    log(f"抓到 {len(all_data)} 筆股票資料")
    
    # 提取我們要的股票
    for code in STOCKS:
        if code in all_data:
            save_cache(code, all_data[code])
            d = all_data[code]
            log(f"OK: {code} {d['name']} PE={d['PE']} PB={d['PB']} Yield={d['yield']}")
            print(f"OK: {code} {d['name']} PE={d['PE']} PB={d['PB']} Yield={d['yield']}")
        else:
            log(f"FAIL: {code} not found")
            print(f"FAIL: {code} not found")
    
    log("=== 基本面收集完成 ===")

if __name__ == "__main__":
    main()