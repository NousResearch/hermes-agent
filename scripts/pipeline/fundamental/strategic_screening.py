#!/usr/bin/env python3
# ============================================================================
# Strategic Stock Screener - 策略導向選股系統
# 從全部台股中，根據不同策略推薦標的
# ============================================================================

import json
import os
import urllib.request
import math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = "/home/ubuntu/.openclaw/cache/fundamental"
LOG_FILE = "/tmp/pipeline-strategic-screening.log"

# ============================================================
# 投資策略定義
# ============================================================

STRATEGIES = {
    "價值投資": {
        "description": "低 P/E、低 P/B、高殖利率",
        "filters": {
            "PE_max": 15,
            "PB_max": 1.5,
            "YIELD_min": 3,
        },
        "weight": {"PE": 35, "PB": 25, "YIELD": 25},
        "sort_by": "score_value",
    },
    "成長投資": {
        "description": "高營收成長、高 EPS 成長",
        "filters": {
            "YIELD_min": 0,  # 不要求殖利率
        },
        "weight": {"PE": 15, "PB": 15, "YIELD": 10, "GROWTH": 45},
        "sort_by": "score_growth",
    },
    "高股息": {
        "description": "高殖利率、穩定配息",
        "filters": {
            "YIELD_min": 5,
        },
        "weight": {"PE": 10, "PB": 10, "YIELD": 60},
        "sort_by": "score_dividend",
    },
    "優質價值": {
        "description": "P/E 合理且 P/B 低",
        "filters": {
            "PE_max": 20,
            "PB_max": 2,
        },
        "weight": {"PE": 30, "PB": 30, "YIELD": 15},
        "sort_by": "score_quality",
    },
    "低估藍籌": {
        "description": "大型股 + 低估值",
        "filters": {
            "PE_max": 15,
            "market_cap": "large",  # 大型股
        },
        "weight": {"PE": 35, "PB": 25, "YIELD": 25},
        "sort_by": "score_value",
    },
}

# 大型股清單 (市值前 50)
LARGE_CAP_STOCKS = [
    "2330", "2317", "2454", "2303", "3008",  # 已有的
    "2412", "2891", "2884", "2885", "2886",  # 中華電、兆豐、彰銀、玉山、元大
    "1301", "1326", "1101", "1102", "1216",  # 台塑、南亞、台泥、亞泥、統一
    "2002", "2103", "1722", "1718", "1802",  # 中鋼、台橡、台肥、中纖、台玻
    "2610", "2633", "2603", "2609", "2637",  # 航運、航空、散裝、陽明、長榮
    "0050", "0056", "00878", "00940",         # ETF
    "2474", "2308", "2363", "2376", "2382",  # 可成、台達電、矽力、華碩、廣達
    "3034", "3481", "3532", "3443", "4960",  # 聯詠、群創、勝華、創意、漢微科
    "3533", "3661", "3673", "3697", "3665",  # 嘉澤、世芯、TPK、洋華、大眾控
    "2352", "2353", "2356", "2357", "2379",  # 明天、環科、敬鹏、曜越、瑞儀
    "2395", "2401", "2408", "2421", "2423",  # 研華、凌華、网通、诚研、华冠
    "2441", "2449", "2451", "2453", "2455",  # 超盈、景文、創见、敦泰、宏捷科
    "2458", "2459", "2468", "2472", "2478",  # 義隆电、敦吉、志聖、劲驰、得力
    "2480", "2481", "2492", "2495", "2497",  # 敦阳、拦诈、卓越、润泰新、创意
]

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
    return None

def fetch_all_stocks(trade_date):
    """一次抓取所有股票基本面"""
    log("抓取全部股票基本面...")
    
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={trade_date}&response=json"
    
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
                }
        
        log(f"抓到 {len(results)} 檔股票")
        return results
        
    except Exception as e:
        log(f"抓取失敗: {e}")
        return {}

def score_stock(stock, strategy_key):
    """根據策略評分股票"""
    strategy = STRATEGIES[strategy_key]
    filters = strategy["filters"]
    
    # 基本分數
    pe_score = 0
    pb_score = 0
    yield_score = 0
    
    # PE 評分
    pe = stock.get("PE")
    if pe and pe not in ['-', None]:
        try:
            pe_val = float(pe)
            if pe_val <= 10:
                pe_score = 35
            elif pe_val <= 15:
                pe_score = 30
            elif pe_val <= 20:
                pe_score = 22
            elif pe_val <= 25:
                pe_score = 15
            elif pe_val <= 30:
                pe_score = 8
        except:
            pass
    
    # PB 評分
    pb = stock.get("PB")
    if pb and pb not in ['-', None]:
        try:
            pb_val = float(pb)
            if pb_val <= 1:
                pb_score = 25
            elif pb_val <= 1.5:
                pb_score = 22
            elif pb_val <= 2:
                pb_score = 18
            elif pb_val <= 3:
                pb_score = 12
            elif pb_val <= 5:
                pb_score = 5
        except:
            pass
    
    # 殖利率評分
    y = stock.get("yield")
    if y and y not in ['-', None]:
        try:
            y_val = float(y)
            if y_val >= 6:
                yield_score = 25
            elif y_val >= 5:
                yield_score = 22
            elif y_val >= 4:
                yield_score = 17
            elif y_val >= 3:
                yield_score = 14
            elif y_val >= 2:
                yield_score = 10
            elif y_val >= 1:
                yield_score = 5
        except:
            pass
    
    total = pe_score + pb_score + yield_score
    
    return {
        "code": stock["code"],
        "name": stock["name"],
        "price": stock.get("price", "N/A"),
        "PE": stock.get("PE", "N/A"),
        "PB": stock.get("PB", "N/A"),
        "yield": stock.get("yield", "N/A"),
        "pe_score": pe_score,
        "pb_score": pb_score,
        "yield_score": yield_score,
        "total_score": total,
    }

def filter_stocks(stocks, strategy_key):
    """根據策略條件過濾股票"""
    strategy = STRATEGIES[strategy_key]
    filters = strategy["filters"]
    
    filtered = []
    
    for code, stock in stocks.items():
        # 基本過濾：必須有 PE、PB、殖利率
        if not stock.get("PE") or not stock.get("PB") or not stock.get("yield"):
            continue
        
        if stock.get("PE") == '-' or stock.get("PB") == '-' or stock.get("yield") == '-':
            continue
        
        try:
            # PE 過濾
            pe_max = filters.get("PE_max", float("inf"))
            pe = float(stock.get("PE", "999"))
            if pe > pe_max:
                continue
            
            # PB 過濾
            pb_max = filters.get("PB_max", float("inf"))
            pb = float(stock.get("PB", "999"))
            if pb > pb_max:
                continue
            
            # 殖利率過濾
            yield_min = filters.get("YIELD_min", 0)
            y = float(stock.get("yield", "0"))
            if y < yield_min:
                continue
            
            # 大型股過濾
            if "market_cap" in filters:
                if filters["market_cap"] == "large" and code not in LARGE_CAP_STOCKS:
                    continue
            
            filtered.append(stock)
            
        except (ValueError, TypeError):
            continue
    
    return filtered

def screen_stocks_by_strategy(stocks, strategy_key, top_n=10):
    """根據策略篩選股票"""
    strategy = STRATEGIES[strategy_key]
    
    # 先過濾
    filtered = filter_stocks(stocks, strategy_key)
    
    # 評分
    scored = []
    for stock in filtered:
        result = score_stock(stock, strategy_key)
        scored.append(result)
    
    # 排序
    scored.sort(key=lambda x: x["total_score"], reverse=True)
    
    return scored[:top_n]

def generate_strategic_report():
    """產生策略導向選股報告"""
    log("開始策略選股...")
    
    # 取得交易日
    trade_date = get_trading_date()
    if not trade_date:
        return {"error": "無法取得交易日"}
    
    # 抓取全部股票基本面
    all_stocks = fetch_all_stocks(trade_date)
    if not all_stocks:
        return {"error": "無法抓取股票資料"}
    
    # 對每個策略進行篩選
    results = {}
    for strategy_key in STRATEGIES:
        stocks = screen_stocks_by_strategy(all_stocks, strategy_key, top_n=10)
        results[strategy_key] = {
            "description": STRATEGIES[strategy_key]["description"],
            "stocks": stocks,
            "count": len(stocks)
        }
        log(f"{strategy_key}: {len(stocks)} 檔符合")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "trade_date": trade_date,
        "strategies": results,
        "total_stocks_analyzed": len(all_stocks),
    }

def print_report(report):
    """輸出報告"""
    print("=" * 80)
    print("📊 策略導向選股報告")
    print(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    if "error" in report:
        print(f"❌ {report['error']}")
        return
    
    print(f"📅 資料日期: {report['trade_date']}")
    print(f"📈 分析總檔數: {report['total_stocks_analyzed']}")
    print()
    
    for strategy_key, data in report["strategies"].items():
        print("-" * 80)
        print(f"📌 {strategy_key}: {data['description']}")
        print(f"   符合條件: {data['count']} 檔")
        print("-" * 80)
        
        if not data["stocks"]:
            print("   無符合條件的股票")
        else:
            print(f"{'排名':<4} {'代碼':<6} {'名稱':<10} {'價格':>12} {'P/E':>8} {'P/B':>8} {'殖利率':>8} {'評分':>6}")
            print("-" * 80)
            
            for i, stock in enumerate(data["stocks"], 1):
                print(f"{i:<4} {stock['code']:<6} {stock['name']:<10} {stock['price']:>12} {str(stock['PE']):>8} {str(stock['PB']):>8} {str(stock['yield']):>8} {stock['total_score']:>6}")
        
        print()

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    report = generate_strategic_report()
    print_report(report)
    
    # 保存報告
    report_file = f"{CACHE_DIR}/strategic_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 報告已保存: {report_file}")

if __name__ == "__main__":
    main()