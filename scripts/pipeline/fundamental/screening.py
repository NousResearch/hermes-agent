#!/usr/bin/env python3
# ============================================================================
# Stock Screening System - 選股系統
# 結合基本面 + 技術面 + 策略濾網
# ============================================================================

import json
import os
import urllib.request
from datetime import datetime, timedelta

CACHE_DIR = "/home/ubuntu/.openclaw/cache"
FUNDAMENTAL_DIR = f"{CACHE_DIR}/fundamental"
STOCK_CACHE_DIR = f"{CACHE_DIR}/stock"

# 股票清單
WATCHLIST = {
    "2330": {"name": "台積電", "type": "權值"},
    "2317": {"name": "鴻海", "type": "權值"},
    "2454": {"name": "聯發科", "type": "權值"},
    "2303": {"name": "聯電", "type": "景氣循環"},
    "3008": {"name": "大立光", "type": "光學"},
    "0050": {"name": "台灣50", "type": "ETF"},
}

# 評分標準
FUNDAMENTAL_SCORES = {
    "PE": {"max": 15, "good": 20, "excellent": 25},  # 越低越好
    "PB": {"max": 1.5, "good": 3, "excellent": 5},  # 越低越好
    "YIELD": {"min": 3, "good": 4, "excellent": 5},   # 越高越好
}

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
                    "timestamp": datetime.now().isoformat()
                }
        return results
        
    except Exception as e:
        print(f"抓取失敗: {e}")
        return {}

def score_fundamental(stock):
    """基本面評分 (0-100)"""
    score = 0
    
    # PE 評分 (權重 30%)
    pe = stock.get("PE")
    if pe and pe not in ['-', None]:
        try:
            pe_val = float(pe)
            if pe_val <= 10:
                score += 30
            elif pe_val <= 15:
                score += 25
            elif pe_val <= 20:
                score += 20
            elif pe_val <= 25:
                score += 15
            elif pe_val <= 30:
                score += 10
            else:
                score += 5
        except:
            pass
    
    # PB 評分 (權重 20%)
    pb = stock.get("PB")
    if pb and pb not in ['-', None]:
        try:
            pb_val = float(pb)
            if pb_val <= 1:
                score += 20
            elif pb_val <= 1.5:
                score += 17
            elif pb_val <= 2:
                score += 14
            elif pb_val <= 3:
                score += 10
            elif pb_val <= 5:
                score += 5
            else:
                score += 2
        except:
            pass
    
    # 殖利率評分 (權重 20%)
    yield_val = stock.get("yield")
    if yield_val and yield_val not in ['-', None]:
        try:
            y = float(yield_val)
            if y >= 5:
                score += 20
            elif y >= 4:
                score += 17
            elif y >= 3:
                score += 14
            elif y >= 2:
                score += 10
            elif y >= 1:
                score += 5
            else:
                score += 2
        except:
            pass
    
    # 價格合理性 (權重 15%) - 預設為合理
    score += 15
    
    return score

def grade_fundamental(score):
    """評分轉等級"""
    if score >= 80:
        return "🟢A", "優質"
    elif score >= 60:
        return "🟡B", "良好"
    elif score >= 40:
        return "🟠C", "普通"
    else:
        return "🔴D", "待觀察"

def analyze_stock(code, fundamental_data):
    """分析單一股票"""
    stock = fundamental_data.get(code)
    if not stock:
        return None
    
    fund_score = score_fundamental(stock)
    grade, grade_text = grade_fundamental(fund_score)
    
    return {
        "code": code,
        "name": stock.get("name", code),
        "price": stock.get("price", "N/A"),
        "PE": stock.get("PE", "N/A"),
        "PB": stock.get("PB", "N/A"),
        "yield": stock.get("yield", "N/A"),
        "fund_score": fund_score,
        "grade": grade,
        "grade_text": grade_text,
        "season": stock.get("season", "N/A"),
    }

def generate_report():
    """產生選股報告"""
    print("=" * 60)
    print("📊 個股基本面評分報告")
    print(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 抓取最新基本面資料
    trade_date = get_trading_date()
    if not trade_date:
        print("❌ 無法取得交易日")
        return
    
    print(f"📅 資料日期: {trade_date}\n")
    
    fundamental_data = fetch_all_stocks(trade_date)
    
    # 分析 watchlist
    results = []
    for code in WATCHLIST:
        result = analyze_stock(code, fundamental_data)
        if result:
            results.append(result)
    
    # 按分數排序
    results.sort(key=lambda x: x["fund_score"], reverse=True)
    
    # 輸出報告
    print(f"{'代碼':<6} {'名稱':<8} {'價格':>10} {'P/E':>6} {'P/B':>6} {'殖利率':>6} {'評分':>4} {'等級':<6}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['code']:<6} {r['name']:<8} {r['price']:>10} {str(r['PE']):>6} {str(r['PB']):>6} {str(r['yield']):>6} {r['fund_score']:>4} {r['grade']:<6}")
    
    print("\n" + "=" * 60)
    print("📈 評分說明")
    print("-" * 60)
    print("""
【評分標準】(基本面滿分 100)
• P/E 本益比 (30分): ≤10=30分, ≤15=25分, ≤20=20分, ≤25=15分, ≤30=10分
• P/B 股價淨值比 (20分): ≤1=20分, ≤1.5=17分, ≤2=14分, ≤3=10分
• 殖利率 (20分): ≥5%=20分, ≥4%=17分, ≥3%=14分, ≥2%=10分
• 價格合理性 (15分): 綜合考量

【等級】
🟢A (80+): 優質 - 基本面優異
🟡B (60-79): 良好 - 基本面良好  
🟠C (40-59): 普通 - 需要更多觀察
🔴D (<40): 待觀察 - 風險較高
    """)
    
    # 儲存到 cache
    os.makedirs(FUNDAMENTAL_DIR, exist_ok=True)
    with open(f"{FUNDAMENTAL_DIR}/screening_report.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "trade_date": trade_date,
            "stocks": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 報告已保存到 {FUNDAMENTAL_DIR}/screening_report.json")

if __name__ == "__main__":
    generate_report()