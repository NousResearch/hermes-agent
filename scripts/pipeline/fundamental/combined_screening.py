#!/usr/bin/env python3
# ============================================================================
# Combined Stock Screening - 綜合選股系統
# 基本面 + 技術面 + 策略濾網
# ============================================================================

import json
import os
import urllib.request
from datetime import datetime, timedelta

CACHE_DIR = "/home/ubuntu/.openclaw/cache"
FUNDAMENTAL_DIR = f"{CACHE_DIR}/fundamental"

# ============================================================
# 觀察名單設定
# ============================================================
WATCHLIST = {
    "2330": {"name": "台積電", "sector": "半導體", "weight": 1.0},
    "2317": {"name": "鴻海", "sector": "電子代工", "weight": 0.8},
    "2454": {"name": "聯發科", "sector": "IC設計", "weight": 0.9},
    "2303": {"name": "聯電", "sector": "晶圓代工", "weight": 0.7},
    "3008": {"name": "大立光", "sector": "光學", "weight": 0.6},
}

# ============================================================
# 評分標準
# ============================================================

# 基本面評分標準
FUNDAMENTAL_RULES = {
    "PE": {"max_ok": 20, "max_good": 15, "weight": 35},
    "PB": {"max_ok": 3, "max_good": 2, "weight": 25},
    "YIELD": {"min_ok": 2, "min_good": 4, "weight": 25},
}

# 技術面評分標準 (需要有歷史資料)
TECHNICAL_RULES = {
    "TREND": {"weight": 40},
    "RSI": {"weight": 30},
    "MOMENTUM": {"weight": 30},
}

# ============================================================
# 工具函數
# ============================================================

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

# ============================================================
# 評分函數
# ============================================================

def score_PE(pe_str):
    """P/E 評分 (滿分 35)"""
    if not pe_str or pe_str == '-':
        return 0, "無資料"
    
    try:
        pe = float(pe_str)
        if pe <= 10:
            return 35, f"{pe:.1f} (極低)"
        elif pe <= 15:
            return 30, f"{pe:.1f} (低)"
        elif pe <= 20:
            return 22, f"{pe:.1f} (適中)"
        elif pe <= 25:
            return 15, f"{pe:.1f} (稍高)"
        elif pe <= 30:
            return 8, f"{pe:.1f} (偏高)"
        else:
            return 0, f"{pe:.1f} (極高)"
    except:
        return 0, "N/A"

def score_PB(pb_str):
    """P/B 評分 (滿分 25)"""
    if not pb_str or pb_str == '-':
        return 0, "無資料"
    
    try:
        pb = float(pb_str)
        if pb <= 1:
            return 25, f"{pb:.2f} (極低)"
        elif pb <= 1.5:
            return 22, f"{pb:.2f} (低)"
        elif pb <= 2:
            return 18, f"{pb:.2f} (適中)"
        elif pb <= 3:
            return 12, f"{pb:.2f} (稍高)"
        elif pb <= 5:
            return 5, f"{pb:.2f} (偏高)"
        else:
            return 0, f"{pb:.2f} (極高)"
    except:
        return 0, "N/A"

def score_YIELD(yield_str):
    """殖利率評分 (滿分 25)"""
    if not yield_str or yield_str == '-':
        return 0, "無資料"
    
    try:
        y = float(yield_str)
        if y >= 6:
            return 25, f"{y:.2f}% (極高)"
        elif y >= 4:
            return 22, f"{y:.2f}% (高)"
        elif y >= 3:
            return 17, f"{y:.2f}% (適中)"
        elif y >= 2:
            return 10, f"{y:.2f}% (低)"
        elif y >= 1:
            return 5, f"{y:.2f}% (極低)"
        else:
            return 0, f"{y:.2f}% (無)"
    except:
        return 0, "N/A"

def calculate_fundamental_score(stock):
    """計算基本面總分"""
    pe_score, pe_note = score_PE(stock.get("PE"))
    pb_score, pb_note = score_PB(stock.get("PB"))
    yield_score, yield_note = score_YIELD(stock.get("yield"))
    
    total = pe_score + pb_score + yield_score
    
    return {
        "total": total,
        "max": 85,
        "percentage": round(total / 85 * 100, 1),
        "breakdown": {
            "PE": {"score": pe_score, "note": pe_note},
            "PB": {"score": pb_score, "note": pb_note},
            "YIELD": {"score": yield_score, "note": yield_note}
        }
    }

def get_grade(score, max_score=85):
    """評分轉等級"""
    pct = score / max_score * 100
    
    if pct >= 85:
        return "🟢A", "優質"
    elif pct >= 70:
        return "🟡B+", "良好"
    elif pct >= 55:
        return "🟡B", "普通"
    elif pct >= 40:
        return "🟠C", "待觀察"
    else:
        return "🔴D", "風險高"

def get_recommendation(score, pct):
    """根據分數給建議"""
    if pct >= 85:
        return "✅ 值得關注，基本面優異"
    elif pct >= 70:
        return "✅ 基本面良好，可持續追蹤"
    elif pct >= 55:
        return "⚠️ 基本面普通，需要更多資訊"
    elif pct >= 40:
        return "⚠️ 基本面偏弱，謹慎關注"
    else:
        return "❌ 基本面較差，建議觀望"

# ============================================================
# 主要報告產生
# ============================================================

def generate_combined_report():
    """產生綜合選股報告"""
    report = []
    
    # 取得交易日
    trade_date = get_trading_date()
    if not trade_date:
        return {"error": "無法取得交易日"}
    
    # 抓取基本面資料
    all_stocks = fetch_all_stocks(trade_date)
    
    # 分析觀察名單
    for code, info in WATCHLIST.items():
        stock = all_stocks.get(code)
        
        if not stock:
            report.append({
                "code": code,
                "name": info["name"],
                "sector": info["sector"],
                "weight": info["weight"],
                "error": "無基本面資料"
            })
            continue
        
        # 基本面評分
        fund_score = calculate_fundamental_score(stock)
        grade, grade_text = get_grade(fund_score["total"])
        recommendation = get_recommendation(fund_score["total"], fund_score["percentage"])
        
        report.append({
            "code": code,
            "name": info["name"],
            "sector": info["sector"],
            "weight": info["weight"],
            "price": stock.get("price", "N/A"),
            "PE": stock.get("PE", "N/A"),
            "PB": stock.get("PB", "N/A"),
            "yield": stock.get("yield", "N/A"),
            "fund_score": fund_score["total"],
            "fund_max": fund_score["max"],
            "fund_percentage": fund_score["percentage"],
            "grade": grade,
            "grade_text": grade_text,
            "breakdown": fund_score["breakdown"],
            "recommendation": recommendation
        })
    
    # 按分數排序
    report.sort(key=lambda x: x.get("fund_score", 0), reverse=True)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "trade_date": trade_date,
        "stocks": report
    }

def print_report(report):
    """輸出報告"""
    print("=" * 70)
    print("📊 綜合選股報告")
    print(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    if "error" in report:
        print(f"❌ {report['error']}")
        return
    
    print(f"📅 資料日期: {report['trade_date']}\n")
    
    # 基本面評分表
    print("-" * 70)
    print("【基本面評分】")
    print("-" * 70)
    print(f"{'代碼':<6} {'名稱':<8} {'價格':>12} {'P/E':>8} {'P/B':>8} {'殖利率':>8} {'分數':>6} {'等級':<6}")
    print("-" * 70)
    
    for stock in report["stocks"]:
        if "error" in stock:
            print(f"{stock['code']:<6} {stock['name']:<8} {'N/A':>12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'⚠️':<6}")
            continue
        
        print(f"{stock['code']:<6} {stock['name']:<8} {stock['price']:>12} {str(stock['PE']):>8} {str(stock['PB']):>8} {str(stock['yield']):>8} {stock['fund_score']:>6} {stock['grade']:<6}")
    
    # 詳細分析
    print("\n" + "=" * 70)
    print("【詳細分析】")
    print("=" * 70)
    
    for stock in report["stocks"]:
        if "error" in stock:
            continue
        
        print(f"\n▶ {stock['code']} {stock['name']} ({stock['sector']})")
        print(f"  價格: {stock['price']}")
        print(f"  基本面分數: {stock['fund_score']}/{stock['fund_max']} ({stock['fund_percentage']}%) {stock['grade']} {stock['grade_text']}")
        
        bd = stock["breakdown"]
        print(f"  • P/E: {bd['PE']['note']} (+{bd['PE']['score']}分)")
        print(f"  • P/B: {bd['PB']['note']} (+{bd['PB']['score']}分)")
        print(f"  • 殖利率: {bd['YIELD']['note']} (+{bd['YIELD']['score']}分)")
        print(f"  📌 {stock['recommendation']}")
    
    # 評分說明
    print("\n" + "=" * 70)
    print("【評分說明】")
    print("=" * 70)
    print("""
基本面評分 (滿分 85):
• P/E 本益比 (35分): ≤10=35分, ≤15=30分, ≤20=22分, ≤25=15分, ≤30=8分
• P/B 股價淨值比 (25分): ≤1=25分, ≤1.5=22分, ≤2=18分, ≤3=12分, ≤5=5分
• 殖利率 (25分): ≥6%=25分, ≥4%=22分, ≥3%=17分, ≥2%=10分, ≥1%=5分

等級標準:
🟢A (85%+): 優質 - 基本面極佳
🟡B+ (70-84%): 良好 - 基本面良好
🟡B (55-69%): 普通 - 基本面一般
🟠C (40-54%): 待觀察 - 基本面偏弱
🔴D (<40%): 風險高 - 基本面較差
    """)

def main():
    # 確保目錄存在
    os.makedirs(FUNDAMENTAL_DIR, exist_ok=True)
    
    # 產生報告
    report = generate_combined_report()
    
    # 輸出報告
    print_report(report)
    
    # 保存報告
    report_file = f"{FUNDAMENTAL_DIR}/combined_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 報告已保存: {report_file}")

if __name__ == "__main__":
    main()