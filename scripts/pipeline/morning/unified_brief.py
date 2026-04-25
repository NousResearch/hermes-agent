#!/usr/bin/env python3
# ============================================================================
# Unified Morning Brief - 整合晨報
# 基本面 + 技術面 + 策略選股
# ============================================================================

import urllib.request
import json
import os
from datetime import datetime, timedelta

CACHE_DIR = "/home/ubuntu/.openclaw/cache"
FUNDAMENTAL_DIR = f"{CACHE_DIR}/fundamental"
TECHNICAL_DIR = f"{CACHE_DIR}/technical"

FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMy0yNyAxNTozNzo0NyIsInVzZXJfaWQiOiJwdXBweTA4MDgiLCJlbWFpbCI6InB1cHB5MDgwOEBnbWFpbC5jb20iLCJpcCI6IjExNC4xMzcuMTI5LjIxIn0.xD9y66Ggd4FY5HnEmutFoC-7L_qizPZb_mPn6TAAQj0"

WATCHLIST = {
    "2330": {"name": "台積電", "sector": "半導體"},
    "2317": {"name": "鴻海", "sector": "電子代工"},
    "2454": {"name": "聯發科", "sector": "IC設計"},
    "2303": {"name": "聯電", "sector": "晶圓代工"},
    "3008": {"name": "大立光", "sector": "光學"},
}

# ============================================================
# 基本面取得
# ============================================================

def get_trading_date():
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

def fetch_fundamental(trade_date):
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
                    "PE": row[5] if row[5] != '-' else None,
                    "PB": row[6] if row[6] != '-' else None,
                }
        return results
    except:
        return {}

# ============================================================
# 技術面取得
# ============================================================

def fetch_technical(code):
    url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={code}&start_date=2026-02-01&end_date=2026-04-18&token={FINMIND_TOKEN}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        if data.get('status') != 200:
            return None
        return data.get('data', [])
    except:
        return None

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
    return round(macd, 2), round(macd * 0.9, 2)

# ============================================================
# 策略評分
# ============================================================

def score_fundamental(stock):
    score = 0
    
    pe = stock.get("PE")
    if pe and pe not in ['-', None]:
        try:
            pe_val = float(pe)
            if pe_val <= 10: score += 35
            elif pe_val <= 15: score += 30
            elif pe_val <= 20: score += 22
            elif pe_val <= 25: score += 15
            elif pe_val <= 30: score += 8
        except: pass
    
    pb = stock.get("PB")
    if pb and pb not in ['-', None]:
        try:
            pb_val = float(pb)
            if pb_val <= 1: score += 25
            elif pb_val <= 1.5: score += 22
            elif pb_val <= 2: score += 18
            elif pb_val <= 3: score += 12
            elif pb_val <= 5: score += 5
        except: pass
    
    y = stock.get("yield")
    if y and y not in ['-', None]:
        try:
            y_val = float(y)
            if y_val >= 6: score += 25
            elif y_val >= 4: score += 22
            elif y_val >= 3: score += 17
            elif y_val >= 2: score += 10
            elif y_val >= 1: score += 5
        except: pass
    
    return score

def analyze_stock(code, info, fundamental_data, technical_data):
    fund = fundamental_data.get(code, {})
    tech = technical_data.get(code, {})
    
    fund_score = score_fundamental(fund)
    
    # 技術面評估
    trend = "整理"
    if tech and len(tech) >= 20:
        tech.sort(key=lambda x: x['date'])
        ma5 = calc_ma(tech, 5)
        ma10 = calc_ma(tech, 10)
        ma20 = calc_ma(tech, 20)
        
        if ma5 and ma10 and ma20:
            if ma5 > ma10 > ma20:
                trend = "多頭"
            elif ma5 < ma10 < ma20:
                trend = "空頭"
        
        rsi = calc_rsi(tech, 14)
        macd, signal = calc_macd(tech)
        
        price = tech[-1]['close'] if tech else None
        current_ma5 = ma5
    else:
        price = fund.get("price")
        rsi = None
        macd = None
        current_ma5 = None
    
    # 綜合評級
    pct = (fund_score / 85) * 100
    if pct >= 85: grade = "🟢A"
    elif pct >= 70: grade = "🟡B+"
    elif pct >= 55: grade = "🟡B"
    elif pct >= 40: grade = "🟠C"
    else: grade = "🔴D"
    
    return {
        "code": code,
        "name": info["name"],
        "sector": info["sector"],
        "price": price,
        "PE": fund.get("PE", "N/A"),
        "PB": fund.get("PB", "N/A"),
        "yield": fund.get("yield", "N/A"),
        "fund_score": fund_score,
        "fund_pct": round(pct, 1),
        "grade": grade,
        "trend": trend,
        "rsi": rsi,
        "macd": macd,
    }

# ============================================================
# 主要輸出
# ============================================================

def generate_report():
    print("📊 晨報產生中...")
    
    # 取得資料
    trade_date = get_trading_date()
    fundamental_data = fetch_fundamental(trade_date) if trade_date else {}
    
    # 取得技術面
    technical_data = {}
    for code in WATCHLIST:
        data = fetch_technical(code)
        if data:
            technical_data[code] = data
    
    # 分析
    results = []
    for code, info in WATCHLIST.items():
        result = analyze_stock(code, info, fundamental_data, technical_data)
        results.append(result)
    
    # 按基本面分數排序
    results.sort(key=lambda x: x["fund_score"], reverse=True)
    
    return results, trade_date

def format_report(results, trade_date):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    msg = f"📊 **投資晨報** {now}\n"
    msg += f"資料日期: {trade_date}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 基本面摘要
    msg += "**【基本面 + 技術面】**\n"
    msg += "```\n"
    msg += f"{'代碼':<6} {'名稱':<8} {'價格':>10} {'P/E':>6} {'殖利率':>7} {'評分':>5} {'技術':<6}\n"
    msg += f"{'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*7} {'-'*5} {'-'*6}\n"
    
    for r in results:
        pe = str(r['PE'])[:6] if r['PE'] else 'N/A'
        y = str(r['yield'])[:7] if r['yield'] else 'N/A'
        trend_icon = "📈" if r['trend'] == "多頭" else "📉" if r['trend'] == "空頭" else "➡️"
        msg += f"{r['code']:<6} {r['name']:<8} {str(r['price']):>10} {pe:>6} {y:>7} {r['grade']:<5} {trend_icon} {r['trend']:<5}\n"
    
    msg += "```\n"
    
    # 技術面燈號
    msg += "**【技術面燈號】**\n"
    for r in results:
        if not r.get('rsi'):
            continue
        
        signals = []
        
        # RSI
        if r['rsi'] > 70:
            signals.append("RSI超買")
        elif r['rsi'] < 30:
            signals.append("RSI超賣")
        
        # MACD
        if r['macd']:
            if r['macd'] > 0:
                signals.append("MACD多")
            else:
                signals.append("MACD空")
        
        signal_str = " / ".join(signals) if signals else "中性"
        
        msg += f"• {r['code']} {r['name']}: {signal_str}\n"
    
    # 策略建議
    msg += "\n**【策略建議】**\n"
    
    # 價值投資
    value_picks = [r for r in results if r['fund_pct'] >= 60 and r['trend'] == '多頭']
    if value_picks:
        picks_str = ', '.join([f"{r['code']}{r['name']}" for r in value_picks[:3]])
        msg += f"價值 + 多頭: {picks_str}\n"
    
    # 低估藍籌
    bluechip = [r for r in results if r['fund_pct'] >= 50 and r['code'] in ['2330', '2317']]
    if bluechip:
        blue_str = ', '.join([f"{r['code']}{r['name']}" for r in bluechip[:3]])
        msg += f"藍籌關注: {blue_str}\n"
    
    msg += f"\n_每日 08:00 自動更新_"
    
    return msg

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    results, trade_date = generate_report()
    msg = format_report(results, trade_date)
    
    print(msg)
    
    # 保存
    with open(f"{FUNDAMENTAL_DIR}/morning_report.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "trade_date": trade_date,
            "stocks": results
        }, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 晨報完成")

if __name__ == "__main__":
    main()