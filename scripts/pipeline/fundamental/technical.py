#!/usr/bin/env python3
# ============================================================================
# Technical Analysis - 技術面分析
# 計算均線、RSI、MACD 等指標
# ============================================================================

import json
import urllib.request
from datetime import datetime, timedelta

def get_trading_dates(count=60):
    """取得最近 N 個交易日"""
    dates = []
    d = datetime.now()
    while len(dates) < count:
        if d.weekday() < 5:  # 平日
            dates.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return dates

def fetch_price_history(code, days=60):
    """抓取個股歷史價格"""
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=&eventDate=&stockNo={code}&response=json"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        # 解析價格資料
        price_data = []
        for row in data.get("data10", [])[1:]:  # 跳過標題列
            if len(row) >= 11:
                try:
                    date = row[0]
                    close = float(row[2].replace(",", ""))
                    volume = int(row[8].replace(",", ""))
                    price_data.append({
                        "date": date,
                        "close": close,
                        "volume": volume
                    })
                except:
                    pass
        
        return price_data[-days:]  # 只取最近 N 天
        
    except Exception as e:
        print(f"抓取 {code} 價格失敗: {e}")
        return []

def calc_ma(prices, period):
    """計算移動平均線"""
    if len(prices) < period:
        return None
    return sum(p["close"] for p in prices[-period:]) / period

def calc_rsi(prices, period=14):
    """計算 RSI"""
    if len(prices) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i]["close"] - prices[i-1]["close"]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return None
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calc_macd(prices, fast=12, slow=26, signal=9):
    """計算 MACD"""
    if len(prices) < slow:
        return None, None, None
    
    # 計算 EMA
    def calc_ema(period, prices):
        ema = prices[0]["close"]
        k = 2 / (period + 1)
        for p in prices[1:]:
            ema = p["close"] * k + ema * (1 - k)
        return ema
    
    ema_fast = calc_ema(fast, prices)
    ema_slow = calc_ema(slow, prices)
    macd_line = ema_fast - ema_slow
    
    # Signal line 是 MACD 的 EMA (簡化版)
    signal_line = macd_line * 0.9  # 簡化
    
    return macd_line, signal_line, ema_fast

def analyze_technical(code):
    """技術面分析"""
    prices = fetch_price_history(code, 60)
    
    if len(prices) < 30:
        return None
    
    # 計算均線
    ma5 = calc_ma(prices, 5)
    ma10 = calc_ma(prices, 10)
    ma20 = calc_ma(prices, 20)
    ma60 = calc_ma(prices, 60) if len(prices) >= 60 else None
    
    # 計算 RSI
    rsi = calc_rsi(prices, 14)
    
    # 計算 MACD
    macd, signal, ema12 = calc_macd(prices)
    
    # 當前價格
    current_price = prices[-1]["close"] if prices else None
    
    # 趨勢判斷
    trend = "整理"
    if all([ma5, ma10, ma20]):
        if current_price > ma5 > ma10 > ma20:
            trend = "多頭"
        elif current_price < ma5 < ma10 < ma20:
            trend = "空頭"
    
    # RSI 判斷
    rsi_status = "中性"
    if rsi:
        if rsi > 70:
            rsi_status = "超買"
        elif rsi < 30:
            rsi_status = "超賣"
    
    # MACD 判斷
    macd_status = "中性"
    if macd and signal:
        if macd > signal:
            macd_status = "多方"
        elif macd < signal:
            macd_status = "空方"
    
    # 綜合評估
    signals = []
    if ma5 and current_price:
        if current_price > ma5:
            signals.append("✅ 價格站上 MA5")
        else:
            signals.append("❌ 價格跌破 MA5")
    
    if ma10 and current_price:
        if current_price > ma10:
            signals.append("✅ 價格站上 MA10")
        else:
            signals.append("❌ 價格跌破 MA10")
    
    if rsi:
        if rsi > 70:
            signals.append("⚠️ RSI 超買")
        elif rsi < 30:
            signals.append("💡 RSI 超賣")
    
    if macd and signal:
        if macd > signal:
            signals.append("✅ MACD 金叉")
        elif macd < signal:
            signals.append("❌ MACD 死叉")
    
    return {
        "code": code,
        "price": current_price,
        "ma5": round(ma5, 2) if ma5 else None,
        "ma10": round(ma10, 2) if ma10 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "rsi": round(rsi, 1) if rsi else None,
        "macd": round(macd, 2) if macd else None,
        "macd_signal": round(signal, 2) if signal else None,
        "trend": trend,
        "rsi_status": rsi_status,
        "macd_status": macd_status,
        "signals": signals
    }

def main():
    print("=" * 60)
    print("📈 技術面分析報告")
    print(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    stocks = ["2330", "2317", "2454", "2303", "3008"]
    results = []
    
    for code in stocks:
        print(f"\n分析 {code}...")
        result = analyze_technical(code)
        if result:
            results.append(result)
    
    # 輸出結果
    print("\n" + "=" * 60)
    print(f"{'代碼':<6} {'現在價':>10} {'MA5':>10} {'MA10':>10} {'MA20':>10} {'RSI':>6} {'趨勢':<6}")
    print("-" * 70)
    
    for r in results:
        price = r["price"]
        ma5 = r["ma5"]
        ma10 = r["ma10"]
        ma20 = r["ma20"]
        rsi = r["rsi"]
        
        print(f"{r['code']:<6} {price:>10.2f} {ma5:>10.2f} {ma10:>10.2f} {ma20:>10.2f} {rsi:>6.1f} {r['trend']:<6}")
    
    print("\n" + "-" * 60)
    print("📊 詳細訊號")
    print("-" * 60)
    
    for r in results:
        print(f"\n【{r['code']}】")
        for signal in r["signals"]:
            print(f"  {signal}")
        print(f"  RSI: {r['rsi']} ({r['rsi_status']})")
        print(f"  MACD: {r['macd']} ({r['macd_status']})")

if __name__ == "__main__":
    main()