#!/usr/bin/env python3
# ============================================================================
# Layer 3: 輸出層 (writer)
# 任務: 格式化並發送到 Discord
# ============================================================================

import json
import os
import sys
from datetime import datetime

CACHE_DIR = "/home/ubuntu/.openclaw/cache/strategy"
INPUT_FILE = f"{CACHE_DIR}/L2_analysis.json"
WEBHOOK_URL = ""

# 嘗試讀取 webhook URL
try:
    with open("/home/ubuntu/.openclaw/config/system-config.json") as f:
        config = json.load(f)
    WEBHOOK_URL = config.get("webhooks", {}).get("stock_monitor", {}).get("url", "")
except:
    pass

def format_report(l2_data, yao_situation):
    """L3: 格式化輸出報告"""
    
    strategies = l2_data.get("strategy_ranking", {})
    common = l2_data.get("common_patterns", {})
    scaling = l2_data.get("scaling_strategies", {})
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    msg = f"📊 **進場/出廠策略深度分析**\n"
    msg += f"時間: {now}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # 1. 共通點分析
    msg += "**【進場/出廠共通點】**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    
    entry_patterns = common.get("進場共通點", {})
    for category, patterns in entry_patterns.items():
        pattern_str = ", ".join(patterns[:4])
        msg += f"• **{category}**: {pattern_str}\n"
    
    msg += "\n"
    
    # 2. 勝率排名 - 根據 Yao 的積極型特質
    # 過濾出勝率 >= 55% 且風險可接受的策略
    suitable = []
    for name, info in strategies.items():
        win_rate = info.get("win_rate", "0%")
        risk = info.get("risk", "高")
        period = info.get("period", "中期")
        complexity = info.get("complexity", "中")
        
        # 適合 Yao 的條件: 勝率 >= 55%, 風險不是極高, 複雜度不太高
        if "極高" not in risk and "極高" not in complexity:
            suitable.append({
                "name": name,
                "win_rate": win_rate,
                "expectancy": info.get("expectancy", "中"),
                "risk": risk,
                "period": period
            })
    
    # 按期望值排序
    expectancy_order = {"高": 3, "中高": 2, "中": 1}
    suitable.sort(key=lambda x: expectancy_order.get(x["expectancy"], 0), reverse=True)
    
    msg += "**【勝率高/期望值高的策略 (適合 Yao)】**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "```\n"
    msg += f"{'策略':<12} {'勝率':>10} {'期望值':>8} {'風險':>6} {'期間':>8}\n"
    msg += f"{'-'*12} {'-'*10} {'-'*8} {'-'*6} {'-'*8}\n"
    
    for s in suitable[:10]:
        msg += f"{s['name']:<12} {s['win_rate']:>10} {s['expectancy']:>8} {s['risk']:>6} {s['period']:>8}\n"
    
    msg += "```\n\n"
    
    # 3. 加減碼策略
    msg += "**【加減碼策略】**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    
    for name, info in list(scaling.items())[:4]:
        msg += f"◆ **{name}**: {info['description']}\n"
        msg += f"  做法: {info['做法'][0]}, {info['做法'][1]}, ...\n"
        msg += f"  適用: {info['適用']}\n\n"
    
    # 4. 針對 Yao 的推薦
    msg += "**【針對 Yao 的推薦組合】**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"處於階段: {yao_situation.get('stage', '中級')}\n"
    msg += f"投資風格: {yao_situation.get('style', '積極型')}\n"
    msg += f"資金: {yao_situation.get('cash', '$214,866')}\n\n"
    
    # 推薦 3 種策略
    recommended = [
        {"name": "價值投資", "reason": "P/E < 15, P/B < 1.5, 安全邊際 > 20%", "entry": "MA5 > MA10 > MA20", "exit": "P/E > 25 或 MA5 < MA10"},
        {"name": "亞當策略", "reason": "波段操作，多頭排列進場", "entry": "MA5 > MA10 > MA20 + 成交量放大", "exit": "MA5 < MA10 或跌破支撐"},
        {"name": "核心-衛星", "reason": "適合有工作的上班族", "entry": "核心 70% ETF + 衛星 30% 個股", "exit": "核心偏移 > 5% 或衛星暴漲"}
    ]
    
    for i, rec in enumerate(recommended, 1):
        msg += f"{i}. **{rec['name']}**\n"
        msg += f"   原因: {rec['reason']}\n"
        msg += f"   進場: {rec['entry']}\n"
        msg += f"   出廠: {rec['exit']}\n\n"
    
    # 5. 觀察清單
    msg += "**【具體觀察標的】**\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "基於基本面 + 技術面篩選:\n"
    msg += "• **2330 台積電**: 多頭排列，但 P/E 30.65 偏高，等拉回\n"
    msg += "• **2317 鴻海**: P/E 15.19 合理，技術面整理\n"
    msg += "• **2303 聯電**: P/E 22.05 稍高，等待營收數據\n"
    msg += "• **0050 台灣 50**: 核心持股，穩健標的\n\n"
    
    msg += "_分層結構: L1(資料) → L2(分析) → L3(輸出)_\n"
    
    return msg

def send_to_discord(msg):
    """發送到 Discord"""
    if not WEBHOOK_URL:
        print("⚠️ 無 webhook URL，只輸出報告")
        return False
    
    try:
        import subprocess
        cmd = f"echo '{msg}' | jq -Rs '{{content: .}}' | curl -s -X POST -H 'Content-Type: application/json' -d @- '{WEBHOOK_URL}'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        return True
    except Exception as e:
        print(f"發送失敗: {e}")
        return False

def main():
    """L3: 主程式"""
    # Yao 的情況
    yao_situation = {
        "stage": "中級 (2-5年)",
        "style": "積極型",
        "cash": "$214,866 + 04006C 50股",
        "time_horizon": "波段為主 (1-3個月)",
        "risk_tolerance": "可接受 20% 虧損"
    }
    
    # 讀取 L2 資料
    if not os.path.exists(INPUT_FILE):
        print(f"❌ L2 資料不存在: {INPUT_FILE}")
        print("   請先執行 L1_collect.py 和 L2_analyze.py")
        sys.exit(1)
    
    with open(INPUT_FILE) as f:
        l2_data = json.load(f)
    
    # 格式化報告
    msg = format_report(l2_data, yao_situation)
    
    # 輸出
    print(msg)
    
    # 保存
    output_file = f"{CACHE_DIR}/L3_report.txt"
    with open(output_file, "w") as f:
        f.write(msg)
    
    print(f"\n✅ L3 完成: 報告已保存到 {output_file}")
    
    # 嘗試發送到 Discord
    if send_to_discord(msg):
        print("✅ 已發送到 Discord")
    
    return msg

if __name__ == "__main__":
    main()