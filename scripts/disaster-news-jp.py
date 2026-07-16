#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
災害・安全保障ニュース速報スクリプト
気象庁地震情報JSONから過去1時間の震度3以上の地震を抽出しTelegramに通知
"""
import urllib.request
import json
from datetime import datetime, timezone, timedelta

def fetch_jma_quake_list():
    url = "https://www.jma.go.jp/bosai/quake/data/list.json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; HermesAgent/1.0)'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        return data
    except Exception as e:
        return {"error": f"JMA取得失敗: {e}"}

def parse_quakes(data):
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    results = []
    if isinstance(data, dict) and "error" in data:
        return results
    if not isinstance(data, list):
        return results
    for entry in data:
        try:
            time_str = entry.get("time") or entry.get("at")  # some entries have 'time', others 'at'
            if not time_str:
                continue
            # normalize to ISO format with timezone
            if time_str.endswith('Z'):
                time_str = time_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(time_str)
            if dt < one_hour_ago:
                continue
            # Determine magnitude and max intensity (shindo)
            magnitude = "不明"
            max_scale = 0  # 震度
            name = "不明"
            depth = "不明"
            latitude = None
            longitude = None
            # Earthquake and Seismic Intensity Information (VXSE5k) or Earthquake Information (VXSE52)
            if "earthquake" in entry and isinstance(entry["earthquake"], dict):
                eq = entry["earthquake"]
                hypo = eq.get("hypocenter", {})
                name = hypo.get("name", "不明")
                depth = hypo.get("depth", "不明")
                magnitude = hypo.get("magnitude", "不明")
                latitude = hypo.get("latitude")
                longitude = hypo.get("longitude")
                max_scale = eq.get("maxScale", 0)
                # If maxScale is empty string, treat as 0
                if isinstance(max_scale, str):
                    if max_scale.isdigit():
                        max_scale = int(max_scale)
                    else:
                        max_scale = 0
            # Seismic Intensity Information (VXSE51) - no earthquake object but has maxi at top
            elif "maxi" in entry and entry["maxi"] not in ("", None):
                try:
                    max_scale = int(entry["maxi"])
                except (ValueError, TypeError):
                    max_scale = 0
                # For intensity-only, we may not have magnitude/location; keep defaults
            # else skip
            if max_scale >= 3:
                results.append({
                    "time": dt,
                    "name": name,
                    "depth": depth,
                    "magnitude": magnitude,
                    "max_scale": max_scale,
                    "latitude": latitude,
                    "longitude": longitude,
                })
        except Exception:
            # Skip problematic entries
            continue
    # 新しい順にソート
    results.sort(key=lambda x: x["time"], reverse=True)
    return results

def format_jst(dt_utc):
    jst = dt_utc.astimezone(timezone(timedelta(hours=9)))
    return jst.strftime('%Y-%m-%d %H:%M JST')

def main():
    data = fetch_jma_quake_list()
    quakes = parse_quakes(data)
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    lines = []
    lines.append(f"【災害・安全保障速報｜{now_jst.strftime('%Y-%m-%d %H:%M JST')}]")
    lines.append("")
    lines.append("■ 判定")
    if quakes:
        lines.append(f"- 確認済み速報: {len(quakes)}件（過去1時間以内、震度3以上）")
        for i, q in enumerate(quakes[:5], 1):  # 上位5件表示
            name = q['name']
            if len(name) > 20:
                name = name[:17] + "..."
            mag = q['magnitude']
            depth = q['depth']
            scale = q['max_scale']
            time_str = format_jst(q['time'])
            lat = q['latitude']
            lon = q['longitude']
            loc = f"{name}"
            if lat is not None and lon is not None:
                loc += f" ({lat:.2f}°, {lon:.2f}°)"
            lines.append(f"  {i}. {loc} M{mag} 深さ{depth}km 震度{scale} ({time_str})")
        if len(quakes) > 5:
            lines.append(f"  他 {len(quakes)-5} 件")
    else:
        lines.append("- 確認済み速報: なし（過去1時間以内に震度3以上の地震なし）")
    lines.append(f"- 根拠時刻: {now_jst.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append("")
    lines.append("■ 影響")
    if quakes:
        lines.append("- 地震が発生しています。津波情報にもご注意ください。")
    else:
        lines.append("- 現在のところ、注目すべき地震はありません。")
    lines.append("")
    lines.append("■ 次に取る行動")
    lines.append("- 最新の情報は気象庁ウェブサイト等でご確認ください。")
    message = "\n".join(lines)
    print(message)

if __name__ == "__main__":
    main()