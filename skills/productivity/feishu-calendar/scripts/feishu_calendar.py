#!/usr/bin/env python3
"""
飞书日历 API 调用脚本（使用 OAuth user_access_token）
支持：创建日程、读取日程、删除日程

用法：
    python feishu_calendar.py create --title "会议" --start "2026-04-16 09:00" --end "2026-04-16 10:00"
    python feishu_calendar.py list --date "2026-04-16"
    python feishu_calendar.py delete --event_id "xxx"

输出：JSON 格式
"""
import json
import sys
import os
import time
import requests
from datetime import datetime
import subprocess

# OAuth token 文件
TOKEN_FILE = os.path.expanduser("~/.hermes/.feishu_user_token.json")

def get_user_access_token(auto_refresh=True):
    """
    获取有效的 user_access_token（自动刷新）
    """
    if not os.path.exists(TOKEN_FILE):
        return None
    
    with open(TOKEN_FILE, "r") as f:
        token_data = json.load(f)
    
    now = int(time.time())
    created_at = token_data.get("created_at", 0)
    expires_in = token_data.get("expires_in", 7200)
    refresh_expires_in = token_data.get("refresh_expires_in", 2592000)
    
    access_expires_at = created_at + expires_in - 300  # 提前 5 分钟
    refresh_expires_at = created_at + refresh_expires_in - 3600
    
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    
    # access_token 还有效
    if now < access_expires_at:
        return access_token
    
    # 需要刷新
    if now >= refresh_expires_at:
        return None  # refresh_token 也过期了
    
    # 自动刷新
    if auto_refresh:
        try:
            result = subprocess.run(
                ["python3", os.path.join(os.path.dirname(__file__), "feishu_oauth.py"), "refresh"],
                capture_output=True,
                text=True
            )
            # 重新读取
            with open(TOKEN_FILE, "r") as f:
                token_data = json.load(f)
            return token_data.get("access_token")
        except:
            return None
    
    return None

def get_calendar_id(access_token):
    """获取用户主日历 ID"""
    url = "https://open.feishu.cn/open-apis/calendar/v4/calendars"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    result = response.json()
    if result.get("code") == 0:
        for cal in result["data"].get("calendar_list", []):
            if cal.get("type") == "primary":
                return cal["calendar_id"]
        if result["data"].get("calendar_list"):
            return result["data"]["calendar_list"][0]["calendar_id"]
        raise Exception("没有找到日历")
    else:
        raise Exception(f"获取日历失败：{result}")

def create_event(access_token, calendar_id, title, start_time, end_time, description=""):
    """创建日程"""
    url = f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{calendar_id}/events"
    
    time_formats = ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
    
    start_dt = None
    end_dt = None
    
    for fmt in time_formats:
        try:
            start_dt = datetime.strptime(start_time, fmt)
            break
        except ValueError:
            continue
    
    for fmt in time_formats:
        try:
            end_dt = datetime.strptime(end_time, fmt)
            break
        except ValueError:
            continue
    
    if not start_dt or not end_dt:
        return {"success": False, "error": f"时间格式错误，支持格式：YYYY-MM-DD HH:MM 或 YYYY-MM-DDTHH:MM"}
    
    payload = {
        "summary": title,
        "description": description,
        "start_time": {
            "timestamp": int(start_dt.timestamp()),
            "time_zone": "Asia/Shanghai"
        },
        "end_time": {
            "timestamp": int(end_dt.timestamp()),
            "time_zone": "Asia/Shanghai"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    result = response.json()
    
    if result.get("code") == 0:
        event = result.get("data", {}).get("event", {})
        return {
            "success": True, 
            "event_id": event.get("event_id", ""), 
            "message": "日程创建成功",
            "app_link": event.get("app_link", "")
        }
    else:
        return {"success": False, "error": result.get("msg", "未知错误"), "detail": result.get("data", {})}

def list_events(access_token, calendar_id, date):
    """读取日程"""
    url = f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{calendar_id}/events"
    
    start_dt = datetime.strptime(date, "%Y-%m-%d")
    end_dt = start_dt.replace(hour=23, minute=59, second=59)
    
    params = {
        "time_min": int(start_dt.timestamp()),
        "time_max": int(end_dt.timestamp())
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, params=params, headers=headers)
    result = response.json()
    
    if result.get("code") == 0:
        events = []
        for item in result.get("data", {}).get("items", []):
            events.append({
                "event_id": item.get("event_id"),
                "title": item.get("summary"),
                "start_time": item.get("start_time", {}).get("timestamp"),
                "end_time": item.get("end_time", {}).get("timestamp")
            })
        return {"success": True, "events": events}
    else:
        return {"success": False, "error": result.get("msg", "未知错误")}

def delete_event(access_token, calendar_id, event_id):
    """删除日程"""
    url = f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{calendar_id}/events/{event_id}"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.delete(url, headers=headers)
    result = response.json()
    
    if result.get("code") == 0:
        return {"success": True, "message": "日程删除成功"}
    else:
        return {"success": False, "error": result.get("msg", "未知错误")}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "用法：python feishu_calendar.py <command> [args]"}))
        return
    
    command = sys.argv[1]
    
    try:
        # 获取 user_access_token（自动刷新）
        token = get_user_access_token(auto_refresh=True)
        if not token:
            print(json.dumps({
                "success": False, 
                "error": "OAuth token 无效或已过期，请运行：python3 feishu_oauth.py generate_link"
            }, ensure_ascii=False))
            return
        
        if command == "create":
            args = {}
            for i in range(2, len(sys.argv), 2):
                if i+1 < len(sys.argv):
                    key = sys.argv[i].lstrip("-")
                    args[key] = sys.argv[i+1]
            
            calendar_id = get_calendar_id(token)
            result = create_event(
                token,
                calendar_id,
                args.get("title", "无标题"),
                args.get("start", ""),
                args.get("end", ""),
                args.get("description", "")
            )
            print(json.dumps(result, ensure_ascii=False))
        
        elif command == "list":
            date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y-%m-%d")
            calendar_id = get_calendar_id(token)
            result = list_events(token, calendar_id, date)
            print(json.dumps(result, ensure_ascii=False))
        
        elif command == "delete":
            event_id = sys.argv[2] if len(sys.argv) > 2 else ""
            if not event_id:
                print(json.dumps({"success": False, "error": "需要 event_id"}))
                return
            calendar_id = get_calendar_id(token)
            result = delete_event(token, calendar_id, event_id)
            print(json.dumps(result, ensure_ascii=False))
        
        else:
            print(json.dumps({"success": False, "error": f"未知命令：{command}"}))
    
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    main()