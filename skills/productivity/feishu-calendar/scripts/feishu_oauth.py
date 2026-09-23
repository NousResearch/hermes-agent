#!/usr/bin/env python3
"""
飞书 OAuth Token 管理模块
- 自动生成授权链接
- 用 code 换取 token
- 自动刷新 access_token（2 小时过期）
- 自动刷新 refresh_token（30 天过期）

用法：
    python feishu_oauth.py generate_link          # 生成授权链接
    python feishu_oauth.py exchange <code>        # 用 code 换取 token
    python feishu_oauth.py refresh                # 手动刷新 token
    python feishu_oauth.py status                 # 查看 token 状态
    python feishu_oauth.py ensure_valid           # 确保 token 有效（自动刷新）
"""
import os
import sys
import json
import requests
import time
from datetime import datetime

TOKEN_FILE = os.path.expanduser("~/.hermes/.feishu_user_token.json")
ENV_FILE = os.path.expanduser("~/.hermes/.env")

def get_credentials():
    """从.env 文件读取凭证"""
    creds = {}
    with open(ENV_FILE, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                creds[key] = value
    return creds

def get_app_access_token(app_id, app_secret):
    """获取 app_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
    payload = {"app_id": app_id, "app_secret": app_secret}
    resp = requests.post(url, json=payload)
    result = resp.json()
    if result.get("code") == 0:
        return result.get("app_access_token")
    else:
        raise Exception(f"获取 app_access_token 失败：{result}")

def generate_auth_url(app_id, redirect_uri="http://127.0.0.1:18080/callback"):
    """生成授权链接"""
    state = f"hermes_oauth_{int(time.time())}"
    # 添加日历权限 scope
    scope = "calendar:calendar:readonly calendar:calendar"
    auth_url = (
        f"https://open.feishu.cn/open-apis/authen/v1/authorize"
        f"?app_id={app_id}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&response_type=code"
        f"&scope={scope}"
    )
    return auth_url, state

def exchange_code_for_token(app_id, app_secret, app_access_token, code, redirect_uri="http://127.0.0.1:18080/callback"):
    """用 authorization code 换取 user_access_token"""
    url = "https://open.feishu.cn/open-apis/authen/v1/oidc/access_token"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri
    }
    headers = {
        "Authorization": f"Bearer {app_access_token}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, json=payload, headers=headers)
    result = resp.json()
    if result.get("code") == 0:
        return result.get("data", {})
    else:
        raise Exception(f"换取 token 失败：{result}")

def refresh_user_token(app_id, app_secret, app_access_token, refresh_token):
    """用 refresh_token 刷新 access_token"""
    url = "https://open.feishu.cn/open-apis/authen/v1/oidc/refresh_access_token"
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    headers = {
        "Authorization": f"Bearer {app_access_token}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, json=payload, headers=headers)
    result = resp.json()
    if result.get("code") == 0:
        return result.get("data", {})
    else:
        raise Exception(f"刷新 token 失败：{result}")

def load_token():
    """加载保存的 token"""
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE, "r") as f:
        return json.load(f)

def save_token(token_data):
    """保存 token 到文件"""
    token_data["updated_at"] = int(time.time())
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    os.chmod(TOKEN_FILE, 0o600)
    return True

def get_user_access_token(auto_refresh=True):
    """
    获取有效的 user_access_token
    
    Args:
        auto_refresh: 是否自动刷新（默认 True）
    
    Returns:
        access_token 字符串，如果失败返回 None
    """
    creds = get_credentials()
    app_id = creds.get("FEISHU_APP_ID", "")
    app_secret = creds.get("FEISHU_APP_SECRET", "")
    
    token_data = load_token()
    if not token_data:
        print("❌ 未找到 OAuth token，请先授权")
        print("运行：python3 feishu_oauth.py generate_link")
        return None
    
    now = int(time.time())
    created_at = token_data.get("created_at", 0)
    expires_in = token_data.get("expires_in", 7200)  # 默认 2 小时
    refresh_expires_in = token_data.get("refresh_expires_in", 2592000)  # 默认 30 天
    
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    
    # 检查 access_token 是否过期（提前 5 分钟刷新）
    access_expires_at = created_at + expires_in - 300
    refresh_expires_at = created_at + refresh_expires_in - 3600  # 提前 1 小时
    
    if now < access_expires_at:
        # access_token 还有效
        return access_token
    
    if not auto_refresh:
        print("⚠️ access_token 已过期，需要刷新")
        return None
    
    # 需要刷新 access_token
    if now < refresh_expires_at:
        # refresh_token 还有效，自动刷新
        try:
            app_access_token = get_app_access_token(app_id, app_secret)
            new_token_data = refresh_user_token(app_id, app_secret, app_access_token, refresh_token)
            
            # 更新 token
            token_data["access_token"] = new_token_data.get("access_token")
            token_data["refresh_token"] = new_token_data.get("refresh_token")
            token_data["expires_in"] = new_token_data.get("expires_in")
            token_data["refresh_expires_in"] = new_token_data.get("refresh_expires_in")
            token_data["created_at"] = int(time.time())
            
            save_token(token_data)
            
            print(f"✅ access_token 已自动刷新（有效期 {new_token_data.get('expires_in')} 秒）")
            return new_token_data.get("access_token")
            
        except Exception as e:
            print(f"❌ 自动刷新失败：{e}")
            return None
    else:
        # refresh_token 也过期了，需要重新授权
        print("❌ refresh_token 已过期，需要重新授权")
        print("运行：python3 feishu_oauth.py generate_link")
        return None

def cmd_generate_link():
    """生成授权链接"""
    creds = get_credentials()
    app_id = creds.get("FEISHU_APP_ID", "")
    
    if not app_id:
        print("❌ 请在 ~/.hermes/.env 中设置 FEISHU_APP_ID")
        return
    
    auth_url, state = generate_auth_url(app_id)
    
    print("\n🔐 飞书 OAuth 授权链接\n")
    print("=" * 80)
    print(auth_url)
    print("=" * 80)
    print(f"\n步骤：")
    print(f"1. 复制链接到浏览器打开")
    print(f"2. 登录飞书并同意授权")
    print(f"3. 授权后 URL 会包含 code=xxx")
    print(f"4. 运行：python3 feishu_oauth.py exchange <code>")

def cmd_exchange(code):
    """用 code 换取 token"""
    creds = get_credentials()
    app_id = creds.get("FEISHU_APP_ID", "")
    app_secret = creds.get("FEISHU_APP_SECRET", "")
    
    if not app_id or not app_secret:
        print("❌ 请在 ~/.hermes/.env 中设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        return
    
    try:
        print("\n🔄 正在换取 token...")
        app_access_token = get_app_access_token(app_id, app_secret)
        token_data = exchange_code_for_token(app_id, app_secret, app_access_token, code)
        
        # 保存 token
        save_data = {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "expires_in": token_data.get("expires_in"),
            "refresh_expires_in": token_data.get("refresh_expires_in"),
            "created_at": int(time.time())
        }
        save_token(save_data)
        
        print(f"\n✅ 授权成功！")
        print(f"   access_token 有效期：{token_data.get('expires_in')} 秒（约 {token_data.get('expires_in', 0) // 3600} 小时）")
        print(f"   refresh_token 有效期：{token_data.get('refresh_expires_in')} 秒（约 {token_data.get('refresh_expires_in', 0) // 86400} 天）")
        print(f"\n后续创建日历事件将自动使用并刷新此 token")
        
    except Exception as e:
        print(f"❌ 换取 token 失败：{e}")

def cmd_refresh():
    """手动刷新 token"""
    token_data = load_token()
    if not token_data:
        print("❌ 未找到 token，请先授权")
        return
    
    creds = get_credentials()
    app_id = creds.get("FEISHU_APP_ID", "")
    app_secret = creds.get("FEISHU_APP_SECRET", "")
    
    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        print("❌ 未找到 refresh_token")
        return
    
    try:
        print("\n🔄 正在刷新 token...")
        app_access_token = get_app_access_token(app_id, app_secret)
        new_token_data = refresh_user_token(app_id, app_secret, app_access_token, refresh_token)
        
        # 更新
        token_data["access_token"] = new_token_data.get("access_token")
        token_data["refresh_token"] = new_token_data.get("refresh_token")
        token_data["expires_in"] = new_token_data.get("expires_in")
        token_data["refresh_expires_in"] = new_token_data.get("refresh_expires_in")
        token_data["created_at"] = int(time.time())
        
        save_token(token_data)
        
        print(f"\n✅ Token 刷新成功！")
        print(f"   新 access_token 有效期：{new_token_data.get('expires_in')} 秒")
        
    except Exception as e:
        print(f"❌ 刷新失败：{e}")

def cmd_status():
    """查看 token 状态"""
    token_data = load_token()
    if not token_data:
        print("❌ 未找到 OAuth token")
        print("运行：python3 feishu_oauth.py generate_link")
        return
    
    now = int(time.time())
    created_at = token_data.get("created_at", 0)
    expires_in = token_data.get("expires_in", 7200)
    refresh_expires_in = token_data.get("refresh_expires_in", 2592000)
    
    access_expires_at = created_at + expires_in
    refresh_expires_at = created_at + refresh_expires_in
    
    access_remaining = access_expires_at - now
    refresh_remaining = refresh_expires_at - now
    
    print("\n📊 OAuth Token 状态\n")
    print(f"access_token:")
    print(f"   剩余有效期：{access_remaining // 60} 分钟")
    print(f"   过期时间：{datetime.fromtimestamp(access_expires_at).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   状态：{'✅ 有效' if access_remaining > 0 else '❌ 已过期'}")
    print(f"\nrefresh_token:")
    print(f"   剩余有效期：{refresh_remaining // 86400} 天")
    print(f"   过期时间：{datetime.fromtimestamp(refresh_expires_at).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   状态：{'✅ 有效' if refresh_remaining > 0 else '❌ 已过期'}")

def cmd_ensure_valid():
    """确保 token 有效（用于其他脚本调用）"""
    token = get_user_access_token(auto_refresh=True)
    if token:
        # 只输出 token，供其他脚本使用
        print(token)
        return True
    return False

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    
    if command == "generate_link":
        cmd_generate_link()
    elif command == "exchange":
        if len(sys.argv) < 3:
            print("用法：python3 feishu_oauth.py exchange <code>")
            return
        cmd_exchange(sys.argv[2])
    elif command == "refresh":
        cmd_refresh()
    elif command == "status":
        cmd_status()
    elif command == "ensure_valid":
        cmd_ensure_valid()
    else:
        print(f"❌ 未知命令：{command}")
        print(__doc__)

if __name__ == "__main__":
    main()
