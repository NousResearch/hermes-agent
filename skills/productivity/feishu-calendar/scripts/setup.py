#!/usr/bin/env python3
"""
飞书日历技能 - 设置脚本
检查配置状态和授权状态

用法：
    python setup.py --check      # 检查配置和授权状态
    python setup.py --guide      # 显示配置指南
"""
import os
import sys
import json

HERMES_HOME = os.path.expanduser("~/.hermes")
ENV_FILE = os.path.join(HERMES_HOME, ".env")
TOKEN_FILE = os.path.join(HERMES_HOME, ".feishu_user_token.json")

def check_env():
    """检查环境变量配置"""
    if not os.path.exists(ENV_FILE):
        return False, "NOT_CONFIGURED", f".env 文件不存在：{ENV_FILE}"
    
    creds = {}
    with open(ENV_FILE, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                creds[key] = value
    
    required = ["FEISHU_APP_ID", "FEISHU_APP_SECRET"]
    missing = [k for k in required if not creds.get(k)]
    
    if missing:
        return False, "NOT_CONFIGURED", f"缺少环境变量：{', '.join(missing)}"
    
    # 检查是否为占位符
    app_id = creds.get("FEISHU_APP_ID", "")
    app_secret = creds.get("FEISHU_APP_SECRET", "")
    
    if app_id == "your_app_id" or app_secret == "your_app_secret":
        return False, "NOT_CONFIGURED", "请替换为实际的 App ID 和 App Secret"
    
    if app_id.startswith("cli_") and len(app_id) >= 18:
        return True, "ENV_OK", "环境变量配置正确"
    else:
        return False, "NOT_CONFIGURED", "FEISHU_APP_ID 格式不正确（应为 cli_xxxxxxxxxxxxx）"

def check_token():
    """检查 OAuth token"""
    if not os.path.exists(TOKEN_FILE):
        return False, "NOT_AUTHENTICATED", "未找到 OAuth token"
    
    try:
        with open(TOKEN_FILE, "r") as f:
            token_data = json.load(f)
        
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        created_at = token_data.get("created_at", 0)
        expires_in = token_data.get("expires_in", 7200)
        refresh_expires_in = token_data.get("refresh_expires_in", 2592000)
        
        import time
        now = int(time.time())
        
        # 检查 access_token
        access_expires_at = created_at + expires_in
        if now >= access_expires_at:
            # access_token 过期，检查 refresh_token
            refresh_expires_at = created_at + refresh_expires_in
            if now >= refresh_expires_at:
                return False, "TOKEN_EXPIRED", "refresh_token 已过期，需要重新授权"
            else:
                return True, "NEED_REFRESH", f"access_token 已过期，可自动刷新（refresh_token 剩余 {(refresh_expires_at - now) // 86400} 天）"
        else:
            remaining = (access_expires_at - now) // 60
            return True, "AUTHENTICATED", f"已授权（access_token 剩余 {remaining} 分钟）"
    
    except Exception as e:
        return False, "TOKEN_ERROR", f"Token 文件损坏：{e}"

def cmd_check():
    """检查配置状态"""
    print("\n📊 飞书日历技能配置检查\n")
    
    # 检查环境变量
    env_ok, env_status, env_msg = check_env()
    print(f"1. 环境变量配置：{env_status}")
    print(f"   {env_msg}")
    
    if not env_ok:
        print("\n❌ 请先配置环境变量")
        print("   在 ~/.hermes/.env 中添加：")
        print("   FEISHU_APP_ID=cli_xxxxxxxxxxxxx")
        print("   FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx")
        print("   FEISHU_REDIRECT_URI=http://127.0.0.1:18080/callback")
        return
    
    # 检查 token
    token_ok, token_status, token_msg = check_token()
    print(f"\n2. OAuth 授权状态：{token_status}")
    print(f"   {token_msg}")
    
    if token_status == "NOT_AUTHENTICATED":
        print("\n⚠️ 需要授权")
        print("   运行：python3 feishu_oauth.py generate_link")
    elif token_status == "NEED_REFRESH":
        print("\n✅ 已授权（会自动刷新）")
    elif token_status == "AUTHENTICATED":
        print("\n✅ 配置完成，可以使用")
    elif token_status == "TOKEN_EXPIRED":
        print("\n⚠️ Token 已过期")
        print("   运行：python3 feishu_oauth.py generate_link")
    
    print()

def cmd_guide():
    """显示配置指南"""
    guide = """
📖 飞书日历技能配置指南

═══════════════════════════════════════════════════

第一步：飞书开放平台配置

1. 访问 https://open.feishu.cn/app
2. 创建自定义应用（或编辑现有应用）
3. 记录 App ID 和 App Secret

第二步：配置权限

在 权限管理 中添加：
- calendar:calendar:readonly
- calendar:calendar

第三步：配置重定向 URL

在 安全设置 中添加：
http://127.0.0.1:18080/callback

第四步：配置环境变量

编辑 ~/.hermes/.env，添加：
FEISHU_APP_ID=cli_xxxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx
FEISHU_REDIRECT_URI=http://127.0.0.1:18080/callback

第五步：OAuth 授权

运行：python3 feishu_oauth.py generate_link
复制链接到浏览器，登录飞书并同意授权
复制 code 值，运行：python3 feishu_oauth.py exchange <code>

第六步：验证

运行：python3 setup.py --check
应显示 AUTHENTICATED

═══════════════════════════════════════════════════
"""
    print(guide)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    if "--check" in sys.argv:
        cmd_check()
    elif "--guide" in sys.argv:
        cmd_guide()
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
