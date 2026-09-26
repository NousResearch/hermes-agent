#!/usr/bin/env python3
"""
AI 足迹 Agent API 工具脚本 — 零配置，装完即用

用法：
  python footprints.py me [--json]
  python footprints.py search <query> [--limit <n>] [--json]
  python footprints.py list [--category-id <id>] [--limit <n>] [--offset <n>] [--json]
  python footprints.py get <id> [--json]
  python footprints.py add <url> --title <title> [--description <desc>] [--category-ids <ids>] [--tags <tags>] [--json]
  python footprints.py update <id> [--title <t>] [--description <d>] [--category-ids <ids>] [--tags <tags>] [--json]
  python footprints.py batch-update '<json-array>' [--json]
  python footprints.py categories [--json]
  python footprints.py create-category <name> [--category-set-id <id>] [--json]
  python footprints.py tags [--json]
  python footprints.py category-sets [--json]
  python footprints.py create-category-set <name> [--json]
  python footprints.py copy <id> --category-ids <ids> [--json]
  python footprints.py shared-categories [--json]
  python footprints.py create-shared-category <name> --mode <cocreate|subscribe> [--color <hex>] [--description <desc>] [--json]
  python footprints.py join-shared-category <code> [--json]
  python footprints.py add-to-shared <sc_id> --collection-id <id> [--json]
  python footprints.py remove-from-shared <sc_id> --collection-id <id> [--json]
  python footprints.py create-invite-link <sc_id> [--duration-hours <h>] [--json]
  python footprints.py agent_register [--json]
  python footprints.py agent_magic_link [--json]

Token 管理：
  - 优先读环境变量 FOOTPRINTS_TOKEN
  - 其次读 profile-scoped 的 .token 文件
  - 都没有则需要运行 agent_register 获取新 token（需用户明确同意）
"""
import os
import sys
import json
import urllib.request
import urllib.error

# ── 零配置自动引导 ──

# Hermes profile-scoped token path: $HERMES_HOME/profiles/<profile>/.footprints_token
# Falls back to ~/.hermes/profiles/default/.footprints_token
def _get_hermes_token_dir():
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    profile = os.environ.get("HERMES_PROFILE", "default")
    return os.path.join(home, "profiles", profile)

_HERMES_DIR = _get_hermes_token_dir()
TOKEN_FILE = os.path.join(_HERMES_DIR, ".footprints_token")
ENDPOINT = os.environ.get("FOOTPRINTS_ENDPOINT", "https://ai.ocean94.com")
# 安全：仅允许白名单内的端点，防止环境变量注入劫持网络请求
_ALLOWED_ENDPOINTS = {
    "https://ai.ocean94.com",
    "http://192.168.0.88:8080",
    "http://localhost:8000",
}
_ep = ENDPOINT.rstrip("/")
if _ep not in _ALLOWED_ENDPOINTS:
    ENDPOINT = "https://ai.ocean94.com"

# 全局开关：是否以 JSON 模式输出
JSON_MODE = False


def _get_token():
    """读取 TOKEN：env → .token 文件。无 token 返回空字符串，不触发自动注册。"""
    token = os.environ.get("FOOTPRINTS_TOKEN", "")
    if token:
        return token

    # 尝试从 profile-scoped .token 文件读取
    try:
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
        if token:
            os.environ["FOOTPRINTS_TOKEN"] = token
            return token
    except FileNotFoundError:
        pass

    # 无 token — 调用者需先执行 agent_register（需用户明确同意）
    return ""


def _raw_api(path, method="GET", data=None, no_auth=False):
    """底层 HTTP 调用（不依赖 _get_token，避免循环）"""
    token = os.environ.get("FOOTPRINTS_TOKEN", "")
    url = f"{ENDPOINT.rstrip('/')}/api/v1/agent{path}"
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif not no_auth:
        return {"error": "无法获取 Token — 自动注册失败"}
    body = None
    if data:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        try:
            err = json.loads(err_body)
            return {"error": err.get("detail", str(e))}
        except:
            return {"error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"error": f"网络错误: {e}"}


def _output(result):
    """统一输出：JSON 模式打印 JSON，否则只返回数据由各函数自行 print"""
    if JSON_MODE:
        print(json.dumps(result, ensure_ascii=False))


def _echo(*args, **kwargs):
    """只在非 JSON 模式下打印（人类可读输出）"""
    if not JSON_MODE:
        print(*args, **kwargs)


def api(path, method="GET", data=None, no_auth=False):
    """API 调用 + token 自动管理"""
    token = _get_token()
    if not token and not no_auth:
        err = {"error": "未登录 — 请先运行 agent_register 创建账号（需要用户明确同意）"}
        _output(err)
        return err
    return _raw_api(path, method=method, data=data, no_auth=no_auth)


# ── 用户 ──

def me():
    result = api("/me")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"用户名: {result.get('username')}")
    _echo(f"昵称: {result.get('nickname', '')}")
    return result


# ── 分类集 ──

def category_sets():
    result = api("/category-sets")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    for s in result:
        tag = "🔗" if s.get("is_shared") else "📁"
        _echo(f"  {tag} [{s['id']}] {s['name']}")
    return result


def create_category_set(name):
    result = api("/category-sets", method="POST", data={"name": name})
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"✅ 分类集已创建: [{result['id']}] {result['name']}")
    return result


# ── 分类 ──

def categories():
    result = api("/categories")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    # 按分类集分组
    from collections import defaultdict
    grouped = defaultdict(list)
    for cat in result:
        cs_name = cat.get("category_set_name") or "其他"
        grouped[cs_name].append(cat)
    for cs_name, cats in grouped.items():
        n = len(cats)
        _echo(f"\n📁 分类集「{cs_name}」ID:{cats[0].get('category_set_id') or '?'} ({n}个分类)")
        for cat in cats:
            mode_tag = ""
            if cat.get("mode") == "cocreate":
                mode_tag = " [共创]"
            elif cat.get("mode") == "subscribe":
                mode_tag = " [订阅]"
            _echo(f"  [{cat['id']}] {cat['name']}{mode_tag}")
    return result


def create_category(name, category_set_id=None):
    body = {"name": name}
    if category_set_id is not None:
        body["category_set_id"] = category_set_id
    result = api("/categories", method="POST", data=body)
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    verb = "已创建" if result.get("created") else "已存在"
    _echo(f"✅ {verb}: [{result['id']}] {result['name']}")
    return result


# ── 标签 ──

def tags():
    result = api("/tags")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    for tag in result:
        _echo(f"  [{tag['id']}] {tag['name']}")
    return result


# ── 足迹 ──

def search(query, limit=20):
    import urllib.parse
    q = urllib.parse.quote(query)
    result = api(f"/collections?q={q}&limit={limit}")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    items = result.get("items", [])
    if not items:
        _echo("未找到匹配的足迹")
        return result
    for i, item in enumerate(items):
        _echo(f"{i+1}. {item['title']}")
        _echo(f"   {item['url']}")
        if item.get("description"):
            _echo(f"   {item['description'][:120]}")
        cats = ", ".join(item.get("category_names", []))
        if cats:
            _echo(f"   分类: {cats}")
        tags_list = ", ".join(item.get("tags", []))
        if tags_list:
            _echo(f"   标签: {tags_list}")
        _echo()
    return result


def list_collections(category_id=None, limit=20, offset=0):
    path = f"/collections?limit={limit}&offset={offset}"
    if category_id:
        path += f"&category_id={category_id}"
    result = api(path)
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    items = result.get("items", [])
    total = result.get("total", len(items))
    if not items:
        _echo("暂无足迹")
        return result
    _echo(f"共 {total} 条" + (f"，显示第 {offset+1}-{offset+len(items)} 条" if total > limit else ""))
    for i, item in enumerate(items):
        _echo(f"{i+1}. {item['title']}")
        _echo(f"   {item['url']}")
        cats = ", ".join(item.get("category_names", []))
        if cats:
            _echo(f"   分类: {cats}")
        _echo()
    return result


def get_collection(collection_id):
    """获取单条足迹详情"""
    result = api(f"/collections/{collection_id}")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"标题: {result.get('title')}")
    _echo(f"链接: {result.get('url')}")
    if result.get("description"):
        _echo(f"摘要: {result['description'][:200]}")
    cats = ", ".join(result.get("category_names", []))
    if cats:
        _echo(f"分类: {cats}")
    tags_list = ", ".join(result.get("tags", []))
    if tags_list:
        _echo(f"标签: {tags_list}")
    return result


def content_types():
    """列出当前用户使用过的所有 content_type"""
    result = api("/content-types")
    if not JSON_MODE and isinstance(result, list):
        for ct in result:
            _echo(f"  {ct['value']} ({ct['count']}条)")
    return result


def add(url, title=None, description=None, category_ids=None, tags=None, content_type=None):
    data = {"url": url}
    if title:
        data["title"] = title
    if description:
        data["description"] = description
    if category_ids:
        data["category_ids"] = [int(x) for x in category_ids.split(",")]
    if tags:
        data["tag_names"] = [t.strip() for t in tags.split(",")]
    if content_type:
        data["content_type"] = content_type
    result = api("/collections", method="POST", data=data)
    if "error" in result:
        _echo(f"❌ {result['error']}")
    else:
        _echo(f"✅ 已保存: {result.get('title', url)}")
        cats = ", ".join(result.get("category_names", []))
        if cats:
            _echo(f"   分类: {cats}")
    return result


def update_collection(collection_id, title=None, description=None, category_ids=None, tags=None, content_type=None):
    """更新足迹（全部字段可选，传 None 不修改）"""
    data = {}
    if title is not None:
        data["title"] = title
    if description is not None:
        data["description"] = description
    if category_ids is not None:
        data["category_ids"] = [int(x.strip()) for x in category_ids.split(",")]
    if tags is not None:
        data["tag_names"] = [t.strip() for t in tags.split(",") if t.strip()]
    if content_type is not None:
        data["content_type"] = content_type
    result = api(f"/collections/{collection_id}", method="PUT", data=data)
    if "error" in result:
        _echo(f"❌ {result['error']}")
    else:
        _echo(f"✅ 已更新: {result.get('title', collection_id)}")
    return result


def copy_collection(collection_id, category_ids):
    """将共享足迹复制到个人分类"""
    data = {"category_ids": [int(x) for x in category_ids.split(",")]}
    result = api(f"/collections/{collection_id}/copy", method="POST", data=data)
    if "error" in result:
        _echo(f"❌ {result['error']}")
    else:
        _echo(f"✅ 已复制到个人分类: {result.get('title', collection_id)}")
        cats = ", ".join(result.get("category_names", []))
        if cats:
            _echo(f"   分类: {cats}")
    return result


def batch_update_collections(updates_json):
    """批量更新足迹 — 接受 JSON 数组字符串"""
    try:
        updates = json.loads(updates_json)
    except json.JSONDecodeError as e:
        _echo(f"❌ JSON 解析失败: {e}")
        return {"error": str(e)}
    if not isinstance(updates, list):
        _echo("❌ 参数应为 JSON 数组")
        return {"error": "不是数组"}
    result = api("/collections/batch", method="PUT", data={"updates": updates})
    if "error" in result:
        _echo(f"❌ {result['error']}")
    else:
        _echo(f"✅ 成功 {result['ok']} 条" + (f"，失败 {result['errors']} 条" if result.get('errors') else ""))
        for e in result.get("error_details", []):
            _echo(f"   ❌ {e['id']}: {e['detail']}")
    return result


def show_help():
    _echo(__doc__)



# ── 共享分类 ──

def shared_categories():
    """列出共享分类（通过分类接口过滤 mode != null 的条目）"""
    result = api("/categories")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    shared = [c for c in result if c.get("mode")]
    if not shared:
        _echo("暂无共享分类")
        return shared
    for c in shared:
        mode_tag = "[共创]" if c.get("mode") == "cocreate" else "[订阅]"
        _echo(f"  [{c['id']}] {c['name']} {mode_tag}")
    return shared


def create_shared_category(name, mode, color=None, description=None):
    """创建共享分类"""
    data = {"name": name, "mode": mode}
    if color:
        data["color"] = color
    if description:
        data["description"] = description
    result = api("/shared-categories", method="POST", data=data)
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"✅ 共享分类已创建: [{result['id']}] {result['name']}")
    if result.get("invite_code"):
        _echo(f"   邀请码: {result['invite_code']}")
    if result.get("mode"):
        _echo(f"   模式: {result['mode']}")
    return result


def join_shared_category(code):
    """通过邀请码加入共享分类"""
    result = api("/shared-categories/join", method="POST", data={"code": code})
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"✅ 已加入共享分类")
    return result


def add_to_shared_category(sc_id, collection_id):
    """将足迹加入共享分类"""
    result = api(f"/shared-categories/{sc_id}/collections", method="POST",
                 data={"collection_id": str(collection_id)})
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"✅ 足迹已加入共享分类 [{sc_id}]")
    return result


def remove_from_shared_category(sc_id, collection_id):
    """将足迹移出共享分类"""
    result = api(f"/shared-categories/{sc_id}/collections/{collection_id}",
                 method="DELETE")
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"✅ 足迹已移出共享分类 [{sc_id}]")
    return result


def create_invite_link(sc_id, duration_hours=24):
    """为共享分类生成邀请链接，可发给人类或其他 Agent"""
    data = {}
    if duration_hours is not None:
        data["duration_hours"] = duration_hours
    result = api(f"/shared-categories/{sc_id}/invite-link", method="POST", data=data)
    if "error" in result:
        _echo(f"❌ {result['error']}")
        return result
    _echo(f"✅ 邀请链接已生成:")
    _echo(f"   URL: {result.get('url')}")
    if result.get("code"):
        _echo(f"   邀请码: {result['code']}")
    if result.get("expires_at"):
        _echo(f"   有效期至: {result['expires_at']}")
    _echo(f"\n📨 可将链接或邀请码发送给人类或其他 Agent，对方加入后即可协作")
    return result


def agent_register():
    """Agent 自主注册：创建新账号并返回访问令牌。需用户明确同意后才调用。"""
    # 直接用 _raw_api，不经过 api()→_get_token()，避免双重注册
    result = _raw_api("/register", method="POST", no_auth=True)
    if "error" in result:
        _echo(f"❌ 注册失败: {result['error']}")
        return result
    token = result["token"]
    # 持久化 token
    os.environ["FOOTPRINTS_TOKEN"] = token
    try:
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            f.write(token)
        if sys.platform != "win32":
            os.chmod(TOKEN_FILE, 0o600)
    except Exception:
        pass
    _echo(f"Token: {token}")
    _echo(f"\n{result.get('message', '')}")
    return result


def agent_magic_link():
    """生成一次性登录链接"""
    result = api("/magic-link", method="POST")
    if "error" in result:
        _echo(f"❌ 生成链接失败: {result['error']}")
        return result
    _echo(f"链接: {result['url']}")
    _echo(f"有效期: {result['expires_in']} 秒（{result['expires_in'] // 60} 分钟）")
    return result


if __name__ == "__main__":
    # 第一步：提取 --json 全局开关
    JSON_MODE = "--json" in sys.argv
    if JSON_MODE:
        sys.argv.remove("--json")

    if len(sys.argv) < 2:
        if JSON_MODE:
            print(json.dumps({"error": "缺少命令", "usage": "python footprints.py <me|search|list|get|add|update|...>"}))
        else:
            show_help()
        sys.exit(0)

    cmd = sys.argv[1]

    import argparse

    # 统一分发：每个分支执行后调用 _output(result) 输出 JSON
    if cmd == "me":
        result = me()
        _output(result)
    elif cmd == "category-sets":
        result = category_sets()
        _output(result)
    elif cmd == "create-category-set":
        if len(sys.argv) < 3:
            err = {"error": "用法: footprints.py create-category-set <名称>"}
            if not JSON_MODE: print(err["error"])
            _output(err)
            sys.exit(1)
        result = create_category_set(sys.argv[2])
        _output(result)
    elif cmd == "categories":
        result = categories()
        _output(result)
    elif cmd == "create-category":
        parser = argparse.ArgumentParser()
        parser.add_argument("name")
        parser.add_argument("--category-set-id", type=int, default=None)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = create_category(args.name, args.category_set_id)
        _output(result)
    elif cmd == "tags":
        result = tags()
        _output(result)
    elif cmd == "search":
        parser = argparse.ArgumentParser()
        parser.add_argument("query", nargs="+")
        parser.add_argument("--limit", type=int, default=20)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = search(" ".join(args.query), limit=args.limit)
        _output(result)
    elif cmd == "list":
        parser = argparse.ArgumentParser()
        parser.add_argument("--category-id", type=int, default=None)
        parser.add_argument("--limit", type=int, default=20)
        parser.add_argument("--offset", type=int, default=0)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = list_collections(category_id=args.category_id, limit=args.limit, offset=args.offset)
        _output(result)
    elif cmd == "get":
        if len(sys.argv) < 3:
            err = {"error": "用法: footprints.py get <足迹ID>"}
            if not JSON_MODE: print(err["error"])
            _output(err)
            sys.exit(1)
        result = get_collection(sys.argv[2])
        _output(result)
    elif cmd == "add":
        parser = argparse.ArgumentParser()
        parser.add_argument("url", help="URL to save")
        parser.add_argument("--title", default=None)
        parser.add_argument("--description", default=None)
        parser.add_argument("--category-ids", default=None)
        parser.add_argument("--tags", default=None)
        parser.add_argument("--content-type", default=None, choices=["article","video","image","audio","page"])
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = add(args.url, args.title, args.description, args.category_ids, args.tags, args.content_type)
        _output(result)
    elif cmd == "update":
        parser = argparse.ArgumentParser()
        parser.add_argument("id")
        parser.add_argument("--title", default=None)
        parser.add_argument("--description", default=None)
        parser.add_argument("--category-ids", default=None)
        parser.add_argument("--tags", default=None)
        parser.add_argument("--content-type", default=None, choices=["article","video","image","audio","page"])
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = update_collection(args.id, args.title, args.description, args.category_ids, args.tags, args.content_type)
        _output(result)
    elif cmd == "copy":
        parser = argparse.ArgumentParser()
        parser.add_argument("id")
        parser.add_argument("--category-ids", required=True)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = copy_collection(args.id, args.category_ids)
        _output(result)
    elif cmd == "batch-update":
        if len(sys.argv) < 3:
            err = {"error": "用法: footprints.py batch-update '<json-array>'"}
            if not JSON_MODE: print(err["error"])
            _output(err)
            sys.exit(1)
        result = batch_update_collections(sys.argv[2])
        _output(result)
    elif cmd == "agent_register":
        result = agent_register()
        _output(result)
    elif cmd == "agent_magic_link":
        result = agent_magic_link()
        _output(result)
    elif cmd == "shared-categories":
        result = shared_categories()
        _output(result)
    elif cmd == "create-shared-category":
        parser = argparse.ArgumentParser()
        parser.add_argument("name")
        parser.add_argument("--mode", required=True, choices=["subscribe", "cocreate"])
        parser.add_argument("--color", default=None)
        parser.add_argument("--description", default=None)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = create_shared_category(args.name, args.mode, args.color, args.description)
        _output(result)
    elif cmd == "join-shared-category":
        if len(sys.argv) < 3:
            err = {"error": "用法: footprints.py join-shared-category <邀请码>"}
            if not JSON_MODE: print(err["error"])
            _output(err)
            sys.exit(1)
        result = join_shared_category(sys.argv[2])
        _output(result)
    elif cmd == "add-to-shared":
        parser = argparse.ArgumentParser()
        parser.add_argument("sc_id", type=int)
        parser.add_argument("--collection-id", required=True)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = add_to_shared_category(args.sc_id, args.collection_id)
        _output(result)
    elif cmd == "remove-from-shared":
        parser = argparse.ArgumentParser()
        parser.add_argument("sc_id", type=int)
        parser.add_argument("--collection-id", required=True)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = remove_from_shared_category(args.sc_id, args.collection_id)
        _output(result)
    elif cmd == "create-invite-link":
        parser = argparse.ArgumentParser()
        parser.add_argument("sc_id", type=int)
        parser.add_argument("--duration-hours", type=int, default=24)
        args, _ = parser.parse_known_args(sys.argv[2:])
        result = create_invite_link(args.sc_id, args.duration_hours)
        _output(result)
    elif cmd == "content-types":
        result = content_types()
        _output(result)
    elif cmd in ("help", "-h", "--help"):
        show_help()
    else:
        err = {"error": f"未知命令: {cmd}"}
        if not JSON_MODE: print(err["error"]); show_help()
        _output(err)
        sys.exit(1)