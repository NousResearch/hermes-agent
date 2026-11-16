#!/usr/bin/env python3
"""由 pdr/config.yaml 生成管理端「智能体配置 / 系统配置」页使用的表单数据。

输入：config.yaml 中每个配置项上方的 `#:` 元数据行（中文名｜说明｜范围:…｜生效:…）与当前取值；
hermes 段内以 `# ---- 分组名 ----` 注释分组，PDR 自身的各顶层段各为一组。
输出：pdr/web/admin/src/data/config-schema.json。后端完成后由 /api/admin/v1/config/schema 返回同样结构，
示例数据模式下管理端直接读取这个文件。修改 config.yaml 的配置项或元数据后重新运行本脚本：

    uv run python pdr/scripts/gen_admin_config_schema.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "pdr" / "config.yaml"
OUT = ROOT / "pdr" / "web" / "admin" / "src" / "data" / "config-schema.json"

# PDR 自身各段的中文分组名（页面左侧分组列表）
PDR_SECTIONS = {
    "app": "系统基本信息",
    "admin_web": "管理端",
    "portal_web": "用户端",
    "install": "安装与下载源",
    "database": "数据库",
    "redis": "Redis",
    "storage": "文件存储",
    "security": "安全与登录",
    "mineru": "MinerU 解析",
    "media": "音视频处理",
    "pipeline": "内容流水线",
    "knowledge": "知识库",
    "agent_runtime": "智能体运行",
    "logging": "日志",
}

META = re.compile(r"^\s*#:\s*(?P<body>.+)$")
GROUP = re.compile(r"^  # ---- (?P<name>.+?) ----\s*$")
KEY = re.compile(r"^(?P<indent>\s*)(?P<key>\"[^\"]+\"|[A-Za-z0-9_.\-]+):(?:\s|$)")
NUM_RANGE = re.compile(r"^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$")


def parse_meta(body: str) -> dict:
    parts = [p.strip() for p in body.split("｜")]
    meta = {"label": parts[0], "desc": parts[1] if len(parts) > 1 else "", "range": "", "reload": ""}
    for part in parts[2:]:
        if part.startswith("范围:"):
            meta["range"] = part[3:].strip()
        elif part.startswith("生效:"):
            meta["reload"] = part[3:].strip()
    return meta


def lookup(data, path: list[str]):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def control(meta: dict, value) -> dict:
    """由取值范围推断表单控件：开关、下拉、数字、JSON 或文本。"""
    rng, desc = meta["range"], meta["desc"]
    if rng == "true/false":
        return {"kind": "bool"}
    match = NUM_RANGE.match(rng)
    if match:
        return {"kind": "number", "min": float(match.group(1)), "max": float(match.group(2))}
    if rng in ("整数", "小数"):
        return {"kind": "number"}
    if rng in ("列表", "键值对") or isinstance(value, (list, dict)):
        return {"kind": "json"}
    if "|" in rng and " " not in rng:
        options = []
        for opt in rng.split("|"):
            found = re.search(r"(?:^|[\s：:，,])" + re.escape(opt) + r"-([^\s，,；;]+)", desc)
            options.append({"value": opt, "label": f"{opt} · {found.group(1)}" if found else opt})
        return {"kind": "enum", "options": options}
    return {"kind": "text", "nullable": rng == "可空"}


def build() -> dict:
    text = CONFIG.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    groups: list[dict] = []
    index: dict[str, dict] = {}
    stack: list[tuple[int, str]] = []
    pending: dict | None = None
    hermes_group = "通用"

    def group_for(key: str, name: str, scope: str) -> dict:
        if key not in index:
            index[key] = {"key": key, "name": name, "scope": scope, "items": []}
            groups.append(index[key])
        return index[key]

    for line in text.splitlines():
        if m := GROUP.match(line):
            if stack and stack[0][1] == "hermes":
                hermes_group = m.group("name")
            continue
        if m := META.match(line):
            pending = parse_meta(m.group("body"))
            continue
        if line.lstrip().startswith("#") or not line.strip():
            continue
        m = KEY.match(line)
        if not m:
            continue
        indent = len(m.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m.group("key").strip('"')))
        if pending is None:
            continue
        path = [k for _, k in stack]
        value = lookup(data, path)
        item = {"path": ".".join(path), **pending, **control(pending, value), "value": value}
        top = path[0]
        if top == "hermes":
            group_for("hermes:" + hermes_group, hermes_group, "agent")["items"].append(item)
        else:
            group_for(top, PDR_SECTIONS.get(top, top), "pdr")["items"].append(item)
        pending = None

    return {
        "source": "pdr/config.yaml",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "itemQty": sum(len(g["items"]) for g in groups),
        "groups": groups,
    }


def main() -> None:
    schema = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pdr = sum(1 for g in schema["groups"] if g["scope"] == "pdr")
    print(f"已生成 {OUT.relative_to(ROOT)}：{schema['itemQty']} 项，"
          f"PDR {pdr} 组，智能体 {len(schema['groups']) - pdr} 组")


if __name__ == "__main__":
    main()
