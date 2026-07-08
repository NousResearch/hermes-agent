#!/usr/bin/env python3
"""三模型第十人入口脚本 — 小青 + DS + Qwen + 火山 四方审计

用法:
  python3 triple_tenth_man.py --scheme-file /path/to/scheme.md --topic "主题"

三路并行调用:
  1. DS (DeepSeek V4 Pro)
  2. Qwen (CLOUD-PLATFORM qwen-plus, enable_search)
  3. Volc (火山 2.0 Pro, Responses API + web_search)

产出: ~/kb/审计/{topic}_审计_{ds|qwen|volc}_{日期}.md
成本: ~¥0.20/次 (DS ¥0.05 + Qwen ¥0.07 + Volc ¥0.08)
"""

import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Prompt 模板（复用 dual_tenth_man_audit.py 的设计）──
PROMPT_TEMPLATE = """# 你的任务

你是外部第十人——一个独立审计员。USER_X委派你对下面这个方案进行独立的第十人审计。

## 你要做什么

找一个漏洞——任何一个被忽略的失败模式、逻辑跳跃、未标注的假设、被遗漏的替代方案。

## 方案全文

{scheme_content}

## 环境快照

- HOST 单机 (Ubuntu 24.04)，无多机部署
- SSD-only 操作，NAS 仅做备份
- DS Flash + DS V4 Pro 双模型分层

## 用户偏好快照

- USER_X：架构师视角，先穷举再收敛，逐条过不跳步，准确度优先速度
- 不喜欢给未来留坑，发现立刻修，不写 TODO

## 审计要求

1. 逐方案/逐节点找漏洞和盲区
2. 找被遗漏的替代方案
3. 标注置信度 🔴🟡🟢⚪
4. 输出 Markdown 完整报告，不要省略

格式:
# 外部第十人审计：{topic}
## 决策链理解
## 节点审计（假设→反方→置信度）
## 盲区发现
## 被遗漏的替代方案
## 修正建议
## 总置信度"""


def load_env_key(var_name):
    """从 <config-dir>/.env 读 API key"""
    with open(os.path.expanduser('<config-dir>/.env')) as f:
        for line in f:
            if line.strip().startswith('#'):
                continue
            if var_name in line:
                return line.split('=', 1)[1].strip().strip('"')
    raise KeyError(f'{var_name} not found in .env')


def load.provider_b_key():
    """从 <config-dir>/.provider_b_key 读CLOUD-PLATFORM key"""
    with open(os.path.expanduser('<config-dir>/.provider_b_key')) as f:
        return f.read().strip()


def call_ds(scheme_content, topic):
    """DeepSeek V4 Pro — Chat Completions"""
    start = time.time()
    try:
        key = load_env_key('API_KEY_PROVIDER_A')
        prompt = PROMPT_TEMPLATE.format(scheme_content=scheme_content, topic=topic)
        payload = json.dumps({
            'model': 'deepseek-chat',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 6000,
            'temperature': 0.7,
        }).encode()
        req = urllib.request.Request(
            'https://api.deepseek.com/v1/chat/completions',
            data=payload, method='POST',
        )
        req.add_header('Authorization', f'Bearer {key}')
        req.add_header('Content-Type', 'application/json')
        resp = urllib.request.urlopen(req, timeout=300)
        content = json.loads(resp.read())['choices'][0]['message']['content']
        return True, content, time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def call_qwen(scheme_content, topic):
    """CLOUD-PLATFORM Qwen — Chat Completions + enable_search"""
    start = time.time()
    try:
        key = load.provider_b_key()
        prompt = PROMPT_TEMPLATE.format(scheme_content=scheme_content, topic=topic)
        payload = json.dumps({
            'model': 'qwen-plus',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 6000,
            'temperature': 0.7,
            'enable_search': True,
        }).encode()
        req = urllib.request.Request(
            'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            data=payload, method='POST',
        )
        req.add_header('Authorization', f'Bearer {key}')
        req.add_header('Content-Type', 'application/json')
        resp = urllib.request.urlopen(req, timeout=300)
        content = json.loads(resp.read())['choices'][0]['message']['content']
        return True, content, time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def call_volc(scheme_content, topic):
    """火山 2.0 Pro — Responses API + web_search"""
    start = time.time()
    try:
        key = load_env_key('VOLC_API_KEY')
        prompt = PROMPT_TEMPLATE.format(scheme_content=scheme_content, topic=topic)
        payload = json.dumps({
            'model': 'doubao-seed-2-0-pro-260215',
            'input': [{'role': 'user', 'content': prompt}],
            'tools': [{'type': 'web_search', 'max_keyword': 5}],
            'max_output_tokens': 8192,
        }).encode()
        req = urllib.request.Request(
            'https://ark.cn-beijing.volces.com/api/v3/responses',
            data=payload, method='POST',
        )
        req.add_header('Authorization', f'Bearer {key}')
        req.add_header('Content-Type', 'application/json')
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        output_texts = []
        for item in result.get('output', []):
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') == 'output_text':
                        output_texts.append(c['text'])
        content = '\n'.join(output_texts)
        return True, content, time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def main():
    parser = argparse.ArgumentParser(description='三模型第十人审计入口')
    parser.add_argument('--scheme-file', required=True, help='方案文档 .md 路径')
    parser.add_argument('--topic', required=True, help='审计主题')
    args = parser.parse_args()

    # 读方案全文
    with open(args.scheme_file) as f:
        scheme_content = f.read()

    date_str = datetime.now().strftime('%Y-%m-%d')
    outdir = os.path.expanduser('~/local/USER/审计')
    os.makedirs(outdir, exist_ok=True)

    # 安全文件名
    safe_topic = args.topic.replace('/', '-').replace(' ', '_')
    prefix = f'{safe_topic}_审计'

    print(f'🔍 三模型第十人审计启动: {args.topic}')
    print(f'   方案: {args.scheme_file} ({len(scheme_content)} chars)')
    print()

    # 三路并行
    tasks = {
        'ds': call_ds,
        'qwen': call_qwen,
        'volc': call_volc,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn, scheme_content, args.topic): name
                   for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            ok, content, dur = future.result()
            outfile = os.path.join(outdir, f'{prefix}_{name}_{date_str}.md')
            header = f'# {name.upper()} 外部第十人审计：{args.topic}\n\n> 模型: {name} | 时间: {datetime.now().strftime("%Y-%m-%d %H:%M")} | 耗时: {dur:.0f}s\n> 独立审计，未接触其他审计报告\n\n'

            if ok:
                with open(outfile, 'w') as f:
                    f.write(header + content)
                print(f'  ✅ {name:5s} ({dur:3.0f}s, {len(content):5d} chars) → {outfile}')
                results[name] = True
            else:
                with open(outfile, 'w') as f:
                    f.write(header + f'❌ FAILED\n\n{content}')
                print(f'  ❌ {name:5s} ({dur:3.0f}s) → {outfile}')
                results[name] = False

    print()
    ok_count = sum(results.values())
    print(f'完成: {ok_count}/3 成功 ({len(scheme_content)} chars 方案, ~¥0.20)')
    sys.exit(0 if ok_count >= 2 else 1)


if __name__ == '__main__':
    main()

# @hermes:patch 2026-07-06 | session:20260705_192249_4675c803
